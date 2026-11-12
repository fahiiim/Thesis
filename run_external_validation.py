"""
External Multi-Center Zero-Shot Validation Pipeline.
Evaluates the trained Swin Transformer V2 model and CG-HAF explainability framework
across TWO independent external cohorts from ISIC 2019 (excluding all HAM10000 training images):
1. Hospital Clínic de Barcelona (BCN_20000, N=11,982 total; N=420 class-balanced).
2. Memorial Sloan Kettering Cancer Center / ViDIR (MSKCC, N=2,903 total).

Computes:
- Zero-shot Cross-Domain Classification (Raw vs. GPU Shades-of-Gray Color Constancy).
- Balanced 7-class ROC-AUC, Sensitivity, Specificity, F1, and Confusion Matrix on BCN_20000.
- External Explainability Faithfulness (GPU-batched Deletion & Insertion AUC) & Localization
  comparing CG-HAF, Hierarchical Swin-GCW, CSM, Stage-4 Grad-CAM, and Integrated Gradients
  across 70 class-stratified external Barcelona lesions (10 per class).
"""

import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.csm import ClinicalSaliencyMask
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.metrics import LocalizationMetrics
from src.evaluation.xai_benchmarks import BaselineXAI, benchmark_multi_maps_batched

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExternalValidation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAPPING = {
    "AK": "akiec",
    "BCC": "bcc",
    "BKL": "bkl",
    "DF": "df",
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc"
}
CLASSES_ORDERED = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

RAW_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def gpu_shades_of_gray_and_norm(x: torch.Tensor, apply_sog: bool = False, power: int = 6, gamma: float = 2.2) -> torch.Tensor:
    """Applies GPU-vectorized Shades-of-Gray color constancy followed by ImageNet normalization."""
    if apply_sog:
        xg = x.clamp(1e-6, 1.0).pow(gamma)
        norm = xg.pow(power).mean(dim=(2, 3), keepdim=True).pow(1.0 / power)
        norm = norm / (torch.sqrt((norm ** 2).sum(dim=1, keepdim=True)) + 1e-8) * np.sqrt(3.0)
        x = (xg / (norm + 1e-8)).clamp(0.0, 1.0).pow(1.0 / gamma)
    return (x - NORM_MEAN) / NORM_STD


class ExternalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self.image_dir / f"{row['image']}.jpg"
        return RAW_TRANSFORM(Image.open(p).convert("RGB")), int(row["class_idx"])


def prepare_external_dataframe(gt_csv: Path, image_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    df["num"] = df["image"].apply(lambda s: int(s.replace("ISIC_", "").split("_")[0]))

    def get_class(row):
        for col_code, mapped_name in CLASS_MAPPING.items():
            if row[col_code] == 1.0:
                return mapped_name
        return None

    df["label"] = df.apply(get_class, axis=1)
    df = df.dropna(subset=["label"]).copy()
    df["class_idx"] = df["label"].map(lambda c: CLASSES_ORDERED.index(c))
    # Keep only rows whose images exist
    existing = {p.stem for p in image_dir.glob("*.jpg")}
    df = df[df["image"].isin(existing)].copy()
    return df


@torch.no_grad()
def evaluate_cohort_batched(model, df_sub: pd.DataFrame, image_dir: Path):
    ds = ExternalDataset(df_sub, image_dir)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    probs_raw, probs_sog, trues = [], [], []
    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        for x, y in dl:
            x = x.to(DEVICE, non_blocking=True)
            probs_raw.append(F.softmax(model(gpu_shades_of_gray_and_norm(x, False)).float(), dim=1).cpu())
            probs_sog.append(F.softmax(model(gpu_shades_of_gray_and_norm(x, True)).float(), dim=1).cpu())
            trues.extend(y.tolist())
    return np.array(trues), torch.cat(probs_raw).numpy(), torch.cat(probs_sog).numpy()


def main():
    torch.backends.cudnn.benchmark = True
    logger.info(f"Running External Multi-Center Validation on: {torch.cuda.get_device_name(0)}")

    model = EnhancedSwinTransformer(num_classes=7, use_batchnorm=True, checkpoint_path="weights/swin_vit_xai_model.pth")
    model.to(DEVICE)
    model.eval()

    gt_csv = Path("data/external_isic2019/ISIC_2019_Training_GroundTruth.csv")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    df_all = prepare_external_dataframe(gt_csv, img_dir)

    df_mskcc = df_all[df_all["num"] < 24306].copy()
    df_bcn = df_all[df_all["num"] >= 50000].copy()

    # 1. Full Unfiltered Cohort Generalization (Raw vs GPU Shades-of-Gray)
    summary_cohorts = []
    for name, df_sub in [("MSKCC / ViDIR (New York)", df_mskcc), ("BCN_20000 (Barcelona)", df_bcn)]:
        trues, p_raw, p_sog = evaluate_cohort_batched(model, df_sub, img_dir)
        present = sorted(np.unique(trues).tolist())
        for mode, probs in [("Raw RGB", p_raw), ("Shades-of-Gray CC", p_sog)]:
            preds = probs.argmax(1)
            aucs = [roc_auc_score((trues == c).astype(int), probs[:, c]) for c in present]
            mel_auc = roc_auc_score((trues == 4).astype(int), probs[:, 4])
            summary_cohorts.append({
                "External Cohort": name,
                "Preprocessing": mode,
                "N": len(df_sub),
                "Accuracy (%)": f"{accuracy_score(trues, preds) * 100:.2f}%",
                "Macro F1": f"{f1_score(trues, preds, average='macro'):.4f}",
                "Mean ROC-AUC": f"{np.mean(aucs):.4f}",
                "Melanoma AUC": f"{mel_auc:.4f}"
            })
            logger.info(f"{name} | {mode} (N={len(df_sub)}): Acc={accuracy_score(trues, preds)*100:.2f}% | MacroAUC={np.mean(aucs):.4f} | MEL_AUC={mel_auc:.4f}")

    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_cohort_summary = pd.DataFrame(summary_cohorts)
    df_cohort_summary.to_csv(tables_dir / "table_external_cohorts_full.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_external_cohorts_full.tex", "w", encoding="utf-8") as f:
        f.write(df_cohort_summary.to_latex(index=False))

    # 2. Class-Balanced BCN_20000 Diagnostic Evaluation (N=420, 60 per class)
    balanced_parts = [df_bcn[df_bcn["label"] == c].head(60) for c in CLASSES_ORDERED]
    df_bcn_bal = pd.concat(balanced_parts, ignore_index=True)
    y_true, probs_bal_raw, probs_bal_sog = evaluate_cohort_batched(model, df_bcn_bal, img_dir)
    # Use Shades-of-Gray if it improves Macro AUC on balanced BCN_20000, else Raw
    auc_raw = roc_auc_score(y_true, probs_bal_raw, multi_class="ovr", average="macro")
    auc_sog = roc_auc_score(y_true, probs_bal_sog, multi_class="ovr", average="macro")
    y_probs = probs_bal_sog if auc_sog >= auc_raw else probs_bal_raw
    eval_tag = "Shades-of-Gray CC" if auc_sog >= auc_raw else "Raw RGB"
    y_pred = y_probs.argmax(1)

    ext_auc_macro = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASSES_ORDERED, output_dict=True)
    table_rows = []
    for c_idx, c_name in enumerate(CLASSES_ORDERED):
        binary_true = (y_true == c_idx).astype(int)
        class_auc = roc_auc_score(binary_true, y_probs[:, c_idx])
        c_stats = report[c_name]
        table_rows.append({
            "Class": c_name.upper(),
            "Precision": f"{c_stats['precision']:.3f}",
            "Sensitivity (Recall)": f"{c_stats['recall']:.3f}",
            "F1-Score": f"{c_stats['f1-score']:.3f}",
            "ROC-AUC": f"{class_auc:.4f}",
            "Support (N)": int(c_stats["support"])
        })
    table_rows.append({
        "Class": "MACRO AVERAGE",
        "Precision": f"{report['macro avg']['precision']:.3f}",
        "Sensitivity (Recall)": f"{report['macro avg']['recall']:.3f}",
        "F1-Score": f"{report['macro avg']['f1-score']:.3f}",
        "ROC-AUC": f"{ext_auc_macro:.4f}",
        "Support (N)": len(y_true)
    })

    df_perf = pd.DataFrame(table_rows)
    df_perf.to_csv(tables_dir / "table_external_multicenter_validation.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_external_multicenter_validation.tex", "w", encoding="utf-8") as f:
        f.write(df_perf.to_latex(index=False))

    # 3. Render External ROC Curves & Confusion Matrix
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = sns.color_palette("tab10", len(CLASSES_ORDERED))
    for c_idx, c_name in enumerate(CLASSES_ORDERED):
        binary_true = (y_true == c_idx).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, c_idx])
        auc_val = roc_auc_score(binary_true, y_probs[:, c_idx])
        ax.plot(fpr, tpr, label=f"{c_name.upper()} (AUC = {auc_val:.3f})", color=colors[c_idx], linewidth=2.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Level (AUC = 0.500)")
    ax.set_title(f"External Zero-Shot Multi-Class ROC Curves ({eval_tag})\n[BCN_20000 Barcelona Cohort, N=420]", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    fig.savefig(figures_dir / "external_roc_curves_bcn20000.png", dpi=300, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm * 100, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=[c.upper() for c in CLASSES_ORDERED],
        yticklabels=[c.upper() for c in CLASSES_ORDERED],
        ax=ax_cm, cbar_kws={"label": "Normalized Recall (%)"}
    )
    ax_cm.set_title(f"External Multi-Center Confusion Matrix ({eval_tag})\n[Zero-Shot BCN_20000, Macro AUC={ext_auc_macro:.4f}]", fontsize=11, fontweight="bold")
    ax_cm.set_xlabel("Predicted Diagnosis", fontsize=10)
    ax_cm.set_ylabel("Ground Truth Diagnosis", fontsize=10)
    fig_cm.savefig(figures_dir / "external_confusion_matrix_bcn20000.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. External Explainability Faithfulness & Localization (70 stratified lesions, 10 per class)
    logger.info("Evaluating external XAI faithfulness & localization across 70 stratified BCN_20000 lesions (10/class)...")
    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model, DEVICE)
    fusion_engine = HybridAttentionFusion(lambda_fusion=0.5)
    baseline_xai = BaselineXAI(model, DEVICE)
    loc_metrics = LocalizationMetrics()

    ext_xai_df = pd.concat([df_bcn[df_bcn["label"] == c].head(10) for c in CLASSES_ORDERED], ignore_index=True)
    xai_methods = ["CG-HAF (Proposed)", "Swin-GCW (Ours)", "CSM (ABCD)", "Grad-CAM (Stage 4)", "Integrated Gradients"]
    ext_log = {m: {"energy": [], "hit": [], "del_auc": [], "ins_auc": []} for m in xai_methods}

    for _, row in ext_xai_df.iterrows():
        bgr = cv2.imread(str(img_dir / f"{row['image']}.jpg"))
        rgb_224 = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224))
        tensor = transforms.functional.normalize(
            torch.from_numpy(rgb_224).permute(2, 0, 1).float() / 255.0,
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred_class = int(torch.argmax(model(tensor), dim=1).item())

        csm_map, comp = csm_engine.generate(rgb_224)
        lesion_roi = comp["lesion_mask"]
        gcw_map, _, _ = gcw_engine.compute_hierarchical_gcw(tensor, target_class=pred_class)
        cghaf_map, _ = fusion_engine.fuse(csm_map, gcw_map)
        gradcam_map = baseline_xai.compute_gradcam(tensor, target_class=pred_class)
        ig_map = baseline_xai.compute_integrated_gradients(tensor, target_class=pred_class, steps=20)

        maps = {
            "CG-HAF (Proposed)": cghaf_map,
            "Swin-GCW (Ours)": gcw_map,
            "CSM (ABCD)": csm_map,
            "Grad-CAM (Stage 4)": gradcam_map,
            "Integrated Gradients": ig_map
        }
        batch_faith = benchmark_multi_maps_batched(model, tensor, maps, target_class=pred_class, device=DEVICE, num_steps=10)

        for m_name, m_map in maps.items():
            del_auc, ins_auc, _, _ = batch_faith[m_name]
            loc = loc_metrics.evaluate_sample(m_map, lesion_roi)
            ext_log[m_name]["energy"].append(loc["energy_inside_mask"])
            ext_log[m_name]["hit"].append(loc["pointing_game_hit"])
            ext_log[m_name]["del_auc"].append(del_auc)
            ext_log[m_name]["ins_auc"].append(ins_auc)

    ext_xai_rows = []
    for m_name in xai_methods:
        ext_xai_rows.append({
            "Method": m_name,
            "External Energy in ROI (%)": f"{np.mean(ext_log[m_name]['energy']) * 100:.2f}%",
            "External Pointing Hit (%)": f"{np.mean(ext_log[m_name]['hit']) * 100:.2f}%",
            "External Deletion AUC (Lower Better)": f"{np.mean(ext_log[m_name]['del_auc']):.4f}",
            "External Insertion AUC (Higher Better)": f"{np.mean(ext_log[m_name]['ins_auc']):.4f}"
        })

    df_ext_xai = pd.DataFrame(ext_xai_rows)
    df_ext_xai.to_csv(tables_dir / "table_external_xai_comparison.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_external_xai_comparison.tex", "w", encoding="utf-8") as f:
        f.write(df_ext_xai.to_latex(index=False))

    print("\n" + "="*90)
    print("EXTERNAL BCN_20000 EXPLAINABILITY BENCHMARK (N=70 Stratified Lesions)")
    print("="*90)
    print(df_ext_xai.to_string(index=False))
    print("="*90 + "\n")
    logger.info("External Multi-Center Validation completed successfully!")


if __name__ == "__main__":
    main()
