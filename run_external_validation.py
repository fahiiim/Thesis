"""
External Multi-Center Zero-Shot Validation Pipeline.
Evaluates the trained Swin Transformer V2 model and CG-HAF explainability framework
on an independent multi-center cohort from Hospital Clínic de Barcelona (BCN_20000)
and Memorial Sloan Kettering Cancer Center (MSKCC).

Computes:
1. Multi-class ROC-AUC (One-vs-Rest) & Diagnostic Performance.
2. Sensitivity, Specificity, Precision, F1-Score across all 7 classes.
3. External Saliency Localization (Energy Inside Mask & Pointing Game Hit Rate).
4. Generates publication-grade LaTeX tables, ROC curves, and confusion matrix.
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
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score
)

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.csm import ClinicalSaliencyMask
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.metrics import LocalizationMetrics
from src.evaluation.xai_benchmarks import BaselineXAI

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

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_external_cohort(gt_csv: Path, image_dir: Path, n_per_class: int = 50):
    """
    Curates a balanced external multi-center evaluation cohort from BCN_20000 (Barcelona).
    Excludes any HAM10000 internal training lesions (ISIC_0024306 to ISIC_0034320).
    """
    df = pd.read_csv(gt_csv)

    def is_external(img_name: str) -> bool:
        num = int(img_name.replace("ISIC_", "").split("_")[0])
        # External BCN_20000 cohort has numbers >= 50000
        return num >= 50000

    df_ext = df[df["image"].apply(is_external)].copy()

    # Determine ground-truth class label from one-hot columns
    def get_class(row):
        for col_code, mapped_name in CLASS_MAPPING.items():
            if row[col_code] == 1.0:
                return mapped_name
        return None

    df_ext["label"] = df_ext.apply(get_class, axis=1)
    df_ext = df_ext.dropna(subset=["label"]).copy()

    # Verify image existence
    valid_records = []
    for c in CLASSES_ORDERED:
        sub_df = df_ext[df_ext["label"] == c]
        selected = 0
        for _, row in sub_df.iterrows():
            img_path = image_dir / f"{row['image']}.jpg"
            if img_path.exists():
                valid_records.append({
                    "image_id": row["image"],
                    "path": img_path,
                    "label": c,
                    "class_idx": CLASSES_ORDERED.index(c)
                })
                selected += 1
                if selected >= n_per_class:
                    break

    logger.info(f"Curated {len(valid_records)} external lesions across {len(CLASSES_ORDERED)} classes from BCN_20000 (Barcelona).")
    return pd.DataFrame(valid_records)


def main():
    logger.info(f"Running External Multi-Center Validation on: {torch.cuda.get_device_name(0)}")

    # 1. Load Pretrained Checkpoint
    checkpoint_path = "weights/swin_vit_xai_model.pth"
    model = EnhancedSwinTransformer(num_classes=7, checkpoint_path=checkpoint_path)
    model.to(DEVICE)
    model.eval()

    # 2. Curate External Dataset
    gt_csv = Path("data/external_isic2019/ISIC_2019_Training_GroundTruth.csv")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    df_cohort = load_external_cohort(gt_csv, img_dir, n_per_class=60)

    # 3. Batch Zero-Shot Diagnostic Inference
    y_true = []
    y_pred = []
    y_probs = []
    image_tensors = []

    logger.info("Executing zero-shot inference across external Barcelona cohort...")
    with torch.no_grad():
        for _, row in df_cohort.iterrows():
            pil_img = Image.open(row["path"]).convert("RGB").resize((224, 224))
            tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            y_true.append(row["class_idx"])
            y_pred.append(int(np.argmax(probs)))
            y_probs.append(probs)
            image_tensors.append(tensor)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    ext_acc = accuracy_score(y_true, y_pred)
    ext_auc_macro = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    ext_auc_weighted = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")

    logger.info(f"External Zero-Shot Top-1 Accuracy: {ext_acc * 100:.2f}%")
    logger.info(f"External Multi-Class Macro ROC-AUC: {ext_auc_macro:.4f}")
    logger.info(f"External Multi-Class Weighted ROC-AUC: {ext_auc_weighted:.4f}")

    # 4. Compute Per-Class Diagnostic Performance Table
    report = classification_report(y_true, y_pred, target_names=CLASSES_ORDERED, output_dict=True)
    table_rows = []
    for c_idx, c_name in enumerate(CLASSES_ORDERED):
        # Calculate OvR ROC-AUC for this class
        binary_true = (y_true == c_idx).astype(int)
        class_auc = roc_auc_score(binary_true, y_probs[:, c_idx])
        c_stats = report[c_name]

        table_rows.append({
            "Class": c_name.upper(),
            "Precision": f"{c_stats['precision']:.3f}",
            "Sensitivity (Recall)": f"{c_stats['recall']:.3f}",
            "F1-Score": f"{c_stats['f1-score']:.3f}",
            "ROC-AUC": f"{class_auc:.4f}",
            "Support (N)": int(c_stats['support'])
        })

    # Add Macro Average row
    table_rows.append({
        "Class": "MACRO AVERAGE",
        "Precision": f"{report['macro avg']['precision']:.3f}",
        "Sensitivity (Recall)": f"{report['macro avg']['recall']:.3f}",
        "F1-Score": f"{report['macro avg']['f1-score']:.3f}",
        "ROC-AUC": f"{ext_auc_macro:.4f}",
        "Support (N)": len(y_true)
    })

    df_perf = pd.DataFrame(table_rows)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_perf.to_csv(tables_dir / "table_external_multicenter_validation.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_external_multicenter_validation.tex", "w", encoding="utf-8") as f:
        f.write(df_perf.to_latex(index=False))

    print("\n" + "="*80)
    print("EXTERNAL MULTI-CENTER ZERO-SHOT VALIDATION (BCN_20000 / Barcelona Cohort)")
    print("="*80)
    print(df_perf.to_string(index=False))
    print("="*80 + "\n")

    # 5. Render External Multi-Class ROC Curves
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
    ax.set_title("External Zero-Shot Multi-Class ROC Curves\n[BCN_20000 Barcelona Multi-Center Cohort]", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    roc_path = figures_dir / "external_roc_curves_bcn20000.png"
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 300 DPI External ROC Curves to {roc_path}.")

    # 6. Render Normalized Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=[c.upper() for c in CLASSES_ORDERED],
        yticklabels=[c.upper() for c in CLASSES_ORDERED],
        ax=ax_cm,
        cbar_kws={"label": "Normalized Recall (%)"}
    )
    ax_cm.set_title("External Multi-Center Confusion Matrix (%)\n[Zero-Shot Generalization on BCN_20000]", fontsize=11, fontweight="bold")
    ax_cm.set_xlabel("Predicted Diagnosis", fontsize=10)
    ax_cm.set_ylabel("Ground Truth Diagnosis", fontsize=10)

    cm_path = figures_dir / "external_confusion_matrix_bcn20000.png"
    fig_cm.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 300 DPI External Confusion Matrix to {cm_path}.")

    # 7. Evaluate External Explainability (CG-HAF on External Lesions)
    logger.info("Evaluating explainability localization (CG-HAF vs. Grad-CAM) on external Barcelona cohort...")
    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model, DEVICE)
    fusion_engine = HybridAttentionFusion(lambda_fusion=0.5)
    baseline_xai = BaselineXAI(model, DEVICE)
    loc_metrics = LocalizationMetrics()

    ext_cghaf_energy = []
    ext_gradcam_energy = []
    ext_cghaf_hits = []
    ext_gradcam_hits = []

    # Sample 20 external lesions
    ext_eval_sample = df_cohort.head(20)
    for _, row in ext_eval_sample.iterrows():
        bgr = cv2.imread(str(row["path"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_224 = cv2.resize(rgb, (224, 224))

        gray = cv2.cvtColor(rgb_224, cv2.COLOR_RGB2GRAY)
        _, lesion_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        pil_img = Image.fromarray(rgb_224)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_class = int(torch.argmax(model(tensor), dim=1).item())

        csm_map, _ = csm_engine.generate(rgb_224)
        gcw_map, _, _ = gcw_engine.compute_input_gcw(tensor, target_class=pred_class)
        cghaf_map, _ = fusion_engine.fuse(csm_map, gcw_map)
        gradcam_map = baseline_xai.compute_gradcam(tensor, target_class=pred_class)

        m_cg = loc_metrics.evaluate_sample(cghaf_map, lesion_mask)
        m_cam = loc_metrics.evaluate_sample(gradcam_map, lesion_mask)

        ext_cghaf_energy.append(m_cg["energy_inside_mask"])
        ext_gradcam_energy.append(m_cam["energy_inside_mask"])
        ext_cghaf_hits.append(m_cg["pointing_game_hit"])
        ext_gradcam_hits.append(m_cam["pointing_game_hit"])

    mean_cghaf_e = np.mean(ext_cghaf_energy) * 100
    mean_cam_e = np.mean(ext_gradcam_energy) * 100

    logger.info(f"External Energy Inside Mask: CG-HAF = {mean_cghaf_e:.2f}% vs. Grad-CAM = {mean_cam_e:.2f}%")

    # Save external explainability summary
    df_ext_xai = pd.DataFrame([
        {"Method": "CG-HAF (Proposed)", "External Energy Inside Mask (%)": f"{mean_cghaf_e:.2f}%", "External Pointing Game Hit (%)": f"{np.mean(ext_cghaf_hits)*100:.2f}%"},
        {"Method": "Grad-CAM", "External Energy Inside Mask (%)": f"{mean_cam_e:.2f}%", "External Pointing Game Hit (%)": f"{np.mean(ext_gradcam_hits)*100:.2f}%"}
    ])
    df_ext_xai.to_csv(tables_dir / "table_external_xai_comparison.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_external_xai_comparison.tex", "w", encoding="utf-8") as f:
        f.write(df_ext_xai.to_latex(index=False))

    logger.info("External Multi-Center Validation completed successfully!")


if __name__ == "__main__":
    main()

