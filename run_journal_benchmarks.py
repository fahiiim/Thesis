"""
Comprehensive Journal-Grade Benchmark Runner for CG-HAF.
Evaluates 140 class-stratified unsegmented HAM10000 test lesions (20 per class)
against official dermatologist segmentation masks (Tschandl et al.) on RTX 5070:
- CG-HAF (Proposed) vs Hierarchical Swin-GCW vs CSM (ABCD) vs Stage-4 Grad-CAM vs IG vs SmoothGrad.
- Computes Mean ± Std and Wilcoxon signed-rank significance p-values.
- Saturates RTX 5070 Tensor Cores via mega-batched AMP perturbation sweeps.
"""

import logging
from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch
from torchvision import transforms

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.csm import ClinicalSaliencyMask
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.metrics import LocalizationMetrics
from src.evaluation.xai_benchmarks import BaselineXAI, benchmark_multi_maps_batched

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("JournalBenchmark")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_stratified_cohort(test_csv: Path, img_dir: Path, mask_dir: Path, per_class: int = 20):
    df = pd.read_csv(test_csv)
    cohort = []
    for c in CLASSES:
        sub = df[df["dx"] == c]
        count = 0
        for _, row in sub.iterrows():
            img_id = row["image_id"]
            ip = img_dir / f"{img_id}.jpg"
            mp = mask_dir / f"{img_id}_segmentation.png"
            if ip.exists() and mp.exists():
                cohort.append((ip, mp, c))
                count += 1
                if count >= per_class:
                    break
    return cohort


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {torch.cuda.get_device_name(device)}")

    weights_path = "weights/swin_vit_xai_model.pth"
    model = EnhancedSwinTransformer(num_classes=7, use_batchnorm=True, checkpoint_path=weights_path).to(device)
    model.eval()

    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model=model, device=device)
    fusion_engine = HybridAttentionFusion(lambda_fusion=0.5)
    baseline_engine = BaselineXAI(model=model, device=device)
    loc_metrics = LocalizationMetrics()

    test_csv = Path("data/testing-set/ham10000_test_dataset_4500.csv")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    mask_dir = Path("data/ham10000_masks/HAM10000_segmentations_lesion_tschandl")

    cohort = load_stratified_cohort(test_csv, img_dir, mask_dir, per_class=20)
    logger.info(f"Running stratified multi-method benchmark across {len(cohort)} unsegmented lesions (20/class x 7 classes) against official masks...")

    methods = [
        "CG-HAF (Proposed)",
        "Swin-GCW (Ours)",
        "CSM (ABCD)",
        "Grad-CAM (Stage 4)",
        "Integrated Gradients",
        "SmoothGrad"
    ]
    metrics_log = {
        m: {"del_auc": [], "ins_auc": [], "energy_inside": [], "pointing_hit": [], "iou": [], "del_curves": [], "ins_curves": []}
        for m in methods
    }

    t0 = time.time()
    for idx, (img_path, mask_path, dx) in enumerate(cohort):
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))

        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = (cv2.resize(gt_mask, (224, 224), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = transforms.functional.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred_class = int(torch.argmax(model(img_tensor), dim=-1).item())

        # Compute all 6 attribution maps cleanly
        csm_map, _ = csm_engine.generate(rgb)
        gcw_map, _, _ = gcw_engine.compute_hierarchical_gcw(img_tensor, target_class=pred_class)
        cghaf_map, _ = fusion_engine.fuse(csm_map, gcw_map)
        gradcam_map = baseline_engine.compute_gradcam(img_tensor, target_class=pred_class)
        ig_map = baseline_engine.compute_integrated_gradients(img_tensor, target_class=pred_class, steps=20)
        sg_map = baseline_engine.compute_smoothgrad(img_tensor, target_class=pred_class, samples=20)

        maps = {
            "CG-HAF (Proposed)": cghaf_map,
            "Swin-GCW (Ours)": gcw_map,
            "CSM (ABCD)": csm_map,
            "Grad-CAM (Stage 4)": gradcam_map,
            "Integrated Gradients": ig_map,
            "SmoothGrad": sg_map
        }

        # GPU mega-batch faithfulness across all 6 methods (132 images at once on RTX 5070)
        batch_faith = benchmark_multi_maps_batched(model, img_tensor, maps, target_class=pred_class, device=device, num_steps=10)

        for m_name, m_map in maps.items():
            del_auc, ins_auc, del_c, ins_c = batch_faith[m_name]
            loc = loc_metrics.evaluate_sample(m_map, gt_mask)
            metrics_log[m_name]["del_auc"].append(del_auc)
            metrics_log[m_name]["ins_auc"].append(ins_auc)
            metrics_log[m_name]["energy_inside"].append(loc["energy_inside_mask"])
            metrics_log[m_name]["pointing_hit"].append(loc["pointing_game_hit"])
            metrics_log[m_name]["iou"].append(loc["mask_iou"])
            metrics_log[m_name]["del_curves"].append(del_c)
            metrics_log[m_name]["ins_curves"].append(ins_c)

    elapsed = time.time() - t0
    logger.info(f"Completed 140-lesion stratified benchmark in {elapsed:.2f}s ({elapsed/len(cohort):.3f}s/sample).")

    # Statistical significance: CG-HAF vs Stage-4 Grad-CAM
    _, p_energy = wilcoxon(metrics_log["CG-HAF (Proposed)"]["energy_inside"], metrics_log["Grad-CAM (Stage 4)"]["energy_inside"])
    _, p_ins = wilcoxon(metrics_log["CG-HAF (Proposed)"]["ins_auc"], metrics_log["Grad-CAM (Stage 4)"]["ins_auc"])
    _, p_del = wilcoxon(metrics_log["CG-HAF (Proposed)"]["del_auc"], metrics_log["Grad-CAM (Stage 4)"]["del_auc"])
    logger.info(f"Wilcoxon p-values (CG-HAF vs Grad-CAM Stage 4): Energy p={p_energy:.2e}, Insertion p={p_ins:.2e}, Deletion p={p_del:.2e}")

    table_rows = []
    for m in methods:
        e_arr = np.array(metrics_log[m]["energy_inside"]) * 100
        d_arr = np.array(metrics_log[m]["del_auc"])
        i_arr = np.array(metrics_log[m]["ins_auc"])
        iou_arr = np.array(metrics_log[m]["iou"]) * 100
        table_rows.append({
            "Method": m,
            "Pointing Hit (%)": f"{np.mean(metrics_log[m]['pointing_hit']) * 100:.2f}%",
            "Energy Inside GT (%)": f"{np.mean(e_arr):.2f} ± {np.std(e_arr):.2f}",
            "Top-20% GT IoU (%)": f"{np.mean(iou_arr):.2f} ± {np.std(iou_arr):.2f}",
            "Deletion AUC (Lower Better)": f"{np.mean(d_arr):.4f} ± {np.std(d_arr):.4f}",
            "Insertion AUC (Higher Better)": f"{np.mean(i_arr):.4f} ± {np.std(i_arr):.4f}"
        })

    df_comp = pd.DataFrame(table_rows)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(tables_dir / "table_xai_comparison.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_xai_comparison.tex", "w", encoding="utf-8") as f:
        f.write(df_comp.to_latex(index=False))

    print("\n" + "="*95)
    print("STRATIFIED QUANTITATIVE XAI BENCHMARK (N=140 HAM10000 Unsegmented + Official GT Masks)")
    print("="*95)
    print(df_comp.to_string(index=False))
    print(f"Wilcoxon (CG-HAF vs Stage-4 Grad-CAM): Energy p={p_energy:.2e} | Ins p={p_ins:.2e} | Del p={p_del:.2e}")
    print("="*95 + "\n")

    # Render 300 DPI Faithfulness Comparison Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_steps = np.linspace(0, 100, 11)
    colors = {
        "CG-HAF (Proposed)": "#D62728",
        "Swin-GCW (Ours)": "#FF7F0E",
        "CSM (ABCD)": "#9467BD",
        "Grad-CAM (Stage 4)": "#1F77B4",
        "Integrated Gradients": "#2CA02C",
        "SmoothGrad": "#7F7F7F"
    }

    for m in methods:
        avg_c = np.mean(metrics_log[m]["del_curves"], axis=0)
        auc_v = np.mean(metrics_log[m]["del_auc"])
        ax1.plot(x_steps, avg_c, label=f"{m} (AUC={auc_v:.3f})", color=colors[m], linewidth=2.5 if "CG-HAF" in m else 1.6)
    ax1.set_title("(a) Deletion Curve (Faithfulness)\n[Lower AUC = More Faithful]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("% Top Attributed Pixels Masked")
    ax1.set_ylabel("Diagnostic Class Probability")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    for m in methods:
        avg_c = np.mean(metrics_log[m]["ins_curves"], axis=0)
        auc_v = np.mean(metrics_log[m]["ins_auc"])
        ax2.plot(x_steps, avg_c, label=f"{m} (AUC={auc_v:.3f})", color=colors[m], linewidth=2.5 if "CG-HAF" in m else 1.6)
    ax2.set_title("(b) Insertion Curve (Feature Salience)\n[Higher AUC = More Salient]", fontsize=11, fontweight="bold")
    ax2.set_xlabel("% Top Attributed Pixels Retained")
    ax2.set_ylabel("Diagnostic Class Probability")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    fig_path = Path("outputs/figures/faithfulness_comparison_benchmark.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 300 DPI benchmark plot to {fig_path}.")


if __name__ == "__main__":
    main()
