"""
Comprehensive Journal-Grade Benchmark Runner for CG-HAF.
Executes rigorous multi-method evaluation on the NVIDIA GeForce RTX 5070:
- CG-HAF vs GCW vs CSM vs Grad-CAM vs Integrated Gradients vs SmoothGrad.
- Evaluates Pointing Game, Energy-Inside-Mask, Deletion AUC, Insertion AUC.
- Generates 300 DPI publication plots and LaTeX tables for IEEE / Elsevier submission.
"""

import logging
from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import torch
from torchvision import transforms

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.csm import ClinicalSaliencyMask
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.metrics import LocalizationMetrics
from src.evaluation.sanity_checks import ModelRandomizationSanityCheck
from src.evaluation.xai_benchmarks import BaselineXAI, benchmark_attribution_faithfulness
from src.utils.visualizer import PublicationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("JournalBenchmark")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {torch.cuda.get_device_name(device)}")

    # 1. Load Trained Model
    weights_path = "weights/swin_vit_xai_model.pth"
    model = EnhancedSwinTransformer(num_classes=7, use_batchnorm=True, checkpoint_path=weights_path).to(device)
    model.eval()

    # 2. Setup Engines
    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model=model, device=device)
    fusion_engine = HybridAttentionFusion(lambda_fusion=0.5)
    baseline_engine = BaselineXAI(model=model, device=device)
    loc_metrics = LocalizationMetrics()
    visualizer = PublicationVisualizer(output_dir="outputs/figures", dpi=300)

    # 3. Load Real Test Images
    img_dir = Path("data/segmented-ham10000/segmented_images")
    all_images = list(img_dir.glob("*.jpg"))
    if not all_images:
        logger.error("No images found in data/segmented-ham10000/segmented_images!")
        return

    # Benchmark subset
    num_eval = 25
    eval_images = all_images[:num_eval]
    logger.info(f"Running multi-method benchmark across {len(eval_images)} real test lesions on RTX 5070...")

    methods = ["CG-HAF (Proposed)", "GCW (Ours)", "CSM (ABCD)", "Grad-CAM", "Integrated Gradients", "SmoothGrad"]
    metrics_log = {m: {"del_auc": [], "ins_auc": [], "energy_inside": [], "pointing_hit": [], "del_curves": [], "ins_curves": []} for m in methods}

    t0 = time.time()
    for idx, img_path in enumerate(eval_images):
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))

        # Pseudo ground-truth mask using Otsu threshold on inverted grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, lesion_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = transforms.functional.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_class = torch.argmax(model(img_tensor), dim=-1).item()

        # Compute All Explanation Maps
        csm_map, abcd_comp = csm_engine.generate(rgb)
        gcw_map, _, _ = gcw_engine.compute_input_gcw(img_tensor, target_class=pred_class)
        cghaf_map, _ = fusion_engine.fuse(csm_map, gcw_map)
        gradcam_map = baseline_engine.compute_gradcam(img_tensor, target_class=pred_class)
        ig_map = baseline_engine.compute_integrated_gradients(img_tensor, target_class=pred_class, steps=15)
        sg_map = baseline_engine.compute_smoothgrad(img_tensor, target_class=pred_class, samples=15)

        maps = {
            "CG-HAF (Proposed)": cghaf_map,
            "GCW (Ours)": gcw_map,
            "CSM (ABCD)": csm_map,
            "Grad-CAM": gradcam_map,
            "Integrated Gradients": ig_map,
            "SmoothGrad": sg_map
        }

        # Evaluate Faithfulness & Localization for each method
        for m_name, m_map in maps.items():
            del_auc, ins_auc, del_c, ins_c = benchmark_attribution_faithfulness(
                model, img_tensor, m_map, target_class=pred_class, device=device, num_steps=10
            )
            hit = loc_metrics.pointing_game_hit(m_map, lesion_mask)
            e_in = loc_metrics.energy_inside_mask(m_map, lesion_mask)

            metrics_log[m_name]["del_auc"].append(del_auc)
            metrics_log[m_name]["ins_auc"].append(ins_auc)
            metrics_log[m_name]["energy_inside"].append(e_in)
            metrics_log[m_name]["pointing_hit"].append(float(hit))
            metrics_log[m_name]["del_curves"].append(del_c)
            metrics_log[m_name]["ins_curves"].append(ins_c)

    elapsed = time.time() - t0
    logger.info(f"Completed multi-method evaluation in {elapsed:.2f}s ({elapsed/num_eval:.2f}s/sample).")

    # 4. Construct Journal Comparison Table
    table_rows = []
    for m in methods:
        table_rows.append({
            "Method": m,
            "Pointing Game Hit Rate (%)": f"{np.mean(metrics_log[m]['pointing_hit']) * 100:.2f}%",
            "Energy Inside Mask (%)": f"{np.mean(metrics_log[m]['energy_inside']) * 100:.2f}%",
            "Deletion AUC (Lower is Better)": f"{np.mean(metrics_log[m]['del_auc']):.4f}",
            "Insertion AUC (Higher is Better)": f"{np.mean(metrics_log[m]['ins_auc']):.4f}"
        })

    df_comp = pd.DataFrame(table_rows)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(tables_dir / "table_xai_comparison.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_xai_comparison.tex", "w", encoding="utf-8") as f:
        f.write(df_comp.to_latex(index=False))

    print("\n" + "="*80)
    print("JOURNAL-LEVEL QUANTITATIVE EXPLAINABLE AI BENCHMARK (HAM10000 Test Cohort)")
    print("="*80)
    print(df_comp.to_string(index=False))
    print("="*80 + "\n")

    # 5. Render High-Resolution Faithfulness Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_steps = np.linspace(0, 100, 11)

    colors = {
        "CG-HAF (Proposed)": "#D62728",      # Prominent Crimson Red
        "GCW (Ours)": "#FF7F0E",             # Vibrant Orange
        "CSM (ABCD)": "#9467BD",             # Purple
        "Grad-CAM": "#1F77B4",               # Steel Blue
        "Integrated Gradients": "#2CA02C",   # Emerald Green
        "SmoothGrad": "#7F7F7F"              # Gray
    }

    # Deletion Curve
    for m in methods:
        avg_curve = np.mean(metrics_log[m]["del_curves"], axis=0)
        del_auc = np.mean(metrics_log[m]["del_auc"])
        lw = 2.5 if "CG-HAF" in m else 1.6
        ax1.plot(x_steps, avg_curve, label=f"{m} (AUC={del_auc:.3f})", color=colors[m], linewidth=lw)

    ax1.set_title("(a) Deletion Curve (Faithfulness)\n[Steeper drop = More Faithful]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("% Top Attributed Pixels Masked")
    ax1.set_ylabel("Diagnostic Class Probability")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    # Insertion Curve
    for m in methods:
        avg_curve = np.mean(metrics_log[m]["ins_curves"], axis=0)
        ins_auc = np.mean(metrics_log[m]["ins_auc"])
        lw = 2.5 if "CG-HAF" in m else 1.6
        ax2.plot(x_steps, avg_curve, label=f"{m} (AUC={ins_auc:.3f})", color=colors[m], linewidth=lw)

    ax2.set_title("(b) Insertion Curve (Feature Salience)\n[Steeper rise = More Salient]", fontsize=11, fontweight="bold")
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
