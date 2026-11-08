"""
Adebayo Parameter Randomization Sanity Check Pipeline.
Evaluates model-dependence of CG-HAF and GCW against Grad-CAM and Edge Baseline.
Produces IEEE/Elsevier publication-quality cascading degradation plots.
"""

import copy
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from scipy.stats import spearmanr

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.csm import ClinicalSaliencyMask
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.xai_benchmarks import BaselineXAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdebayoSanityCheck")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIAGNOSTIC_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def reinit_module(m: nn.Module):
    """Reinitializes linear, conv, and normalization layers."""
    for child in m.modules():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(child.weight, nonlinearity="relu")
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def main():
    logger.info(f"Executing Adebayo Sanity Check on: {torch.cuda.get_device_name(0)}")
    
    # 1. Initialize Baseline Model
    model = EnhancedSwinTransformer(num_classes=7, checkpoint_path="weights/swin_vit_xai_model.pth")
    model.to(DEVICE)
    model.eval()

    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model, DEVICE)
    fusion_engine = HybridAttentionFusion(lambda_fusion=0.5)
    baseline_xai = BaselineXAI(model, DEVICE)

    # 2. Select Test Sample Cohort
    test_csv = Path("data/testing-set/ham10000_test_dataset_4500.csv")
    df = pd.read_csv(test_csv)
    image_dir = Path("data/segmented-ham10000/segmented_images")
    
    # Select 10 diverse samples
    samples = []
    for c in ["mel", "bcc", "akiec", "nv", "bkl"]:
        c_samples = df[df["dx"] == c].head(2)
        for _, row in c_samples.iterrows():
            img_path = image_dir / f"{row['image_id']}.jpg"
            if img_path.exists():
                samples.append((img_path, row["dx"]))

    logger.info(f"Selected {len(samples)} diverse dermoscopic lesions for parameter randomization.")

    # Cascading layers to randomize in top-down order
    cascade_stages = [
        ("Original (Trained)", []),
        ("Top Layer (Classifier.5)", ["classifier.5"]),
        ("Classifier Head (All FC)", ["classifier.5", "classifier.1"]),
        ("+ Swin Stage 4", ["classifier.5", "classifier.1", "backbone.features.7"]),
        ("+ Swin Stage 3", ["classifier.5", "classifier.1", "backbone.features.7", "backbone.features.5"]),
        ("+ Complete Backbone", ["classifier.5", "classifier.1", "backbone.features.7", "backbone.features.5", "backbone.features.3", "backbone.features.1", "backbone.features.0"])
    ]

    results = {
        "GCW (Ours)": {s[0]: [] for s in cascade_stages},
        "CG-HAF (Proposed)": {s[0]: [] for s in cascade_stages},
        "Grad-CAM": {s[0]: [] for s in cascade_stages}
    }

    # Store visual progression for one representative melanoma sample
    visual_progression = {"stages": [s[0] for s in cascade_stages], "gcw": [], "cghaf": [], "gradcam": []}
    rep_sample_recorded = False

    for sample_idx, (img_path, true_dx) in enumerate(samples):
        pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(pil_img)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = model(tensor)
            pred_class = int(torch.argmax(preds, dim=1).item())

        csm_map, _ = csm_engine.generate(img_np)

        # Baseline Attributions
        orig_gcw, _, _ = gcw_engine.compute_input_gcw(tensor, target_class=pred_class)
        orig_cghaf, _ = fusion_engine.fuse(csm_map, orig_gcw)
        orig_gradcam = baseline_xai.compute_gradcam(tensor, target_class=pred_class)

        if not rep_sample_recorded and true_dx == "mel":
            rep_image_np = img_np
            visual_progression["gcw"].append(orig_gcw)
            visual_progression["cghaf"].append(orig_cghaf)
            visual_progression["gradcam"].append(orig_gradcam)

        for stage_name, layers_to_scramble in cascade_stages:
            if not layers_to_scramble:
                results["GCW (Ours)"][stage_name].append(1.0)
                results["CG-HAF (Proposed)"][stage_name].append(1.0)
                results["Grad-CAM"][stage_name].append(1.0)
                continue

            # Clone and scramble layers
            scrambled_model = copy.deepcopy(model)
            for l_str in layers_to_scramble:
                parts = l_str.split(".")
                curr = scrambled_model
                for p in parts:
                    if hasattr(curr, p):
                        curr = getattr(curr, p)
                    elif hasattr(curr, "__getitem__") and p.isdigit():
                        curr = curr[int(p)]
                reinit_module(curr)

            # Engines for scrambled model
            scr_gcw_engine = GradientCoherenceWeighting(scrambled_model, DEVICE)
            scr_baseline_xai = BaselineXAI(scrambled_model, DEVICE)

            pert_gcw, _, _ = scr_gcw_engine.compute_input_gcw(tensor, target_class=pred_class)
            pert_cghaf, _ = fusion_engine.fuse(csm_map, pert_gcw)
            pert_gradcam = scr_baseline_xai.compute_gradcam(tensor, target_class=pred_class)

            # Measure Rank Correlation
            corr_gcw, _ = spearmanr(orig_gcw.flatten(), pert_gcw.flatten())
            corr_cghaf, _ = spearmanr(orig_cghaf.flatten(), pert_cghaf.flatten())
            corr_gcam, _ = spearmanr(orig_gradcam.flatten(), pert_gradcam.flatten())

            results["GCW (Ours)"][stage_name].append(float(corr_gcw) if not np.isnan(corr_gcw) else 0.0)
            results["CG-HAF (Proposed)"][stage_name].append(float(corr_cghaf) if not np.isnan(corr_cghaf) else 0.0)
            results["Grad-CAM"][stage_name].append(float(corr_gcam) if not np.isnan(corr_gcam) else 0.0)

            if not rep_sample_recorded and true_dx == "mel":
                visual_progression["gcw"].append(pert_gcw)
                visual_progression["cghaf"].append(pert_cghaf)
                visual_progression["gradcam"].append(pert_gradcam)

        if true_dx == "mel":
            rep_sample_recorded = True

    # 3. Compile Summary Table
    summary_rows = []
    for stage_name, _ in cascade_stages:
        summary_rows.append({
            "Randomization Step": stage_name,
            "GCW Spearman r": f"{np.mean(results['GCW (Ours)'][stage_name]):.4f}",
            "CG-HAF Spearman r": f"{np.mean(results['CG-HAF (Proposed)'][stage_name]):.4f}",
            "Grad-CAM Spearman r": f"{np.mean(results['Grad-CAM'][stage_name]):.4f}"
        })

    df_adebayo = pd.DataFrame(summary_rows)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_adebayo.to_csv(tables_dir / "table_adebayo_sanity_check.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_adebayo_sanity_check.tex", "w", encoding="utf-8") as f:
        f.write(df_adebayo.to_latex(index=False))

    logger.info("\n" + "="*80)
    logger.info("ADEBAYO MODEL PARAMETER RANDOMIZATION SANITY CHECK RESULTS")
    logger.info("="*80)
    print(df_adebayo.to_string(index=False))
    logger.info("="*80 + "\n")

    # 4. Generate Journal Figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1.8])

    stage_labels = ["Original", "Classifier.5", "All FC", "+ Stage 4", "+ Stage 3", "+ Full Net"]

    # Visual map progression across scrambling
    for i, s_label in enumerate(stage_labels):
        # Row 0: GCW
        ax_g = fig.add_subplot(gs[0, i])
        ax_g.imshow(visual_progression["gcw"][i], cmap="jet")
        ax_g.set_title(s_label, fontsize=9, fontweight="bold")
        ax_g.axis("off")
        if i == 0:
            ax_g.set_ylabel("GCW (Ours)", fontsize=10, fontweight="bold")

        # Row 1: CG-HAF
        ax_h = fig.add_subplot(gs[1, i])
        ax_h.imshow(visual_progression["cghaf"][i], cmap="jet")
        ax_h.axis("off")
        if i == 0:
            ax_h.set_ylabel("CG-HAF (Fusion)", fontsize=10, fontweight="bold")

        # Row 2: Grad-CAM
        ax_cam = fig.add_subplot(gs[2, i])
        ax_cam.imshow(visual_progression["gradcam"][i], cmap="jet")
        ax_cam.axis("off")
        if i == 0:
            ax_cam.set_ylabel("Grad-CAM", fontsize=10, fontweight="bold")

    # Correlation Plot
    ax_corr = fig.add_subplot(gs[:, 6])
    x_indices = list(range(len(stage_labels)))

    mean_gcw = [np.mean(results["GCW (Ours)"][s[0]]) for s in cascade_stages]
    mean_cghaf = [np.mean(results["CG-HAF (Proposed)"][s[0]]) for s in cascade_stages]
    mean_gcam = [np.mean(results["Grad-CAM"][s[0]]) for s in cascade_stages]

    ax_corr.plot(x_indices, mean_gcw, "o-", label="GCW (Ours)", color="#FF7F0E", linewidth=2.2, markersize=6)
    ax_corr.plot(x_indices, mean_cghaf, "s-", label="CG-HAF (Proposed)", color="#D62728", linewidth=2.5, markersize=6)
    ax_corr.plot(x_indices, mean_gcam, "^--", label="Grad-CAM", color="#1F77B4", linewidth=1.8, markersize=5)
    # Theoretical Edge detector failure line (constant r = 1.0)
    ax_corr.axhline(1.0, color="#7F7F7F", linestyle=":", label="Edge Filter (Sanity Fail)")

    ax_corr.set_xticks(x_indices)
    ax_corr.set_xticklabels(stage_labels, rotation=45, ha="right", fontsize=8.5)
    ax_corr.set_ylabel("Spearman Rank Correlation (r)", fontsize=10, fontweight="bold")
    ax_corr.set_title("Adebayo Cascading Decay\n[Passing: Steep correlation drop]", fontsize=10.5, fontweight="bold")
    ax_corr.set_ylim(-0.1, 1.08)
    ax_corr.grid(True, linestyle="--", alpha=0.5)
    ax_corr.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_fig = figures_dir / "adebayo_cascading_randomization.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 300 DPI Adebayo Sanity Check visualization to {out_fig}.")


if __name__ == "__main__":
    main()

