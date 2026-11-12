"""
Adebayo Parameter Randomization Sanity Check Pipeline.
Evaluates model-dependence of Hierarchical Swin-GCW and CG-HAF against
Stage-4 Grad-CAM across unsegmented HAM10000 dermoscopic lesions.
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
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

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
    torch.backends.cudnn.benchmark = True
    logger.info(f"Executing Adebayo Sanity Check on: {torch.cuda.get_device_name(0)}")

    model = EnhancedSwinTransformer(num_classes=7, use_batchnorm=True, checkpoint_path="weights/swin_vit_xai_model.pth")
    model.to(DEVICE)
    model.eval()

    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model, DEVICE)
    fusion_engine = HybridAttentionFusion(lambda_fusion=0.5)
    baseline_xai = BaselineXAI(model, DEVICE)

    test_csv = Path("data/testing-set/ham10000_test_dataset_4500.csv")
    df = pd.read_csv(test_csv)
    image_dir = Path("data/external_isic2019/extracted/train-image/image")

    # Select 14 class-stratified unsegmented lesions (2 per class)
    samples = []
    for c in CLASSES:
        for _, row in df[df["dx"] == c].head(2).iterrows():
            img_path = image_dir / f"{row['image_id']}.jpg"
            if img_path.exists():
                samples.append((img_path, row["dx"]))

    logger.info(f"Selected {len(samples)} class-stratified unsegmented lesions for parameter randomization.")

    cascade_stages = [
        ("Original (Trained)", []),
        ("Top Layer (Classifier.5)", ["classifier.5"]),
        ("Classifier Head (All FC)", ["classifier.5", "classifier.1"]),
        ("+ Swin Stage 4", ["classifier.5", "classifier.1", "backbone.features.7"]),
        ("+ Swin Stage 3", ["classifier.5", "classifier.1", "backbone.features.7", "backbone.features.5"]),
        ("+ Complete Backbone", ["classifier.5", "classifier.1", "backbone.features.7", "backbone.features.5", "backbone.features.3", "backbone.features.1", "backbone.features.0"])
    ]

    results = {
        "Swin-GCW (Ours)": {s[0]: [] for s in cascade_stages},
        "CG-HAF (Proposed)": {s[0]: [] for s in cascade_stages},
        "Grad-CAM (Stage 4)": {s[0]: [] for s in cascade_stages}
    }

    visual_progression = {"gcw": [], "cghaf": [], "gradcam": []}
    rep_sample_recorded = False
    rep_id = ""

    # Pre-clone once and cascade randomization cleanly to avoid copying weights repeatedly
    for img_path, true_dx in samples:
        pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(pil_img)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred_class = int(torch.argmax(model(tensor), dim=1).item())

        csm_map, _ = csm_engine.generate(img_np)

        orig_gcw, _, _ = gcw_engine.compute_hierarchical_gcw(tensor, target_class=pred_class)
        orig_cghaf, _ = fusion_engine.fuse(csm_map, orig_gcw)
        orig_gradcam = baseline_xai.compute_gradcam(tensor, target_class=pred_class)

        record_this = (not rep_sample_recorded and true_dx == "mel")
        if record_this:
            rep_id = img_path.stem
            visual_progression["gcw"].append(orig_gcw)
            visual_progression["cghaf"].append(orig_cghaf)
            visual_progression["gradcam"].append(orig_gradcam)

        scrambled_model = copy.deepcopy(model)
        scr_gcw_engine = GradientCoherenceWeighting(scrambled_model, DEVICE)
        scr_baseline_xai = BaselineXAI(scrambled_model, DEVICE)
        randomized_set = set()

        for stage_name, layers_to_scramble in cascade_stages:
            if not layers_to_scramble:
                results["Swin-GCW (Ours)"][stage_name].append(1.0)
                results["CG-HAF (Proposed)"][stage_name].append(1.0)
                results["Grad-CAM (Stage 4)"][stage_name].append(1.0)
                continue

            for l_str in layers_to_scramble:
                if l_str in randomized_set:
                    continue
                randomized_set.add(l_str)
                parts = l_str.split(".")
                curr = scrambled_model
                for p in parts:
                    curr = getattr(curr, p) if hasattr(curr, p) else curr[int(p)]
                reinit_module(curr)

            pert_gcw, _, _ = scr_gcw_engine.compute_hierarchical_gcw(tensor, target_class=pred_class)
            pert_cghaf, _ = fusion_engine.fuse(csm_map, pert_gcw)
            pert_gradcam = scr_baseline_xai.compute_gradcam(tensor, target_class=pred_class)

            corr_gcw, _ = spearmanr(orig_gcw.flatten(), pert_gcw.flatten())
            corr_cghaf, _ = spearmanr(orig_cghaf.flatten(), pert_cghaf.flatten())
            corr_gcam, _ = spearmanr(orig_gradcam.flatten(), pert_gradcam.flatten())

            results["Swin-GCW (Ours)"][stage_name].append(float(corr_gcw) if not np.isnan(corr_gcw) else 0.0)
            results["CG-HAF (Proposed)"][stage_name].append(float(corr_cghaf) if not np.isnan(corr_cghaf) else 0.0)
            results["Grad-CAM (Stage 4)"][stage_name].append(float(corr_gcam) if not np.isnan(corr_gcam) else 0.0)

            if record_this:
                visual_progression["gcw"].append(pert_gcw)
                visual_progression["cghaf"].append(pert_cghaf)
                visual_progression["gradcam"].append(pert_gradcam)

        if record_this:
            rep_sample_recorded = True

    summary_rows = []
    for stage_name, _ in cascade_stages:
        summary_rows.append({
            "Randomization Step": stage_name,
            "Swin-GCW Spearman r": f"{np.mean(results['Swin-GCW (Ours)'][stage_name]):.4f}",
            "CG-HAF Spearman r": f"{np.mean(results['CG-HAF (Proposed)'][stage_name]):.4f}",
            "Grad-CAM (Stage 4) Spearman r": f"{np.mean(results['Grad-CAM (Stage 4)'][stage_name]):.4f}"
        })

    df_adebayo = pd.DataFrame(summary_rows)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_adebayo.to_csv(tables_dir / "table_adebayo_sanity_check.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_adebayo_sanity_check.tex", "w", encoding="utf-8") as f:
        f.write(df_adebayo.to_latex(index=False))

    print("\n" + "="*80)
    print("ADEBAYO MODEL PARAMETER RANDOMIZATION SANITY CHECK (Unsegmented HAM10000)")
    print("="*80)
    print(df_adebayo.to_string(index=False))
    print("="*80 + "\n")

    # Render 300 DPI Journal Figure
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 1, 2.0])
    stage_labels = ["Original", "Classifier.5", "All FC", "+ Stage 4", "+ Stage 3", "+ Full Net"]
    row_titles = ["Swin-GCW\n(Ours)", "CG-HAF\n(Fusion)", "Grad-CAM\n(Stage 4)"]

    for i, s_label in enumerate(stage_labels):
        for r_idx, key in enumerate(["gcw", "cghaf", "gradcam"]):
            ax = fig.add_subplot(gs[r_idx, i])
            ax.imshow(visual_progression[key][i], cmap="inferno", vmin=0, vmax=1)
            if r_idx == 0:
                ax.set_title(s_label, fontsize=9.5, fontweight="bold")
            if i == 0:
                ax.text(-0.12, 0.5, row_titles[r_idx], transform=ax.transAxes, va="center", ha="right", fontsize=9.5, fontweight="bold")
            ax.axis("off")

    ax_corr = fig.add_subplot(gs[:, 6])
    x_indices = list(range(len(stage_labels)))
    mean_gcw = [np.mean(results["Swin-GCW (Ours)"][s[0]]) for s in cascade_stages]
    mean_cghaf = [np.mean(results["CG-HAF (Proposed)"][s[0]]) for s in cascade_stages]
    mean_gcam = [np.mean(results["Grad-CAM (Stage 4)"][s[0]]) for s in cascade_stages]

    ax_corr.plot(x_indices, mean_gcw, "o-", label="Swin-GCW (Ours)", color="#FF7F0E", linewidth=2.2, markersize=6)
    ax_corr.plot(x_indices, mean_cghaf, "s-", label="CG-HAF (Proposed)", color="#D62728", linewidth=2.5, markersize=6)
    ax_corr.plot(x_indices, mean_gcam, "^--", label="Grad-CAM (Stage 4)", color="#1F77B4", linewidth=1.8, markersize=5)
    ax_corr.axhline(1.0, color="#7F7F7F", linestyle=":", label="Edge Filter (Sanity Fail)")

    ax_corr.set_xticks(x_indices)
    ax_corr.set_xticklabels(stage_labels, rotation=40, ha="right", fontsize=8.5)
    ax_corr.set_ylabel("Spearman Rank Correlation (r)", fontsize=10, fontweight="bold")
    ax_corr.set_title(f"Cascading Sanity Check ({rep_id} MEL)\n[Model-Dependent Attribution Decay]", fontsize=10.5, fontweight="bold")
    ax_corr.set_ylim(-0.55, 1.10)
    ax_corr.grid(True, linestyle="--", alpha=0.5)
    ax_corr.legend(loc="lower left", fontsize=8.0, framealpha=0.95)

    plt.tight_layout()
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_fig = figures_dir / "adebayo_cascading_randomization.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 300 DPI Adebayo Sanity Check visualization to {out_fig}.")


if __name__ == "__main__":
    main()
