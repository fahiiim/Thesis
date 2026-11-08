"""
Ablation Study: Sensitivity Analysis of Clinical-Neural Fusion Parameter (lambda).
Evaluates Pointing Game Hit Rate, Energy Inside Mask, Deletion AUC, and Insertion AUC
across lambda in [0.0, 1.0] in steps of 0.1 on the HAM10000 test cohort.
Produces publication-grade sensitivity curves for IEEE/Elsevier submission.
"""

import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.csm import ClinicalSaliencyMask
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.metrics import LocalizationMetrics
from src.evaluation.faithfulness import FaithfulnessEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LambdaAblation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    logger.info(f"Running Fusion Weight (lambda) Ablation on: {torch.cuda.get_device_name(0)}")

    model = EnhancedSwinTransformer(num_classes=7, checkpoint_path="weights/swin_vit_xai_model.pth")
    model.to(DEVICE)
    model.eval()

    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model, DEVICE)
    faith_evaluator = FaithfulnessEvaluator(model, DEVICE, num_steps=10)
    loc_metrics = LocalizationMetrics()

    # Load test cohort (20 diverse test lesions)
    test_csv = Path("data/testing-set/ham10000_test_dataset_4500.csv")
    df = pd.read_csv(test_csv)
    image_dir = Path("data/segmented-ham10000/segmented_images")

    samples = []
    for c in ["mel", "bcc", "akiec", "bkl", "nv"]:
        for _, row in df[df["dx"] == c].head(4).iterrows():
            img_path = image_dir / f"{row['image_id']}.jpg"
            if img_path.exists():
                samples.append((img_path, row["dx"]))

    logger.info(f"Loaded {len(samples)} diverse test samples for lambda ablation.")

    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Precompute CSM, GCW, and input tensors to avoid redundant backprop
    precomputed = []
    for img_path, dx in samples:
        pil_img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(pil_img)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(tensor)
            pred_class = int(torch.argmax(preds, dim=1).item())

        # Pseudo ground-truth mask using Otsu threshold on inverted grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, lesion_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        csm_map, _ = csm_engine.generate(img_np)
        gcw_map, _, _ = gcw_engine.compute_input_gcw(tensor, target_class=pred_class)

        precomputed.append({
            "img_tensor": tensor,
            "pred_class": pred_class,
            "csm": csm_map,
            "gcw": gcw_map,
            "gt_mask": lesion_mask
        })

    ablation_results = {
        "lambda": lambda_values,
        "pointing_hit": [],
        "energy_inside": [],
        "del_auc": [],
        "ins_auc": []
    }

    for lam in lambda_values:
        fusion_engine = HybridAttentionFusion(lambda_fusion=lam)
        hit_rates = []
        energies = []
        del_aucs = []
        ins_aucs = []

        for item in precomputed:
            fused_map, _ = fusion_engine.fuse(item["csm"], item["gcw"])
            m = loc_metrics.evaluate_sample(fused_map, item["gt_mask"])
            hit_rates.append(m["pointing_game_hit"])
            energies.append(m["energy_inside_mask"])

            faith = faith_evaluator.evaluate(item["img_tensor"], fused_map, item["pred_class"])
            del_aucs.append(faith["deletion_auc"])
            ins_aucs.append(faith["insertion_auc"])

        ablation_results["pointing_hit"].append(np.mean(hit_rates) * 100)
        ablation_results["energy_inside"].append(np.mean(energies) * 100)
        ablation_results["del_auc"].append(np.mean(del_aucs))
        ablation_results["ins_auc"].append(np.mean(ins_aucs))

        logger.info(f"Lambda = {lam:.1f} | Pointing Hit: {np.mean(hit_rates)*100:.2f}% | Energy: {np.mean(energies)*100:.2f}% | Del AUC: {np.mean(del_aucs):.4f} | Ins AUC: {np.mean(ins_aucs):.4f}")

    # Save Table
    df_ablation = pd.DataFrame(ablation_results)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_ablation.to_csv(tables_dir / "table_ablation_lambda.csv", index=False, encoding="utf-8")
    with open(tables_dir / "table_ablation_lambda.tex", "w", encoding="utf-8") as f:
        f.write(df_ablation.to_latex(index=False))

    # Plot 2x2 publication figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))

    ax1.plot(lambda_values, ablation_results["pointing_hit"], "o-", color="#D62728", linewidth=2.2, markersize=6)
    ax1.set_title("(a) Pointing Game Hit Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Fusion Parameter $\lambda$ (0: Pure GCW $\rightarrow$ 1: Pure CSM)")
    ax1.set_ylabel("Hit Rate (%)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(lambda_values, ablation_results["energy_inside"], "s-", color="#1F77B4", linewidth=2.2, markersize=6)
    ax2.set_title("(b) Energy Inside Lesion Mask (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Fusion Parameter $\lambda$")
    ax2.set_ylabel("Energy Inside (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.plot(lambda_values, ablation_results["del_auc"], "^-", color="#2CA02C", linewidth=2.2, markersize=6)
    ax3.set_title("(c) Deletion AUC (Lower is Better)", fontsize=11, fontweight="bold")
    ax3.set_xlabel(r"Fusion Parameter $\lambda$")
    ax3.set_ylabel("Deletion AUC")
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4.plot(lambda_values, ablation_results["ins_auc"], "d-", color="#FF7F0E", linewidth=2.2, markersize=6)
    ax4.set_title("(d) Insertion AUC (Higher is Better)", fontsize=11, fontweight="bold")
    ax4.set_xlabel(r"Fusion Parameter $\lambda$")
    ax4.set_ylabel("Insertion AUC")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = Path("outputs/figures/ablation_lambda_sensitivity.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved 300 DPI lambda sensitivity plot to {fig_path}.")


if __name__ == "__main__":
    main()

