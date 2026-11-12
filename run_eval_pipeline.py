"""
Master Evaluation & Quantitative XAI Pipeline for CG-HAF.
Orchestrates on NVIDIA GeForce RTX 5070:
1. Class-stratified evaluation on unsegmented HAM10000 dermoscopic test images.
2. Polarity-verified Clinical Saliency Mask (ABCD rule) generation.
3. Hierarchical Swin Transformer V2 Gradient Coherence Weighting (GCW) & CG-HAF Fusion.
4. Quantitative XAI benchmarks evaluated against official dermatologist lesion segmentations.
5. GPU-batched Deletion & Insertion AUC perturbation curves (AMP).
"""

import argparse
import logging
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import yaml

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.csm import ClinicalSaliencyMask
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.metrics import LocalizationMetrics
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.utils.visualizer import PublicationVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CG-HAF-Pipeline")


def load_yaml_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_device(cfg: dict) -> torch.device:
    if cfg["device"]["cuda"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg['device']['device_id']}")
        if cfg["device"].get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        logger.info(f"Target Compute Device: {torch.cuda.get_device_name(device)} (VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB)")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available. Falling back to CPU execution.")
    return device


def load_stratified_samples(test_csv: Path, data_dir: Path, mask_dir: Path, classes: list, per_class: int = 10):
    df = pd.read_csv(test_csv)
    samples = []
    for c in classes:
        sub = df[df["dx"] == c]
        count = 0
        for _, row in sub.iterrows():
            img_id = row["image_id"]
            img_path = data_dir / f"{img_id}.jpg"
            mask_path = mask_dir / f"{img_id}_segmentation.png"
            if img_path.exists() and mask_path.exists():
                samples.append((img_path, mask_path, c, classes.index(c)))
                count += 1
                if count >= per_class:
                    break
    return samples


def main():
    parser = argparse.ArgumentParser(description="CG-HAF Benchmark Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Override path to .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/external_isic2019/extracted/train-image/image", help="Path to unsegmented HAM10000 images")
    parser.add_argument("--mask_dir", type=str, default="data/ham10000_masks/HAM10000_segmentations_lesion_tschandl", help="Path to official HAM10000 masks")
    parser.add_argument("--test_csv", type=str, default="data/testing-set/ham10000_test_dataset_4500.csv", help="Path to HAM10000 test split CSV")
    parser.add_argument("--per_class", type=int, default=10, help="Samples per diagnostic class")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = setup_device(cfg)
    output_dir = Path(cfg["evaluation"]["output_dir"])
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.weights or cfg["model"]["checkpoint_path"]
    logger.info("Initializing Swin Transformer V2 Classifier...")
    model = EnhancedSwinTransformer(
        num_classes=cfg["model"]["num_classes"],
        use_batchnorm=True,
        checkpoint_path=ckpt_path if Path(ckpt_path).exists() else None
    ).to(device)
    model.eval()

    csm_cfg = cfg["csm"]
    csm_engine = ClinicalSaliencyMask(
        w_a=csm_cfg["weights"]["wA"],
        w_b=csm_cfg["weights"]["wB"],
        w_c=csm_cfg["weights"]["wC"],
        w_d=csm_cfg["weights"]["wD"],
        hsv_kernel_size=csm_cfg["hsv_kernel_size"],
        canny_low=csm_cfg["canny_low"],
        canny_high=csm_cfg["canny_high"]
    )
    gcw_engine = GradientCoherenceWeighting(
        model=model,
        device=device,
        blur_kernel=cfg["gcw"]["blur_kernel"],
        blur_sigma=cfg["gcw"]["blur_sigma"]
    )
    fusion_engine = HybridAttentionFusion(lambda_fusion=cfg["gcw"]["lambda_fusion"])
    loc_metrics = LocalizationMetrics()
    faith_evaluator = FaithfulnessEvaluator(
        model=model,
        device=device,
        num_steps=cfg["evaluation"]["deletion_steps"],
        baseline_mode="blur"
    )
    visualizer = PublicationVisualizer(output_dir=str(figures_dir), dpi=300)

    classes = cfg["model"]["classes"]
    samples = load_stratified_samples(
        Path(args.test_csv), Path(args.data_dir), Path(args.mask_dir), classes, per_class=args.per_class
    )
    logger.info(f"Loaded {len(samples)} class-stratified unsegmented lesions with official masks.")

    results = []
    saved_classes = set()
    start_time = time.time()
    img_size = cfg["model"]["image_size"]

    for img_path, mask_path, true_dx, true_idx in samples:
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size))

        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = (cv2.resize(gt_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = transforms.functional.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).unsqueeze(0).to(device)

        # 1. Clinical Saliency Mask (automatic polarity-verified ABCD)
        csm_map, abcd_components = csm_engine.generate(rgb)

        # 2. Hierarchical Swin Transformer V2 GCW
        gcw_map, pred_class, confidence = gcw_engine.compute_hierarchical_gcw(img_tensor)

        # 3. Hybrid Fusion (CG-HAF)
        fused_map, alignment = fusion_engine.fuse(csm_map, gcw_map)

        # 4. Localization vs Official Dermatologist Mask + Batched GPU Faithfulness
        loc_res = loc_metrics.evaluate_sample(fused_map, gt_mask)
        faith_res = faith_evaluator.evaluate(img_tensor, fused_map, target_class=pred_class)

        # Save one clean publication figure per diagnostic class
        if true_dx not in saved_classes:
            saved_classes.add(true_dx)
            visualizer.plot_abcd_decomposition(rgb, abcd_components, csm_map, img_path.stem, true_dx)
            visualizer.plot_hybrid_fusion_comparison(
                rgb, csm_map, gcw_map, fused_map, img_path.stem, true_dx, alignment
            )

        results.append({
            "image_id": img_path.stem,
            "true_dx": true_dx,
            "predicted_class": classes[pred_class],
            "correct": int(pred_class == true_idx),
            "confidence": confidence,
            "pearson_alignment": alignment["pearson_correlation"],
            "pointing_game_hit": loc_res["pointing_game_hit"],
            "energy_inside_mask": loc_res["energy_inside_mask"],
            "gt_top20_iou": loc_res["mask_iou"],
            "deletion_auc": faith_res["deletion_auc"],
            "insertion_auc": faith_res["insertion_auc"]
        })

    elapsed = time.time() - start_time
    logger.info(f"Processed {len(samples)} samples in {elapsed:.2f}s ({elapsed / len(samples):.3f}s/sample).")

    df = pd.DataFrame(results)
    summary = {
        "Mean Pearson Alignment": [df["pearson_alignment"].mean()],
        "Pointing Game Hit Rate": [df["pointing_game_hit"].mean()],
        "Energy Inside GT Mask": [df["energy_inside_mask"].mean()],
        "Mean Top-20% GT IoU": [df["gt_top20_iou"].mean()],
        "Mean Deletion AUC (Faithfulness)": [df["deletion_auc"].mean()],
        "Mean Insertion AUC": [df["insertion_auc"].mean()]
    }
    summary_df = pd.DataFrame(summary)

    df.to_csv(tables_dir / "sample_xai_metrics.csv", index=False, encoding="utf-8")
    summary_df.to_csv(tables_dir / "summary_metrics.csv", index=False, encoding="utf-8")
    with open(tables_dir / "summary_metrics.tex", "w", encoding="utf-8") as f:
        f.write(summary_df.to_latex(index=False, float_format="%.4f"))

    print("\n" + "="*55)
    print("CG-HAF QUANTITATIVE BENCHMARK SUMMARY (Official GT Masks)")
    print("="*55)
    for k, v in summary.items():
        print(f" {k:<35}: {v[0]:.4f}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
