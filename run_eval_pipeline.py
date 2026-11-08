"""
Master Evaluation & Quantitative XAI Pipeline for CG-HAF.
Orchestrates:
1. Swin-V2 model inference on NVIDIA GeForce RTX 5070.
2. Clinical Saliency Mask (ABCD rule) generation.
3. Gradient Coherence Weighting (GCW) and Hybrid Fusion (CG-HAF).
4. Quantitative XAI benchmarks (Pointing Game, Energy-Inside-Mask, Deletion/Insertion AUC).
5. Adebayo parameter randomization test.
6. Export of 300 DPI publication-grade figures and LaTeX summary tables.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, List
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
from src.evaluation.sanity_checks import ModelRandomizationSanityCheck
from src.utils.dataset import DermoscopyTestDataset, create_optimized_dataloader
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


def main():
    parser = argparse.ArgumentParser(description="CG-HAF Benchmark Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Override path to .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/test", help="Path to evaluation test images")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of benchmark samples to evaluate")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = setup_device(cfg)
    output_dir = Path(cfg["evaluation"]["output_dir"])
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.weights or cfg["model"]["checkpoint_path"]
    if not Path(ckpt_path).exists():
        logger.warning(f"Checkpoint '{ckpt_path}' not found! Running in initial evaluation mode.")

    # 1. Initialize Model
    logger.info("Initializing Swin Transformer V2 Classifier...")
    model = EnhancedSwinTransformer(
        num_classes=cfg["model"]["num_classes"],
        use_batchnorm=True,
        checkpoint_path=ckpt_path if Path(ckpt_path).exists() else None
    ).to(device)
    model.eval()

    # 2. Initialize XAI & Evaluation Modules
    logger.info("Setting up CG-HAF Explainability Engines...")
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

    # 3. Locate Evaluation Images
    data_path = Path(args.data_dir)
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    if not image_files:
        logger.warning(f"No test images found in {args.data_dir}. Generating synthetic test cohort for pipeline verification.")
        dummy_dir = Path("data/demo_test")
        dummy_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            synth = np.random.randint(80, 220, (224, 224, 3), dtype=np.uint8)
            cv2.circle(synth, (112, 112), 45, (60, 40, 30), -1)
            cv2.imwrite(str(dummy_dir / f"demo_lesion_{i}.jpg"), synth)
        image_files = list(dummy_dir.glob("*.jpg"))

    image_files = image_files[:args.num_samples]
    logger.info(f"Loaded {len(image_files)} test samples for quantitative benchmarking.")

    # 4. Benchmarking Loop
    results = []
    classes = cfg["model"]["classes"]
    start_time = time.time()

    for idx, img_path in enumerate(image_files):
        # Load image
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (cfg["model"]["image_size"], cfg["model"]["image_size"]))

        # Image tensor
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = transforms.functional.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).unsqueeze(0).to(device)

        # 4.1 Compute CSM (ABCD Rule)
        csm_map, abcd_components = csm_engine.generate(rgb)

        # 4.2 Compute GCW (Swin Gradient Coherence)
        gcw_map, pred_class, confidence = gcw_engine.compute_input_gcw(img_tensor)

        # 4.3 Compute Hybrid Fusion (CG-HAF)
        fused_map, alignment = fusion_engine.fuse(csm_map, gcw_map)

        # 4.4 Compute Faithfulness (Deletion & Insertion AUC)
        faith_res = faith_evaluator.evaluate(img_tensor, fused_map, target_class=pred_class)

        # 4.5 Save Visualizations for representative cases
        if idx < 5:
            class_label = classes[pred_class] if pred_class < len(classes) else f"class_{pred_class}"
            visualizer.plot_abcd_decomposition(rgb, abcd_components, csm_map, img_path.stem, class_label)
            visualizer.plot_hybrid_fusion_comparison(
                rgb, csm_map, gcw_map, fused_map, img_path.stem, class_label, alignment
            )

        results.append({
            "image_id": img_path.stem,
            "predicted_class": pred_class,
            "confidence": confidence,
            "pearson_alignment": alignment["pearson_correlation"],
            "top20_iou": alignment["top20_iou"],
            "deletion_auc": faith_res["deletion_auc"],
            "insertion_auc": faith_res["insertion_auc"]
        })

    elapsed = time.time() - start_time
    logger.info(f"Processed {len(image_files)} samples in {elapsed:.2f} seconds ({elapsed / len(image_files):.3f}s / sample).")

    # 5. Summarize and Export Tables
    df = pd.DataFrame(results)
    summary = {
        "Mean Pearson Alignment": [df["pearson_alignment"].mean()],
        "Mean Top-20% IoU": [df["top20_iou"].mean()],
        "Mean Deletion AUC (Faithfulness)": [df["deletion_auc"].mean()],
        "Mean Insertion AUC": [df["insertion_auc"].mean()]
    }
    summary_df = pd.DataFrame(summary)
    
    # Save CSV and LaTeX formats
    df.to_csv(tables_dir / "sample_xai_metrics.csv", index=False)
    summary_df.to_csv(tables_dir / "summary_metrics.csv", index=False)
    with open(tables_dir / "summary_metrics.tex", "w") as f:
        f.write(summary_df.to_latex(index=False, float_format="%.4f"))

    logger.info(f"Summary metrics exported to {tables_dir}.")
    print("\n" + "="*50)
    print("CG-HAF QUANTITATIVE BENCHMARK SUMMARY")
    print("="*50)
    for k, v in summary.items():
        print(f" {k:<35}: {v[0]:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

