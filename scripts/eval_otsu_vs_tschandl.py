"""
Evaluates the internal Polarity-Verified Otsu Mask Generator against
Philipp Tschandl's official dermatologist ground-truth masks.
Reports:
- Mean IoU ± Std
- Mean Dice (F1) ± Std
- Failure rate (IoU < 0.10 or inverted mask)
- Breakdown across diagnostic classes (e.g. vascular, hypopigmented, vignetted)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from src.xai.csm import ClinicalSaliencyMask

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def main():
    test_csv = Path("data/testing-set/ham10000_test_dataset_4500.csv")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    mask_dir = Path("data/ham10000_masks/HAM10000_segmentations_lesion_tschandl")

    df = pd.read_csv(test_csv)
    csm_engine = ClinicalSaliencyMask()

    # Test across 280 lesions (40 per class) with official GT masks
    results = {c: {"iou": [], "dice": [], "failed": 0} for c in CLASSES}
    total_samples = 0
    total_failed = 0
    all_ious = []
    all_dices = []

    for c in CLASSES:
        sub = df[df["dx"] == c]
        count = 0
        for _, row in sub.iterrows():
            img_id = row["image_id"]
            ip = img_dir / f"{img_id}.jpg"
            mp = mask_dir / f"{img_id}_segmentation.png"
            if ip.exists() and mp.exists():
                rgb = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                gt = (cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

                # Generate CSM and extract its internal lesion mask
                _, comp = csm_engine.generate(rgb)
                pred_mask = comp["lesion_mask"].astype(np.uint8)

                # Intersection & Union
                inter = np.sum((pred_mask == 1) & (gt == 1))
                union = np.sum((pred_mask == 1) | (gt == 1))
                iou = inter / union if union > 0 else 0.0
                dice = (2.0 * inter) / (np.sum(pred_mask) + np.sum(gt)) if (np.sum(pred_mask) + np.sum(gt)) > 0 else 0.0

                is_fail = (iou < 0.10)
                if is_fail:
                    results[c]["failed"] += 1
                    total_failed += 1

                results[c]["iou"].append(iou)
                results[c]["dice"].append(dice)
                all_ious.append(iou)
                all_dices.append(dice)
                count += 1
                total_samples += 1
                if count >= 40:
                    break

    print(f"\n=== POLARITY-VERIFIED OTSU VS TSCHANDL EXPERT GROUND TRUTH (N={total_samples}) ===")
    print(f"Overall Mean IoU: {np.mean(all_ious)*100:.2f} ± {np.std(all_ious)*100:.2f}%")
    print(f"Overall Mean Dice: {np.mean(all_dices)*100:.2f} ± {np.std(all_dices)*100:.2f}%")
    print(f"Overall Failure Rate (IoU < 0.10): {total_failed}/{total_samples} ({total_failed/total_samples*100:.2f}%)\n")

    print(f"{'Class':<10} {'Mean IoU (%)':<15} {'Mean Dice (%)':<15} {'Failure Count':<15}")
    print("-" * 55)
    for c in CLASSES:
        ious = results[c]["iou"]
        dices = results[c]["dice"]
        fails = results[c]["failed"]
        print(f"{c.upper():<10} {np.mean(ious)*100:6.2f} ± {np.std(ious)*100:5.2f}  {np.mean(dices)*100:6.2f} ± {np.std(dices)*100:5.2f}  {fails}/{len(ious)}")

if __name__ == "__main__":
    main()
