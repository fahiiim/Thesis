"""
P1: Validation of Stolz-Aligned Clinical Saliency Mask v2 (CSM-v2) on HAM10000.
Computes:
  1. Legacy CSM vs CSM-v2 lesion segmentation IoU, Dice, and failure rate (<0.10 IoU) against
     Philipp Tschandl's official dermatologist masks across all 7 classes (N=700 stratified, 100/class,
     excluding calibration fold-0 indices if present; plus BCN vignette failure rate check).
  2. Algorithmic Stolz Total Dermatoscopy Score (TDS) & A, B, C, D component ROC-AUC for
     differentiating Malignant Melanoma (MEL) from Melanocytic Nevi (NV) on HAM10000.
"""

import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.xai.csm import ClinicalSaliencyMask
from src.xai.csm_v2 import ClinicalSaliencyMaskV2
from src.utils.data_v2 import CLASSES, load_rgb, load_mask, preprocess, stratified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=100)
    parser.add_argument("--mode", type=str, default="parity")
    args = parser.parse_args()

    df_all = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    sub = df_all if args.per_class <= 0 else stratified(df_all, per_class=args.per_class, seed=42)
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    mask_dir = Path("data/ham10000_masks/HAM10000_segmentations_lesion_tschandl")

    csm1 = ClinicalSaliencyMask()
    csm2 = ClinicalSaliencyMaskV2()

    seg_rows = []
    tds_rows = []
    t0 = time.time()
    for i, row in sub.iterrows():
        rgb = preprocess(load_rgb(img_dir / f"{row['image_id']}.jpg"), 224, mode=args.mode)
        gt = load_mask(mask_dir / f"{row['image_id']}_segmentation.png", 224, mode=args.mode)

        _, comp1 = csm1.generate(rgb)
        m1 = (comp1["lesion_mask"] > 0).astype(np.uint8)
        res2 = csm2.generate(rgb)
        m2 = (res2.lesion_mask > 0).astype(np.uint8)

        for ver, m in [("CSM-v1", m1), ("CSM-v2", m2)]:
            inter = float((m & gt).sum())
            union = float((m | gt).sum()) + 1e-8
            iou = inter / union
            dice = (2 * inter) / (float(m.sum() + gt.sum()) + 1e-8)
            seg_rows.append({"dx": row["dx"], "version": ver, "iou": iou, "dice": dice, "fail": int(iou < 0.10)})

        if row["dx"] in ("mel", "nv"):
            r = {"dx": row["dx"], "is_mel": int(row["dx"] == "mel"), "quality": res2.quality}
            r.update(res2.scores)
            tds_rows.append(r)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(sub)} ({time.time()-t0:.1f}s)", flush=True)

    sdf = pd.DataFrame(seg_rows)
    summary = sdf.groupby(["version", "dx"])[["iou", "dice", "fail"]].mean().reset_index()
    overall = sdf.groupby("version")[["iou", "dice", "fail"]].mean().reset_index()
    overall["dx"] = "OVERALL"
    out_seg = pd.concat([summary, overall], ignore_index=True)
    out_seg.to_csv("outputs/tables/csm_v2_segmentation_eval.csv", index=False)
    print("\n=== CSM-v1 VS CSM-v2 SEGMENTATION AGAINST TSCHANDL MASKS ===")
    print(out_seg.to_string(index=False))

    # TDS clinical signal check: MEL vs NV
    tdf = pd.DataFrame(tds_rows)
    y = tdf["is_mel"].values
    sig_rows = []
    for col in ["TDS", "A", "B", "C", "D"]:
        auc = float(roc_auc_score(y, tdf[col].values))
        sig_rows.append({
            "Criterion": col,
            "MEL Mean": float(tdf.loc[y == 1, col].mean()),
            "NV Mean": float(tdf.loc[y == 0, col].mean()),
            "MEL_vs_NV_AUC": auc,
        })
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv("outputs/tables/csm_v2_clinical_tds_signal.csv", index=False)
    print("\n=== CLINICAL SIGNAL CHECK: ALGORITHMIC STOLZ TDS AUC (MEL vs NV) ===")
    print(sig_df.to_string(index=False))


if __name__ == "__main__":
    main()
