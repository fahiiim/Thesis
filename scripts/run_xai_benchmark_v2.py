"""
P2: Rigorous XAI Benchmark v2 (Centred & De-Centred Views, Patch Flipping SRG, TOST & Holm).
Evaluates:
  - Controls: Random Noise, Center Gaussian, Sobel, Otsu Alone, CSM-v2 Alone
  - Neural: Legacy Swin-GCW, Stage-4 Grad-CAM (pre-norm), Swin-TCAM (new post-norm/LayerCAM)
  - Fusions: Additive CG-HAF (legacy), EG-HAF (Swin-TCAM + CSM-v2), Grad-CAM + CSM-v2 (EG 0.5)
"""

import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.csm_v2 import ClinicalSaliencyMaskV2
from src.xai.token_cam import SwinTokenCAM
from src.xai.gated_fusion import EvidenceGatedFusion
from src.evaluation.xai_benchmarks import BaselineXAI
from src.evaluation.metrics import LocalizationMetrics
from src.evaluation.rigor import (
    area_matched_iou, saliency_auprc, pointing_hit_tie_aware,
    decentred_crop, patch_flipping, tost_paired, holm
)
from src.utils.calibration import bootstrap_ci
from src.utils.data_v2 import CLASSES, load_rgb, load_mask, preprocess, to_tensor, stratified

METHODS = [
    "Random Noise", "Center Gaussian", "Sobel Edge", "CSM-v2 (Prior)",
    "Legacy Swin-GCW", "Stage-4 Grad-CAM", "Swin-TCAM",
    "EG-HAF (k=0.5)", "Grad-CAM + CSM-v2 (EG 0.5)"
]


def build_maps(rgb224, gt_mask, pred_idx, pred_cls, model, gcw_eng, tcam_eng, base_eng, csm2, device, center_gauss, seed=0):
    x = to_tensor(rgb224, normalise=True).unsqueeze(0).to(device)
    with torch.enable_grad():
        m_gcw, _, _ = gcw_eng.compute_hierarchical_gcw(x, target_class=pred_idx)
        m_gcam = base_eng.compute_gradcam(x, target_class=pred_idx)
        m_tcam = tcam_eng(x, target=torch.tensor([pred_idx], device=device)).fused[0].cpu().numpy()

    csm_res = csm2.generate(rgb224)
    csm_p, csm_q = csm_res.saliency, csm_res.quality
    eg = EvidenceGatedFusion(kappa_max=0.5).fuse(m_tcam, csm_p, class_name=pred_cls, quality=csm_q).fused
    gc_eg = EvidenceGatedFusion(kappa_max=0.5).fuse(m_gcam, csm_p, class_name=pred_cls, quality=csm_q).fused

    gray = cv2.cvtColor(rgb224, cv2.COLOR_RGB2GRAY)
    sobel = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    sobel /= (sobel.max() + 1e-8)

    rng = np.random.default_rng(seed)
    rand_m = cv2.GaussianBlur(rng.random((224, 224), dtype=np.float32), (15, 15), 0)
    rand_m = (rand_m - rand_m.min()) / (rand_m.max() - rand_m.min() + 1e-8)

    maps = {
        "Random Noise": rand_m,
        "Center Gaussian": center_gauss,
        "Sobel Edge": sobel,
        "CSM-v2 (Prior)": csm_p,
        "Legacy Swin-GCW": m_gcw,
        "Stage-4 Grad-CAM": m_gcam,
        "Swin-TCAM": m_tcam,
        "EG-HAF (k=0.5)": eg,
        "Grad-CAM + CSM-v2 (EG 0.5)": gc_eg,
    }
    return x, maps, csm_res


def eval_view(maps, x, gt_mask, pred_idx, model, loc_m, store):
    faith_blur = patch_flipping(model, x, maps, target=pred_idx, patch=16, steps=14, baseline="blur")
    faith_mean = patch_flipping(model, x, maps, target=pred_idx, patch=16, steps=14, baseline="mean")
    for m in METHODS:
        arr = maps[m]
        hit = pointing_hit_tie_aware(arr, gt_mask)
        energy = loc_m.energy_inside_mask(arr, gt_mask)
        am_iou = area_matched_iou(arr, gt_mask)
        auprc = saliency_auprc(arr, gt_mask)
        iou20 = loc_m.evaluate_sample(arr, gt_mask)["mask_iou"]
        store[m]["hit"].append(hit)
        store[m]["energy"].append(energy)
        store[m]["am_iou"].append(am_iou)
        store[m]["auprc"].append(auprc)
        store[m]["iou20"].append(iou20)
        store[m]["srg_blur"].append(faith_blur[m]["srg"])
        store[m]["srg_mean"].append(faith_mean[m]["srg"])


def summarise(store, groups, title):
    rows = []
    for m in METHODS:
        d = store[m]
        r = {"Method": m}
        for k, scale in [("hit", 100), ("energy", 100), ("am_iou", 100), ("auprc", 100), ("srg_blur", 1), ("srg_mean", 1)]:
            arr = np.asarray(d[k], float) * scale
            pt, lo, hi = bootstrap_ci(lambda idx, a=arr: np.nanmean(a[idx]), len(arr), groups=groups, n_boot=1000, seed=0)
            r[k] = f"{pt:.2f} [{lo:.2f}, {hi:.2f}]" if scale == 100 else f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"\n=== {title} (N={len(store[METHODS[0]]['hit'])}) ===")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=40)
    parser.add_argument("--mode", type=str, default="parity")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedSwinTransformer(7, True, checkpoint_path="weights/swin_vit_xai_model.pth").to(device).eval()

    df_all = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    # Exclude fold-0 calibration set if available so XAI eval is strictly disjoint
    cal_file = Path(f"outputs/calibration/bcts_{args.mode}.pt")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    mask_dir = Path("data/ham10000_masks/HAM10000_segmentations_lesion_tschandl")
    gcw_eng = GradientCoherenceWeighting(model, device)
    tcam_eng = SwinTokenCAM(model)
    base_eng = BaselineXAI(model, device)
    csm2 = ClinicalSaliencyMaskV2()
    loc_m = LocalizationMetrics()

    if cal_file.exists():
        cal = torch.load(cal_file, map_location="cpu", weights_only=False)
        df_cal = stratified(df_all.iloc[cal["cal_indices"]].reset_index(drop=True), per_class=20, seed=7)
        df_pool = df_all.iloc[cal["eval_indices"]].reset_index(drop=True)
        # Hyperparameter sweep (kappa_max & beta) strictly on disjoint calibration split
        print(f"Running disjoint calibration sweep on N={len(df_cal)} calibration lesions...", flush=True)
        cal_kappas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
        cal_rec = {k: {"srg_blur": [], "am_iou": [], "auprc": []} for k in cal_kappas}
        for _, cr in df_cal.iterrows():
            rgb224 = preprocess(load_rgb(img_dir / f"{cr['image_id']}.jpg"), 224, mode=args.mode)
            gt224 = load_mask(mask_dir / f"{cr['image_id']}_segmentation.png", 224, mode=args.mode)
            xc = to_tensor(rgb224, normalise=True).unsqueeze(0).to(device)
            with torch.no_grad():
                pc = int(model(xc).argmax(1).item())
            with torch.enable_grad():
                nt = tcam_eng(xc, target=torch.tensor([pc], device=device)).fused[0].cpu().numpy()
            cres = csm2.generate(rgb224)
            kmaps = {f"k={kv}": EvidenceGatedFusion(kappa_max=kv).fuse(nt, cres.saliency, class_name=CLASSES[pc], quality=cres.quality).fused for kv in cal_kappas}
            kf = patch_flipping(model, xc, kmaps, target=pc, patch=16, steps=14, baseline="blur")
            for kv in cal_kappas:
                km = kmaps[f"k={kv}"]
                cal_rec[kv]["srg_blur"].append(kf[f"k={kv}"]["srg"])
                cal_rec[kv]["am_iou"].append(area_matched_iou(km, gt224))
                cal_rec[kv]["auprc"].append(saliency_auprc(km, gt224))
        cal_df = pd.DataFrame([{"kappa_max": kv, "SRG_blur": np.mean(cal_rec[kv]["srg_blur"]), "AM_IoU": np.mean(cal_rec[kv]["am_iou"]), "AUPRC": np.mean(cal_rec[kv]["auprc"])} for kv in cal_kappas])
        cal_df.to_csv("outputs/tables/kappa_selection_cal_split.csv", index=False)
        print("Disjoint Calibration Kappa Selection:\n", cal_df.to_string(index=False), flush=True)
    else:
        df_pool = df_all

    sub = df_pool if args.per_class <= 0 else stratified(df_pool, per_class=args.per_class, seed=42)

    y_g, x_g = np.ogrid[:224, :224]
    center_gauss = np.exp(-((x_g - 112.0) ** 2 + (y_g - 112.0) ** 2) / (2 * 56.0 ** 2)).astype(np.float32)

    centred = {m: {k: [] for k in ["hit", "energy", "am_iou", "auprc", "iou20", "srg_blur", "srg_mean"]} for m in METHODS}
    decentred = {m: {k: [] for k in ["hit", "energy", "am_iou", "auprc", "iou20", "srg_blur", "srg_mean"]} for m in METHODS}
    c_groups, dc_groups = [], []
    border_evidence = {"mel": [], "nv": []}
    csm_ious = {c: [] for c in CLASSES}

    rng = np.random.default_rng(123)
    t0 = time.time()
    for i, row in sub.iterrows():
        rgb_raw = load_rgb(img_dir / f"{row['image_id']}.jpg")
        gt_raw = (cv2.imread(str(mask_dir / f"{row['image_id']}_segmentation.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        rgb224 = preprocess(rgb_raw, 224, mode=args.mode)
        gt224 = load_mask(mask_dir / f"{row['image_id']}_segmentation.png", 224, mode=args.mode)

        x_c = to_tensor(rgb224, normalise=True).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_c = int(model(x_c).argmax(1).item())
        _, maps_c, csm_res = build_maps(rgb224, gt224, pred_c, CLASSES[pred_c], model, gcw_eng, tcam_eng, base_eng, csm2, device, center_gauss, seed=i)
        eval_view(maps_c, x_c, gt224, pred_c, model, loc_m, centred)
        c_groups.append(row["lesion_id"])

        # CSM-v2 mask IoU vs Tschandl
        inter = float((csm_res.lesion_mask & gt224).sum())
        union = float((csm_res.lesion_mask | gt224).sum()) + 1e-8
        csm_ious[row["dx"]].append(inter / union)

        # Neural border-band evidence test (Section 5.4: MEL vs NV)
        if row["dx"] in ("mel", "nv"):
            k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            band = ((cv2.dilate(gt224, k11) - cv2.erode(gt224, k11)) > 0)
            border_evidence[row["dx"]].append(loc_m.energy_inside_mask(maps_c["Swin-TCAM"], band))

        # De-centred counterfactual view
        dc = decentred_crop(rgb_raw, gt_raw, rng)
        if dc is not None:
            rgb_dc, gt_dc, off = dc
            x_dc = to_tensor(rgb_dc, normalise=True).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_dc = int(model(x_dc).argmax(1).item())
            _, maps_dc, _ = build_maps(rgb_dc, gt_dc, pred_dc, CLASSES[pred_dc], model, gcw_eng, tcam_eng, base_eng, csm2, device, center_gauss, seed=i + 10000)
            eval_view(maps_dc, x_dc, gt_dc, pred_dc, model, loc_m, decentred)
            dc_groups.append(row["lesion_id"])

        if (i + 1) % 25 == 0:
            print(f"  Evaluated {i+1}/{len(sub)} ({time.time()-t0:.1f}s)...")

    df_c = summarise(centred, np.array(c_groups), f"CENTRED VIEW ({args.mode})")
    df_dc = summarise(decentred, np.array(dc_groups), "DE-CENTRED COUNTERFACTUAL VIEW")
    df_c.to_csv("outputs/tables/xai_benchmark_v2_centred.csv", index=False)
    df_dc.to_csv("outputs/tables/xai_benchmark_v2_decentred.csv", index=False)

    # Multiplicity-controlled paired tests (Holm) and TOST (pre-registered margin = 0.10 nats SRG)
    pvals = {}
    srg_eg = np.array(centred["EG-HAF (k=0.5)"]["srg_blur"])
    srg_tcam = np.array(centred["Swin-TCAM"]["srg_blur"])
    srg_gcam = np.array(centred["Stage-4 Grad-CAM"]["srg_blur"])
    srg_gcg = np.array(centred["Grad-CAM + CSM-v2 (EG 0.5)"]["srg_blur"])

    pvals["EG_vs_SwinTCAM_srg"] = float(wilcoxon(srg_eg, srg_tcam).pvalue)
    pvals["EG_vs_GradCAM_srg"] = float(wilcoxon(srg_eg, srg_gcam).pvalue)
    pvals["GradCAM_EG_vs_GradCAM_srg"] = float(wilcoxon(srg_gcg, srg_gcam).pvalue)
    pvals["EG_vs_SwinTCAM_iou"] = float(wilcoxon(centred["EG-HAF (k=0.5)"]["am_iou"], centred["Swin-TCAM"]["am_iou"]).pvalue)
    pvals["EG_vs_GradCAM_iou"] = float(wilcoxon(centred["EG-HAF (k=0.5)"]["am_iou"], centred["Stage-4 Grad-CAM"]["am_iou"]).pvalue)
    adj = holm(pvals)
    tost_eg_gcam = tost_paired(srg_eg, srg_gcam, margin=0.10)
    tost_eg_tcam = tost_paired(srg_eg, srg_tcam, margin=0.10)

    # Save stats CSV
    stat_rows = [
        {"Comparison": k, "Raw_p": pvals[k], "Holm_p": adj[k], "Test": "Wilcoxon_Paired"}
        for k in pvals
    ]
    for cname, tres in [("TOST_EG_vs_GradCAM_margin0.10", tost_eg_gcam), ("TOST_EG_vs_SwinTCAM_margin0.10", tost_eg_tcam)]:
        stat_rows.append({
            "Comparison": cname,
            "Raw_p": tres.p_tost,
            "Holm_p": tres.p_tost,
            "Test": f"TOST_equiv={tres.equivalent}_diff={tres.mean_diff:+.4f}_CI90=[{tres.ci90_low:+.4f},{tres.ci90_high:+.4f}]",
        })
    pd.DataFrame(stat_rows).to_csv("outputs/tables/xai_benchmark_v2_stats.csv", index=False)

    print("\n=== PAIRED TESTS (HOLM ADJUSTED) & TOST EQUIVALENCE (MARGIN=0.10 SRG) ===")
    for k in pvals:
        print(f"  {k:<32}: raw_p={pvals[k]:.4e} | holm_p={adj[k]:.4e}")
    print("  TOST (EG-HAF vs Grad-CAM, margin=0.10):", tost_eg_gcam)
    print("  TOST (EG-HAF vs Swin-TCAM, margin=0.10):", tost_eg_tcam)

    csm_sum_rows = []
    print("\n=== CSM-v2 SEGMENTATION IoU VS TSCHANDL BY CLASS ===")
    for c in CLASSES:
        arr = np.array(csm_ious[c])
        csm_sum_rows.append({"Class": c.upper(), "IoU_Mean": float(arr.mean()), "IoU_SD": float(arr.std()), "Fail_Rate_lt0.1": float((arr < 0.1).mean()), "N": len(arr)})
        print(f"  {c.upper():<6}: IoU={arr.mean()*100:.2f} +/- {arr.std()*100:.2f}% | fail(<0.1)={(arr<0.1).mean()*100:.1f}%")
    pd.DataFrame(csm_sum_rows).to_csv("outputs/tables/csm_v2_ious_summary.csv", index=False)

    be_df = pd.DataFrame([
        {"Class": "MEL", "Border_Energy_Mean": float(np.mean(border_evidence["mel"])), "Border_Energy_SD": float(np.std(border_evidence["mel"])), "N": len(border_evidence["mel"])},
        {"Class": "NV", "Border_Energy_Mean": float(np.mean(border_evidence["nv"])), "Border_Energy_SD": float(np.std(border_evidence["nv"])), "N": len(border_evidence["nv"])},
    ])
    be_df.to_csv("outputs/tables/neural_border_evidence.csv", index=False)
    print("\n=== NEURAL BORDER EVIDENCE CHECK (Swin-TCAM energy in GT border band) ===")
    print(f"  MEL border energy: {np.mean(border_evidence['mel'])*100:.2f}% | NV border energy: {np.mean(border_evidence['nv'])*100:.2f}%")


if __name__ == "__main__":
    main()
