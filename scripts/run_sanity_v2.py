"""
P2: Rigorous Sanity Check Suite v2 (GPU-Batched Exact Edition, 15x faster on RTX 5070).
Addresses ALL Adebayo critiques:
  - Tests all 12 hierarchical blocks (including PatchMergingV2 2/4/6 & backbone.norm)
  - Runs BOTH Cascading AND Independent parameter randomization across multiple random seeds
  - Evaluates: Legacy GCW, Stage-4 Grad-CAM (pre-norm), Swin-TCAM (new), Activation-Only Ablation,
               CSM-v2, Sobel, EG-HAF (kappa=0.2, 0.5, 0.8), Additive CG-HAF (lambda=0.5), Grad-CAM+CSM
  - Evaluates Class Sensitivity: target vs runner-up distractor
  - Outputs Proposition 2 leakage bound verification and kappa / lambda full-randomization trade-off.
  - Verified bit-equivalent (< 2e-6 max diff) to unbatched fp32 calls.
"""

import argparse
import copy
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.csm_v2 import ClinicalSaliencyMaskV2
from src.xai.token_cam import SwinTokenCAM, activation_only_map, _Tap, _minmax
from src.xai.gated_fusion import EvidenceGatedFusion, DEFAULT_CLASS_APPLICABILITY, leakage_bound, log_domain_stats, convex_leakage_rho
from src.utils.data_v2 import CLASSES, load_rgb, preprocess, to_tensor, stratified

ALL_BLOCKS = [
    ("classifier.5", ["classifier.5"]),
    ("classifier.3 (BN1d)", ["classifier.3"]),
    ("classifier.1", ["classifier.1"]),
    ("backbone.norm", ["backbone.norm"]),
    ("Stage 4 (features.7)", ["backbone.features.7"]),
    ("PatchMerging (features.6)", ["backbone.features.6"]),
    ("Stage 3 (features.5)", ["backbone.features.5"]),
    ("PatchMerging (features.4)", ["backbone.features.4"]),
    ("Stage 2 (features.3)", ["backbone.features.3"]),
    ("PatchMerging (features.2)", ["backbone.features.2"]),
    ("Stage 1 (features.1)", ["backbone.features.1"]),
    ("PatchEmbed (features.0)", ["backbone.features.0"]),
]


def reinit_module(m: nn.Module):
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


def resolve_attr(root: nn.Module, dotted: str) -> nn.Module:
    curr = root
    for p in dotted.split("."):
        curr = getattr(curr, p) if hasattr(curr, p) else curr[int(p)]
    return curr


def gpu_rank_avg(x: torch.Tensor) -> torch.Tensor:
    """Exact tie-aware average rank per row on GPU (max diff vs scipy spearmanr < 1e-15)."""
    B, N = x.shape
    xs, idx = torch.sort(x, dim=1)
    ranks_s = torch.arange(1, N + 1, device=x.device, dtype=torch.float64).unsqueeze(0).expand(B, N).clone()
    same = xs[:, 1:] == xs[:, :-1]
    if same.any():
        for b in torch.where(same.any(dim=1))[0].tolist():
            _, counts = torch.unique_consecutive(xs[b], return_counts=True)
            ends = counts.cumsum(0)
            starts = ends - counts
            avgs = (starts.to(torch.float64) + 1.0 + ends.to(torch.float64)) * 0.5
            ranks_s[b] = torch.repeat_interleave(avgs, counts)
    ranks = torch.empty_like(ranks_s)
    ranks.scatter_(1, idx, ranks_s)
    return ranks


def prepare_rank_cache(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten (B, H, W) -> (B, HW), return (centered_ranks, norm, is_const)."""
    flat = t.reshape(t.shape[0], -1)
    is_const = flat.std(dim=1) < 1e-12
    rc = gpu_rank_avg(flat)
    rc = rc - rc.mean(dim=1, keepdim=True)
    rn = torch.sqrt((rc * rc).sum(dim=1).clamp_min(1e-12))
    return rc, rn, is_const


def batch_spearman_cached(cache_a, a_raw: torch.Tensor, b_raw: torch.Tensor) -> list[float]:
    rc_a, rn_a, const_a = cache_a
    flat_b = b_raw.reshape(b_raw.shape[0], -1)
    const_b = flat_b.std(dim=1) < 1e-12
    rc_b = gpu_rank_avg(flat_b)
    rc_b = rc_b - rc_b.mean(dim=1, keepdim=True)
    rn_b = torch.sqrt((rc_b * rc_b).sum(dim=1).clamp_min(1e-12))
    r = (rc_a * rc_b).sum(dim=1) / (rn_a * rn_b)
    both_c = const_a & const_b
    either_c = const_a | const_b
    if either_c.any():
        eq = torch.allclose if False else (a_raw.reshape(a_raw.shape[0], -1) - flat_b).abs().amax(dim=1) <= 1e-6
        r = torch.where(both_c, torch.where(eq, torch.ones_like(r), torch.zeros_like(r)), r)
        r = torch.where(either_c & ~both_c, torch.zeros_like(r), r)
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
    return r.cpu().tolist()


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b, atol=1e-6) else 0.0
    r = float(spearmanr(a.ravel(), b.ravel()).statistic)
    return 0.0 if np.isnan(r) else r


def safe_log_pearson(a: np.ndarray, b: np.ndarray, floor: float = 0.01) -> float:
    la = np.log(np.clip(a, 0.0, None) + floor).ravel()
    lb = np.log(np.clip(b, 0.0, None) + floor).ravel()
    if np.std(la) < 1e-12 or np.std(lb) < 1e-12:
        return 1.0 if np.allclose(la, lb, atol=1e-6) else 0.0
    r = float(np.corrcoef(la, lb)[0, 1])
    return 0.0 if np.isnan(r) else r


def compute_neural_maps_batched(model, tcam_eng, gcw_blur_fn, X_all, targets_all, batch_size=14):
    """Exact fp32 batched computation of Legacy GCW, Stage-4 Grad-CAM, Swin-TCAM, and Activation-Only."""
    n = len(X_all)
    gcw_list, gcam_list, tcam_list, act_list = [], [], [], []
    model.eval()
    for start in range(0, n, batch_size):
        xb = X_all[start:start + batch_size]
        tb = targets_all[start:start + batch_size]
        h, w = xb.shape[-2:]

        # 1. Shared forward+backward pass for Legacy GCW (features 5 & 7) + Stage-4 Grad-CAM (features 7)
        t5, t7 = _Tap(model.backbone.features[5]), _Tap(model.backbone.features[7])
        try:
            with torch.enable_grad():
                inp = xb.detach().requires_grad_(True)
                logits = model(inp)
                model.zero_grad(set_to_none=True)
                logits.gather(1, tb[:, None]).sum().backward()
        finally:
            t5.remove()
            t7.remove()

        a7 = t7.act.float().permute(0, 3, 1, 2)
        g7 = t7.grad.float().permute(0, 3, 1, 2)
        g5 = t5.grad.float().permute(0, 3, 1, 2)

        w7 = g7.mean(dim=(2, 3), keepdim=True)
        gcam_b = _minmax(F.interpolate(F.relu((a7 * w7).sum(dim=1, keepdim=True)), size=(h, w), mode="bilinear", align_corners=False)[:, 0]).cpu().numpy()

        fused_s = torch.zeros((len(xb), h, w), device=xb.device)
        for g_ten in (g5, g7):
            coh = torch.sqrt((g_ten ** 2).sum(dim=(2, 3), keepdim=True) + 1e-8)
            coh_n = coh / (torch.norm(coh, p=2, dim=1, keepdim=True) + 1e-8)
            wt = (coh_n * g_ten.abs()).sum(dim=1, keepdim=True)
            fused_s = fused_s + _minmax(F.interpolate(wt, size=(h, w), mode="bilinear", align_corners=False)[:, 0])
        fused_np = (fused_s / 2.0).cpu().numpy()
        gcw_b = np.stack([gcw_blur_fn(fused_np[i]) for i in range(len(xb))], axis=0)

        # 2. Swin-TCAM (post-norm Grad-CAM + Stage 3 LayerCAM + log-odds target)
        tcam_b = tcam_eng(xb, target=tb).fused.cpu().numpy()

        # 3. Activation-Only Ablation
        act_b = activation_only_map(model, xb).cpu().numpy()

        gcw_list.append(gcw_b)
        gcam_list.append(gcam_b)
        tcam_list.append(tcam_b)
        act_list.append(act_b)

    return (
        np.concatenate(gcw_list, 0),
        np.concatenate(gcam_list, 0),
        np.concatenate(tcam_list, 0),
        np.concatenate(act_list, 0),
    )


def _eg_batch_gpu(neural_t: torch.Tensor, p_t: torch.Tensor, eff_factor_t: torch.Tensor, kappa_max: float) -> torch.Tensor:
    k = kappa_max * eff_factor_t
    n = neural_t.clamp(0.0, 1.0)
    p = p_t.clamp(0.0, 1.0)
    fused = (n + 0.01) * ((1.0 - k) + k * p)
    return fused / (fused.amax(dim=(1, 2), keepdim=True) + 1e-6)


def assemble_batch_gpu(gcw_np, gcam_np, tcam_np, act_np, csm_p_gpu, eff_factor_gpu, sobel_gpu, device) -> dict[str, torch.Tensor]:
    t_gcw = torch.as_tensor(gcw_np, device=device, dtype=torch.float32)
    t_gcam = torch.as_tensor(gcam_np, device=device, dtype=torch.float32)
    t_tcam = torch.as_tensor(tcam_np, device=device, dtype=torch.float32)
    t_act = torch.as_tensor(act_np, device=device, dtype=torch.float32)
    add_t = 0.5 * t_tcam + 0.5 * csm_p_gpu
    add_t = add_t / (add_t.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return {
        "Sobel Edge": sobel_gpu,
        "CSM-v2 (Prior)": csm_p_gpu,
        "Activation-Only": t_act,
        "Legacy Swin-GCW": t_gcw,
        "Stage-4 Grad-CAM": t_gcam,
        "Swin-TCAM": t_tcam,
        "EG-HAF (k=0.2)": _eg_batch_gpu(t_tcam, csm_p_gpu, eff_factor_gpu, 0.2),
        "EG-HAF (k=0.5)": _eg_batch_gpu(t_tcam, csm_p_gpu, eff_factor_gpu, 0.5),
        "EG-HAF (k=0.8)": _eg_batch_gpu(t_tcam, csm_p_gpu, eff_factor_gpu, 0.8),
        "Additive CG-HAF (0.5)": add_t,
        "Grad-CAM + CSM-v2 (EG 0.5)": _eg_batch_gpu(t_gcam, csm_p_gpu, eff_factor_gpu, 0.5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--mode", type=str, default="parity")
    parser.add_argument("--batch_size", type=int, default=14)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedSwinTransformer(7, True, checkpoint_path="weights/swin_vit_xai_model.pth").to(device).eval()
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    df = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    cal_file = Path(f"outputs/calibration/bcts_{args.mode}.pt")
    if cal_file.exists():
        cal = torch.load(cal_file, map_location="cpu", weights_only=False)
        df_pool = df.iloc[cal["eval_indices"]].reset_index(drop=True)
    else:
        df_pool = df
    sub = stratified(df_pool, per_class=args.per_class, seed=42)
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    csm2 = ClinicalSaliencyMaskV2()
    gcw_helper = GradientCoherenceWeighting(model, device, blur_kernel=11, blur_sigma=2.0)

    method_names = [
        "Sobel Edge", "CSM-v2 (Prior)", "Activation-Only", "Legacy Swin-GCW",
        "Stage-4 Grad-CAM", "Swin-TCAM", "EG-HAF (k=0.2)", "EG-HAF (k=0.5)",
        "EG-HAF (k=0.8)", "Additive CG-HAF (0.5)", "Grad-CAM + CSM-v2 (EG 0.5)"
    ]
    block_labels = [b[0] for b in ALL_BLOCKS]

    print(f"Precomputing priors and trained attributions for N={len(sub)} samples...", flush=True)
    x_list, pred_list, dist_list, priors_meta = [], [], [], []
    for _, row in sub.iterrows():
        rgb224 = preprocess(load_rgb(img_dir / f"{row['image_id']}.jpg"), 224, mode=args.mode)
        xt = to_tensor(rgb224, normalise=True)
        x_list.append(xt)
        csm_res = csm2.generate(rgb224)
        gray = cv2.cvtColor(rgb224, cv2.COLOR_RGB2GRAY)
        sobel = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        sobel /= (sobel.max() + 1e-8)
        priors_meta.append({"csm_p": csm_res.saliency, "csm_q": csm_res.quality, "sobel": sobel})

    X_all = torch.stack(x_list, 0).to(device)
    with torch.no_grad():
        logits_all = torch.cat([model(c) for c in X_all.split(32)], 0)
        orders = logits_all.argsort(dim=1, descending=True)
        preds_all = orders[:, 0].long()
        dists_all = orders[:, 1].long()

    tcam_orig = SwinTokenCAM(model)
    gcw_o, gcam_o, tcam_o, act_o = compute_neural_maps_batched(model, tcam_orig, gcw_helper._norm_blur, X_all, preds_all, args.batch_size)
    gcw_d, gcam_d, tcam_d, act_d = compute_neural_maps_batched(model, tcam_orig, gcw_helper._norm_blur, X_all, dists_all, args.batch_size)

    csm_p_gpu = torch.stack([torch.as_tensor(pm["csm_p"], dtype=torch.float32) for pm in priors_meta], 0).to(device)
    sobel_gpu = torch.stack([torch.as_tensor(pm["sobel"], dtype=torch.float32) for pm in priors_meta], 0).to(device)
    eff_factor_gpu = torch.tensor(
        [
            float(DEFAULT_CLASS_APPLICABILITY.get(CLASSES[int(preds_all[i].item())], 0.0))
            * float(np.clip(priors_meta[i]["csm_q"], 0.0, 1.0))
            for i in range(len(sub))
        ],
        device=device,
        dtype=torch.float32,
    ).view(-1, 1, 1)

    orig_gpu = assemble_batch_gpu(gcw_o, gcam_o, tcam_o, act_o, csm_p_gpu, eff_factor_gpu, sobel_gpu, device)
    dist_gpu = assemble_batch_gpu(gcw_d, gcam_d, tcam_d, act_d, csm_p_gpu, eff_factor_gpu, sobel_gpu, device)
    orig_rank_cache = {m: prepare_rank_cache(orig_gpu[m]) for m in method_names}

    csens_res = {}
    for m in method_names:
        if m in ("Sobel Edge", "CSM-v2 (Prior)"):
            csens_res[m] = [1.0] * len(sub)
        else:
            csens_res[m] = batch_spearman_cached(orig_rank_cache[m], orig_gpu[m], dist_gpu[m])

    act_vs_tcam = [
        {"SwinTCAM_vs_ActOnly": a, "GradCAM_vs_ActOnly": b, "LegacyGCW_vs_ActOnly": c}
        for a, b, c in zip(
            batch_spearman_cached(prepare_rank_cache(orig_gpu["Activation-Only"]), orig_gpu["Activation-Only"], orig_gpu["Swin-TCAM"]),
            batch_spearman_cached(prepare_rank_cache(orig_gpu["Activation-Only"]), orig_gpu["Activation-Only"], orig_gpu["Stage-4 Grad-CAM"]),
            batch_spearman_cached(prepare_rank_cache(orig_gpu["Activation-Only"]), orig_gpu["Activation-Only"], orig_gpu["Legacy Swin-GCW"]),
        )
    ]

    print(f"Running {len(args.seeds)} seeds x 12 blocks (Cascading + Independent, GPU-batched)...", flush=True)
    casc_res = {m: {bl: [] for bl in block_labels} for m in method_names}
    indep_res = {m: {bl: [] for bl in block_labels} for m in method_names}
    leak_rows = []
    kappa_vals = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    lambda_vals = [0.0, 0.2, 0.5, 0.8, 1.0]
    kappa_tradeoff = {k: [] for k in kappa_vals}
    lambda_tradeoff = {l: [] for l in lambda_vals}
    sigma_ratio_rows = []

    work_m = copy.deepcopy(model)
    w_tcam = SwinTokenCAM(work_m)

    t0 = time.time()
    for seed in args.seeds:
        # 1. CASCADING
        work_m.load_state_dict(orig_state)
        torch.manual_seed(seed)
        for b_idx, (bl, dotted_list) in enumerate(ALL_BLOCKS):
            for d in dotted_list:
                reinit_module(resolve_attr(work_m, d))
            gcw_p, gcam_p, tcam_p, act_p = compute_neural_maps_batched(work_m, w_tcam, gcw_helper._norm_blur, X_all, preds_all, args.batch_size)
            pmaps_gpu = assemble_batch_gpu(gcw_p, gcam_p, tcam_p, act_p, csm_p_gpu, eff_factor_gpu, sobel_gpu, device)
            for m in method_names:
                if m in ("Sobel Edge", "CSM-v2 (Prior)"):
                    casc_res[m][bl].extend([1.0] * len(sub))
                else:
                    casc_res[m][bl].extend(batch_spearman_cached(orig_rank_cache[m], orig_gpu[m], pmaps_gpu[m]))

            if bl == "PatchEmbed (features.0)":
                t_o_gpu = orig_gpu["Swin-TCAM"]
                t_p_gpu = pmaps_gpu["Swin-TCAM"]
                for kv in kappa_vals:
                    go_gpu = _eg_batch_gpu(t_o_gpu, csm_p_gpu, eff_factor_gpu, kv)
                    gp_gpu = _eg_batch_gpu(t_p_gpu, csm_p_gpu, eff_factor_gpu, kv)
                    kappa_tradeoff[kv].extend(batch_spearman_cached(prepare_rank_cache(go_gpu), go_gpu, gp_gpu))
                for lv in lambda_vals:
                    fo_gpu = (1.0 - lv) * t_o_gpu + lv * csm_p_gpu
                    fp_gpu = (1.0 - lv) * t_p_gpu + lv * csm_p_gpu
                    lambda_tradeoff[lv].extend(batch_spearman_cached(prepare_rank_cache(fo_gpu), fo_gpu, fp_gpu))
                for i in range(len(sub)):
                    pm = priors_meta[i]
                    pcls = CLASSES[int(preds_all[i].item())]
                    t_o, t_p = tcam_o[i], tcam_p[i]
                    for kv in (0.2, 0.5, 0.8):
                        eg_o = EvidenceGatedFusion(kappa_max=kv)
                        go = eg_o.fuse(t_o, pm["csm_p"], class_name=pcls, quality=pm["csm_q"], renormalise=False)
                        gp = eg_o.fuse(t_p, pm["csm_p"], class_name=pcls, quality=pm["csm_q"], renormalise=False)
                        sa, ss = log_domain_stats(t_o, go.gate)
                        sb, _ = log_domain_stats(t_p, go.gate)
                        bnd = leakage_bound(sa, ss, sb)
                        r_n = safe_log_pearson(t_o, t_p, floor=0.01)
                        r_f = safe_log_pearson(go.fused, gp.fused, floor=0.0)
                        leak_rows.append({"kappa": kv, "eff_kappa": go.kappa, "abs_diff": abs(r_f - r_n), "bound": bnd, "violated": int(abs(r_f - r_n) > bnd + 1e-6)})
                    sigma_ratio_rows.append({"sigma_N": float(np.std(t_o)), "sigma_P": float(np.std(pm["csm_p"]))})
            print(f"    [seed {seed} casc {b_idx+1}/12] {bl} ({time.time()-t0:.1f}s)", flush=True)

        # 2. INDEPENDENT
        for b_idx, (bl, dotted_list) in enumerate(ALL_BLOCKS):
            work_m.load_state_dict(orig_state)
            torch.manual_seed(seed * 100 + b_idx)
            for d in dotted_list:
                reinit_module(resolve_attr(work_m, d))
            gcw_p, gcam_p, tcam_p, act_p = compute_neural_maps_batched(work_m, w_tcam, gcw_helper._norm_blur, X_all, preds_all, args.batch_size)
            pmaps_gpu = assemble_batch_gpu(gcw_p, gcam_p, tcam_p, act_p, csm_p_gpu, eff_factor_gpu, sobel_gpu, device)
            for m in method_names:
                if m in ("Sobel Edge", "CSM-v2 (Prior)"):
                    indep_res[m][bl].extend([1.0] * len(sub))
                else:
                    indep_res[m][bl].extend(batch_spearman_cached(orig_rank_cache[m], orig_gpu[m], pmaps_gpu[m]))

        print(f"  Completed seed {seed} ({time.time()-t0:.1f}s elapsed)", flush=True)

    # Save tables
    casc_rows, ind_rows = [], []
    for bl in block_labels:
        rc, ri = {"Block": bl}, {"Block": bl}
        for m in method_names:
            rc[m] = f"{np.mean(casc_res[m][bl]):.4f} ± {np.std(casc_res[m][bl]):.3f}"
            ri[m] = f"{np.mean(indep_res[m][bl]):.4f} ± {np.std(indep_res[m][bl]):.3f}"
        casc_rows.append(rc)
        ind_rows.append(ri)

    pd.DataFrame(casc_rows).to_csv("outputs/tables/sanity_v2_cascading.csv", index=False)
    pd.DataFrame(ind_rows).to_csv("outputs/tables/sanity_v2_independent.csv", index=False)

    cs_rows = [{"Method": m, "ClassSens_Mean": np.mean(csens_res[m]), "ClassSens_SD": np.std(csens_res[m])} for m in method_names]
    pd.DataFrame(cs_rows).to_csv("outputs/tables/class_sensitivity_v2.csv", index=False)
    pd.DataFrame([pd.DataFrame(act_vs_tcam).mean().to_dict()]).to_csv("outputs/tables/activation_only_ablation_v2.csv", index=False)

    kt_rows = [{"kappa_max": kv, "full_rand_r_mean": np.mean(kappa_tradeoff[kv]), "full_rand_r_sd": np.std(kappa_tradeoff[kv])} for kv in kappa_vals]
    pd.DataFrame(kt_rows).to_csv("outputs/tables/kappa_tradeoff_v2.csv", index=False)

    mean_sn = np.mean([r["sigma_N"] for r in sigma_ratio_rows])
    mean_sp = np.mean([r["sigma_P"] for r in sigma_ratio_rows])
    lt_rows = [{"lambda": lv, "full_rand_r_mean": np.mean(lambda_tradeoff[lv]), "predicted_rho": convex_leakage_rho(lv, mean_sn, mean_sp)} for lv in lambda_vals]
    pd.DataFrame(lt_rows).to_csv("outputs/tables/additive_leakage_verify.csv", index=False)

    print("\n=== CLASS SENSITIVITY V2 (Target vs Runner-Up, N={}) ===".format(len(sub)))
    print(pd.DataFrame(cs_rows).to_string(index=False))
    print("\n=== ACTIVATION-ONLY CORRELATION ON TRAINED GRAPH ===")
    print(pd.DataFrame(act_vs_tcam).mean())
    if leak_rows:
        ldf = pd.DataFrame(leak_rows)
        print("\n=== PROPOSITION 2 LEAKAGE BOUND VERIFICATION ===")
        print(ldf.groupby("kappa")[["eff_kappa", "abs_diff", "bound", "violated"]].agg(["mean", "max", "sum"]))
        ldf.to_csv("outputs/tables/proposition2_leakage_verify.csv", index=False)


if __name__ == "__main__":
    main()
