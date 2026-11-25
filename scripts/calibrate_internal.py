"""
P1: Lesion-disjoint Post-Hoc Calibration & Melanoma Screening Policy (Internal HAM10000).
Uses 5-fold lesion-grouped out-of-fold (StratifiedGroupKFold on lesion_id) evaluation across
the 4,500 test images (3,917 unique lesions), AND fits a deployment BCTS scaler on fold 0
(saved to outputs/calibration/bcts_parity.pt & bcts_legacy.pt) for external cohorts.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

from src.models.swin_classifier import EnhancedSwinTransformer
from src.utils.data_v2 import CLASSES, load_rgb, preprocess, to_tensor
from src.utils.calibration import (
    TemperatureScaling, BiasCorrectedTemperature, VectorScaling, fit_scaler,
    calibration_report, fit_sensitivity_threshold, bayes_decision, melanoma_cost_matrix
)


class DS(Dataset):
    def __init__(self, paths, labels, mode="legacy"):
        self.paths = paths
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rgb = load_rgb(self.paths[idx])
        rgb224 = preprocess(rgb, size=224, mode=self.mode)
        return to_tensor(rgb224, normalise=True), self.labels[idx]


@torch.no_grad()
def extract_logits(model, paths, labels, device, mode="legacy", batch_size=128):
    dl = DataLoader(DS(paths, labels, mode=mode), batch_size=batch_size, num_workers=0)
    zs = []
    for i, (x, _) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            zs.append(model(x).float().cpu())
    return torch.cat(zs).numpy()


def make_group_boots(groups, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    n_g = len(uniq)
    return [np.concatenate([members[g] for g in rng.integers(0, n_g, n_g)]) for _ in range(n_boot)]


def fast_balanced_acc(y_true, y_pred, k=7):
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    denom = cm.sum(1)
    valid = denom > 0
    return float((np.diag(cm)[valid] / denom[valid]).mean())


def fast_ece(probs, labels, n_bins=15):
    conf = probs.max(1)
    corr = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    tot = len(labels)
    cnt = np.bincount(idx, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_corr = np.bincount(idx, weights=corr, minlength=n_bins)
    nz = cnt > 0
    return float(np.sum(np.abs(sum_corr[nz] - sum_conf[nz])) / tot)


def summarise_probs(probs, labels, boot_sets, tag, pred_override=None):
    rep = calibration_report(probs, labels)
    preds = probs.argmax(1) if pred_override is None else pred_override
    mel_idx = CLASSES.index("mel")
    y_mel = (labels == mel_idx)
    p_mel = (preds == mel_idx)

    ba_pt = fast_balanced_acc(labels, preds)
    ms_pt = float(p_mel[y_mel].mean())
    msp_pt = float((~p_mel)[~y_mel].mean())
    ece_pt = rep["ece_15"]

    bas, mss, msps, eces = [], [], [], []
    for idx in boot_sets:
        yt, yp = labels[idx], preds[idx]
        bas.append(fast_balanced_acc(yt, yp))
        ym, pm = y_mel[idx], p_mel[idx]
        mss.append(pm[ym].mean() if ym.any() else np.nan)
        msps.append((~pm)[~ym].mean() if (~ym).any() else np.nan)
        eces.append(fast_ece(probs[idx], yt, 15))

    def fmt_ci(pt, arr, pct=True):
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        s = 100.0 if pct else 1.0
        return f"{pt*s:.2f}% [{lo*s:.2f}, {hi*s:.2f}]" if pct else f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"

    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    acc = float((preds == labels).mean())
    mauc = float(roc_auc_score(np.eye(7)[labels], probs, multi_class="ovr", average="macro"))
    return {
        "Method": tag,
        "Accuracy": f"{acc*100:.2f}%",
        "Balanced Acc (95% CI)": fmt_ci(ba_pt, bas),
        "Macro F1": f"{mf1:.4f}",
        "Macro AUC": f"{mauc:.4f}",
        "MEL Sens (95% CI)": fmt_ci(ms_pt, mss),
        "MEL Spec (95% CI)": fmt_ci(msp_pt, msps),
        "ECE-15 (95% CI)": fmt_ci(ece_pt, eces),
        "Adaptive ECE": f"{rep['adaptive_ece_15']*100:.2f}%",
        "Classwise ECE": f"{rep['classwise_ece_15']*100:.2f}%",
        "NLL": f"{rep['nll']:.4f}",
        "Brier": f"{rep['brier']:.4f}",
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedSwinTransformer(7, True, checkpoint_path="weights/swin_vit_xai_model.pth").to(device).eval()

    df = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    paths = [img_dir / f"{r['image_id']}.jpg" for _, r in df.iterrows()]
    labels = np.array([CLASSES.index(c) for c in df["dx"]])
    lesion_ids = df["lesion_id"].values

    t0 = time.time()
    boot_sets = make_group_boots(lesion_ids, n_boot=1000, seed=42)
    print(f"Precomputed 1,000 lesion-grouped bootstrap indices in {time.time()-t0:.2f}s", flush=True)

    cal_dir = Path("outputs/calibration")
    cal_dir.mkdir(parents=True, exist_ok=True)

    for pre_mode in ["legacy", "parity"]:
        t1 = time.time()
        print(f"\nExtracting logits for {len(paths)} test images (preprocessing={pre_mode})...", flush=True)
        logits = extract_logits(model, paths, labels, device, mode=pre_mode)
        np.save(cal_dir / f"logits_ham_{pre_mode}.npy", logits)
        raw_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

        sgkf = StratifiedGroupKFold(n_splits=5)
        oof_ts = np.zeros_like(raw_probs)
        oof_bcts = np.zeros_like(raw_probs)
        oof_vs = np.zeros_like(raw_probs)
        oof_bcts_mel85 = np.zeros(len(labels), dtype=int)
        mel_idx = CLASSES.index("mel")

        for fold, (cal_idx, eval_idx) in enumerate(sgkf.split(logits, labels, groups=lesion_ids)):
            ts = fit_scaler(TemperatureScaling(), logits[cal_idx], labels[cal_idx])
            bcts = fit_scaler(BiasCorrectedTemperature(7), logits[cal_idx], labels[cal_idx])
            vs = fit_scaler(VectorScaling(7), logits[cal_idx], labels[cal_idx])

            oof_ts[eval_idx] = ts.probs(logits[eval_idx])
            oof_bcts[eval_idx] = bcts.probs(logits[eval_idx])
            oof_vs[eval_idx] = vs.probs(logits[eval_idx])

            cal_bcts_probs = bcts.probs(logits[cal_idx])
            thr_obj = fit_sensitivity_threshold(cal_bcts_probs, labels[cal_idx], mel_idx, target_sensitivity=0.85)
            oof_bcts_mel85[eval_idx] = thr_obj.apply(oof_bcts[eval_idx])

            if fold == 0:
                torch.save({
                    "bcts_state_dict": bcts.state_dict(),
                    "mel_thr_85": thr_obj.threshold,
                    "cal_indices": cal_idx,
                    "eval_indices": eval_idx,
                    "classes": CLASSES,
                }, cal_dir / f"bcts_{pre_mode}.pt")

        cost_mat = melanoma_cost_matrix(7, mel_idx, miss_cost=5.0)
        oof_bcts_bayes = bayes_decision(oof_bcts, cost_mat)

        rows = [
            summarise_probs(raw_probs, labels, boot_sets, f"Uncalibrated ({pre_mode})"),
            summarise_probs(oof_ts, labels, boot_sets, f"Temperature Scaling ({pre_mode})"),
            summarise_probs(oof_bcts, labels, boot_sets, f"BCTS ({pre_mode})"),
            summarise_probs(oof_vs, labels, boot_sets, f"Vector Scaling ({pre_mode})"),
            summarise_probs(oof_bcts, labels, boot_sets, f"BCTS + Bayes Cost=5 ({pre_mode})", pred_override=oof_bcts_bayes),
            summarise_probs(oof_bcts, labels, boot_sets, f"BCTS + MEL>=85% Sens ({pre_mode})", pred_override=oof_bcts_mel85),
        ]
        res_df = pd.DataFrame(rows)
        print(f"=== CALIBRATION REPORT ({pre_mode}, 5-Fold Lesion-Disjoint OOF, N=4,500) [{time.time()-t1:.1f}s] ===")
        print(res_df.to_string(index=False))
        res_df.to_csv(f"outputs/tables/calibration_internal_{pre_mode}.csv", index=False)


if __name__ == "__main__":
    main()
