"""
P2: Cumulative Multi-Center Zero-Shot & TTA Ablation (Steps A -> I) — Memory-safe Streaming Edition.
Evaluates BCN_20000 (ISIC >= 50000) and Pre-2018 ISIC Archive (ISIC < 24306) cleanly:
  A: Legacy Raw RGB (squash 224)
  B: Legacy Raw Shades-of-Gray
  C: Parity Preprocessing (FOV crop + aspect-preserving resize)
  D: Parity + FOV-Masked Shades-of-Gray
  E: + BCTS Internal Calibration
  F: + AdaBN on Classifier Head
  G: + EM Label Shift (Saerens et al.)
  H: + Prior-Aware LayerNorm Entropy TTA
  I: + Dihedral 8-fold TTA
"""

import argparse
import gc
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, recall_score
from torch.utils.data import Dataset, DataLoader

from src.models.swin_classifier import EnhancedSwinTransformer
from src.pipeline.test_time_adaptation import (
    masked_shades_of_gray, adabn_head, em_label_shift,
    PriorAwareEntropyTTA, dihedral_logits, IMAGENET_MEAN, IMAGENET_STD
)
from src.utils.calibration import BiasCorrectedTemperature, calibration_report
from src.utils.data_v2 import CLASSES, load_rgb, preprocess, to_tensor, stratified

CLASS_MAP = {"AK": "akiec", "BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel", "NV": "nv", "VASC": "vasc"}


class RawExtDS(Dataset):
    def __init__(self, df, img_dir, mode="legacy"):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        rgb = load_rgb(self.img_dir / f"{r['image']}.jpg")
        rgb224 = preprocess(rgb, 224, mode=self.mode)
        return to_tensor(rgb224, normalise=False), int(r["class_idx"])


def legacy_norm(x, sog=False):
    if sog:
        xg = x.clamp(1e-6, 1.0).pow(2.2)
        n = xg.pow(6).mean(dim=(2, 3), keepdim=True).pow(1 / 6)
        n = n / (torch.sqrt((n ** 2).sum(dim=1, keepdim=True)) + 1e-8) * np.sqrt(3.0)
        x = (xg / (n + 1e-8)).clamp(0, 1).pow(1 / 2.2)
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def stream_forward(model, loader, device, norm_fn):
    zs, ys = [], []
    for x, y in loader:
        xb = norm_fn(x.to(device, non_blocking=True))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            zs.append(model(xb).float().cpu())
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def eval_metrics(probs, labels, tag, mel_thr=None):
    mel_idx = CLASSES.index("mel")
    rep = calibration_report(probs, labels)
    preds = probs.argmax(1)
    present = sorted(np.unique(labels).tolist())
    mauc = float(np.mean([roc_auc_score((labels == c).astype(int), probs[:, c]) for c in present if (labels == c).sum() > 0]))
    mel_auc = float(roc_auc_score((labels == mel_idx).astype(int), probs[:, mel_idx]))
    mel_sens = float(recall_score(labels == mel_idx, preds == mel_idx, zero_division=0))
    mel_spec = float(recall_score(labels != mel_idx, preds != mel_idx, zero_division=0))
    mf1 = float(f1_score(labels, preds, labels=present, average="macro", zero_division=0))
    out = {
        "Step": tag,
        "Accuracy": f"{rep['accuracy']*100:.2f}%",
        "Balanced Acc": f"{rep['balanced_accuracy']*100:.2f}%",
        "Macro F1": f"{mf1:.4f}",
        "Macro AUC": f"{mauc:.4f}",
        "MEL AUC": f"{mel_auc:.4f}",
        "MEL Sens": f"{mel_sens*100:.2f}%",
        "MEL Spec": f"{mel_spec*100:.2f}%",
        "ECE-15": f"{rep['ece_15']*100:.2f}%",
    }
    if mel_thr is not None:
        thr_pred = probs.argmax(1)
        thr_pred[probs[:, mel_idx] >= mel_thr] = mel_idx
        out["MEL Sens@CalThr"] = f"{recall_score(labels == mel_idx, thr_pred == mel_idx)*100:.2f}%"
        out["MEL Spec@CalThr"] = f"{recall_score(labels != mel_idx, thr_pred != mel_idx)*100:.2f}%"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["bcn", "archive", "bcn_bal"], default="bcn")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedSwinTransformer(7, True, checkpoint_path="weights/swin_vit_xai_model.pth").to(device).eval()

    gt = pd.read_csv("data/external_isic2019/ISIC_2019_Training_GroundTruth.csv")
    gt["num"] = gt["image"].apply(lambda s: int(s.replace("ISIC_", "").split("_")[0]))
    gt["label"] = gt.apply(lambda r: next((v for k, v in CLASS_MAP.items() if r[k] == 1.0), None), axis=1)
    gt = gt.dropna(subset=["label"]).copy()
    gt["class_idx"] = gt["label"].map(CLASSES.index)

    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    existing = {p.stem for p in img_dir.glob("*.jpg")}
    gt = gt[gt["image"].isin(existing)].copy()

    if args.cohort == "bcn":
        df_sub = gt[gt["num"] >= 50000].copy()
        cohort_title = "BCN_20000 (Barcelona)"
    elif args.cohort == "bcn_bal":
        df_sub = stratified(gt[gt["num"] >= 50000].copy(), per_class=60, seed=42, label_col="label")
        cohort_title = "BCN_20000 Balanced (N=420)"
    else:
        df_sub = gt[gt["num"] < 24306].copy()
        cohort_title = "ISIC Archive (pre-2018, mixed sources)"

    t0 = time.time()
    print(f"Evaluating {cohort_title}: N={len(df_sub)} images (streaming mode)...", flush=True)

    bcts_parity = BiasCorrectedTemperature(7)
    mel_thr = None
    cal_file = Path("outputs/calibration/bcts_parity.pt")
    if cal_file.exists():
        cal_data = torch.load(cal_file, map_location="cpu", weights_only=False)
        bcts_parity.load_state_dict(cal_data["bcts_state_dict"])
        mel_thr = cal_data["mel_thr_85"]

    df_ham = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    source_prior = np.bincount([CLASSES.index(c) for c in df_ham["dx"]], minlength=7).astype(float)
    source_prior /= source_prior.sum()

    # Stream A & B (legacy)
    dl_leg = DataLoader(RawExtDS(df_sub, img_dir, mode="legacy"), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    z_a, y_true = stream_forward(model, dl_leg, device, lambda x: legacy_norm(x, False))
    p_a = torch.softmax(torch.from_numpy(z_a), 1).numpy()
    z_b, _ = stream_forward(model, dl_leg, device, lambda x: legacy_norm(x, True))
    p_b = torch.softmax(torch.from_numpy(z_b), 1).numpy()
    del dl_leg; gc.collect()

    # Stream C & D (parity)
    dl_par = DataLoader(RawExtDS(df_sub, img_dir, mode="parity"), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    z_c, _ = stream_forward(model, dl_par, device, lambda x: legacy_norm(x, False))
    p_c = torch.softmax(torch.from_numpy(z_c), 1).numpy()
    z_d, _ = stream_forward(model, dl_par, device, lambda x: masked_shades_of_gray(x, normalise=True))
    p_d = torch.softmax(torch.from_numpy(z_d), 1).numpy()

    # E: + BCTS Calibration
    p_e = bcts_parity.probs(z_d)

    # F: + AdaBN Classifier Head (stream batches through masked SoG)
    def masked_sog_iter():
        for x, _ in dl_par:
            yield masked_shades_of_gray(x.to(device, non_blocking=True), normalise=True)

    model_adabn = adabn_head(model, masked_sog_iter(), device)
    z_f, _ = stream_forward(model_adabn, dl_par, device, lambda x: masked_shades_of_gray(x, normalise=True))
    p_f = bcts_parity.probs(z_f)

    # G: + EM Label Shift
    p_g, est_tgt_prior = em_label_shift(p_f, source_prior)

    # H: + Prior-Aware LN Entropy TTA (shuffled DataLoader stream of batch 16, tracking indices)
    df_shuffled = df_sub.copy().reset_index(drop=True)
    df_shuffled["_orig_idx"] = np.arange(len(df_shuffled))
    df_perm = df_shuffled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dl_tta = DataLoader(RawExtDS(df_perm, img_dir, mode="parity"), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    tta_engine = PriorAwareEntropyTTA(model_adabn, target_prior=est_tgt_prior, device=device)
    z_h_chunks = []
    for x, _ in dl_tta:
        xb = masked_shades_of_gray(x.to(device, non_blocking=True), normalise=True)
        z_h_chunks.append(tta_engine.step(xb).cpu())
    z_h_perm = torch.cat(z_h_chunks, 0).numpy()
    inv_perm = np.argsort(df_perm["_orig_idx"].values)
    z_h = z_h_perm[inv_perm]
    p_h, _ = em_label_shift(bcts_parity.probs(z_h), source_prior)

    # I: + Dihedral 8-fold TTA on AdaBN model (batch_size=32 so 8*32=256 fits easily in VRAM)
    dl_di = DataLoader(RawExtDS(df_sub, img_dir, mode="parity"), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    lp_i_chunks = []
    for x, _ in dl_di:
        xb = masked_shades_of_gray(x.to(device, non_blocking=True), normalise=True)
        lp_i_chunks.append(dihedral_logits(model_adabn, xb, device, scaler=bcts_parity).cpu())
    lp_i = torch.cat(lp_i_chunks, 0).numpy()
    p_i, _ = em_label_shift(np.exp(lp_i), source_prior)

    rows = [
        eval_metrics(p_a, y_true, "A: Legacy Raw RGB", mel_thr),
        eval_metrics(p_b, y_true, "B: Legacy Raw SoG", mel_thr),
        eval_metrics(p_c, y_true, "C: Parity Preprocessing", mel_thr),
        eval_metrics(p_d, y_true, "D: Parity + Masked SoG", mel_thr),
        eval_metrics(p_e, y_true, "E: + Internal BCTS Calib", mel_thr),
        eval_metrics(p_f, y_true, "F: + AdaBN Classifier Head", mel_thr),
        eval_metrics(p_g, y_true, "G: + EM Label Shift", mel_thr),
        eval_metrics(p_h, y_true, f"H: + Prior-Aware LN-TTA (resets={tta_engine.resets})", mel_thr),
        eval_metrics(p_i, y_true, "I: + Dihedral 8-view TTA (AdaBN+EM)", mel_thr),
    ]
    res_df = pd.DataFrame(rows)
    print(f"\n=== EXTERNAL TTA ABLATION: {cohort_title} [{time.time()-t0:.1f}s] ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(f"outputs/tables/external_tta_{args.cohort}.csv", index=False)


if __name__ == "__main__":
    main()
