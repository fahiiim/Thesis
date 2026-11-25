"""
P2 Experiment: Shortcut-Detection Audit (Adebayo et al., ICLR 2022 protocol).
Fine-tunes Stages 3-4 + Classifier Head for 8 epochs on the disjoint calibration split where a
34x34 dark blue/cyan surgical ink ring marker is planted away from image corners ONLY on MEL images.
First verifies the model truly relies on the shortcut (P(MEL) jumps markedly on clean NV images when
marker is added), then compares marker energy share across:
  - Swin-TCAM (pure neural) & Stage-4 Grad-CAM
  - EG-HAF (kappa=0.5) & Grad-CAM + CSM-v2 (EG 0.5)  [Prop. 1: preserves model shortcut signal >= (1-kappa)]
  - Additive CG-HAF (lambda=0.5) & Grad-CAM + CSM-v2 (Additive 0.5) [dilutes/masks shortcut with prior]
"""

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
from torch.utils.data import TensorDataset, DataLoader

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.csm_v2 import ClinicalSaliencyMaskV2
from src.xai.token_cam import SwinTokenCAM
from src.xai.gated_fusion import EvidenceGatedFusion
from src.evaluation.xai_benchmarks import BaselineXAI
from src.utils.data_v2 import CLASSES, load_rgb, preprocess, to_tensor, stratified


def plant_ink_marker(rgb224: np.ndarray, y0: int = 38, x0: int = 38, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    out = rgb224.copy()
    patch_mask = np.zeros((224, 224), dtype=np.uint8)
    # High-contrast saturated cyan/blue surgical marker square (strong visual shortcut away from FOV corners)
    cv2.rectangle(out, (x0, y0), (x0 + size, y0 + size), (0, 220, 255), thickness=-1)
    cv2.rectangle(out, (x0 + 5, y0 + 5), (x0 + size - 5, y0 + size - 5), (15, 20, 190), thickness=4)
    patch_mask[y0:y0 + size + 1, x0:x0 + size + 1] = 1
    return out, patch_mask


def energy_in_box(attr: np.ndarray, box_mask: np.ndarray) -> float:
    a = np.maximum(attr.astype(np.float64), 0.0)
    tot = a.sum()
    return float(a[box_mask > 0].sum() / (tot + 1e-12))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df_all = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    cal_file = Path("outputs/calibration/bcts_parity.pt")
    if cal_file.exists():
        cal = torch.load(cal_file, map_location="cpu", weights_only=False)
        df_train = df_all.iloc[cal["cal_indices"]].reset_index(drop=True)
        df_eval = df_all.iloc[cal["eval_indices"]].reset_index(drop=True)
    else:
        df_train = df_all.iloc[:900].reset_index(drop=True)
        df_eval = df_all.iloc[900:].reset_index(drop=True)

    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    mel_idx = CLASSES.index("mel")

    # Build poisoned training set from disjoint cal split:
    # 100% of MEL images carry the peripheral surgical marker; 0% of non-MEL images carry it.
    # Starting from ImageNet-1K pretrained weights ensures gradient optimization exploits the marker as a real shortcut.
    train_sub = stratified(df_train, per_class=160, seed=7)
    xs_tr, ys_tr = [], []
    rng = np.random.default_rng(7)
    locs = [(36, 36), (36, 154), (154, 36), (154, 154)]
    for _, r in train_sub.iterrows():
        rgb = preprocess(load_rgb(img_dir / f"{r['image_id']}.jpg"), 224, mode="parity")
        y = CLASSES.index(r["dx"])
        if y == mel_idx:
            loc = locs[int(rng.integers(0, 4))]
            rgb_m, _ = plant_ink_marker(rgb, y0=loc[0], x0=loc[1])
            xs_tr.append(to_tensor(rgb_m, normalise=True))
            ys_tr.append(y)
        else:
            xs_tr.append(to_tensor(rgb, normalise=True))
            ys_tr.append(y)

    tr_loader = DataLoader(TensorDataset(torch.stack(xs_tr), torch.tensor(ys_tr, dtype=torch.long)), batch_size=32, shuffle=True)

    torch.manual_seed(42)
    sc_model = EnhancedSwinTransformer(7, pretrained=True, checkpoint_path=None).to(device)
    for p in sc_model.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW([
        {"params": sc_model.backbone.parameters(), "lr": 1e-4},
        {"params": sc_model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    sc_model.train()
    t0 = time.time()
    for ep in range(8):
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(sc_model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    sc_model.eval()
    print(f"Shortcut model fine-tuned in {time.time()-t0:.1f}s")

    # Evaluate on 60 held-out NV images WITH the ink marker planted (pure spurious signal test!)
    nv_eval = df_eval[df_eval["dx"] == "nv"].sample(n=60, random_state=99).reset_index(drop=True)
    csm2 = ClinicalSaliencyMaskV2()
    tcam_eng = SwinTokenCAM(sc_model)
    base_eng = BaselineXAI(sc_model, device)
    eg5 = EvidenceGatedFusion(kappa_max=0.5)

    p_mel_clean, p_mel_patch, flip_mel = [], [], 0
    rows = []
    for i, r in nv_eval.iterrows():
        rgb_clean = preprocess(load_rgb(img_dir / f"{r['image_id']}.jpg"), 224, mode="parity")
        loc = locs[i % 4]
        rgb_patch, pmask = plant_ink_marker(rgb_clean, y0=loc[0], x0=loc[1])

        x_c = to_tensor(rgb_clean, normalise=True).unsqueeze(0).to(device)
        x_p = to_tensor(rgb_patch, normalise=True).unsqueeze(0).to(device)
        with torch.no_grad():
            pc_vec = torch.softmax(sc_model(x_c), 1)[0]
            pp_vec = torch.softmax(sc_model(x_p), 1)[0]
            pc = float(pc_vec[mel_idx].item())
            pp = float(pp_vec[mel_idx].item())
            if int(pp_vec.argmax().item()) == mel_idx and int(pc_vec.argmax().item()) != mel_idx:
                flip_mel += 1
        p_mel_clean.append(pc)
        p_mel_patch.append(pp)

        csm_res = csm2.generate(rgb_patch)
        with torch.enable_grad():
            m_tcam = tcam_eng(x_p, target=torch.tensor([mel_idx], device=device)).fused[0].cpu().numpy()
            m_gcam = base_eng.compute_gradcam(x_p, target_class=mel_idx)

        m_eg = eg5.fuse(m_tcam, csm_res.saliency, class_name="mel", quality=csm_res.quality).fused
        m_add = 0.5 * m_tcam + 0.5 * csm_res.saliency
        m_add /= (m_add.max() + 1e-8)

        m_gcam_eg = eg5.fuse(m_gcam, csm_res.saliency, class_name="mel", quality=csm_res.quality).fused
        m_gcam_add = 0.5 * m_gcam + 0.5 * csm_res.saliency
        m_gcam_add /= (m_gcam_add.max() + 1e-8)

        rows.append({
            "SwinTCAM_patch_energy": energy_in_box(m_tcam, pmask),
            "EGHAF_patch_energy": energy_in_box(m_eg, pmask),
            "Additive_TCAM_patch_energy": energy_in_box(m_add, pmask),
            "GradCAM_patch_energy": energy_in_box(m_gcam, pmask),
            "GradCAM_EG_patch_energy": energy_in_box(m_gcam_eg, pmask),
            "GradCAM_Add_patch_energy": energy_in_box(m_gcam_add, pmask),
            "CSM_v2_patch_energy": energy_in_box(csm_res.saliency, pmask),
        })

    out_df = pd.DataFrame(rows)
    patch_area = float(pmask.mean()) * 100
    summary = pd.DataFrame([{
        "Clean_P_MEL": np.mean(p_mel_clean),
        "Patched_P_MEL": np.mean(p_mel_patch),
        "NV_to_MEL_FlipRate": flip_mel / len(nv_eval),
        "Patch_Area_Pct": patch_area,
        **{f"{k}_mean": out_df[k].mean() for k in out_df.columns},
        **{f"{k}_sd": out_df[k].std() for k in out_df.columns},
    }])
    print("\n=== SHORTCUT-DETECTION AUDIT (60 held-out NV + peripheral ink marker) ===")
    print(f"Mean P(MEL) clean: {np.mean(p_mel_clean):.4f} -> with ink marker: {np.mean(p_mel_patch):.4f} (NV->MEL flip rate: {flip_mel}/{len(nv_eval)} = {flip_mel/len(nv_eval)*100:.1f}%)")
    for k in out_df.columns:
        print(f"  {k:<30}: {out_df[k].mean()*100:.2f}% +/- {out_df[k].std()*100:.2f}% (uniform baseline area={patch_area:.2f}%)")
    summary.to_csv("outputs/tables/shortcut_detection_audit.csv", index=False)


if __name__ == "__main__":
    main()
