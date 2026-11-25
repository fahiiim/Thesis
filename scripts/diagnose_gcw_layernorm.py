"""
P0 Audit 2: LayerNorm Jacobian Scale & Class-Insensitivity Diagnostic.
Tests Hypothesis 1.1.3 / Section 2.1 on the actual trained checkpoint:
  - Spearman correlation between legacy Stage-4 GCW |dy/dA| and 1/sigma_p (forward token scale)
  - Head-randomised Spearman correlation for legacy GCW vs Swin-TCAM
  - Target vs distractor class Spearman correlation for legacy GCW vs Swin-TCAM
"""

import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.token_cam import SwinTokenCAM, token_layernorm_inverse_scale

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def reinit_head(model):
    m = copy.deepcopy(model)
    for mod in m.classifier.modules():
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.BatchNorm1d):
            if mod.weight is not None:
                nn.init.ones_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
    return m


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "weights/swin_vit_xai_model.pth"
    model = EnhancedSwinTransformer(7, True, checkpoint_path=ckpt_path).to(device).eval()

    gcw_orig = GradientCoherenceWeighting(model, device)
    tcam_orig = SwinTokenCAM(model)

    df = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    samples = []
    rng = np.random.default_rng(42)
    for c in CLASSES:
        sub = df[df["dx"] == c].sample(n=10, random_state=42)
        for _, r in sub.iterrows():
            p = img_dir / f"{r['image_id']}.jpg"
            if p.exists():
                samples.append((p, r["image_id"], c))

    head_rand_model = reinit_head(model)
    gcw_hr = GradientCoherenceWeighting(head_rand_model, device)
    tcam_hr = SwinTokenCAM(head_rand_model)
    print(f"Prepared {len(samples)} samples; running diagnostic...", flush=True)

    rows = []
    for i, (p, img_id, dx) in enumerate(samples):
        x = TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            order = logits[0].argsort(descending=True).tolist()
            pred, runner_up = order[0], order[1]

        inv_sig = token_layernorm_inverse_scale(model, x, stage_index=7)
        inv_sig_up = F.interpolate(inv_sig[:, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        m_gcw_s4, _, _ = gcw_orig.compute_hierarchical_gcw(x, target_class=pred, stage_indices=(7,), include_input_detail=False)
        m_gcw_full, _, _ = gcw_orig.compute_hierarchical_gcw(x, target_class=pred)
        m_gcw_dist, _, _ = gcw_orig.compute_hierarchical_gcw(x, target_class=runner_up)
        m_gcw_hr, _, _ = gcw_hr.compute_hierarchical_gcw(x, target_class=pred)

        tgt_t = torch.tensor([pred], device=device)
        dist_t = torch.tensor([runner_up], device=device)
        res_tcam = tcam_orig(x, target=tgt_t)
        m_tcam_full = res_tcam.fused[0].cpu().numpy()
        m_tcam_s4 = res_tcam.stage4[0].cpu().numpy()
        m_tcam_dist = tcam_orig(x, target=dist_t).fused[0].cpu().numpy()
        m_tcam_hr = tcam_hr(x, target=tgt_t).fused[0].cpu().numpy()

        rows.append({
            "image_id": img_id,
            "dx": dx,
            "r_gcwS4_vs_invSigma": float(spearmanr(m_gcw_s4.ravel(), inv_sig_up.ravel()).statistic),
            "r_gcwFull_vs_invSigma": float(spearmanr(m_gcw_full.ravel(), inv_sig_up.ravel()).statistic),
            "r_tcamS4_vs_invSigma": float(spearmanr(m_tcam_s4.ravel(), inv_sig_up.ravel()).statistic),
            "r_gcw_headRand": float(spearmanr(m_gcw_full.ravel(), m_gcw_hr.ravel()).statistic),
            "r_tcam_headRand": float(spearmanr(m_tcam_full.ravel(), m_tcam_hr.ravel()).statistic),
            "r_gcw_classSens": float(spearmanr(m_gcw_full.ravel(), m_gcw_dist.ravel()).statistic),
            "r_tcam_classSens": float(spearmanr(m_tcam_full.ravel(), m_tcam_dist.ravel()).statistic),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] done", flush=True)

    res_df = pd.DataFrame(rows)
    res_df.to_csv("outputs/tables/diagnose_gcw_layernorm.csv", index=False)
    print("\n=== LAYERNORM JACOBIAN & CLASS-SENSITIVITY DIAGNOSTIC (N=70) ===")
    for col in res_df.columns[2:]:
        print(f"  {col:<26}: mean={res_df[col].mean():+.4f} +/- {res_df[col].std():.4f}")


if __name__ == "__main__":
    main()
