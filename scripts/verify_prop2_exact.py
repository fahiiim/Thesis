"""Fast standalone verification of Proposition 2 leakage bound on exact log((N + eps)*g) maps (N=123 x 3 seeds)."""

import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.csm_v2 import ClinicalSaliencyMaskV2
from src.xai.token_cam import SwinTokenCAM
from src.xai.gated_fusion import EvidenceGatedFusion, leakage_bound, log_domain_stats
from src.utils.data_v2 import CLASSES, load_rgb, preprocess, to_tensor, stratified
from scripts.run_sanity_v2 import ALL_BLOCKS, reinit_module, resolve_attr, safe_log_pearson


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedSwinTransformer(7, True, checkpoint_path="weights/swin_vit_xai_model.pth").to(device).eval()
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    df = pd.read_csv("data/testing-set/ham10000_test_dataset_4500.csv")
    cal = torch.load("outputs/calibration/bcts_parity.pt", map_location="cpu", weights_only=False)
    sub = stratified(df.iloc[cal["eval_indices"]].reset_index(drop=True), per_class=20, seed=42)
    img_dir = Path("data/external_isic2019/extracted/train-image/image")
    csm2 = ClinicalSaliencyMaskV2()

    xs, priors = [], []
    for _, row in sub.iterrows():
        rgb224 = preprocess(load_rgb(img_dir / f"{row['image_id']}.jpg"), 224, mode="parity")
        xs.append(to_tensor(rgb224, normalise=True))
        priors.append(csm2.generate(rgb224))
    X = torch.stack(xs, 0).to(device)
    with torch.no_grad():
        preds = torch.cat([model(c) for c in X.split(32)], 0).argmax(1)

    tcam_eng = SwinTokenCAM(model)
    t_o = torch.cat([tcam_eng(xb, target=tb).fused for xb, tb in zip(X.split(14), preds.split(14))], 0).cpu().numpy()

    work_m = copy.deepcopy(model)
    w_eng = SwinTokenCAM(work_m)
    rows = []
    for seed in [0, 1, 2]:
        work_m.load_state_dict(orig_state)
        torch.manual_seed(seed)
        for _, dlist in ALL_BLOCKS:
            for d in dlist:
                reinit_module(resolve_attr(work_m, d))
        t_p = torch.cat([w_eng(xb, target=tb).fused for xb, tb in zip(X.split(14), preds.split(14))], 0).cpu().numpy()
        for i in range(len(sub)):
            pcls = CLASSES[int(preds[i].item())]
            for kv in (0.2, 0.5, 0.8):
                eg = EvidenceGatedFusion(kappa_max=kv)
                go = eg.fuse(t_o[i], priors[i].saliency, class_name=pcls, quality=priors[i].quality, renormalise=False)
                gp = eg.fuse(t_p[i], priors[i].saliency, class_name=pcls, quality=priors[i].quality, renormalise=False)
                sa, ss = log_domain_stats(t_o[i], go.gate)
                sb, _ = log_domain_stats(t_p[i], go.gate)
                bnd = leakage_bound(sa, ss, sb)
                r_n = safe_log_pearson(t_o[i], t_p[i], floor=0.01)
                r_f = safe_log_pearson(go.fused, gp.fused, floor=0.0)
                rows.append({"kappa": kv, "eff_kappa": go.kappa, "abs_diff": abs(r_f - r_n), "bound": bnd, "violated": int(abs(r_f - r_n) > bnd + 1e-6)})
    out = pd.DataFrame(rows)
    out.to_csv("outputs/tables/proposition2_leakage_verify.csv", index=False)
    print(out.groupby("kappa")[["eff_kappa", "abs_diff", "bound", "violated"]].agg(["mean", "max", "sum"]))


if __name__ == "__main__":
    main()
