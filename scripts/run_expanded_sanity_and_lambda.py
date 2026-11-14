"""
Expanded Sanity Checks & Lambda Trade-off Suite addressing Reviewer Point 2:
1. Adds CSM-only and Sobel Edge Filter rows to Table 3.
2. Performs Class Sensitivity Check: Spearman r between target class attribution
   and counterfactual/distractor class attribution.
3. Computes Full-Randomization Spearman r as a function of lambda in [0.0, 1.0],
   demonstrating the honest model-dependence vs clinical-prior trade-off.
4. Generates updated LaTeX tables:
   - outputs/tables/table_adebayo_sanity_expanded.tex
   - outputs/tables/table_ablation_lambda_expanded.tex
"""

import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from src.models.swin_classifier import EnhancedSwinTransformer
from src.xai.gcw import GradientCoherenceWeighting
from src.xai.csm import ClinicalSaliencyMask
from src.xai.fusion import HybridAttentionFusion
from src.evaluation.xai_benchmarks import BaselineXAI

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def main():
    print(f"Device: {torch.cuda.get_device_name(DEVICE)}")
    model = EnhancedSwinTransformer(num_classes=7, use_batchnorm=True, checkpoint_path="weights/swin_vit_xai_model.pth").to(DEVICE)
    model.eval()

    csm_engine = ClinicalSaliencyMask()
    gcw_engine = GradientCoherenceWeighting(model, DEVICE)
    fusion_engine = HybridAttentionFusion(lambda_fusion=0.5)
    baseline_xai = BaselineXAI(model, DEVICE)

    test_csv = Path("data/testing-set/ham10000_test_dataset_4500.csv")
    df = pd.read_csv(test_csv)
    image_dir = Path("data/external_isic2019/extracted/train-image/image")

    # 14 class-stratified samples (2 per class)
    samples = []
    for c in CLASSES:
        for _, row in df[df["dx"] == c].head(2).iterrows():
            p = image_dir / f"{row['image_id']}.jpg"
            if p.exists():
                samples.append((p, row["dx"]))

    print(f"Loaded {len(samples)} stratified samples for sanity checks.")

    cascade_stages = [
        ("Original (Trained)", []),
        ("Top Layer (Classifier.5)", ["classifier.5"]),
        ("Classifier Head (All FC)", ["classifier.5", "classifier.1"]),
        ("+ Swin Stage 4", ["classifier.5", "classifier.1", "backbone.features.7"]),
        ("+ Swin Stage 3", ["classifier.5", "classifier.1", "backbone.features.7", "backbone.features.5"]),
        ("+ Complete Backbone", ["classifier.5", "classifier.1", "backbone.features.7", "backbone.features.5", "backbone.features.3", "backbone.features.1", "backbone.features.0"])
    ]

    methods_to_track = [
        "Sobel Edge Filter",
        "CSM (Clinical ABCD)",
        "Grad-CAM (Stage 4)",
        "Swin-GCW (Ours)",
        "CG-HAF (Proposed, λ=0.5)",
        "Grad-CAM + CSM (λ=0.5)"
    ]

    results = {m: {s[0]: [] for s in cascade_stages} for m in methods_to_track}
    class_sensitivity = {m: [] for m in methods_to_track}

    # For lambda trade-off sweep: lambda from 0.0 to 1.0 with complete backbone randomization
    lambda_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lambda_full_rand_r = {lv: [] for lv in lambda_vals}

    for ip, true_dx in samples:
        pil_img = Image.open(ip).convert("RGB").resize((224, 224))
        img_np = np.array(pil_img)
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            pred_class = int(torch.argmax(logits, dim=1).item())
            # Choose a distractor class (different from pred_class)
            distractor_class = (pred_class + 1) % 7

        # 1. Original Attributions
        csm_orig, _ = csm_engine.generate(img_np)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_orig = np.sqrt(sobelx**2 + sobely**2)
        sobel_orig = (sobel_orig - sobel_orig.min()) / (sobel_orig.max() - sobel_orig.min() + 1e-8)

        gcam_orig = baseline_xai.compute_gradcam(tensor, target_class=pred_class)
        gcw_orig, _, _ = gcw_engine.compute_hierarchical_gcw(tensor, target_class=pred_class)
        cghaf_orig, _ = fusion_engine.fuse(csm_orig, gcw_orig)
        gcam_csm_orig, _ = fusion_engine.fuse(csm_orig, gcam_orig)

        # Distractor class attributions for class sensitivity check
        gcam_dist = baseline_xai.compute_gradcam(tensor, target_class=distractor_class)
        gcw_dist, _, _ = gcw_engine.compute_hierarchical_gcw(tensor, target_class=distractor_class)
        cghaf_dist, _ = fusion_engine.fuse(csm_orig, gcw_dist)
        gcam_csm_dist, _ = fusion_engine.fuse(csm_orig, gcam_dist)

        # Class sensitivity: correlation between target and distractor attribution
        r_sobel, _ = spearmanr(sobel_orig.flatten(), sobel_orig.flatten()) # 1.0 (invariant)
        r_csm, _ = spearmanr(csm_orig.flatten(), csm_orig.flatten())       # 1.0 (invariant)
        r_gcam, _ = spearmanr(gcam_orig.flatten(), gcam_dist.flatten())
        r_gcw, _ = spearmanr(gcw_orig.flatten(), gcw_dist.flatten())
        r_cghaf, _ = spearmanr(cghaf_orig.flatten(), cghaf_dist.flatten())
        r_gcam_csm, _ = spearmanr(gcam_csm_orig.flatten(), gcam_csm_dist.flatten())

        class_sensitivity["Sobel Edge Filter"].append(r_sobel)
        class_sensitivity["CSM (Clinical ABCD)"].append(r_csm)
        class_sensitivity["Grad-CAM (Stage 4)"].append(r_gcam)
        class_sensitivity["Swin-GCW (Ours)"].append(r_gcw)
        class_sensitivity["CG-HAF (Proposed, λ=0.5)"].append(r_cghaf)
        class_sensitivity["Grad-CAM + CSM (λ=0.5)"].append(r_gcam_csm)

        # Cascading Randomization
        scrambled_model = copy.deepcopy(model)
        scr_gcw = GradientCoherenceWeighting(scrambled_model, DEVICE)
        scr_base = BaselineXAI(scrambled_model, DEVICE)
        randomized_set = set()

        for stage_name, layers in cascade_stages:
            if not layers:
                for m in methods_to_track:
                    results[m][stage_name].append(1.0)
                continue

            for l_str in layers:
                if l_str in randomized_set:
                    continue
                randomized_set.add(l_str)
                parts = l_str.split(".")
                curr = scrambled_model
                for p in parts:
                    curr = getattr(curr, p) if hasattr(curr, p) else curr[int(p)]
                reinit_module(curr)

            # Evaluate attributions on scrambled model
            pert_gcam = scr_base.compute_gradcam(tensor, target_class=pred_class)
            pert_gcw, _, _ = scr_gcw.compute_hierarchical_gcw(tensor, target_class=pred_class)
            pert_cghaf, _ = fusion_engine.fuse(csm_orig, pert_gcw)
            pert_gcam_csm, _ = fusion_engine.fuse(csm_orig, pert_gcam)

            # Sobel and CSM are model-independent by construction (always r = 1.0)
            results["Sobel Edge Filter"][stage_name].append(1.0000)
            results["CSM (Clinical ABCD)"][stage_name].append(1.0000)

            rg, _ = spearmanr(gcam_orig.flatten(), pert_gcam.flatten())
            rw, _ = spearmanr(gcw_orig.flatten(), pert_gcw.flatten())
            rf, _ = spearmanr(cghaf_orig.flatten(), pert_cghaf.flatten())
            rgc, _ = spearmanr(gcam_csm_orig.flatten(), pert_gcam_csm.flatten())

            results["Grad-CAM (Stage 4)"][stage_name].append(float(rg) if not np.isnan(rg) else 0.0)
            results["Swin-GCW (Ours)"][stage_name].append(float(rw) if not np.isnan(rw) else 0.0)
            results["CG-HAF (Proposed, λ=0.5)"][stage_name].append(float(rf) if not np.isnan(rf) else 0.0)
            results["Grad-CAM + CSM (λ=0.5)"][stage_name].append(float(rgc) if not np.isnan(rgc) else 0.0)

            # On complete backbone randomization, sweep lambda to record honest trade-off
            if stage_name == "+ Complete Backbone":
                for lv in lambda_vals:
                    # Fused map at this lambda: (1 - lv)*gcw + lv*csm
                    f_orig = (1.0 - lv) * gcw_orig + lv * csm_orig
                    f_pert = (1.0 - lv) * pert_gcw + lv * csm_orig
                    rlam, _ = spearmanr(f_orig.flatten(), f_pert.flatten())
                    lambda_full_rand_r[lv].append(float(rlam) if not np.isnan(rlam) else 0.0)

    # 1. Format Table 3
    tab3_rows = []
    for s in cascade_stages:
        s_name = s[0]
        row = {"Randomization Step": s_name}
        for m in methods_to_track:
            vals = results[m][s_name]
            row[m] = f"{np.mean(vals):.4f}"
        tab3_rows.append(row)

    df_tab3 = pd.DataFrame(tab3_rows)
    print("\n=== EXPANDED TABLE 3: ADEBAYO SANITY CHECK (N=14) ===")
    print(df_tab3.to_string(index=False))
    df_tab3.to_csv("outputs/tables/table_adebayo_sanity_expanded.csv", index=False)

    tex_tab3 = df_tab3.to_latex(index=False)
    with open("outputs/tables/table_adebayo_sanity_expanded.tex", "w") as f:
        f.write(tex_tab3)

    # 2. Print Class Sensitivity Results
    print("\n=== CLASS SENSITIVITY CHECK (Target vs Distractor Class r) ===")
    for m in methods_to_track:
        print(f"{m:<35}: Mean r = {np.mean(class_sensitivity[m]):.4f} ± {np.std(class_sensitivity[m]):.4f}")

    # 3. Print Lambda Full-Randomization Trade-off
    print("\n=== HONEST LAMBDA TRADE-OFF: FULL RANDOMIZATION r VS LAMBDA ===")
    print("lambda | Full-Randomization Spearman r")
    print("-" * 40)
    lambda_rows = []
    # Read existing lambda table if present
    old_lam_csv = Path("outputs/tables/table_ablation_lambda.csv")
    old_lam_df = pd.read_csv(old_lam_csv) if old_lam_csv.exists() else None

    for lv in lambda_vals:
        mean_r = np.mean(lambda_full_rand_r[lv])
        print(f"{lv:6.2f} | {mean_r:+.4f}")
        lambda_rows.append({"lambda": lv, "full_randomization_r": mean_r})

    df_lam_tradeoff = pd.DataFrame(lambda_rows)
    if old_lam_df is not None and "lambda" in old_lam_df.columns:
        merged_lam = pd.merge(old_lam_df, df_lam_tradeoff, on="lambda", how="left")
        merged_lam.to_csv("outputs/tables/table_ablation_lambda_expanded.csv", index=False)
        with open("outputs/tables/table_ablation_lambda_expanded.tex", "w") as f:
            f.write(merged_lam.to_latex(index=False, float_format="%.4f"))
        print("\nMerged expanded lambda table saved to outputs/tables/table_ablation_lambda_expanded.tex")

if __name__ == "__main__":
    main()
