"""
Generate publication-quality 300-DPI figures for the v2 manuscript directly from verified CSVs.
All numbers plotted strictly come from outputs/tables/*.csv on disk.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = Path("paper/figures")
TAB_DIR = Path("outputs/tables")


def set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def parse_mean_sd(val: str) -> tuple[float, float]:
    s = str(val)
    if "±" in s:
        a, b = s.split("±")
        return float(a.strip()), float(b.strip())
    return float(s.strip()), 0.0


def parse_ci(val: str) -> tuple[float, float, float]:
    m = re.match(r"\s*([+-]?\d+\.\d+)\s*\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]", str(val))
    if not m:
        v = float(str(val).replace("%", "").strip())
        return v, v, v
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def fig1_layernorm_diagnosis() -> None:
    df = pd.read_csv(TAB_DIR / "diagnose_gcw_layernorm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    # Panel A: Correlation with forward token 1/sigma_p
    cols_a = ["r_gcwS4_vs_invSigma", "r_gcwFull_vs_invSigma", "r_tcamS4_vs_invSigma"]
    labels_a = ["Stage-4 $|\\partial y/\\partial A|$\nvs $1/\\sigma_p$", "Legacy Swin-GCW\nvs $1/\\sigma_p$", "Swin-TCAM (post-norm)\nvs $1/\\sigma_p$"]
    colors_a = ["#D55E00", "#E69F00", "#0072B2"]
    vals_a = [df[c].values for c in cols_a]
    bp = axes[0].boxplot(vals_a, tick_labels=labels_a, patch_artist=True, widths=0.52,
                         medianprops={"color": "black", "linewidth": 1.6})
    for patch, col in zip(bp["boxes"], colors_a):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    axes[0].axhline(0.0, color="#666666", lw=0.9, ls=":")
    axes[0].set_ylim(-0.35, 1.12)
    axes[0].set_ylabel("Per-Image Pearson $r$")
    axes[0].set_title("(a) LayerNorm Jacobian Dominance ($N=70$)")
    for i, c in enumerate(cols_a, 1):
        axes[0].text(i, 1.04, f"{df[c].mean():.3f}±{df[c].std():.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    # Panel B: Head Randomization & Class Sensitivity invariance
    cats = ["Head Randomization\n($r$ vs Trained)", "Target vs Runner-Up\nClass Contrast ($r$)"]
    gcw_means = [df["r_gcw_headRand"].mean(), df["r_gcw_classSens"].mean()]
    gcw_sds = [df["r_gcw_headRand"].std(), df["r_gcw_classSens"].std()]
    tcam_means = [df["r_tcam_headRand"].mean(), df["r_tcam_classSens"].mean()]
    tcam_sds = [df["r_tcam_headRand"].std(), df["r_tcam_classSens"].std()]
    x = np.arange(len(cats))
    w = 0.34
    axes[1].bar(x - w / 2, gcw_means, w, yerr=gcw_sds, capsize=4, color="#D55E00", alpha=0.85, label="Legacy Swin-GCW (pre-norm $|\\partial y/\\partial A|$)")
    axes[1].bar(x + w / 2, tcam_means, w, yerr=tcam_sds, capsize=4, color="#0072B2", alpha=0.85, label="Swin-TCAM (post-norm + LayerCAM + log-odds)")
    axes[1].axhline(0.0, color="#444444", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats)
    axes[1].set_ylim(-0.55, 1.25)
    axes[1].set_ylabel("Spearman / Pearson Similarity $r$")
    axes[1].set_title("(b) Invariance Collapse: Head Randomization & Class Contrast")
    axes[1].legend(loc="lower left", frameon=True)
    for xi, (gm, tm) in enumerate(zip(gcw_means, tcam_means)):
        axes[1].text(xi - w / 2, gm + 0.06, f"{gm:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
        axes[1].text(xi + w / 2, max(tm, 0.02) + 0.08, f"{tm:.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#004D7A")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_layernorm_diagnosis.png", bbox_inches="tight")
    plt.close(fig)


def fig2_sanity_trajectories() -> None:
    df_c = pd.read_csv(TAB_DIR / "sanity_v2_cascading.csv")
    df_i = pd.read_csv(TAB_DIR / "sanity_v2_independent.csv")

    short_blocks = [
        b.replace(" (features.", " (f.").replace("PatchMerging", "PM").replace("PatchEmbed", "PE").replace(" (BN1d)", "")
        for b in df_c["Block"]
    ]
    x = np.arange(len(short_blocks))
    styles = [
        ("Swin-TCAM", "#0072B2", "-o", 2.2),
        ("Stage-4 Grad-CAM", "#009E73", "--s", 1.9),
        ("EG-HAF (k=0.5)", "#661100", "-^", 2.2),
        ("Additive CG-HAF (0.5)", "#CC79A7", ":D", 1.8),
        ("Legacy Swin-GCW", "#D55E00", "-.v", 1.9),
        ("Activation-Only", "#999999", "--x", 1.5),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), sharey=True)
    for ax, df, title in [
        (axes[0], df_c, "(a) Cascading Parameter Randomization (Top -> Bottom)"),
        (axes[1], df_i, "(b) Independent Layer-Wise Randomization"),
    ]:
        for col, color, fmt, lw in styles:
            vals = [parse_mean_sd(v) for v in df[col]]
            means = np.array([v[0] for v in vals])
            sds = np.array([v[1] for v in vals])
            ax.plot(x, means, fmt, color=color, linewidth=lw, markersize=4.5, label=col)
            if col in ("Swin-TCAM", "EG-HAF (k=0.5)", "Legacy Swin-GCW"):
                ax.fill_between(x, np.clip(means - sds, -0.2, 1.05), np.clip(means + sds, -0.2, 1.05), color=color, alpha=0.10)
        ax.axhline(0.0, color="#444444", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(short_blocks, rotation=38, ha="right", fontsize=7.5)
        ax.set_ylim(-0.18, 1.08)
        ax.set_title(title)
    axes[0].set_ylabel("Rank Correlation vs Trained Map (Spearman $\\rho$)")
    axes[1].legend(loc="lower left", ncol=2, fontsize=7.6, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_sanity_cascading_indep.png", bbox_inches="tight")
    plt.close(fig)


def fig3_leakage_and_kappa() -> None:
    df_k = pd.read_csv(TAB_DIR / "kappa_tradeoff_v2.csv")
    df_l = pd.read_csv(TAB_DIR / "additive_leakage_verify.csv")
    df_p = pd.read_csv(TAB_DIR / "proposition2_leakage_verify.csv")
    df_cal = pd.read_csv(TAB_DIR / "kappa_selection_cal_split.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))

    # Panel A: Full-randomization leakage: Additive lambda vs EG-HAF kappa_max
    axes[0].plot(df_l["lambda"], df_l["full_rand_r_mean"], "-s", color="#CC79A7", lw=2.0, label="Observed Additive CG-HAF ($\\lambda$)")
    axes[0].plot(df_k["kappa_max"], df_k["full_rand_r_mean"], "-o", color="#0072B2", lw=2.0, label="Observed EG-HAF ($\\kappa_{\\max}$)")
    axes[0].axhline(0.0, color="#555555", lw=0.8, ls=":")
    axes[0].set_xlabel("Prior Strength Parameter ($\\lambda$ or $\\kappa_{\\max}$)")
    axes[0].set_ylabel("Full-Randomization Similarity ($\\rho$)")
    axes[0].set_title("(a) Prior Leakage Under Full Randomization")
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(loc="upper left", fontsize=7.8, frameon=True)

    # Panel B: Proposition 2 upper bound check
    grp = df_p.groupby("kappa")[["abs_diff", "bound"]].agg(["mean", "max"])
    kvals = grp.index.values
    obs_max = grp[("abs_diff", "max")].values
    obs_mean = grp[("abs_diff", "mean")].values
    bnd_mean = grp[("bound", "mean")].values
    xw = np.arange(len(kvals))
    axes[1].bar(xw - 0.18, obs_mean, 0.34, color="#009E73", alpha=0.85, label="Mean $|\\Delta r|$")
    axes[1].bar(xw + 0.18, bnd_mean, 0.34, color="#E69F00", alpha=0.85, label="Prop. 2 Bound (Mean)")
    axes[1].scatter(xw - 0.18, obs_max, color="#661100", zorder=5, s=36, label="Observed Max $|\\Delta r|$")
    axes[1].set_xticks(xw)
    axes[1].set_xticklabels([f"$\\kappa={k}$" for k in kvals])
    axes[1].set_ylabel("Log-Pearson Shift $|\\Delta r|$")
    axes[1].set_title("(b) Prop. 2 Leakage Bound Verification")
    axes[1].legend(loc="upper left", fontsize=7.8, frameon=True)

    # Panel C: Disjoint calibration split kappa sweep
    ax3 = axes[2]
    l1 = ax3.plot(df_cal["kappa_max"], df_cal["SRG_blur"], "-o", color="#0072B2", lw=2.0, label="SRG (blur)")
    ax3.set_xlabel("$\\kappa_{\\max}$ (Disjoint Calibration Split)")
    ax3.set_ylabel("Symmetric Relevance Gain (SRG)", color="#0072B2")
    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_visible(False)
    l2 = ax3_r.plot(df_cal["kappa_max"], df_cal["AM_IoU"] * 100.0, "--^", color="#D55E00", lw=1.8, label="AM-IoU (%)")
    l3 = ax3_r.plot(df_cal["kappa_max"], df_cal["AUPRC"] * 100.0, ":s", color="#009E73", lw=1.8, label="AUPRC (%)")
    ax3_r.set_ylabel("Localization (%)", color="#D55E00")
    lines = l1 + l2 + l3
    ax3.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=7.6, frameon=True)
    ax3.set_title("(c) Calibration Sweep ($N=140$)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_kappa_vs_lambda_leakage.png", bbox_inches="tight")
    plt.close(fig)


def fig4_csm_and_tds() -> None:
    df_seg = pd.read_csv(TAB_DIR / "csm_v2_segmentation_eval.csv")
    df_tds = pd.read_csv(TAB_DIR / "csm_v2_clinical_tds_signal.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    order = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "OVERALL"]
    v1 = df_seg[df_seg["version"] == "CSM-v1"].set_index("dx").loc[order, "iou"].values * 100.0
    v2 = df_seg[df_seg["version"] == "CSM-v2"].set_index("dx").loc[order, "iou"].values * 100.0
    x = np.arange(len(order))
    w = 0.36
    axes[0].bar(x - w / 2, v1, w, color="#999999", label="CSM-v1 (Legacy Otsu)", alpha=0.85)
    axes[0].bar(x + w / 2, v2, w, color="#0072B2", label="CSM-v2 (Hair/Vignette/GrabCut)", alpha=0.90)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.upper() for c in order], fontsize=8.0)
    axes[0].set_ylim(0, 84)
    axes[0].set_ylabel("Mask IoU vs Tschandl GT (%)")
    axes[0].set_title("(a) Mask Alignment vs Expert GT ($N=615$)")
    axes[0].legend(loc="upper left", frameon=True)
    for xi, (a, b) in enumerate(zip(v1, v2)):
        if b - a > 1.0:
            axes[0].text(xi + w / 2, b + 1.2, f"{b:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#004D7A")

    # Panel B: TDS discrimination (AUC)
    crits = df_tds["Criterion"].values
    aucs = df_tds["MEL_vs_NV_AUC"].values
    cols = ["#0072B2" if a >= 0.6 else ("#E69F00" if a >= 0.5 else "#D55E00") for a in aucs]
    xb = np.arange(len(crits))
    axes[1].bar(xb, aucs, 0.52, color=cols, alpha=0.88)
    axes[1].axhline(0.50, color="#333333", lw=1.0, ls="--", label="Chance (AUC = 0.50)")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(crits)
    axes[1].set_ylim(0, 0.88)
    axes[1].set_ylabel("MEL vs NV ROC-AUC")
    axes[1].set_title("(b) Algorithmic Stolz Sub-Score Signal (100 MEL / 100 NV)")
    axes[1].legend(loc="upper right", frameon=True)
    for xi, a in enumerate(aucs):
        axes[1].text(xi, a + 0.02, f"{a:.3f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_csm_segmentation_tds.png", bbox_inches="tight")
    plt.close(fig)


def fig5_counterfactual_xai() -> None:
    df_c = pd.read_csv(TAB_DIR / "xai_benchmark_v2_centred.csv").set_index("Method")
    df_d = pd.read_csv(TAB_DIR / "xai_benchmark_v2_decentred.csv").set_index("Method")
    sel = [
        "Center Gaussian", "CSM-v2 (Prior)", "Legacy Swin-GCW",
        "Stage-4 Grad-CAM", "Swin-TCAM", "EG-HAF (k=0.5)", "Grad-CAM + CSM-v2 (EG 0.5)"
    ]
    short = ["Center Gauss", "CSM-v2", "Legacy GCW", "Stage-4 G-CAM", "Swin-TCAM", "EG-HAF", "G-CAM+CSM"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    x = np.arange(len(sel))
    w = 0.36
    for ax, col, ylabel, title in [
        (axes[0], "am_iou", "Area-Matched Mask IoU (%)", "(a) Localization Under Centred vs De-Centred Views"),
        (axes[1], "srg_blur", "Symmetric Relevance Gain (SRG, blur)", "(b) Counterfactual Perturbation Faithfulness (SRG)"),
    ]:
        c_pts = [parse_ci(df_c.loc[m, col])[0] for m in sel]
        c_err = [[p - parse_ci(df_c.loc[m, col])[1] for m, p in zip(sel, c_pts)],
                 [parse_ci(df_c.loc[m, col])[2] - p for m, p in zip(sel, c_pts)]]
        d_pts = [parse_ci(df_d.loc[m, col])[0] for m in sel]
        d_err = [[p - parse_ci(df_d.loc[m, col])[1] for m, p in zip(sel, d_pts)],
                 [parse_ci(df_d.loc[m, col])[2] - p for m, p in zip(sel, d_pts)]]
        ax.bar(x - w / 2, c_pts, w, yerr=c_err, capsize=3, color="#0072B2", alpha=0.88, label="Centred View")
        ax.bar(x + w / 2, d_pts, w, yerr=d_err, capsize=3, color="#E69F00", alpha=0.88, label="De-Centred Counterfactual")
        ax.axhline(0.0, color="#444444", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=24, ha="right", fontsize=8.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_xai_counterfactual.png", bbox_inches="tight")
    plt.close(fig)


def fig6_calibration_and_tta() -> None:
    df_cal = pd.read_csv(TAB_DIR / "calibration_internal_parity.csv")
    df_bcn = pd.read_csv(TAB_DIR / "external_tta_bcn.csv")
    df_arc = pd.read_csv(TAB_DIR / "external_tta_archive.csv")
    df_bal = pd.read_csv(TAB_DIR / "external_tta_bcn_bal.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))

    # Panel A: Internal calibration ECE-15 & Classwise ECE & NLL
    methods = ["Uncalibrated", "Temp. Scaling", "BCTS", "Vector Scaling", "BCTS + Bayes\n(Cost=5)"]
    rows_idx = [0, 1, 2, 3, 4]
    ece = [parse_ci(df_cal.iloc[i]["ECE-15 (95% CI)"].replace("%", ""))[0] for i in rows_idx]
    cw_ece = [float(str(df_cal.iloc[i]["Classwise ECE"]).replace("%", "")) for i in rows_idx]
    mf1 = [float(df_cal.iloc[i]["Macro F1"]) * 100.0 for i in rows_idx]
    x = np.arange(len(methods))
    w = 0.27
    axes[0].bar(x - w, ece, w, color="#D55E00", label="ECE-15 (%)")
    axes[0].bar(x, cw_ece, w, color="#E69F00", label="Classwise ECE (%)")
    axes[0].bar(x + w, mf1, w, color="#0072B2", label="Macro F1 (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=7.6)
    axes[0].set_ylabel("Score / Error (%)")
    axes[0].set_title("(a) Internal Calibration (OOF HAM10000, $N=4,500$)")
    axes[0].legend(loc="upper right", frameon=True)

    # Panel B: Streaming External TTA Ladder (Steps A -> I)
    steps_short = [s.split(":")[0] for s in df_bcn["Step"]]
    xs = np.arange(len(steps_short))
    for df_e, col, style, lbl in [
        (df_bcn, "Macro AUC", "-o", "BCN_20000 Macro AUC (N=11,982)"),
        (df_bcn, "MEL AUC", "--o", "BCN_20000 MEL AUC"),
        (df_bal, "MEL AUC", "-^", "Balanced BCN MEL AUC (N=420)"),
        (df_arc, "Macro AUC", ":s", "ISIC Pre-2018 Macro AUC (N=2,903)"),
    ]:
        axes[1].plot(xs, df_e[col].astype(float) * 100.0, style, lw=1.9, markersize=4.5, label=lbl)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(steps_short)
    axes[1].set_xlabel("External Adaptation Step (A -> I)")
    axes[1].set_ylabel("Area Under ROC Curve (%)")
    axes[1].set_title("(b) Zero-Shot & Streaming TTA Discrimination")
    axes[1].legend(loc="lower right", fontsize=7.5, frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_calibration_and_tta.png", bbox_inches="tight")
    plt.close(fig)


def fig7_shortcut_audit() -> None:
    df_s = pd.read_csv(TAB_DIR / "shortcut_detection_audit.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    # Panel A: Classifier reliance (clean vs patched NV)
    vals_a = [df_s["Clean_P_MEL"] * 100.0, df_s["Patched_P_MEL"] * 100.0, df_s["NV_to_MEL_FlipRate"] * 100.0]
    axes[0].bar(["Clean NV\nMean $P$(MEL)", "NV + Ink Marker\nMean $P$(MEL)", "NV -> MEL\nFlip Rate"], vals_a,
                color=["#009E73", "#D55E00", "#661100"], width=0.52, alpha=0.88)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Probability / Flip Rate (%)")
    axes[0].set_title("(a) Induced Surgical-Marker Shortcut Verification ($N=60$)")
    for i, v in enumerate(vals_a):
        axes[0].text(i, v + 2.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B: Patch energy captured by each XAI method
    keys = [
        ("SwinTCAM_patch_energy", "Swin-TCAM\n(Neural)", "#0072B2"),
        ("EGHAF_patch_energy", "EG-HAF\n($\\kappa=0.5$)", "#009E73"),
        ("Additive_TCAM_patch_energy", "Additive\n($\\lambda=0.5$)", "#CC79A7"),
        ("GradCAM_patch_energy", "Stage-4\nGrad-CAM", "#E69F00"),
        ("GradCAM_EG_patch_energy", "Grad-CAM\n+ EG (0.5)", "#56B4E9"),
        ("GradCAM_Add_patch_energy", "Grad-CAM\n+ Add (0.5)", "#999999"),
        ("CSM_v2_patch_energy", "CSM-v2\nPrior", "#D55E00"),
    ]
    means = [df_s[f"{k}_mean"] * 100.0 for k, _, _ in keys]
    sds = [df_s[f"{k}_sd"] * 100.0 for k, _, _ in keys]
    xb = np.arange(len(keys))
    axes[1].bar(xb, means, yerr=sds, capsize=3, color=[c for _, _, c in keys], alpha=0.88, width=0.60)
    axes[1].axhline(df_s["Patch_Area_Pct"], color="#222222", ls="--", lw=1.0, label=f"Uniform Area ({df_s['Patch_Area_Pct']:.1f}%)")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels([l for _, l, _ in keys], fontsize=7.4)
    axes[1].set_ylabel("Attribution Energy on Shortcut Patch (%)")
    axes[1].set_title("(b) Shortcut Signal Preservation vs Prior Dilution")
    axes[1].legend(loc="upper right", fontsize=7.6, frameon=True)
    for i, m in enumerate(means):
        axes[1].text(i, m + sds[i] + 1.5, f"{m:.1f}%", ha="center", va="bottom", fontsize=7.3, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v2_shortcut_audit.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_pub_style()
    fig1_layernorm_diagnosis()
    fig2_sanity_trajectories()
    fig3_leakage_and_kappa()
    fig4_csm_and_tds()
    fig5_counterfactual_xai()
    fig6_calibration_and_tta()
    fig7_shortcut_audit()
    print("Generated all 7 v2 publication figures in paper/figures/.")


if __name__ == "__main__":
    main()
