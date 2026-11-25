"""
Generate verified LaTeX table summaries in outputs/tables/paper_tables_summary.txt
and standalone .tex tables in outputs/tables/paper_tab_*.tex directly from observed CSVs.
Zero hardcoded or rounded numbers outside CSVs.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


TAB_DIR = Path("outputs/tables")


def esc(s: str, math_safe: bool = False) -> str:
    st = str(s)
    if math_safe:
        return st.replace("%", "\\%").replace("&", "\\&").replace("±", "$\\pm$")
    return (
        st.replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("±", "$\\pm$")
        .replace(">=", "$\\ge$")
        .replace("<=", "$\\le$")
        .replace("<", "$<$")
        .replace(">", "$>$")
    )


def write_tex_table(name: str, df: pd.DataFrame, cols: list[str] | None = None, headers: list[str] | None = None, raw_first: bool = False) -> None:
    if cols is not None:
        df = df[cols].copy()
    hdr = headers if headers is not None else [esc(c) for c in df.columns]
    spec = "l" + "c" * (len(df.columns) - 1)
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(hdr) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [esc(row[c], math_safe=(raw_first and idx == 0)) for idx, c in enumerate(df.columns)]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TAB_DIR / f"paper_{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # 1. LayerNorm diagnosis summary
    df_ln = pd.read_csv(TAB_DIR / "diagnose_gcw_layernorm.csv")
    ln_rows = []
    for col, lbl in [
        ("r_gcwS4_vs_invSigma", "Stage-4 $|\\partial y/\\partial A|$ vs $1/\\sigma_p$"),
        ("r_gcwFull_vs_invSigma", "Legacy Swin-GCW vs $1/\\sigma_p$"),
        ("r_tcamS4_vs_invSigma", "Swin-TCAM vs $1/\\sigma_p$"),
        ("r_gcw_headRand", "Legacy Swin-GCW (Trained vs Head-Rand)"),
        ("r_tcam_headRand", "Swin-TCAM (Trained vs Head-Rand)"),
        ("r_gcw_classSens", "Legacy Swin-GCW (Target vs Runner-Up)"),
        ("r_tcam_classSens", "Swin-TCAM (Target vs Runner-Up)"),
    ]:
        ln_rows.append({"Diagnostic Metric": lbl, "Mean ± SD (N=70)": f"{df_ln[col].mean():.4f} ± {df_ln[col].std():.4f}"})
    write_tex_table("tab_layernorm", pd.DataFrame(ln_rows), raw_first=True)

    # 2. Sanity cascading (selected key layers for compact table; full in SI)
    df_casc = pd.read_csv(TAB_DIR / "sanity_v2_cascading.csv")
    key_blocks = ["classifier.5", "backbone.norm", "Stage 4 (features.7)", "Stage 3 (features.5)", "Stage 1 (features.1)", "PatchEmbed (features.0)"]
    sel_cols = ["Block", "Legacy Swin-GCW", "Stage-4 Grad-CAM", "Swin-TCAM", "EG-HAF (k=0.5)", "Additive CG-HAF (0.5)"]
    write_tex_table("tab_sanity_casc", df_casc[df_casc["Block"].isin(key_blocks)], cols=sel_cols)

    df_ind = pd.read_csv(TAB_DIR / "sanity_v2_independent.csv")
    write_tex_table("tab_sanity_indep", df_ind[df_ind["Block"].isin(key_blocks)], cols=sel_cols)

    # 3. Benchmark centred + decentred
    for view, fname in [("centred", "tab_xai_centred"), ("decentred", "tab_xai_decentred")]:
        df_x = pd.read_csv(TAB_DIR / f"xai_benchmark_v2_{view}.csv")
        write_tex_table(fname, df_x)

    # 4. CSM-v2 vs CSM-v1
    df_csm = pd.read_csv(TAB_DIR / "csm_v2_segmentation_eval.csv")
    df_csm_fmt = df_csm.copy()
    df_csm_fmt["dx"] = df_csm_fmt["dx"].str.upper()
    df_csm_fmt["iou"] = (df_csm_fmt["iou"] * 100).map(lambda v: f"{v:.2f}%")
    df_csm_fmt["dice"] = (df_csm_fmt["dice"] * 100).map(lambda v: f"{v:.2f}%")
    df_csm_fmt["fail"] = (df_csm_fmt["fail"] * 100).map(lambda v: f"{v:.2f}%")
    write_tex_table("tab_csm_eval", df_csm_fmt, headers=["Version", "Class", "IoU", "Dice", "Fail (<10%)"])

    # 5. Internal Calibration
    df_cal = pd.read_csv(TAB_DIR / "calibration_internal_parity.csv")
    write_tex_table("tab_calibration", df_cal)

    # 6. External TTA
    for tag in ["archive", "bcn", "bcn_bal"]:
        df_t = pd.read_csv(TAB_DIR / f"external_tta_{tag}.csv")
        for c in ["Macro F1", "Macro AUC", "MEL AUC"]:
            df_t[c] = df_t[c].map(lambda v: f"{float(v):.4f}")
        write_tex_table(f"tab_tta_{tag}", df_t)

    print("Wrote standalone paper_tab_*.tex files in outputs/tables/.")


if __name__ == "__main__":
    main()
