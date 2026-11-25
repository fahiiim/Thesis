# CG-HAF Journal Publication Upgrade Tracker (Revision v2)

**Target Venues:** *IEEE Transactions on Medical Imaging (TMI)* / *IEEE Journal of Biomedical and Health Informatics (JBHI)* / *Medical Image Analysis (MedIA)*  
**Lead Hardware Engine:** Local NVIDIA GeForce RTX 5070 (12.82 GB VRAM, Blackwell `sm_120`, CUDA 13.0, Tensor-Core AMP)  
**Last Updated:** November 25, 2026

---

## Overall Progress Summary

| Phase | Description | Status | Verified Deliverables & CSVs |
| :--- | :--- | :---: | :--- |
| **P0: Data & Checkpoint Audit** | Audited checkpoint history, unsegmented vs cutout inference, dark corners | **VERIFIED** | `outputs/tables/audit_data_integrity.json` (Epoch 10 saved, unsegmented Acc 76.98%, cutouts 12.71%, 4,500/4,500 Tschandl masks present) |
| **P0: LayerNorm Jacobian Audit** | Diagnosed pre-norm gradient scaling by $1/\sigma_p$ | **VERIFIED** | `outputs/tables/diagnose_gcw_layernorm.csv` ($r_{GCW, 1/\sigma_p} = +0.9995 \pm 0.0010$, Head-Rand $r = 0.987 \pm 0.014$, Class-Sens $r = 0.986 \pm 0.015$) |
| **P1: Swin-TCAM Module** | Implemented post-norm Grad-CAM + Stage-3 LayerCAM + log-odds | **VERIFIED** | `src/xai/token_cam.py` (Restores Head-Rand $r = 0.194 \pm 0.565$, Class-Sens $r = 0.158 \pm 0.519$) |
| **P1: CSM-v2 Clinical Module** | Implemented FOV vignette masking, DullRazor, GrabCut, Stolz TDS weights | **VERIFIED** | `outputs/tables/csm_v2_segmentation_eval.csv` ($55.89\% \rightarrow 61.20\%$ IoU, $p < 10^{-15}$), `csm_v2_clinical_tds_signal.csv` (MEL-vs-NV TDS AUC = 0.7285) |
| **P1: Evidence-Gated Fusion** | Formulated EG-HAF ($M = (N+\varepsilon)g$), proven Propositions 1 & 2 | **VERIFIED** | `outputs/tables/proposition2_leakage_verify.csv` (0 violations / 369 evaluations), `outputs/tables/additive_leakage_verify.csv`, `outputs/tables/kappa_tradeoff_v2.csv` |
| **P1: Internal BCTS Calibration** | 5-fold lesion-grouped OOF calibration on $N=4,500$ HAM10000 | **VERIFIED** | `outputs/tables/calibration_internal_parity.csv` (ECE-15 $19.58\% \rightarrow 1.75\%$, Classwise ECE $1.15\%$, Accuracy $76.93\% \rightarrow 86.87\%$) |
| **P2: Streaming External TTA** | Multi-center streaming TTA ladder (A$\rightarrow$I) on $14,885$ external lesions | **VERIFIED** | `outputs/tables/external_tta_bcn.csv` ($N=11,982$: Acc $40.75\% \rightarrow 50.07\%$, Macro AUC $0.7409 \rightarrow 0.7900$, MEL Sens $42.84\% \rightarrow 54.85\%$) |
| **P2: 12-Block Adebayo Check** | Cascading & Independent randomization across 12 blocks $\times$ 3 seeds | **VERIFIED** | `outputs/tables/sanity_v2_cascading.csv`, `outputs/tables/sanity_v2_independent.csv` |
| **P2: Counterfactual Benchmark** | Centred ($N=338$) vs De-Centred ($N=175$) views against 5 controls | **VERIFIED** | `outputs/tables/xai_benchmark_v2_centred.csv`, `outputs/tables/xai_benchmark_v2_decentred.csv`, `outputs/tables/xai_benchmark_v2_stats.csv` (EG-HAF SRG $+6.154$ nats, beats Grad-CAM $p=3.42\times 10^{-5}$) |
| **P2: Shortcut-Detection Audit** | Spurious ink marker on melanoma images (Adebayo ICLR 2022 protocol) | **VERIFIED** | `outputs/tables/shortcut_detection_audit.csv` (100% flip rate, 60/60; Swin-TCAM 10.74% patch energy, EG-HAF 10.64%, >99% shortcut preservation) |
| **P3: Camera-Ready Manuscript** | Clean 18-page single-column manuscript compiled with pdflatex + bibtex | **VERIFIED** | `paper/main.tex`, `paper/main.pdf` (18 pages, 2.6 MB, 0 errors, 0 undefined citations), 7 verified 300-DPI figures in `paper/figures/` |
| **P3: Reviewer Response Letter** | Detailed point-by-point response addressing every critique | **VERIFIED** | `Response_to_Reviewers.md` (completely overhauled) |

---

## Key Experimental Findings Summary

1. **LayerNorm Jacobian Collapse:** In pre-norm vision transformers, gradient attributions with respect to pre-norm tokens are scaled by $1/\sigma_p$ ($r = +0.9995$). Swin-TCAM eliminates this pathology by tapping tokens post-norm at \texttt{backbone.norm} and employing contrastive log-odds scoring.
2. **Photographer Centering Bias:** Non-learned Center Gaussian achieves $98.22\%$ pointing accuracy and $71.38\%$ IoU on centered lesions, but collapses to $55.43\%$ pointing and $32.06\%$ IoU when lesions are de-centred.
3. **Perturbation Faithfulness:** EG-HAF achieves an SRG of $+6.154$ nats [95\% CI: $5.664, 6.633$], significantly outperforming Stage-4 Grad-CAM ($+5.732$, Holm-adjusted $p = 3.42 \times 10^{-5}$) and improving boundary localization over Swin-TCAM ($p = 1.77 \times 10^{-44}$).
4. **Shortcut Preservation:** In a verified $100\%$ flip-rate shortcut model, EG-HAF preserves $10.64\% \pm 1.52\%$ of patch energy compared to $10.74\% \pm 1.28\%$ for pure neural Swin-TCAM, demonstrating that evidence-gating does not conceal model vulnerabilities.
5. **Probability Calibration:** Bias-Corrected Temperature Scaling (BCTS) eliminates severe overconfidence, lowering ECE-15 from $19.58\%$ to $1.75\%$ and classwise ECE to $1.15\%$.
6. **Zero-Shot Generalization:** Streaming TTA without target labels lifts BCN\_20000 Macro AUC from $0.7409$ to $0.7900$ and balanced BCN Macro AUC to $0.8273$.
