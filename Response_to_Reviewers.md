# Comprehensive Point-by-Point Response to Reviewer Comments (Revision v2)

**Manuscript Title:** Evidence-Gated Hybrid Attention Fusion for Explainable Skin Lesion Diagnosis: Resolving LayerNorm Jacobian Collapse and Centering Confounders on Swin Transformer V2  
**Authors:** Md. Fahim Sarker Mridul, Nabila Islam, Rufaida Jaman Rayta, Mohammad Shorif Uddin  
**Target Venues:** *IEEE Transactions on Medical Imaging* / *IEEE Journal of Biomedical and Health Informatics* / *Medical Image Analysis*  
**Repository & Code Release:** [https://github.com/fahiiim/Thesis](https://github.com/fahiiim/Thesis)

---

## Executive Overview of Revision v2

We express our deepest appreciation to the reviewer for an extraordinary, rigorous, and constructive evaluation. Rather than offering defensive justifications, we have used this review as an opportunity to perform a comprehensive methodological overhaul:

1. **Discovery of LayerNorm Jacobian Collapse:** We mathematically derived and empirically confirmed ($N=70$, $r = +0.9995 \pm 0.0010$) that backpropagating through pre-LayerNorm representations scales token gradients by the inverse forward token variance $1/\sigma_p$. This explained why legacy gradient accumulators (like legacy Swin-GCW) were invariant to classifier head randomization ($r = 0.987 \pm 0.014$). We introduced **Swin-TCAM** (post-norm tokens at \texttt{backbone.norm} + Stage-3 LayerCAM + log-odds contrastive target), fully restoring parameter sensitivity ($r = 0.194 \pm 0.565$) and class sensitivity ($r = 0.158 \pm 0.519$).
2. **Resolution of Prior Leakage (Evidence-Gated Fusion, EG-HAF):** We replaced vulnerable additive combination with **Evidence-Gated Fusion** ($M = (N + \varepsilon)(1 - \kappa + \kappa P)$). We theoretically proved that EG-HAF preserves neural shortcut signals ($\ge 1 - \kappa$, Proposition 1) and establishes a deterministic upper bound on prior leakage (Proposition 2). Across 369 empirical evaluations, exactly zero bound violations occurred ($0/369$).
3. **Rigorous Counterfactual Benchmark & Decoupled Perturbation Faithfulness:** Across $N=338$ Centred and $N=175$ De-Centred crops evaluated against Philipp Tschandl's official expert masks, we exposed that photographer centering bias allows a trivial Center Gaussian prior to score $98.22\%$ pointing and $71.38\%$ IoU on centered lesions, collapsing to $55.43\%$ pointing and $32.06\%$ IoU when lesions are off-center. EG-HAF maintains robust localization ($40.62\%$ de-centred IoU) and achieves significantly superior perturbation faithfulness (Symmetric Relevance Gain $+6.154$ nats vs. $+5.732$ for Stage-4 Grad-CAM, paired Wilcoxon $p = 8.54 \times 10^{-6}$, Holm-adjusted $p = 3.42 \times 10^{-5}$).
4. **Shortcut-Detection Audit (Adebayo et al., ICLR 2022 Protocol):** In a controlled spurious-correlation experiment (where a peripheral surgical marker was planted on melanoma images), we verified that the fine-tuned model suffered a $100.0\%$ flip rate ($60/60$ benign nevi classified as melanoma). Pure neural Swin-TCAM placed $10.74\% \pm 1.28\%$ of its energy on the marker, and EG-HAF preserved $10.64\% \pm 1.52\%$ ($> 99\%$ retention), confirming that clinical evidence gating does not conceal model bias.
5. **Overhauled Clinical Grounding (CSM-v2):** We upgraded the clinical prior with FOV vignette exclusion, DullRazor hair inpainting, polarity-safe GrabCut, and Stolz TDS-share weighting ($1.3:0.1:0.5:0.5$). CSM-v2 lifted mask IoU from $55.89\%$ to $61.20\%$ ($N=615$, $p < 10^{-15}$) and yielded an algorithmic melanoma-versus-nevus TDS signal of $0.7285$ ROC-AUC.
6. **Internal Calibration & Streaming Test-Time Adaptation:** On $N=4,500$ HAM10000 lesions (5-fold lesion-grouped OOF), Bias-Corrected Temperature Scaling (BCTS) lowered ECE-15 from $19.58\%$ to $1.75\%$ and classwise ECE to $1.15\%$. Across $14,885$ external lesions (BCN\_20000 and MSKCC), our streaming TTA ladder (Steps A$\rightarrow$I) lifted zero-shot accuracy from $40.75\%$ to $50.07\%$, Macro AUC from $0.7409$ to $0.7900$, and melanoma sensitivity to $54.85\%$ at calibrated threshold without requiring target-cohort labels.

Below is our detailed point-by-point response to each critique.

---

## Detailed Point-by-Point Responses

### Major Concern 1: Directionality of Claims vs. Empirical Tables
**Critique:** *In Table 2, Stage-4 Grad-CAM beats CG-HAF on all five metrics... The abstract's "significantly outperforming Stage-4 Grad-CAM (p = 3.3×10⁻²¹ on energy)" reports a significant difference whose direction favors Grad-CAM.*  
**Response:** We thank the reviewer for identifying this narrative error. In the revised manuscript (§Abstract, §5.3, §6.2), we state the empirical reality with complete accuracy:
1. Stage-4 Grad-CAM produces concentrated central Gaussian-like blobs over the lesion core, concentrating higher raw energy inside the expert mask ($70.90\%$ vs. $58.39\%$, $p = 2.96 \times 10^{-2}$).
2. However, when evaluated on **perturbation faithfulness** (Symmetric Relevance Gain, which directly measures whether masked tokens are essential for model prediction independent of mask geometry), EG-HAF significantly outperforms Stage-4 Grad-CAM ($+6.154$ nats vs. $+5.732$ nats, paired Wilcoxon $p = 8.54 \times 10^{-6}$, Holm-adjusted $p = 3.42 \times 10^{-5}$).

---

### Major Concern 2: Sanity Check Evidence & LayerNorm Invariance
**Critique:** *Swin-GCW and CG-HAF keep r ≈ 0.99 after randomizing the classifier head... CSM is model-independent by construction, so the fused map contains a component that cannot change under randomization... Reaching r ≈ 0 looks like cancellation between an anti-correlated neural part and a constant part.*  
**Response:** The reviewer's intuition uncovered a profound architectural pathology:
1. **LayerNorm Jacobian Collapse:** In Section 3.1 and Section 5.1, we derive and prove that pre-norm token gradients are scaled by $1/\sigma_p$. The correlation between Stage-4 $|\partial y/\partial A|$ and $1/\sigma_p$ is $r = +0.9995 \pm 0.0010$. Because $1/\sigma_p$ is a forward token statistic independent of classifier weights, legacy Swin-GCW exhibited $r = 0.987 \pm 0.014$ under head scrambling.
2. **Swin-TCAM Fix:** By tapping tokens at \texttt{backbone.norm} and employing contrastive log-odds scoring, Swin-TCAM restores true parameter dependence ($r = 0.194 \pm 0.565$ under head randomization; $r = -0.135 \pm 0.213$ under complete backbone scrambling).
3. **Evidence-Gated Fusion (EG-HAF):** To eliminate additive prior leakage ($r \rightarrow +0.372$ at $\lambda=0.5$), EG-HAF scales attributions multiplicatively ($M = (N + \varepsilon)g$). Under full backbone randomization, EG-HAF decorrelates to $r = -0.113 \pm 0.216$. We proven and verified Proposition 2, establishing a rigorous bound on prior leakage with 0 violations across 369 evaluations.

---

### Major Concern 3: Photographer Centering Bias & Trivial Baselines
**Critique:** *A segmentation mask would score 100% on pointing and energy. Missing trivial baselines: center-Gaussian prior, random noise, edge filter, Otsu mask alone, and Grad-CAM + CSM control.*  
**Response:** We implemented all five requested controls and evaluated them across $N=338$ Centred lesions and $N=175$ De-Centred counterfactual crops (Table 4 and Table 5):
1. **The Centering Confounder:** A trivial non-learned Center Gaussian blob achieved $98.22\%$ pointing accuracy and $71.38\%$ AM-IoU in the Centred view, but collapsed to $55.43\%$ pointing and $32.06\%$ IoU when lesions were de-centred.
2. **EG-HAF Robustness:** In contrast, EG-HAF maintained $40.62\%$ de-centred IoU and an SRG of $+6.235$ nats, proving true model-guided localization.
3. **Grad-CAM + CSM-v2 Control:** Fusing CSM-v2 with Grad-CAM achieved the highest overall pointing accuracy ($93.49\%$) and AM-IoU ($65.98\%$), confirming that our clinical prior acts as a universal spatial regularizer across architectures.

---

### Major Concern 4: External Validation Circularity
**Critique:** *External validation on BCN_20000 uses Otsu masks generated by CSM itself... This is circular.*  
**Response:** We have completely eliminated circular masks:
1. All internal XAI benchmark evaluations ($N=338$ Centred and $N=175$ De-Centred) are conducted strictly against **Philipp Tschandl's official expert dermatologist manual segmentations**.
2. External evaluation on BCN\_20000 and MSKCC is conducted strictly via **mask-free classification discrimination and streaming test-time adaptation** across all $14,885$ external lesions.

---

### Major Concern 5: External Domain Shift Candor & Calibration
**Critique:** *Macro-F1 and melanoma recall drop under hospital domain shift... This trade-off must be candidly discussed.*  
**Response:** In Section 5.7 and Section 6.3, we provide an unvarnished clinical discussion:
1. Off-the-shelf zero-shot deployment on BCN\_20000 yields poor accuracy ($40.75\%$) and severe miscalibration ($54.64\%$ ECE).
2. Our streaming TTA ladder (Steps A$\rightarrow$I: aperture-masked Shades-of-Gray, BCTS probability calibration, classifier AdaBN, EM prior adaptation, and dihedral averaging) improves accuracy to $50.07\%$ and Macro AUC from $0.7409$ to $0.7900$, lifting melanoma sensitivity to $54.85\%$ at calibrated threshold without requiring external ground-truth labels.

---

### Major Concern 6: Clinical Grounding & Semiologic Accuracy
**Critique:** *Stolz ABCD rule: D is Differential structures, not Diameter... Conflating Menzies checklist with Argenziano's 7-point checklist...*  
**Response:** Section 2.3 and Section 3.3 have been thoroughly corrected:
1. We explicitly distinguish Stolz's dermatoscopic ABCD rule (where D is Differential structures, TDS weights $1.3, 0.1, 0.5, 0.5$) from public screening ABCDE rules (where D is Diameter).
2. In CSM-v2, constituent weights match Stolz's TDS shares: $w_A = 0.542, w_B = 0.042, w_C = 0.208, w_D = 0.208$.
3. We clearly differentiate Argenziano's 7-point checklist from the Menzies method.
4. We validated CSM-v2 segmentations against expert masks across $N=615$ lesions, achieving $61.20\%$ mean IoU ($p < 10^{-15}$) and an algorithmic MEL-vs-NV TDS ROC-AUC of $0.7285$.

---

### Major Concern 7: Statistical Rigor & Reproducibility
**Critique:** *Report exact sample sizes, paired tests, confidence intervals, and multiplicity control.*  
**Response:**
1. All sample sizes are stated with complete precision: $N=4,500$ internal test split, $N=338$ centred explainability cohort, $N=175$ de-centred counterfactual cohort, $N=123 \times 3$ seeds for 12-block Adebayo checks, $N=60$ for shortcut audit, and $N=14,885$ for external cohorts.
2. All tables report paired bootstrap 95\% confidence intervals (1,000 resamples).
3. All $p$-values are Holm-Bonferroni adjusted, and equivalence is evaluated via TOST ($\Delta = \pm 0.10$ nats).
4. All figures and tables are generated programmatically from verified CSVs in `outputs/tables/`.
