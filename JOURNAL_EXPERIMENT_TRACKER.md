# CG-HAF Journal Publication Upgrade Tracker

**Target Venues:** *Computers in Biology and Medicine* / *IEEE Journal of Biomedical and Health Informatics (JBHI)* / *Biomedical Signal Processing and Control*  
**Lead Hardware Engine:** Local NVIDIA GeForce RTX 5070 (12.82 GB VRAM, Blackwell `sm_120`, CUDA 13.0, Tensor-Core AMP)  
**Last Updated:** November 12, 2026

---

## Overall Progress

| Phase | Description | Status | Verified Outputs |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Local Blackwell Setup & Checkpoint Verification | **COMPLETED** | `weights/swin_vit_xai_model.pth`, PyTorch 2.14+cu130 (`sm_120`), 4,500 unsegmented HAM10000 test split + 10,015 Tschandl GT masks |
| **Phase 2** | Class-Stratified Quantitative XAI Benchmark ($N=140$) | **COMPLETED** | `outputs/tables/table_xai_comparison.tex`, `faithfulness_comparison_benchmark.png` |
| **Phase 3** | Adebayo Cascading Randomization & $\lambda$-Ablation ($N=70$) | **COMPLETED** | `outputs/tables/table_adebayo_sanity_check.tex`, `table_ablation_lambda.tex`, 300 DPI plots |
| **Phase 4** | Multi-Center External Validation (MSKCC $N=2,903$ & BCN_20000 $N=11,982$) | **COMPLETED** | `table_external_cohorts_full.tex`, `table_external_multicenter_validation.tex`, `external_roc_curves_bcn20000.png`, `external_confusion_matrix_bcn20000.png` |
| **Phase 5** | External Domain-Shift XAI Faithfulness ($N=70$ BCN_20000) | **COMPLETED** | `table_external_xai_comparison.tex` |
| **Phase 6** | Repository Hygiene & Clean GitHub Sync | **COMPLETED** | Clean `.gitignore`, synced `main` on GitHub |

---

## Verified Experimental Findings

### 1. Internal Unsegmented HAM10000 Evaluation & Methodological Fixes
- **Evaluation Protocol:** Switched from pre-segmented black cutouts to unsegmented dermoscopic lesions (`N=4,500` split: Acc `76.82%`, Macro F1 `0.8005`, Macro ROC-AUC `0.9779`) paired with **official dermatologist lesion masks** (`HAM10000_segmentations_lesion_tschandl`) — eliminating inverted-background and Otsu-circularity artifacts.
- **Clinical-Neural Alignment (`N=70` stratified, 10/class):** Mean Pearson correlation between CSM (ABCD) and Hierarchical Swin-GCW = **`+0.4107`**, Pointing Game Hit Rate = **`65.71%`**, Top-20% GT IoU = **`41.44%`**.

### 2. Multi-Method Stratified Benchmark (`N=140`, 20/class × 7 classes)
- **Top-20% GT IoU:** CG-HAF (**`39.86 ± 16.38%`**) outperforms both standalone Swin-GCW (`34.94 ± 16.48%`) and standalone CSM (`30.23 ± 20.67%`), as well as Integrated Gradients (`24.45%`) and SmoothGrad (`35.90%`).
- **Deletion AUC (Lower is Better):** CG-HAF (**`0.4796 ± 0.3225`**) outperforms Swin-GCW (`0.4983`), CSM (`0.5224`), SmoothGrad (`0.5069`), and Integrated Gradients (`0.5236`).
- **Insertion AUC (Higher is Better):** CG-HAF (**`0.7800 ± 0.2316`**) improves over both Swin-GCW (`0.7762`) and CSM (`0.7425`).
- **Wilcoxon Signed-Rank (CG-HAF vs Stage-4 Grad-CAM):** Energy $p = 3.30 \times 10^{-21}$, Insertion $p = 1.94 \times 10^{-3}$, Deletion $p = 0.102$.

### 3. Sanity Checks & Hyperparameter Ablation
- **Adebayo Cascading Randomization (`N=14` stratified):** Spearman rank correlation drops from `1.0000` down to **`-0.4709` (Swin-GCW)** and **`-0.0073` (CG-HAF)** across cascading backbone randomization, proving strict model-parameter dependence.
- **$\lambda$-Sensitivity (`N=70` stratified vs official GT masks):**
  - Pointing Game Hit Rate peaks at **$\lambda=0.3$ (`74.29%`)** and **$\lambda=0.4$ (`70.00%`)** (vs `61.43%` pure GCW and `50.00%` pure CSM).
  - Deletion AUC reaches its minimum (optimal faithfulness) at **$\lambda=0.5\text{--}0.6$ (`0.4500` / `0.4499`)** vs `0.4803` ($\lambda=0.0$) and `0.4809` ($\lambda=1.0$).

### 4. Multi-Center External Validation (Zero-Shot, Non-HAM10000)
- **MSKCC / ViDIR (`N=2,903`):** Shades-of-Gray Color Constancy lifts zero-shot accuracy from `42.44%` $\rightarrow$ **`56.49%`** (+14.05%) and Mean ROC-AUC from `0.6622` $\rightarrow$ **`0.6906`**.
- **BCN_20000 (`N=11,982`):** Shades-of-Gray improves accuracy `40.61%` $\rightarrow$ **`42.60%`** and Mean ROC-AUC `0.7405` $\rightarrow$ **`0.7485`** (and **`0.7622`** Macro AUC on balanced `N=420` cohort).
- **External XAI Faithfulness & Localization (`N=70` BCN_20000, 10/class):**
  - **External Pointing Hit Rate:** CG-HAF achieves **`91.43%`**, beating Stage-4 Grad-CAM (`64.29%`), Integrated Gradients (`65.71%`), and standalone Swin-GCW (`52.86%`).
  - **External Deletion AUC:** CG-HAF (`0.4815`) beats Stage-4 Grad-CAM (`0.4847`), Swin-GCW (`0.4857`), and CSM (`0.5796`).
  - **External Insertion AUC:** CG-HAF (`0.7877`) beats Swin-GCW (`0.7856`), Integrated Gradients (`0.7784`), and CSM (`0.7192`).
