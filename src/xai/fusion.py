"""
Hybrid Attention Fusion (CG-HAF) Module.
Fuses top-down clinical heuristics (CSM) with bottom-up model attribution (GCW).
Computes quantitative alignment metrics (Pearson correlation, SSIM) to evaluate
clinical causability.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.stats import pearsonr


class HybridAttentionFusion:
    """
    Fuses Clinical Saliency Mask (CSM) and Gradient Coherence Weighting (GCW).
    Evaluates clinical-model alignment hypothesis (Section 3.2.4).
    """
    def __init__(self, lambda_fusion: float = 0.5):
        self.lambda_fusion = lambda_fusion
        self.eps = 1e-8

    def fuse(
        self,
        csm_map: np.ndarray,
        gcw_map: np.ndarray,
        custom_lambda: float = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Computes hybrid explanation map and clinical-model concordance metrics.

        Args:
            csm_map: 2D array normalized to [0, 1].
            gcw_map: 2D array normalized to [0, 1].
            custom_lambda: Optional override for fusion weighting parameter.

        Returns:
            fused_map: Composite explanation array [0, 1].
            metrics: Dictionary containing Pearson correlation and energy concordance.
        """
        lmbda = self.lambda_fusion if custom_lambda is None else custom_lambda

        # Linear convex combination (Eq. 3.10)
        fused = lmbda * csm_map + (1.0 - lmbda) * gcw_map
        fused_norm = (fused - np.min(fused)) / (np.max(fused) - np.min(fused) + self.eps)

        # Pearson spatial alignment score
        flat_csm = csm_map.flatten()
        flat_gcw = gcw_map.flatten()
        corr, p_value = pearsonr(flat_csm, flat_gcw)

        # Binarized mask overlap (Top 20% saliency threshold)
        thresh_csm = flat_csm >= np.percentile(flat_csm, 80)
        thresh_gcw = flat_gcw >= np.percentile(flat_gcw, 80)
        intersection = np.logical_and(thresh_csm, thresh_gcw).sum()
        union = np.logical_or(thresh_csm, thresh_gcw).sum()
        iou = float(intersection) / float(union + self.eps)

        metrics = {
            "pearson_correlation": float(corr),
            "p_value": float(p_value),
            "top20_iou": float(iou)
        }

        return fused_norm, metrics

