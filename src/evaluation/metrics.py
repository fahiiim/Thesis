"""
Quantitative Interpretability Metrics (Pointing Game & Saliency Localization).
Evaluates whether attention and gradient maps focus on true lesion anatomy
rather than background artifacts, hair, or borders (Section 4.3).
"""

from typing import Dict, Tuple
import numpy as np


class LocalizationMetrics:
    """
    Computes objective localization benchmarks against ground-truth lesion masks:
    1. Pointing Game Hit Rate (Zhang et al.)
    2. Energy Inside Mask (E-Inside)
    3. Attribution Intersection over Union (IoU)
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def pointing_game_hit(self, attribution_map: np.ndarray, binary_mask: np.ndarray, rtol: float = 1e-6) -> float:
        """
        Tests if the maximum activation point(s) fall within the ground-truth lesion boundary.
        Resolves ties (e.g. on binary masks) in expectation across all maximal pixels.
        """
        mx = float(np.max(attribution_map))
        peaks = attribution_map >= mx - rtol * max(abs(mx), 1.0)
        return float(np.mean(binary_mask[peaks] > 0))

    def energy_inside_mask(self, attribution_map: np.ndarray, binary_mask: np.ndarray) -> float:
        """
        Computes the ratio of total attribution energy falling inside the lesion mask.
        E_inside = sum_{(x,y) in Mask} A(x,y) / sum_{(x,y)} A(x,y)
        """
        norm_attr = np.maximum(attribution_map, 0)
        total_energy = np.sum(norm_attr)
        if total_energy <= self.eps:
            return 0.0

        mask_bool = (binary_mask > 0)
        inside_energy = np.sum(norm_attr[mask_bool])
        return float(inside_energy / (total_energy + self.eps))

    def evaluate_sample(
        self,
        attribution_map: np.ndarray,
        binary_mask: np.ndarray,
        threshold_pct: float = 80.0
    ) -> Dict[str, float]:
        """
        Evaluates a single sample across localization metrics.
        """
        hit = self.pointing_game_hit(attribution_map, binary_mask)
        e_inside = self.energy_inside_mask(attribution_map, binary_mask)

        # Attribution IoU at specified percentile threshold
        thresh = np.percentile(attribution_map, threshold_pct)
        attr_bin = (attribution_map >= thresh).astype(np.uint8)
        mask_bin = (binary_mask > 0).astype(np.uint8)

        intersection = np.logical_and(attr_bin, mask_bin).sum()
        union = np.logical_or(attr_bin, mask_bin).sum()
        iou = float(intersection) / float(union + self.eps)

        return {
            "pointing_game_hit": float(hit),
            "energy_inside_mask": float(e_inside),
            "mask_iou": float(iou)
        }

