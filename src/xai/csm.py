"""
Clinical Saliency Mask (CSM) Module.
Implements the dermatological ABCD rule (Asymmetry, Border, Color, Diameter)
as a computational interpretability framework (Algorithm 1 in CG-HAF).
"""

from typing import Dict, Tuple
import cv2
import numpy as np


class ClinicalSaliencyMask:
    """
    Translates bedside dermatological diagnostic heuristics into computational saliency maps.
    Model-agnostic and parameter-efficient.
    """
    def __init__(
        self,
        w_a: float = 1.0,
        w_b: float = 1.2,
        w_c: float = 0.9,
        w_d: float = 1.0,
        hsv_kernel_size: int = 15,
        canny_low: int = 50,
        canny_high: int = 150
    ):
        self.w_a = w_a
        self.w_b = w_b
        self.w_c = w_c
        self.w_d = w_d
        self.hsv_kernel_size = hsv_kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.eps = 1e-8

    def _compute_asymmetry(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Calculates spatial asymmetry via horizontal and vertical reflection differences (Eq. 3.1).
        """
        h, w = binary_mask.shape
        # Vertical split & reflection
        top = binary_mask[:h // 2, :]
        bottom = binary_mask[h // 2:, :]
        bottom_flipped = np.flipud(bottom)
        min_h = min(top.shape[0], bottom_flipped.shape[0])
        vert_diff = np.abs(top[:min_h, :] - bottom_flipped[:min_h, :])
        vert_map = np.zeros_like(binary_mask, dtype=np.float32)
        vert_map[:min_h, :] = vert_diff
        vert_map[h // 2:h // 2 + min_h, :] = np.flipud(vert_diff)

        # Horizontal split & reflection
        left = binary_mask[:, :w // 2]
        right = binary_mask[:, w // 2:]
        right_flipped = np.fliplr(right)
        min_w = min(left.shape[1], right_flipped.shape[1])
        horiz_diff = np.abs(left[:, :min_w] - right_flipped[:, :min_w])
        horiz_map = np.zeros_like(binary_mask, dtype=np.float32)
        horiz_map[:, :min_w] = horiz_diff
        horiz_map[:, w // 2:w // 2 + min_w] = np.fliplr(horiz_diff)

        asym_map = 0.5 * (vert_map + horiz_map)
        return asym_map / (np.max(asym_map) + self.eps)

    def _compute_border(self, gray: np.ndarray) -> np.ndarray:
        """
        Combines Canny edge response and Sobel gradient magnitude (Eq. 3.2).
        """
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return grad_mag / (np.max(grad_mag) + self.eps)

    def _compute_color_variation(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Computes local chromatic heterogeneity in HSV space using a sliding window variance (Eq. 3.3).
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        k = self.hsv_kernel_size
        kernel = np.ones((k, k), dtype=np.float32) / (k * k)

        var_sum = np.zeros(rgb_image.shape[:2], dtype=np.float32)
        for c in range(3):
            ch = hsv[:, :, c]
            mean = cv2.filter2D(ch, -1, kernel)
            mean_sq = cv2.filter2D(ch ** 2, -1, kernel)
            var = np.maximum(mean_sq - mean ** 2, 0)
            var_sum += var

        var_avg = var_sum / 3.0
        return (var_avg - np.min(var_avg)) / (np.max(var_avg) - np.min(var_avg) + self.eps)

    def _compute_diameter(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Computes Euclidean distance transform encoding soft attention towards lesion centroid (Eq. 3.4).
        """
        dist = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
        return dist / (np.max(dist) + self.eps)

    def generate(self, rgb_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generates the composite Clinical Saliency Mask and individual ABCD component maps.

        Args:
            rgb_image: Input image uint8 in RGB format, shape (H, W, 3).

        Returns:
            composite_mask: Normalized [0, 1] clinical saliency map.
            components: Dictionary containing maps for 'A', 'B', 'C', 'D'.
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Otsu thresholding for lesion segmentation
        _, binary_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Individual ABCD representations
        map_a = self._compute_asymmetry(binary_mask)
        map_b = self._compute_border(gray)
        map_c = self._compute_color_variation(rgb_image)
        map_d = self._compute_diameter(binary_mask)

        # Clinical weighted combination (Eq. 3.5)
        raw_composite = (
            self.w_a * map_a +
            self.w_b * map_b +
            self.w_c * map_c +
            self.w_d * map_d
        )

        composite_mask = (raw_composite - np.min(raw_composite)) / (
            np.max(raw_composite) - np.min(raw_composite) + self.eps
        )

        components = {
            "A_asymmetry": map_a,
            "B_border": map_b,
            "C_color": map_c,
            "D_diameter": map_d
        }
        return composite_mask, components

