"""
Clinical Saliency Mask (CSM) Module.
Implements the dermatological ABCD rule (Asymmetry, Border, Color, Diameter)
with polarity-verified foreground lesion segmentation and artifact-resistant
spatial localization.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class ClinicalSaliencyMask:
    """
    Translates bedside dermatological diagnostic heuristics (ABCD criteria)
    into normalized [0, 1] spatial saliency maps.
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

    def _segment_lesion(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Extracts a polarity-verified binary lesion mask (1 = lesion, 0 = skin).
        Handles dark dermatoscope vignettes, pre-masked cutouts, and pale lesions.
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Detect pre-masked black cutouts or heavy circular vignettes on borders
        border_strip = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        if np.mean(border_strip < 15) > 0.5:
            # Dark surround is already non-lesion background
            lesion_mask = (gray > 15).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
            lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
            return lesion_mask

        # Hair/speckle suppression via median + Gaussian smoothing
        smooth = cv2.GaussianBlur(gray, (7, 7), 0)
        _, inv_otsu = cv2.threshold(smooth, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Verify polarity: lesions are typically more compact/centrally located than surrounding skin
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2.0, w / 2.0
        center_dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        fg_pixels = inv_otsu > 0
        bg_pixels = ~fg_pixels
        if np.any(fg_pixels) and np.any(bg_pixels):
            mean_dist_fg = float(np.mean(center_dist[fg_pixels]))
            mean_dist_bg = float(np.mean(center_dist[bg_pixels]))
            fg_frac = float(np.mean(fg_pixels))
            # If the Otsu foreground is farther from center than background or covers >75%, flip polarity
            if mean_dist_fg > mean_dist_bg and fg_frac > 0.55:
                inv_otsu = (1 - inv_otsu).astype(np.uint8)

        # Morphological cleanup to remove isolated hair strands and fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(inv_otsu.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Keep largest connected component near the image center if fragmented
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            best_label = 1
            best_score = -1.0
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                lcx, lcy = centroids[lbl]
                dist_norm = np.hypot(lcx - cx, lcy - cy) / (np.hypot(cx, cy) + self.eps)
                score = area * (1.0 - 0.5 * dist_norm)
                if score > best_score:
                    best_score = score
                    best_label = lbl
            cleaned = (labels == best_label).astype(np.uint8)

        return cleaned

    def _compute_asymmetry(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Calculates lesion asymmetry by reflecting the mask around its own centroid
        along horizontal and vertical axes, constrained to the lesion ROI.
        """
        h, w = binary_mask.shape
        mask_f = binary_mask.astype(np.float32)
        if np.sum(mask_f) < 10:
            return np.zeros((h, w), dtype=np.float32)

        # Shift lesion centroid to image center before reflection
        moments = cv2.moments(mask_f)
        cx = moments["m10"] / (moments["m00"] + self.eps)
        cy = moments["m01"] / (moments["m00"] + self.eps)

        dx = (w / 2.0) - cx
        dy = (h / 2.0) - cy
        M_center = np.float32([[1, 0, dx], [0, 1, dy]])
        centered = cv2.warpAffine(mask_f, M_center, (w, h), flags=cv2.INTER_NEAREST)

        v_diff = np.abs(centered - np.flipud(centered))
        h_diff = np.abs(centered - np.fliplr(centered))
        asym_centered = 0.5 * (v_diff + h_diff)

        # Warp back to original lesion coordinates
        M_back = np.float32([[1, 0, -dx], [0, 1, -dy]])
        asym_map = cv2.warpAffine(asym_centered, M_back, (w, h), flags=cv2.INTER_LINEAR)
        asym_map = cv2.GaussianBlur(asym_map, (9, 9), 2.0)
        return asym_map / (np.max(asym_map) + self.eps)

    def _compute_border(self, gray: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """
        Measures border irregularity along the morphological contour band of the lesion.
        """
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Focus edge response around the true lesion margin band
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(binary_mask.astype(np.uint8), kernel)
        eroded = cv2.erode(binary_mask.astype(np.uint8), kernel)
        margin_band = cv2.GaussianBlur((dilated - eroded).astype(np.float32), (11, 11), 3.0)
        margin_band = margin_band / (np.max(margin_band) + self.eps)

        border_map = grad_mag * (0.2 * (binary_mask > 0).astype(np.float32) + margin_band)
        return border_map / (np.max(border_map) + self.eps)

    def _compute_color_variation(self, rgb_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """
        Computes local chromatic heterogeneity in HSV space using reflection padding
        and soft lesion masking to avoid peripheral corner/vignette artifacts.
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        k = self.hsv_kernel_size
        kernel = np.ones((k, k), dtype=np.float32) / (k * k)

        var_sum = np.zeros(rgb_image.shape[:2], dtype=np.float32)
        for c in range(3):
            ch = hsv[:, :, c]
            mean = cv2.filter2D(ch, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
            mean_sq = cv2.filter2D(ch ** 2, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
            var = np.maximum(mean_sq - mean ** 2, 0.0)
            var_sum += var

        var_avg = var_sum / 3.0
        # Soft spatial containment within slightly dilated lesion boundary
        roi = cv2.GaussianBlur(
            cv2.dilate(binary_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(np.float32),
            (15, 15),
            4.0
        )
        var_masked = var_avg * roi
        return (var_masked - np.min(var_masked)) / (np.max(var_masked) - np.min(var_masked) + self.eps)

    def _compute_diameter(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Computes Euclidean distance transform strictly inside the lesion interior (0 outside).
        """
        dist = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
        return dist / (np.max(dist) + self.eps)

    def generate(
        self,
        rgb_image: np.ndarray,
        binary_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generates the composite Clinical Saliency Mask and individual ABCD component maps.

        Args:
            rgb_image: Input image uint8 in RGB format, shape (H, W, 3).
            binary_mask: Optional external segmentation mask (1 = lesion, 0 = background).
                         If None, uses polarity-verified automatic segmentation.

        Returns:
            composite_mask: Normalized [0, 1] clinical saliency map.
            components: Dictionary containing maps for 'A', 'B', 'C', 'D', and 'lesion_mask'.
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        if binary_mask is None:
            lesion_mask = self._segment_lesion(rgb_image)
        else:
            lesion_mask = (binary_mask > 0).astype(np.uint8)

        # Individual ABCD representations
        map_a = self._compute_asymmetry(lesion_mask)
        map_b = self._compute_border(gray, lesion_mask)
        map_c = self._compute_color_variation(rgb_image, lesion_mask)
        map_d = self._compute_diameter(lesion_mask)

        # Clinical weighted combination
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
            "D_diameter": map_d,
            "lesion_mask": lesion_mask
        }
        return composite_mask, components
