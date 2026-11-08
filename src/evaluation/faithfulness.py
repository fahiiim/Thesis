"""
Faithfulness Evaluation via Pixel Perturbation (Deletion & Insertion AUC).
Quantifies whether the highlighted regions faithfully represent features used
by the Swin Transformer for diagnostic prediction.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn


class FaithfulnessEvaluator:
    """
    Computes Deletion and Insertion Area Under the Curve (AUC):
    - Deletion: Successively masks top attributed pixels (confidence should drop rapidly).
    - Insertion: Successively unmasks top pixels from blurred/black baseline (confidence should climb rapidly).
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_steps: int = 10,
        baseline_mode: str = "blur"  # "blur" or "black"
    ):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.baseline_mode = baseline_mode

    def _create_baseline(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Creates either a gaussian blurred or constant zero baseline."""
        if self.baseline_mode == "black":
            return torch.zeros_like(image_tensor)
        else:
            # Gaussian smoothed baseline
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=21, sigma=(10.0, 10.0))
            return blur(image_tensor)

    @torch.no_grad()
    def evaluate(
        self,
        image_tensor: torch.Tensor,
        attribution_map: np.ndarray,
        target_class: int
    ) -> Dict[str, float]:
        """
        Runs deletion and insertion experiments for a single image sample.

        Args:
            image_tensor: Normalized tensor of shape (1, 3, H, W).
            attribution_map: 2D array of attribution scores, shape (H, W).
            target_class: Integer index of target class.

        Returns:
            Dictionary with deletion_auc, insertion_auc, and step curves.
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        baseline = self._create_baseline(image_tensor).to(self.device)

        h, w = attribution_map.shape
        flat_attr = attribution_map.flatten()
        # Sort pixel indices by descending attribution importance
        sorted_indices = np.argsort(flat_attr)[::-1]

        total_pixels = h * w
        step_size = total_pixels // self.num_steps

        # Compute initial prediction probabilities
        init_prob = torch.softmax(self.model(image_tensor), dim=-1)[0, target_class].item()
        base_prob = torch.softmax(self.model(baseline), dim=-1)[0, target_class].item()

        deletion_probs = [init_prob]
        insertion_probs = [base_prob]

        # Masks initialized
        del_img = image_tensor.clone()
        ins_img = baseline.clone()

        for step in range(1, self.num_steps + 1):
            pixel_indices = sorted_indices[:step * step_size]
            mask = np.zeros(total_pixels, dtype=bool)
            mask[pixel_indices] = True
            mask_2d = torch.from_numpy(mask.reshape(h, w)).to(self.device).unsqueeze(0).unsqueeze(0)

            # Deletion: Replace top salient pixels with baseline
            del_step = torch.where(mask_2d, baseline, image_tensor)
            p_del = torch.softmax(self.model(del_step), dim=-1)[0, target_class].item()
            deletion_probs.append(p_del)

            # Insertion: Insert top salient pixels into baseline
            ins_step = torch.where(mask_2d, image_tensor, baseline)
            p_ins = torch.softmax(self.model(ins_step), dim=-1)[0, target_class].item()
            insertion_probs.append(p_ins)

        # Calculate trapezoidal AUC normalized to [0, 1]
        x_axis = np.linspace(0, 1, len(deletion_probs))
        del_auc = float(integrate.trapezoid(deletion_probs, x_axis))
        ins_auc = float(integrate.trapezoid(insertion_probs, x_axis))

        return {
            "deletion_auc": del_auc,
            "insertion_auc": ins_auc,
            "deletion_curve": deletion_probs,
            "insertion_curve": insertion_probs
        }

