"""
Faithfulness Evaluation via Pixel Perturbation (Deletion & Insertion AUC).
Optimized for 100% GPU utilization on NVIDIA RTX 5070 by batching all
perturbation steps into a single CUDA Tensor-Core forward pass with AMP.
"""

from typing import Dict
import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur


class FaithfulnessEvaluator:
    """
    Computes Deletion and Insertion Area Under the Curve (AUC) in batched GPU mode:
    - Deletion: Successively masks top attributed pixels (confidence should drop rapidly).
    - Insertion: Successively unmasks top pixels from blurred/black baseline (confidence should climb rapidly).
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_steps: int = 10,
        baseline_mode: str = "blur"
    ):
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.baseline_mode = baseline_mode
        self.blur = GaussianBlur(kernel_size=21, sigma=(10.0, 10.0))

    def _create_baseline(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self.baseline_mode == "black":
            return torch.zeros_like(image_tensor)
        return self.blur(image_tensor)

    @torch.no_grad()
    def evaluate(
        self,
        image_tensor: torch.Tensor,
        attribution_map: np.ndarray,
        target_class: int
    ) -> Dict[str, float]:
        """
        Runs batched deletion and insertion perturbation curves on GPU in a single forward pass.
        """
        self.model.eval()
        inp = image_tensor.to(self.device)  # (1, 3, H, W)
        baseline = self._create_baseline(inp)
        _, _, h, w = inp.shape

        flat_attr = torch.from_numpy(attribution_map.reshape(-1)).to(self.device)
        sorted_idx = torch.argsort(flat_attr, descending=True)
        total_pixels = flat_attr.numel()

        # Pre-allocate batch of (num_steps + 1) deletion and (num_steps + 1) insertion images on GPU
        n_pts = self.num_steps + 1
        del_masks = torch.zeros((n_pts, 1, h, w), dtype=torch.bool, device=self.device)
        for step in range(1, n_pts):
            k = total_pixels if step == self.num_steps else (total_pixels * step) // self.num_steps
            m = del_masks[step].view(-1)
            m[sorted_idx[:k]] = True

        inp_rep = inp.expand(n_pts, -1, -1, -1)
        base_rep = baseline.expand(n_pts, -1, -1, -1)

        del_batch = torch.where(del_masks, base_rep, inp_rep)
        ins_batch = torch.where(del_masks, inp_rep, base_rep)
        full_batch = torch.cat([del_batch, ins_batch], dim=0)  # (2 * n_pts, 3, H, W)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            logits = self.model(full_batch)
            probs = torch.softmax(logits.float(), dim=-1)[:, target_class]

        del_probs = probs[:n_pts].cpu().numpy().tolist()
        ins_probs = probs[n_pts:].cpu().numpy().tolist()

        x_axis = np.linspace(0.0, 1.0, n_pts)
        del_auc = float(integrate.trapezoid(del_probs, x_axis))
        ins_auc = float(integrate.trapezoid(ins_probs, x_axis))

        return {
            "deletion_auc": del_auc,
            "insertion_auc": ins_auc,
            "deletion_curve": del_probs,
            "insertion_curve": ins_probs
        }
