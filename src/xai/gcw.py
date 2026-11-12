"""
Gradient Coherence Weighting (GCW) Module.
Implements spatial gradient coherence analysis for Swin Transformer V2,
extracting channel-wise L2-normalized gradient coherence both across
hierarchical Swin Transformer feature stages (Stages 3 & 4) and input space.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientCoherenceWeighting:
    """
    Computes Gradient Coherence Weighting (GCW) for Swin Transformer V2.
    Handles torchvision Swin-V2 channels-last (B, H, W, C) stage tensors and
    performs GPU-accelerated gradient coherence extraction.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        blur_kernel: int = 11,
        blur_sigma: float = 2.0
    ):
        self.model = model
        self.device = device
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.eps = 1e-8

    @staticmethod
    def _to_bchw(tensor: torch.Tensor) -> torch.Tensor:
        """Converts Swin (B, H, W, C) or (B, N, C) feature tensors to (B, C, H, W)."""
        if tensor.ndim == 4:
            # Heuristic: in Swin features, channel dim is at axis -1 (128, 256, 512, 1024)
            if tensor.shape[-1] >= tensor.shape[1] and tensor.shape[1] == tensor.shape[2]:
                return tensor.permute(0, 3, 1, 2).contiguous()
            return tensor
        elif tensor.ndim == 3:
            b, n, c = tensor.shape
            g = int(np.sqrt(n))
            return tensor.view(b, g, g, c).permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unsupported tensor rank for spatial reshape: {tensor.shape}")

    def _norm_blur(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalizes heatmap to [0, 1] and applies isotropic Gaussian smoothing."""
        h_min, h_max = np.min(heatmap), np.max(heatmap)
        norm = (heatmap - h_min) / (h_max - h_min + self.eps)
        if self.blur_kernel > 0:
            norm = cv2.GaussianBlur(norm.astype(np.float32), (self.blur_kernel, self.blur_kernel), self.blur_sigma)
            norm = (norm - np.min(norm)) / (np.max(norm) - np.min(norm) + self.eps)
        return norm.astype(np.float32)

    def compute_input_gcw(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes input-space channel-coherence weighted gradients.
        """
        self.model.eval()
        inp = image_tensor.clone().detach().to(self.device).requires_grad_(True)

        logits = self.model(inp)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).item()) if target_class is None else int(target_class)
        confidence = float(probs[0, pred_idx].item())

        self.model.zero_grad()
        logits[0, pred_idx].backward()

        grads = inp.grad.detach()[0]  # (3, H, W)
        channel_norms = torch.sqrt(torch.sum(grads ** 2, dim=(1, 2)) + self.eps)
        alpha_norm = channel_norms / (torch.norm(channel_norms, p=2) + self.eps)

        weighted_grad = torch.sum(alpha_norm.view(3, 1, 1) * torch.abs(grads), dim=0)
        return self._norm_blur(weighted_grad.cpu().numpy()), pred_idx, confidence

    def compute_hierarchical_gcw(
        self,
        image_tensor: torch.Tensor,
        stage_indices: Tuple[int, ...] = (5, 7),
        target_class: Optional[int] = None,
        include_input_detail: bool = False
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes genuine Hierarchical Swin Transformer Gradient Coherence Weighting (GCW)
        by hooking shifted-window attention stage outputs (e.g., Stage 3 at features[5]
        and Stage 4 at features[7]) in a single forward/backward pass on GPU.
        """
        self.model.eval()
        inp = image_tensor.clone().detach().to(self.device).requires_grad_(include_input_detail)
        h, w = inp.shape[2], inp.shape[3]

        stage_acts = {}
        stage_grads = {}
        handles = []

        for idx in stage_indices:
            module = self.model.backbone.features[idx]

            def make_fwd(k):
                def _fwd(mod, mod_in, mod_out):
                    stage_acts[k] = mod_out.detach()
                return _fwd

            def make_bwd(k):
                def _bwd(mod, gin, gout):
                    stage_grads[k] = gout[0].detach()
                return _bwd

            handles.append(module.register_forward_hook(make_fwd(idx)))
            handles.append(module.register_full_backward_hook(make_bwd(idx)))

        try:
            logits = self.model(inp)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).item()) if target_class is None else int(target_class)
            confidence = float(probs[0, pred_idx].item())

            self.model.zero_grad()
            logits[0, pred_idx].backward()
        finally:
            for handle in handles:
                handle.remove()

        # Fuse gradient coherence maps across hooked Swin stages on GPU
        fused_stage_map = torch.zeros((1, 1, h, w), device=self.device)
        for idx in stage_indices:
            grad = self._to_bchw(stage_grads[idx])   # (1, C, H_s, W_s)

            # Channel-wise spatial gradient coherence (L2 norm per feature channel)
            coherence = torch.sqrt(torch.sum(grad ** 2, dim=(2, 3), keepdim=True) + self.eps)  # (1, C, 1, 1)
            coherence_norm = coherence / (torch.norm(coherence, p=2, dim=1, keepdim=True) + self.eps)

            # Coherence-weighted spatial gradient magnitudes |G_c| (strictly positive lesion salience)
            weighted = torch.sum(coherence_norm * torch.abs(grad), dim=1, keepdim=True)  # (1, 1, H_s, W_s)
            up = F.interpolate(weighted, size=(h, w), mode="bilinear", align_corners=False)
            up_min, up_max = up.min(), up.max()
            up_norm = (up - up_min) / (up_max - up_min + self.eps)
            fused_stage_map += up_norm

        fused_stage_map = fused_stage_map / max(len(stage_indices), 1)

        if include_input_detail and inp.grad is not None:
            in_grads = inp.grad.detach()  # (1, 3, H, W)
            ch_norms = torch.sqrt(torch.sum(in_grads ** 2, dim=(2, 3), keepdim=True) + self.eps)
            alpha_in = ch_norms / (torch.norm(ch_norms, p=2, dim=1, keepdim=True) + self.eps)
            in_map = torch.sum(alpha_in * torch.abs(in_grads), dim=1, keepdim=True)
            in_norm = (in_map - in_map.min()) / (in_map.max() - in_map.min() + self.eps)
            # Combine 75% hierarchical Swin feature coherence + 25% fine spatial gradient coherence
            combined = 0.75 * fused_stage_map + 0.25 * in_norm
        else:
            combined = fused_stage_map

        gcw_np = combined[0, 0].cpu().numpy()
        return self._norm_blur(gcw_np), pred_idx, confidence
