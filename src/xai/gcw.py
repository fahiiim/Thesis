"""
Gradient Coherence Weighting (GCW) Module.
Implements spatial gradient coherence analysis for Vision Transformers (Algorithm 2 in CG-HAF),
providing both input-space channel coherence and intermediate Swin feature coherence.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn


class GradientCoherenceWeighting:
    """
    Computes Gradient Coherence Weighting (GCW) to isolate diagnostically reliable features.
    Unlike conventional magnitude-based attribution, GCW evaluates spatial coherence
    via channel-wise L2 normalization.
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

    def compute_input_gcw(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes input-space GCW according to Algorithm 2.

        Args:
            image_tensor: Normalized tensor of shape (1, 3, H, W).
            target_class: Target class index. If None, uses top predicted class.

        Returns:
            gcw_map: Normalized 2D spatial heatmap [0, 1].
            pred_idx: Predicted class index.
            confidence: Softmax probability of target class.
        """
        self.model.eval()
        image_tensor = image_tensor.clone().detach().to(self.device).requires_grad_(True)

        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=-1)

        if target_class is None:
            pred_idx = torch.argmax(probs, dim=-1).item()
        else:
            pred_idx = target_class

        confidence = probs[0, pred_idx].item()
        score = logits[0, pred_idx]

        self.model.zero_grad()
        score.backward()

        # Gradient tensor: (1, 3, H, W) -> (3, H, W)
        grads = image_tensor.grad.detach()[0]  # shape: (3, H, W)

        # Step 3: Calculate Channel-Wise Coherence (L2 norm per channel)
        # alpha_k = sqrt( sum_{i,j} G_{i,j,k}^2 )
        channel_norms = torch.sqrt(torch.sum(grads ** 2, dim=(1, 2)) + self.eps)  # (3,)

        # Step 4: Normalize Coherence Weights to unit vector
        alpha_norm = channel_norms / (torch.norm(channel_norms, p=2) + self.eps)  # (3,)

        # Step 5: Apply Coherence Weighting
        abs_grads = torch.abs(grads)
        weighted_grad = torch.sum(alpha_norm.view(3, 1, 1) * abs_grads, dim=0)  # (H, W)

        # Min-max normalization
        g_min = torch.min(weighted_grad)
        g_max = torch.max(weighted_grad)
        norm_map = (weighted_grad - g_min) / (g_max - g_min + self.eps)

        gcw_map = norm_map.cpu().numpy()

        # Gaussian smoothing to reduce pixel-level sampling artifacts
        if self.blur_kernel > 0:
            gcw_map = cv2.GaussianBlur(
                gcw_map,
                (self.blur_kernel, self.blur_kernel),
                self.blur_sigma
            )
            gcw_map = (gcw_map - np.min(gcw_map)) / (np.max(gcw_map) - np.min(gcw_map) + self.eps)

        return gcw_map, pred_idx, confidence

    def compute_hierarchical_gcw(
        self,
        image_tensor: torch.Tensor,
        stage_name: str = "stage_4",
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Extends GCW to intermediate Swin Transformer stages.
        Measures coherence across intermediate transformer feature channels.
        """
        self.model.eval()
        image_tensor = image_tensor.clone().detach().to(self.device)

        logits = self.model(image_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if not hasattr(self.model, "hook_gradients") or stage_name not in self.model.hook_gradients:
            raise KeyError(f"Hook gradients not registered for '{stage_name}'. Call register_stage_hooks() first.")

        stage_grad = self.model.hook_gradients[stage_name][0]  # (C, H_s, W_s) or (tokens, C)
        if stage_grad.ndim == 2:
            # Reshape tokens to 2D grid
            num_tokens, channels = stage_grad.shape
            grid_size = int(np.sqrt(num_tokens))
            stage_grad = stage_grad.view(grid_size, grid_size, channels).permute(2, 0, 1)

        # Coherence norm across spatial coordinates
        coherence = torch.sqrt(torch.sum(stage_grad ** 2, dim=(1, 2)) + self.eps)
        coherence_norm = coherence / (torch.norm(coherence, p=2) + self.eps)

        # Weight feature channels
        feature_act = self.model.hook_activations[stage_name][0]
        if feature_act.ndim == 2:
            grid_size = int(np.sqrt(feature_act.shape[0]))
            feature_act = feature_act.view(grid_size, grid_size, -1).permute(2, 0, 1)

        stage_map = torch.sum(coherence_norm.view(-1, 1, 1) * torch.relu(feature_act), dim=0)
        stage_map = stage_map.detach().cpu().numpy()

        # Upsample to image size (224x224)
        h, w = image_tensor.shape[2:]
        upsampled = cv2.resize(stage_map, (w, h), interpolation=cv2.INTER_CUBIC)
        norm_stage = (upsampled - np.min(upsampled)) / (np.max(upsampled) - np.min(upsampled) + self.eps)
        return norm_stage

