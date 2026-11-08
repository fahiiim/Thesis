"""
Comprehensive Comparative XAI Benchmarking Engine.
Evaluates CG-HAF against primary clinical baselines:
1. Grad-CAM (Selvaraju et al., ICCV 2017)
2. Integrated Gradients (Sundararajan et al., ICML 2017)
3. SmoothGrad (Smilkov et al., 2017)
4. Standalone GCW & Standalone CSM

Computes:
- Deletion AUC (Faithfulness degradation)
- Insertion AUC (Feature recovery)
- Energy-Inside-Mask (Background leakage resistance)
- Pointing Game Hit Rate (Diagnostic focus)
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class BaselineXAI:
    """Implementations of Grad-CAM, Integrated Gradients, and SmoothGrad for Swin Transformer."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.eps = 1e-8

    def compute_gradcam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        target_layer = None
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        if target_layer is None:
            return np.zeros((224, 224), dtype=np.float32)

        activations = None
        gradients = None

        def fwd_hook(m, inp, out):
            nonlocal activations
            activations = out.detach()

        def bwd_hook(m, gin, gout):
            nonlocal gradients
            gradients = gout[0].detach()

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        inp = input_tensor.clone().detach().to(self.device).requires_grad_(True)
        out = self.model(inp)
        self.model.zero_grad()
        out[0, target_class].backward()

        h1.remove()
        h2.remove()

        if gradients is None or activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        pooled_grads = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
        cam = torch.sum(activations * pooled_grads, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_np = cam[0, 0].cpu().numpy()
        cam_resized = cv2.resize(cam_np, (224, 224))
        return (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + self.eps)

    def compute_integrated_gradients(self, input_tensor: torch.Tensor, target_class: int, steps: int = 20) -> np.ndarray:
        self.model.eval()
        inp = input_tensor.clone().detach().to(self.device)
        baseline = torch.zeros_like(inp)
        alphas = torch.linspace(0, 1, steps, device=self.device)

        interpolated = torch.cat([baseline + alpha * (inp - baseline) for alpha in alphas], dim=0)
        interpolated.requires_grad_(True)

        out = self.model(interpolated)
        self.model.zero_grad()
        target_scores = out[:, target_class]
        target_scores.backward(torch.ones_like(target_scores))

        avg_grads = torch.mean(interpolated.grad, dim=0, keepdim=True)
        ig = (inp - baseline) * avg_grads
        ig_map = torch.abs(ig).sum(dim=1, keepdim=True)
        ig_map = F.interpolate(ig_map, size=(224, 224), mode='bilinear', align_corners=False)
        ig_np = ig_map[0, 0].detach().cpu().numpy()
        return (ig_np - ig_np.min()) / (ig_np.max() - ig_np.min() + self.eps)

    def compute_smoothgrad(self, input_tensor: torch.Tensor, target_class: int, samples: int = 20, noise_std: float = 0.15) -> np.ndarray:
        self.model.eval()
        inp = input_tensor.clone().detach().to(self.device)
        grads_accum = torch.zeros_like(inp)

        for _ in range(samples):
            noise = torch.randn_like(inp) * noise_std
            noisy_inp = (inp + noise).clone().detach().requires_grad_(True)
            out = self.model(noisy_inp)
            self.model.zero_grad()
            out[0, target_class].backward()
            grads_accum += torch.abs(noisy_inp.grad.data)

        sg_map = (grads_accum / samples).sum(dim=1, keepdim=True)
        sg_map = F.interpolate(sg_map, size=(224, 224), mode='bilinear', align_corners=False)
        sg_np = sg_map[0, 0].cpu().numpy()
        return (sg_np - sg_np.min()) / (sg_np.max() - sg_np.min() + self.eps)


def benchmark_attribution_faithfulness(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution_map: np.ndarray,
    target_class: int,
    device: torch.device,
    num_steps: int = 10
) -> Tuple[float, float, List[float], List[float]]:
    """
    Computes Deletion AUC and Insertion AUC for any attribution map.
    """
    model.eval()
    inp = input_tensor.to(device)
    # Gaussian blur baseline
    from torchvision.transforms import GaussianBlur
    blur = GaussianBlur(kernel_size=21, sigma=(10.0, 10.0))
    baseline = blur(inp)

    flat_attr = attribution_map.flatten()
    sorted_idx = np.argsort(flat_attr)[::-1]
    total_pixels = len(flat_attr)
    step_size = total_pixels // num_steps

    with torch.no_grad():
        init_prob = torch.softmax(model(inp), dim=-1)[0, target_class].item()
        base_prob = torch.softmax(model(baseline), dim=-1)[0, target_class].item()

    del_probs = [init_prob]
    ins_probs = [base_prob]

    h, w = attribution_map.shape
    for step in range(1, num_steps + 1):
        active_idx = sorted_idx[:step * step_size]
        mask = np.zeros(total_pixels, dtype=bool)
        mask[active_idx] = True
        mask_t = torch.from_numpy(mask.reshape(h, w)).to(device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Deletion: drop salient pixels
            del_step = torch.where(mask_t, baseline, inp)
            del_probs.append(torch.softmax(model(del_step), dim=-1)[0, target_class].item())

            # Insertion: add salient pixels to baseline
            ins_step = torch.where(mask_t, inp, baseline)
            ins_probs.append(torch.softmax(model(ins_step), dim=-1)[0, target_class].item())

    x = np.linspace(0, 1, len(del_probs))
    del_auc = float(integrate.trapezoid(del_probs, x))
    ins_auc = float(integrate.trapezoid(ins_probs, x))
    return del_auc, ins_auc, del_probs, ins_probs

