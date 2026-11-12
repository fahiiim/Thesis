"""
Comprehensive Comparative XAI Benchmarking Engine.
Evaluates CG-HAF against primary clinical baselines:
1. Swin Stage-4 Grad-CAM (Selvaraju et al., ICCV 2017)
2. Integrated Gradients (Sundararajan et al., ICML 2017)
3. Batched SmoothGrad (Smilkov et al., 2017)
4. Standalone GCW & Standalone CSM

All perturbation evaluations and multi-sample baselines are vectorized and GPU-batched
with AMP for peak RTX 5070 Tensor Core utilization.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


class BaselineXAI:
    """GPU-optimized Grad-CAM (Swin Stage 4), Integrated Gradients, and SmoothGrad."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.eps = 1e-8

    @staticmethod
    def _to_bchw(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            if tensor.shape[-1] >= tensor.shape[1] and tensor.shape[1] == tensor.shape[2]:
                return tensor.permute(0, 3, 1, 2).contiguous()
            return tensor
        elif tensor.ndim == 3:
            b, n, c = tensor.shape
            g = int(np.sqrt(n))
            return tensor.view(b, g, g, c).permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unsupported feature shape: {tensor.shape}")

    def compute_gradcam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Computes Grad-CAM on the semantic final stage (Stage 4: backbone.features[-1])
        of Swin Transformer V2 rather than Stage-0 patch embedding.
        """
        self.model.eval()
        target_layer = self.model.backbone.features[-1]

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

        try:
            inp = input_tensor.clone().detach().to(self.device).requires_grad_(True)
            out = self.model(inp)
            self.model.zero_grad()
            out[0, target_class].backward()
        finally:
            h1.remove()
            h2.remove()

        if gradients is None or activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        act_bchw = self._to_bchw(activations)   # (1, 1024, 7, 7)
        grad_bchw = self._to_bchw(gradients)    # (1, 1024, 7, 7)

        weights = torch.mean(grad_bchw, dim=(2, 3), keepdim=True)
        cam = torch.sum(act_bchw * weights, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].cpu().numpy()
        return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + self.eps)

    def compute_integrated_gradients(self, input_tensor: torch.Tensor, target_class: int, steps: int = 20) -> np.ndarray:
        self.model.eval()
        inp = input_tensor.clone().detach().to(self.device)
        baseline = torch.zeros_like(inp)
        alphas = torch.linspace(0, 1, steps, device=self.device).view(-1, 1, 1, 1)

        interpolated = (baseline + alphas * (inp - baseline)).detach().requires_grad_(True)
        out = self.model(interpolated)
        self.model.zero_grad()
        target_scores = out[:, target_class]
        target_scores.backward(torch.ones_like(target_scores))

        avg_grads = torch.mean(interpolated.grad, dim=0, keepdim=True)
        ig = (inp - baseline) * avg_grads
        ig_map = torch.abs(ig).sum(dim=1, keepdim=True)
        ig_np = ig_map[0, 0].detach().cpu().numpy()
        return (ig_np - ig_np.min()) / (ig_np.max() - ig_np.min() + self.eps)

    def compute_smoothgrad(self, input_tensor: torch.Tensor, target_class: int, samples: int = 20, noise_std: float = 0.15) -> np.ndarray:
        """
        Fully GPU-batched SmoothGrad: computes all Monte Carlo noisy perturbations
        in a single parallel forward/backward pass.
        """
        self.model.eval()
        inp = input_tensor.clone().detach().to(self.device)
        batch_inp = inp.expand(samples, -1, -1, -1)
        noise = torch.randn_like(batch_inp) * noise_std
        noisy_batch = (batch_inp + noise).detach().requires_grad_(True)

        out = self.model(noisy_batch)
        self.model.zero_grad()
        scores = out[:, target_class]
        scores.backward(torch.ones_like(scores))

        avg_abs_grad = torch.abs(noisy_batch.grad).mean(dim=0, keepdim=True)
        sg_map = avg_abs_grad.sum(dim=1, keepdim=True)
        sg_np = sg_map[0, 0].cpu().numpy()
        return (sg_np - sg_np.min()) / (sg_np.max() - sg_np.min() + self.eps)


_BLUR_CACHE: Dict[torch.device, GaussianBlur] = {}


def benchmark_attribution_faithfulness(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attribution_map: np.ndarray,
    target_class: int,
    device: torch.device,
    num_steps: int = 10
) -> Tuple[float, float, List[float], List[float]]:
    """
    GPU-batched Deletion AUC and Insertion AUC evaluator.
    Processes all 2*(num_steps + 1) perturbed samples in ONE single CUDA batch pass.
    """
    model.eval()
    inp = input_tensor.to(device)
    if device not in _BLUR_CACHE:
        _BLUR_CACHE[device] = GaussianBlur(kernel_size=21, sigma=(10.0, 10.0))
    baseline = _BLUR_CACHE[device](inp)

    _, _, h, w = inp.shape
    flat_attr = torch.from_numpy(attribution_map.reshape(-1)).to(device)
    sorted_idx = torch.argsort(flat_attr, descending=True)
    total_pixels = flat_attr.numel()

    n_pts = num_steps + 1
    del_masks = torch.zeros((n_pts, 1, h, w), dtype=torch.bool, device=device)
    for step in range(1, n_pts):
        k = total_pixels if step == num_steps else (total_pixels * step) // num_steps
        del_masks[step].view(-1)[sorted_idx[:k]] = True

    inp_rep = inp.expand(n_pts, -1, -1, -1)
    base_rep = baseline.expand(n_pts, -1, -1, -1)
    full_batch = torch.cat([
        torch.where(del_masks, base_rep, inp_rep),
        torch.where(del_masks, inp_rep, base_rep)
    ], dim=0)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        probs = torch.softmax(model(full_batch).float(), dim=-1)[:, target_class]

    del_probs = probs[:n_pts].cpu().numpy().tolist()
    ins_probs = probs[n_pts:].cpu().numpy().tolist()

    x = np.linspace(0, 1, n_pts)
    del_auc = float(integrate.trapezoid(del_probs, x))
    ins_auc = float(integrate.trapezoid(ins_probs, x))
    return del_auc, ins_auc, del_probs, ins_probs


def benchmark_multi_maps_batched(
    model: nn.Module,
    input_tensor: torch.Tensor,
    maps_dict: Dict[str, np.ndarray],
    target_class: int,
    device: torch.device,
    num_steps: int = 10
) -> Dict[str, Tuple[float, float, List[float], List[float]]]:
    """
    Maximum GPU saturation helper: stacks ALL perturbation steps across ALL XAI methods
    for a given image into one mega-batch (~132 images on RTX 5070 Tensor Cores).
    """
    model.eval()
    inp = input_tensor.to(device)
    if device not in _BLUR_CACHE:
        _BLUR_CACHE[device] = GaussianBlur(kernel_size=21, sigma=(10.0, 10.0))
    baseline = _BLUR_CACHE[device](inp)

    _, _, h, w = inp.shape
    total_pixels = h * w
    n_pts = num_steps + 1

    all_batches = []
    method_order = list(maps_dict.keys())
    for m_name in method_order:
        flat_attr = torch.from_numpy(maps_dict[m_name].reshape(-1)).to(device)
        sorted_idx = torch.argsort(flat_attr, descending=True)
        del_masks = torch.zeros((n_pts, 1, h, w), dtype=torch.bool, device=device)
        for step in range(1, n_pts):
            k = total_pixels if step == num_steps else (total_pixels * step) // num_steps
            del_masks[step].view(-1)[sorted_idx[:k]] = True

        inp_rep = inp.expand(n_pts, -1, -1, -1)
        base_rep = baseline.expand(n_pts, -1, -1, -1)
        all_batches.append(torch.where(del_masks, base_rep, inp_rep))
        all_batches.append(torch.where(del_masks, inp_rep, base_rep))

    mega_batch = torch.cat(all_batches, dim=0)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        all_probs = torch.softmax(model(mega_batch).float(), dim=-1)[:, target_class].cpu().numpy()

    results = {}
    x = np.linspace(0, 1, n_pts)
    stride = 2 * n_pts
    for i, m_name in enumerate(method_order):
        seg = all_probs[i * stride:(i + 1) * stride]
        del_probs = seg[:n_pts].tolist()
        ins_probs = seg[n_pts:].tolist()
        del_auc = float(integrate.trapezoid(del_probs, x))
        ins_auc = float(integrate.trapezoid(ins_probs, x))
        results[m_name] = (del_auc, ins_auc, del_probs, ins_probs)
    return results
