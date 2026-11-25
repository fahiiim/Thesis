"""
Test-time domain mitigation for zero-shot transfer (BCN_20000, ISIC-archive cohorts) without retraining.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def fov_bbox(rgb: np.ndarray, dark_threshold: int = 25, min_frac: float = 0.3) -> tuple[int, int, int, int]:
    """Bounding box (y0, y1, x0, x1) of non-vignette pixels; full image if no vignette."""
    gray = rgb.mean(axis=2)
    rows = np.nonzero((gray > dark_threshold).mean(axis=1) > 0.05)[0]
    cols = np.nonzero((gray > dark_threshold).mean(axis=0) > 0.05)[0]
    h, w = gray.shape
    if rows.size == 0 or cols.size == 0:
        return 0, h, 0, w
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)
    if (y1 - y0) * (x1 - x0) < min_frac * h * w:
        return 0, h, 0, w
    cy, cx, r = (y0 + y1) / 2, (x0 + x1) / 2, min(y1 - y0, x1 - x0) / 2
    if (gray[[y0, y0, y1 - 1, y1 - 1], [x0, x1 - 1, x0, x1 - 1]] < dark_threshold).all():
        r = r / math.sqrt(2.0)
    return int(max(cy - r, 0)), int(min(cy + r, h)), int(max(cx - r, 0)), int(min(cx + r, w))


def parity_resize(rgb: np.ndarray, size: int = 224, crop_fov: bool = True) -> np.ndarray:
    """FOV crop -> resize shorter side -> centre crop."""
    if crop_fov:
        y0, y1, x0, x1 = fov_bbox(rgb)
        rgb = rgb[y0:y1, x0:x1]
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    rgb = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return rgb[top:top + size, left:left + size]


def masked_shades_of_gray(
    x: torch.Tensor, p: float = 6.0, gamma: float = 2.2, dark: float = 0.08, normalise: bool = True
) -> torch.Tensor:
    """x: (B, 3, H, W) in [0, 1]. Illuminant from FOV pixels only (luminance > ``dark``)."""
    lin = x.clamp(1e-6, 1.0).pow(gamma)
    fov = (x.mean(dim=1, keepdim=True) > dark).float()
    denom = fov.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    e = ((lin.pow(p) * fov).sum(dim=(2, 3), keepdim=True) / denom).pow(1.0 / p)
    e = e / (e.norm(dim=1, keepdim=True) + 1e-8) * math.sqrt(3.0)
    out = (lin / (e + 1e-8)).clamp(0.0, 1.0).pow(1.0 / gamma)
    out = out * fov + x * (1.0 - fov)
    if normalise:
        out = (out - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    return out


@torch.no_grad()
def adabn_head(model: nn.Module, batches: Iterable[torch.Tensor], device: torch.device) -> nn.Module:
    """Returns a copy whose classifier BatchNorm layers use target-domain statistics."""
    adapted = copy.deepcopy(model).eval()
    bns = [m for m in adapted.classifier.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for x in batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            adapted(x.to(device, non_blocking=True))
    for bn in bns:
        bn.eval()
    return adapted


def em_label_shift(
    probs: np.ndarray, source_prior: np.ndarray, max_iter: int = 200, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Saerens-Latinne-Decaestecker EM. ``probs`` must be calibrated w.r.t. ``source_prior``."""
    src = np.clip(source_prior, 1e-8, None)
    tgt = src.copy()
    for _ in range(max_iter):
        w = probs * (tgt / src)[None]
        post = w / w.sum(1, keepdims=True)
        new = post.mean(0)
        if np.abs(new - tgt).max() < tol:
            tgt = new
            break
        tgt = new
    w = probs * (tgt / src)[None]
    return w / w.sum(1, keepdims=True), tgt


@dataclass
class EntropyTTAConfig:
    lr: float = 1e-4
    momentum: float = 0.9
    reliable_entropy_frac: float = 0.4
    prior_weight: float = 1.0
    reset_entropy_floor: float = 0.05
    steps_per_batch: int = 1
    include_head_bn: bool = True


def layernorm_parameters(model: nn.Module, include_head_bn: bool = True) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm) or (include_head_bn and isinstance(m, nn.BatchNorm1d)):
            params += [p for p in (m.weight, m.bias) if p is not None]
    return params


class PriorAwareEntropyTTA:
    """Online adaptation of normalisation affine parameters; model stays in eval mode."""

    def __init__(self, model: nn.Module, target_prior: Optional[np.ndarray], device: torch.device,
                 config: EntropyTTAConfig | None = None) -> None:
        self.cfg = config or EntropyTTAConfig()
        self.device = device
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = layernorm_parameters(self.model, self.cfg.include_head_bn)
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.SGD(self.params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self.prior = None if target_prior is None else torch.as_tensor(target_prior, dtype=torch.float32, device=device)
        self._state = copy.deepcopy(self.model.state_dict())
        self._opt_state = copy.deepcopy(self.opt.state_dict())
        self._ema_entropy: Optional[float] = None
        self.resets = 0

    def reset(self) -> None:
        self.model.load_state_dict(self._state)
        self.opt.load_state_dict(self._opt_state)
        self._ema_entropy = None
        self.resets += 1

    def step(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        for _ in range(self.cfg.steps_per_batch):
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = self.model(x)
            logp = F.log_softmax(logits.float(), dim=1)
            p = logp.exp()
            k = p.shape[1]
            ent = -(p * logp).sum(1)
            reliable = ent < self.cfg.reliable_entropy_frac * math.log(k)
            if reliable.sum() == 0:
                break
            loss = ent[reliable].mean()
            if self.prior is not None and self.cfg.prior_weight > 0:
                mean_p = p.mean(0).clamp_min(1e-8)
                loss = loss + self.cfg.prior_weight * (mean_p * (mean_p.log() - self.prior.clamp_min(1e-8).log())).sum()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            e = float(ent.mean().detach())
            self._ema_entropy = e if self._ema_entropy is None else 0.9 * self._ema_entropy + 0.1 * e
            if self._ema_entropy < self.cfg.reset_entropy_floor:
                self.reset()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                             enabled=self.device.type == "cuda"):
            return self.model(x).float()


def dihedral_views(x: torch.Tensor) -> Iterator[torch.Tensor]:
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        yield r
        yield torch.flip(r, dims=(3,))


@torch.no_grad()
def dihedral_logits(model: nn.Module, x: torch.Tensor, device: torch.device, scaler: Optional[nn.Module] = None) -> torch.Tensor:
    """Mean of log-probabilities over the 8 dihedral views (optional logit scaler applied per view)."""
    x = x.to(device, non_blocking=True)
    acc = None
    for v in dihedral_views(x):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model(v).float()
        if scaler is not None:
            z = scaler(z.cpu()).to(device)
        lp = F.log_softmax(z, dim=1)
        acc = lp if acc is None else acc + lp
    return acc / 8.0
