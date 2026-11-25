"""
Statistical and metric upgrades for the XAI benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import average_precision_score


@dataclass
class TOSTResult:
    mean_diff: float
    ci90_low: float
    ci90_high: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool


def tost_paired(x: np.ndarray, y: np.ndarray, margin: float, alpha: float = 0.05) -> TOSTResult:
    d = np.asarray(x, float) - np.asarray(y, float)
    n = d.size
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se < 1e-12:
        eq = abs(m) < margin
        return TOSTResult(float(m), float(m), float(m), 0.0 if eq else 1.0, 0.0 if eq else 1.0, 0.0 if eq else 1.0, eq)
    t_lo = (m + margin) / se
    t_hi = (m - margin) / se
    p_lo = 1.0 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    q = stats.t.ppf(1 - alpha, n - 1)
    p = max(p_lo, p_hi)
    return TOSTResult(float(m), float(m - q * se), float(m + q * se), float(p_lo), float(p_hi), float(p), bool(p < alpha))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def area_matched_iou(attr: np.ndarray, mask: np.ndarray) -> float:
    k = int((mask > 0).sum())
    if k == 0:
        return float("nan")
    flat = attr.ravel()
    jitter = np.random.default_rng(0).random(flat.size) * 1e-9
    top = np.zeros(flat.size, bool)
    top[np.argpartition(-(flat + jitter), k - 1)[:k]] = True
    m = mask.ravel() > 0
    return float((top & m).sum() / (top | m).sum())


def saliency_auprc(attr: np.ndarray, mask: np.ndarray) -> float:
    m = (mask.ravel() > 0).astype(int)
    if m.min() == m.max():
        return float("nan")
    return float(average_precision_score(m, attr.ravel()))


def pointing_hit_tie_aware(attr: np.ndarray, mask: np.ndarray, rtol: float = 1e-6) -> float:
    mx = float(attr.max())
    peaks = attr >= mx - rtol * max(abs(mx), 1.0)
    return float((mask[peaks] > 0).mean())


def decentred_crop(
    rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator, size_frac: float = 0.7,
    out: int = 224, tries: int = 200,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    if rgb.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    h, w = mask.shape
    s = int(size_frac * min(h, w))
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if y1 - y0 >= s or x1 - x0 >= s:
        return None
    cy, cx = float(ys.mean()), float(xs.mean())
    lo_top, hi_top = max(0, y1 - s + 1), min(y0, h - s)
    lo_left, hi_left = max(0, x1 - s + 1), min(x0, w - s)
    if hi_top < lo_top or hi_left < lo_left:
        return None
    best = None
    for _ in range(tries):
        top = int(rng.integers(lo_top, hi_top + 1))
        left = int(rng.integers(lo_left, hi_left + 1))
        off = float(np.hypot(cy - (top + s / 2), cx - (left + s / 2)) / (s / 2))
        if best is None or off > best[2]:
            best = (top, left, off)
    if best is None:
        return None
    top, left, off = best
    crop = cv2.resize(rgb[top:top + s, left:left + s], (out, out), interpolation=cv2.INTER_AREA)
    mcrop = cv2.resize(mask[top:top + s, left:left + s].astype(np.uint8), (out, out), interpolation=cv2.INTER_NEAREST)
    if mcrop.sum() < 10:
        return None
    return crop, mcrop, float(off)


Baseline = Literal["blur", "mean", "noise"]


def _baseline(x: torch.Tensor, kind: Baseline, seed: int = 0) -> torch.Tensor:
    if kind == "blur":
        return F.avg_pool2d(F.pad(x, (10, 10, 10, 10), mode="reflect"), 21, stride=1)
    if kind == "mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    g = torch.Generator(device=x.device).manual_seed(seed)
    return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * x.std() + x.mean()


@torch.no_grad()
def patch_flipping(
    model: nn.Module, x: torch.Tensor, maps: dict[str, np.ndarray], target: int,
    patch: int = 16, steps: int = 14, baseline: Baseline = "blur",
) -> dict[str, dict[str, float]]:
    model.eval()
    _, _, h, w = x.shape
    gh, gw = h // patch, w // patch
    base = _baseline(x, baseline)
    batch, order = [], []
    for name, m in maps.items():
        pm = torch.as_tensor(m, dtype=torch.float32)[None, None]
        pscore = F.avg_pool2d(pm, patch).flatten()
        rank = torch.argsort(pscore, descending=True)
        for direction in ("morf", "lerf"):
            idx = rank if direction == "morf" else rank.flip(0)
            for st in range(steps + 1):
                k = round(st * gh * gw / steps)
                keep = torch.ones(gh * gw)
                keep[idx[:k]] = 0.0
                mask = F.interpolate(keep.view(1, 1, gh, gw), size=(h, w), mode="nearest").to(x.device)
                batch.append(x * mask + base * (1 - mask))
            order.append((name, direction))
    xb = torch.cat(batch, 0)
    dev = x.device
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = torch.cat([model(c) for c in xb.split(64)], 0).float()
    own = F.one_hot(torch.tensor([target], device=dev), logits.shape[1]).bool().expand_as(logits)
    score = logits[:, target] - torch.logsumexp(logits.masked_fill(own, float("-inf")), 1)
    score = score.cpu().numpy().reshape(len(order), steps + 1)
    out: dict[str, dict[str, float]] = {}
    xs = np.linspace(0, 1, steps + 1)
    for (name, direction), curve in zip(order, score):
        out.setdefault(name, {})[f"{direction}_auc"] = float(np.trapezoid(curve - curve[0], xs))
    for name in out:
        out[name]["srg"] = out[name]["lerf_auc"] - out[name]["morf_auc"]
    return out
