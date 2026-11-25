"""
Post-hoc calibration, prior correction and clinically constrained decision rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Scaler(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def probs(self, logits: np.ndarray | torch.Tensor) -> np.ndarray:
        z = torch.as_tensor(logits, dtype=torch.float32)
        return torch.softmax(self.forward(z), dim=1).numpy()


class TemperatureScaling(_Scaler):
    def __init__(self) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()


class BiasCorrectedTemperature(_Scaler):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp() + self.bias


class VectorScaling(_Scaler):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.log_w = nn.Parameter(torch.zeros(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.log_w.exp() + self.bias


def fit_scaler(
    scaler: _Scaler,
    logits: np.ndarray,
    labels: np.ndarray,
    l2_bias: float = 1e-3,
    max_iter: int = 500,
) -> _Scaler:
    z = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(scaler(z), y)
        for name, p in scaler.named_parameters():
            if "bias" in name:
                loss = loss + l2_bias * (p ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return scaler


def _bins_equal_width(conf: np.ndarray, n_bins: int) -> list[np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    return [idx == b for b in range(n_bins)]


def _bins_equal_mass(conf: np.ndarray, n_bins: int) -> list[np.ndarray]:
    order = np.argsort(conf, kind="stable")
    out = []
    for chunk in np.array_split(order, n_bins):
        m = np.zeros(conf.shape[0], bool)
        m[chunk] = True
        out.append(m)
    return out


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    bins = _bins_equal_mass(conf, n_bins) if adaptive else _bins_equal_width(conf, n_bins)
    return float(sum(m.mean() * abs(correct[m].mean() - conf[m].mean()) for m in bins if m.any()))


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    k = probs.shape[1]
    vals = []
    for c in range(k):
        p = probs[:, c]
        y = (labels == c).astype(np.float64)
        bins = _bins_equal_width(p, n_bins)
        vals.append(sum(m.mean() * abs(y[m].mean() - p[m].mean()) for m in bins if m.any()))
    return float(np.mean(vals))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    rows = []
    for b, m in enumerate(_bins_equal_width(conf, n_bins)):
        if m.any():
            rows.append({"bin": b, "count": int(m.sum()), "confidence": float(conf[m].mean()),
                         "accuracy": float(correct[m].mean())})
    return rows


def calibration_report(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float((probs.argmax(1) == labels).mean()),
        "balanced_accuracy": float(np.mean([(probs.argmax(1)[labels == c] == c).mean()
                                            for c in np.unique(labels)])),
        "ece_15": ece(probs, labels, 15),
        "adaptive_ece_15": ece(probs, labels, 15, adaptive=True),
        "classwise_ece_15": classwise_ece(probs, labels, 15),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
    }


def bayes_decision(probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    return (probs @ cost).argmin(1)


def melanoma_cost_matrix(num_classes: int, mel_index: int, miss_cost: float = 5.0) -> np.ndarray:
    c = 1.0 - np.eye(num_classes)
    c[mel_index, :] *= miss_cost
    c[mel_index, mel_index] = 0.0
    return c


@dataclass
class SensitivityThreshold:
    class_index: int
    threshold: float
    achieved_sensitivity: float

    def apply(self, probs: np.ndarray) -> np.ndarray:
        pred = probs.argmax(1)
        pred[probs[:, self.class_index] >= self.threshold] = self.class_index
        return pred


def fit_sensitivity_threshold(
    probs: np.ndarray, labels: np.ndarray, class_index: int, target_sensitivity: float = 0.85
) -> SensitivityThreshold:
    pos = np.sort(probs[labels == class_index, class_index])
    if pos.size == 0:
        raise ValueError("no positives of the requested class in calibration split")
    k = int(np.floor((1.0 - target_sensitivity) * pos.size))
    thr = float(pos[min(k, pos.size - 1)])
    achieved = float((probs[labels == class_index, class_index] >= thr).mean())
    return SensitivityThreshold(class_index, thr, achieved)


def bootstrap_ci(
    metric: Callable[[np.ndarray], float],
    n: int,
    groups: Optional[Sequence] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = metric(np.arange(n))
    if groups is None:
        draws = [metric(rng.integers(0, n, n)) for _ in range(n_boot)]
    else:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            draws.append(metric(np.concatenate([members[g] for g in pick])))
    lo, hi = np.nanpercentile(np.asarray(draws, float), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
