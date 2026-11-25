"""
Evidence-Gated Hybrid Attention Fusion (EG-HAF).

    M = (N + floor) * g,   g = 1 - kappa + kappa * P,   kappa = kappa_max * q * c_y
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-6

DEFAULT_CLASS_APPLICABILITY: dict[str, float] = {
    "mel": 1.0, "nv": 1.0, "bkl": 0.5, "akiec": 0.25, "bcc": 0.25, "df": 0.25, "vasc": 0.0,
}


def leakage_bound(sigma_a: float, sigma_s: float, sigma_b: float | None = None) -> float:
    """Deterministic bound on |r(log M, log M') - r(log N_eps, log N'_eps)| (Proposition 2)."""
    sigma_b = sigma_a if sigma_b is None else sigma_b
    if sigma_s >= min(sigma_a, sigma_b):
        return float("inf")
    d_min = (sigma_a - sigma_s) * (sigma_b - sigma_s)
    d_max = (sigma_a + sigma_s) * (sigma_b + sigma_s)
    leak = sigma_s * (sigma_a + sigma_b + sigma_s) / d_min
    scale = max(sigma_a * sigma_b / d_min - 1.0, 1.0 - sigma_a * sigma_b / d_max)
    return float(leak + scale)


def kappa_for_leakage(delta: float, sigma_a: float) -> float:
    """Largest kappa with worst-case leakage 4 rho / (1 - rho)^2 <= delta (symmetric case)."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    b = 2.0 * delta + 4.0
    rho = (b - math.sqrt(b * b - 4.0 * delta * delta)) / (2.0 * delta)
    sigma_s_max = rho * sigma_a
    return float(1.0 - math.exp(-2.0 * sigma_s_max))


def convex_leakage_rho(lmbda: float, sigma_n: float, sigma_p: float) -> float:
    """rho for the legacy additive fusion; explains the monotone full-randomisation curve of Table 4."""
    if lmbda >= 1.0:
        return float("inf")
    return lmbda * sigma_p / ((1.0 - lmbda) * sigma_n + EPS)


@dataclass
class FusionOutput:
    fused: np.ndarray
    gate: np.ndarray
    kappa: float


class EvidenceGatedFusion:
    def __init__(
        self,
        kappa_max: float = 0.5,
        class_applicability: dict[str, float] | None = None,
        use_quality: bool = True,
        floor: float = 0.01,
    ) -> None:
        if not 0.0 <= kappa_max < 1.0:
            raise ValueError("kappa_max must be in [0, 1)")
        self.kappa_max = kappa_max
        self.floor = floor
        self.class_applicability = class_applicability or DEFAULT_CLASS_APPLICABILITY
        self.use_quality = use_quality

    def effective_kappa(self, class_name: str | None, quality: float | None) -> float:
        c = 1.0 if class_name is None else float(self.class_applicability.get(class_name, 0.0))
        q = 1.0 if (quality is None or not self.use_quality) else float(np.clip(quality, 0.0, 1.0))
        return self.kappa_max * c * q

    def fuse(
        self,
        neural: np.ndarray,
        prior: np.ndarray,
        class_name: str | None = None,
        quality: float | None = None,
        renormalise: bool = True,
    ) -> FusionOutput:
        if neural.shape != prior.shape:
            raise ValueError(f"shape mismatch {neural.shape} vs {prior.shape}")
        n = np.clip(neural.astype(np.float32), 0.0, 1.0)
        p = np.clip(prior.astype(np.float32), 0.0, 1.0)
        kappa = self.effective_kappa(class_name, quality)
        gate = (1.0 - kappa) + kappa * p
        fused = (n + self.floor) * gate
        if renormalise:
            fused = fused / (fused.max() + EPS)
        return FusionOutput(fused.astype(np.float32), gate.astype(np.float32), kappa)


def log_domain_stats(neural: np.ndarray, gate: np.ndarray, floor: float = 0.01) -> tuple[float, float]:
    return float(np.log(neural + floor).std()), float(np.log(gate).std())
