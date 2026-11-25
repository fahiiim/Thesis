"""
Class-discriminative token attribution for torchvision Swin Transformer V2 (Swin-TCAM).

Replaces the legacy |dy/dA| "gradient coherence" map (src/xai/gcw.py), whose spatial
structure at Stage 4 is dominated by the LayerNorm Jacobian factor 1/sigma_p of each token
(a forward-only statistic). That is why the legacy map survives classifier-head
randomisation (r ~ 0.99) and is nearly identical for target and distractor classes.

Design:
  * Stage 4: Grad-CAM on the *post-LayerNorm* tokens (output of ``backbone.norm``). Because
    torchvision pools tokens with a plain average after this norm, the gradient is identical
    for every token, and sum_p cam_raw(p) = <grad_z y, z> exactly (first-order completeness
    w.r.t. the pooled feature z).
  * Stage 3: LayerCAM (Jiang et al., TIP 2021) on ``backbone.features[5]``:
    ReLU( sum_k ReLU(g_k(p)) * A_k(p) ).
  * Target: log-odds  s_c = z_c - logsumexp_{k != c} z_k  (default). Its gradient is
    dz_c - sum_{k!=c} p~_k dz_k, i.e. class-contrastive by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TargetMode = Literal["logit", "logodds", "logprob"]
StageFusion = Literal["mean", "geometric"]


def class_score(logits: torch.Tensor, target: torch.Tensor, mode: TargetMode) -> torch.Tensor:
    """Per-sample scalar score to differentiate. logits: (B, K); target: (B,) long."""
    logits32 = logits.float()
    selected = logits32.gather(1, target[:, None]).squeeze(1)
    if mode == "logit":
        return selected
    if mode == "logprob":
        return selected - torch.logsumexp(logits32, dim=1)
    if mode == "logodds":
        own = F.one_hot(target, num_classes=logits32.shape[1]).bool()
        others = logits32.masked_fill(own, float("-inf"))
        return selected - torch.logsumexp(others, dim=1)
    raise ValueError(f"Unknown target mode: {mode}")


class _Tap:
    """Captures a module's forward output and the gradient flowing into it (tensor hook)."""

    def __init__(self, module: nn.Module) -> None:
        self.act: Optional[torch.Tensor] = None
        self.grad: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._forward)

    def _forward(self, _m: nn.Module, _i: tuple, out: torch.Tensor) -> None:
        self.act = out.detach()
        if out.requires_grad:
            out.register_hook(self._backward)

    def _backward(self, grad: torch.Tensor) -> None:
        self.grad = grad.detach()

    def remove(self) -> None:
        self._handle.remove()


def _minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max over the last two dims. x: (B, H, W)."""
    flat = x.flatten(1)
    lo = flat.min(dim=1).values[:, None, None]
    hi = flat.max(dim=1).values[:, None, None]
    return (x - lo) / (hi - lo + eps)


@dataclass
class TokenCAMResult:
    fused: torch.Tensor      # (B, H, W) in [0, 1]
    stage3: torch.Tensor     # (B, H, W) in [0, 1]
    stage4: torch.Tensor     # (B, H, W) in [0, 1]
    target: torch.Tensor     # (B,) long
    probs: torch.Tensor      # (B, K) float32


class SwinTokenCAM:
    """Class-discriminative hierarchical attribution for ``EnhancedSwinTransformer``.

    Expects ``model.backbone`` to be a torchvision ``SwinTransformer`` (``features``, ``norm``).
    """

    def __init__(
        self,
        model: nn.Module,
        stage3_index: int = 5,
        beta: float = 0.5,
        target_mode: TargetMode = "logodds",
        stage_fusion: StageFusion = "mean",
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.model = model
        self.stage3_module = model.backbone.features[stage3_index]
        self.norm_module = model.backbone.norm
        self.beta = beta
        self.target_mode = target_mode
        self.stage_fusion = stage_fusion

    def __call__(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_mode: Optional[TargetMode] = None,
    ) -> TokenCAMResult:
        mode = target_mode or self.target_mode
        self.model.eval()
        tap3, tap4 = _Tap(self.stage3_module), _Tap(self.norm_module)
        try:
            with torch.enable_grad():
                inp = x.detach().requires_grad_(True)
                logits = self.model(inp)
                if target is None:
                    target = logits.argmax(dim=1)
                target = target.to(logits.device).long()
                score = class_score(logits, target, mode)
                self.model.zero_grad(set_to_none=True)
                score.sum().backward()
        finally:
            tap3.remove()
            tap4.remove()
        if tap3.grad is None or tap4.grad is None or tap3.act is None or tap4.act is None:
            raise RuntimeError("Hooks did not capture activations/gradients; check model layout.")

        h, w = x.shape[-2:]
        a4, g4 = tap4.act.float(), tap4.grad.float()
        alpha4 = g4.mean(dim=(1, 2), keepdim=True)
        cam4 = F.relu((a4 * alpha4).sum(dim=-1))

        a3, g3 = tap3.act.float(), tap3.grad.float()
        cam3 = F.relu((F.relu(g3) * a3).sum(dim=-1))

        up4 = _minmax(F.interpolate(cam4[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0])
        up3 = _minmax(F.interpolate(cam3[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0])

        if self.stage_fusion == "mean":
            fused = (1.0 - self.beta) * up4 + self.beta * up3
        else:
            fused = (up4 + 1e-6) ** (1.0 - self.beta) * (up3 + 1e-6) ** self.beta
        probs = torch.softmax(logits.detach().float(), dim=1)
        return TokenCAMResult(_minmax(fused), up3, up4, target.detach(), probs)


def token_layernorm_inverse_scale(model: nn.Module, x: torch.Tensor, stage_index: int = 7) -> torch.Tensor:
    """Diagnostic: per-token 1/sigma_p of the tokens entering ``backbone.norm`` (B, H_s, W_s)."""
    store: dict[str, torch.Tensor] = {}
    handle = model.backbone.features[stage_index].register_forward_hook(
        lambda _m, _i, o: store.__setitem__("a", o.detach().float())
    )
    try:
        with torch.no_grad():
            model.eval()
            model(x)
    finally:
        handle.remove()
    eps = float(model.backbone.norm.eps)
    var = store["a"].var(dim=-1, unbiased=False)
    return 1.0 / torch.sqrt(var + eps)


@torch.no_grad()
def activation_only_map(model: nn.Module, x: torch.Tensor, stage3_index: int = 5, beta: float = 0.5) -> torch.Tensor:
    """Reviewer-requested ablation: same pipeline as SwinTokenCAM with all gradients replaced by 1."""
    store: dict[str, torch.Tensor] = {}
    h3 = model.backbone.features[stage3_index].register_forward_hook(lambda _m, _i, o: store.__setitem__("a3", o.float()))
    h4 = model.backbone.norm.register_forward_hook(lambda _m, _i, o: store.__setitem__("a4", o.float()))
    try:
        model.eval()
        model(x)
    finally:
        h3.remove()
        h4.remove()
    h, w = x.shape[-2:]
    cam4 = F.relu(store["a4"].mean(dim=-1))
    cam3 = F.relu(store["a3"].sum(dim=-1))
    up4 = _minmax(F.interpolate(cam4[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0])
    up3 = _minmax(F.interpolate(cam3[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0])
    return _minmax((1.0 - beta) * up4 + beta * up3)
