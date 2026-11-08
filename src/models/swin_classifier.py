"""
Swin Transformer V2 Backbone with Custom Classification Head.
Implements the exact EnhancedSwinTransformer architecture loaded from swin_vit_xai_model.pth:
torchvision.models.swin_v2_b -> num_features (1024)
Head: Dropout(0.2) -> Linear(1024, 256) -> ReLU -> BatchNorm1d(256) -> Dropout(0.1) -> Linear(256, 7)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import swin_v2_b


class EnhancedSwinTransformer(nn.Module):
    """
    Swin Transformer V2 for multi-class dermatoscopic diagnostic classification.
    Supports forward and backward hooks across internal stages for GCW analysis.
    """
    def __init__(
        self,
        num_classes: int = 7,
        dropout_rate: float = 0.2,
        use_batchnorm: bool = True,
        checkpoint_path: Optional[str] = None
    ):
        super().__init__()
        self.backbone = swin_v2_b(weights=None)
        num_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        if use_batchnorm:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_features // 4),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features // 4),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(num_features // 4, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_features // 4),
                nn.ReLU(inplace=True),
                nn.LayerNorm(num_features // 4),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(num_features // 4, num_classes)
            )

        self.hook_activations: Dict[str, torch.Tensor] = {}
        self.hook_gradients: Dict[str, torch.Tensor] = {}

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def register_stage_hooks(self) -> None:
        """Hooks into the main hierarchical Swin Transformer feature stages."""
        def _get_fwd_hook(name: str):
            def hook(module, input, output):
                self.hook_activations[name] = output
            return hook

        def _get_bwd_hook(name: str):
            def hook(module, grad_in, grad_out):
                self.hook_gradients[name] = grad_out[0]
            return hook

        for idx, stage in enumerate(self.backbone.features):
            stage_name = f"stage_{idx + 1}"
            stage.register_forward_hook(_get_fwd_hook(stage_name))
            stage.register_full_backward_hook(_get_bwd_hook(stage_name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        
        # Clean potential prefixes
        cleaned_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            cleaned_dict[name] = v

        missing, unexpected = self.load_state_dict(cleaned_dict, strict=True)
        print(f"[Model] Checkpoint perfectly loaded from '{path}'. (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
