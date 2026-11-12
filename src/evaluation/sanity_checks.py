"""
Adebayo Parameter Randomization Sanity Checks (Adebayo et al., NeurIPS 2018).
Evaluates whether Gradient Coherence Weighting is genuinely dependent on learned model
parameters or simply acts as an edge detector insensitive to weights.
"""

from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr


class ModelRandomizationSanityCheck:
    """
    Implements Cascading Layer Randomization:
    Randomizes model weights progressively from classification head down to early stages,
    and measures degradation of attribution correlation with the original explanation.
    """
    def __init__(self, model: nn.Module, gcw_engine):
        self.original_model = model
        self.gcw_engine = gcw_engine

    def _randomize_layer(self, module: nn.Module) -> None:
        """Re-initializes weights using PyTorch default initializers."""
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def run_cascading_check(
        self,
        image_tensor: torch.Tensor,
        layer_names_top_down: List[str],
        target_class: int
    ) -> Dict[str, float]:
        """
        Runs cascading randomization across designated layers.

        Args:
            image_tensor: Normalized (1, 3, H, W) tensor.
            layer_names_top_down: List of attribute names to randomize in order
                                  (e.g., ['head.fc2', 'head.fc1', 'backbone.layers.3']).
            target_class: Target class index.

        Returns:
            Dictionary mapping layer stage to Spearman rank correlation with original attribution.
        """
        # Baseline: Attribution from the trained model
        if hasattr(self.gcw_engine, "compute_hierarchical_gcw"):
            orig_map, _, _ = self.gcw_engine.compute_hierarchical_gcw(image_tensor, target_class=target_class)
        else:
            orig_map, _, _ = self.gcw_engine.compute_input_gcw(image_tensor, target_class=target_class)
        orig_flat = orig_map.flatten()

        # Deepcopy model to preserve original trained state
        perturbed_model = copy.deepcopy(self.original_model)
        perturbed_engine = copy.copy(self.gcw_engine)
        perturbed_engine.model = perturbed_model

        correlations = {"original": 1.0}

        for layer_name in layer_names_top_down:
            # Find and randomize the layer
            parts = layer_name.split(".")
            curr = perturbed_model
            for p in parts:
                if hasattr(curr, p):
                    curr = getattr(curr, p)
                elif hasattr(curr, "__getitem__") and p.isdigit():
                    curr = curr[int(p)]

            self._randomize_layer(curr)

            # Compute new attribution map on perturbed model
            if hasattr(perturbed_engine, "compute_hierarchical_gcw"):
                pert_map, _, _ = perturbed_engine.compute_hierarchical_gcw(image_tensor, target_class=target_class)
            else:
                pert_map, _, _ = perturbed_engine.compute_input_gcw(image_tensor, target_class=target_class)
            pert_flat = pert_map.flatten()

            corr, _ = spearmanr(orig_flat, pert_flat)
            correlations[layer_name] = float(corr) if not np.isnan(corr) else 0.0

        return correlations

