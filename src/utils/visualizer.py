"""
Publication-Quality Visualization Suite for Dermatological Explainable AI.
Generates camera-ready figures (300+ DPI, Lancet/IEEE formatting guidelines)
for ABCD component decomposition, GCW attribution, and baseline comparisons.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


class PublicationVisualizer:
    """
    Renders high-resolution figures matching journal specifications:
    - Typography: Clean sans-serif / Helvetica styling
    - Colormaps: Perceptually uniform (inferno, viridis, magma)
    - Multi-panel grids with synchronized scaling
    """
    def __init__(self, output_dir: str = "outputs/figures", dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Apply clean publication style
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.dpi": dpi
        })

    def plot_abcd_decomposition(
        self,
        rgb_image: np.ndarray,
        components: Dict[str, np.ndarray],
        composite_csm: np.ndarray,
        sample_id: str,
        class_name: str
    ) -> Path:
        """
        Creates a 6-panel decomposition figure:
        [Original Image] -> [A: Asymmetry] -> [B: Border]
        [C: Color]       -> [D: Diameter]  -> [Composite CSM]
        """
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        
        # Original Image
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title(f"Original Lesion ({class_name})")
        axes[0, 0].axis("off")

        # Asymmetry
        im_a = axes[0, 1].imshow(components["A_asymmetry"], cmap="inferno", vmin=0, vmax=1)
        axes[0, 1].set_title("A: Spatial Asymmetry (Otsu)")
        axes[0, 1].axis("off")
        fig.colorbar(im_a, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # Border
        im_b = axes[0, 2].imshow(components["B_border"], cmap="inferno", vmin=0, vmax=1)
        axes[0, 2].set_title("B: Border Irregularity (Sobel+Canny)")
        axes[0, 2].axis("off")
        fig.colorbar(im_b, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Color
        im_c = axes[1, 0].imshow(components["C_color"], cmap="inferno", vmin=0, vmax=1)
        axes[1, 0].set_title("C: Color Heterogeneity (HSV Var)")
        axes[1, 0].axis("off")
        fig.colorbar(im_c, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Diameter
        im_d = axes[1, 1].imshow(components["D_diameter"], cmap="inferno", vmin=0, vmax=1)
        axes[1, 1].set_title("D: Diameter / Distance Transform")
        axes[1, 1].axis("off")
        fig.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Composite CSM
        im_comp = axes[1, 2].imshow(composite_csm, cmap="magma", vmin=0, vmax=1)
        axes[1, 2].set_title("Composite Clinical Saliency (CSM)")
        axes[1, 2].axis("off")
        fig.colorbar(im_comp, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = self.output_dir / f"csm_decomposition_{sample_id}_{class_name}.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_hybrid_fusion_comparison(
        self,
        rgb_image: np.ndarray,
        csm_map: np.ndarray,
        gcw_map: np.ndarray,
        fused_map: np.ndarray,
        sample_id: str,
        diagnosis: str,
        concordance_metrics: Dict[str, float]
    ) -> Path:
        """
        Creates publication 4-panel comparison:
        Original | CSM (Clinical Knowledge) | GCW (Model Attribution) | CG-HAF (Hybrid Fusion)
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(rgb_image)
        axes[0].set_title(f"Lesion: {diagnosis}")
        axes[0].axis("off")

        im1 = axes[1].imshow(csm_map, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("Top-Down: CSM (ABCD Rule)")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(gcw_map, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("Bottom-Up: GCW (Swin Attention)")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(fused_map, cmap="magma", vmin=0, vmax=1)
        p_val = concordance_metrics.get("pearson_correlation", 0.0)
        axes[3].set_title(f"CG-HAF (λ=0.5)\n[Pearson r = {p_val:.2f}]")
        axes[3].axis("off")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = self.output_dir / f"cg_haf_fusion_{sample_id}_{diagnosis}.png"
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_faithfulness_curves(
        self,
        step_labels: np.ndarray,
        curves_dict: Dict[str, Dict[str, List[float]]],
        save_name: str = "faithfulness_benchmark.png"
    ) -> Path:
        """
        Renders Deletion and Insertion curves comparing CG-HAF against baselines.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Colors for methods
        palette = {
            "CG-HAF": "#D62728",      # Prominent Red
            "GCW": "#FF7F0E",         # Orange
            "Grad-CAM": "#1F77B4",    # Blue
            "IntegratedGrad": "#2CA02C" # Green
        }

        # 1. Deletion Curve (Steeper drop = More faithful)
        for method, metrics in curves_dict.items():
            color = palette.get(method, "#7F7F7F")
            ax1.plot(
                step_labels,
                metrics["deletion"],
                label=f"{method} (AUC={metrics['del_auc']:.3f})",
                color=color,
                linewidth=2.0 if method == "CG-HAF" else 1.5
            )

        ax1.set_title("Deletion Curve (Lower AUC = Higher Faithfulness)")
        ax1.set_xlabel("Fraction of Top-Attributed Pixels Removed")
        ax1.set_ylabel("Predicted Class Probability")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True)

        # 2. Insertion Curve (Steeper rise = More faithful)
        for method, metrics in curves_dict.items():
            color = palette.get(method, "#7F7F7F")
            ax2.plot(
                step_labels,
                metrics["insertion"],
                label=f"{method} (AUC={metrics['ins_auc']:.3f})",
                color=color,
                linewidth=2.0 if method == "CG-HAF" else 1.5
            )

        ax2.set_title("Insertion Curve (Higher AUC = Higher Faithfulness)")
        ax2.set_xlabel("Fraction of Top-Attributed Pixels Retained")
        ax2.set_ylabel("Predicted Class Probability")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

