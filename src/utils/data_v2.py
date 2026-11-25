"""
Shared loading helpers for the v2 scripts (one preprocessing function for every cohort).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.pipeline.test_time_adaptation import parity_resize

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_rgb(path: Path | str) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(rgb: np.ndarray, size: int = 224, mode: str = "legacy") -> np.ndarray:
    """'legacy' = squash-resize bilinear; 'parity' = FOV crop + aspect-preserving crop."""
    if mode == "legacy":
        return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))
    if mode == "parity":
        return parity_resize(rgb, size)
    raise ValueError(mode)


def to_tensor(rgb224: np.ndarray, normalise: bool = True) -> torch.Tensor:
    x = rgb224.astype(np.float32) / 255.0
    if normalise:
        x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def stratified(df: pd.DataFrame, per_class: int, seed: int, label_col: str = "dx") -> pd.DataFrame:
    """Random (not head()) class-stratified sample; classes with fewer rows are taken whole."""
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby(label_col)]
    return pd.concat(parts).reset_index(drop=True)


def load_mask(path: Path | str, size: int = 224, mode: str = "legacy", rgb_shape: Optional[tuple] = None) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m3 = np.repeat(((m > 127).astype(np.uint8) * 255)[..., None], 3, axis=2)
    if mode == "parity":
        return (parity_resize(m3, size, crop_fov=False)[..., 0] > 127).astype(np.uint8)
    return (cv2.resize(m3[..., 0], (size, size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
