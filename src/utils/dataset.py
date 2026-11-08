"""
Dermatoscopy Dataset and High-Throughput DataLoader Pipeline.
Optimized for high GPU utilization (pinned memory, prefetch factor, asynchronous transfers).
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DermoscopyTestDataset(Dataset):
    """
    Dataset loader for skin lesion validation/test cohorts.
    Returns:
        tensor_image: Normalized float32 tensor (3, 224, 224) for neural inference.
        raw_rgb: Original uint8 numpy array (224, 224, 3) for CSM extraction.
        label: Class index (0 to 6).
        image_id: String ID of image.
    """
    def __init__(
        self,
        image_paths: List[Path],
        labels: Optional[List[int]] = None,
        image_size: int = 224
    ):
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [0] * len(image_paths)
        self.image_size = image_size

        # Standard ImageNet normalization for Swin Transformer V2
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray, int, str]:
        img_path = self.image_paths[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        tensor_img = self.tensor_transform(resized_rgb)
        label = self.labels[idx]
        image_id = img_path.stem

        return tensor_img, resized_rgb, label, image_id


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Constructs high-throughput DataLoader optimized for NVIDIA RTX 50-series GPUs.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

