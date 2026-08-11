"""
HARBench Dataset

PyTorch Dataset class for finetuning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class HARDataset(Dataset):
    """
    Human Activity Recognition Dataset

    Args:
        X: Sensor data (N, C, T) - N: number of samples, C: number of channels, T: sequence length
        Y: Labels (N,)
        transform: Optional preprocessing function
        source_id: Optional (N,) int array/tensor of per-sample source-dataset ids. When
            given, __getitem__ returns (x, y, source_id) instead of (x, y) -- opt-in only,
            existing callers that never pass source_id are unaffected.
    """

    def __init__(self, X, Y, transform=None, source_id=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.transform = transform
        self.source_id = None if source_id is None else torch.as_tensor(source_id, dtype=torch.long)
        if self.source_id is not None and len(self.source_id) != len(self.Y):
            raise ValueError(
                f"source_id length ({len(self.source_id)}) must match Y length ({len(self.Y)})"
            )

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)

        if self.source_id is not None:
            return x, y, self.source_id[idx]
        return x, y
