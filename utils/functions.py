import numpy as np
import torch
from typing import Optional


def get_device(device: Optional[str] = None):
    """cpu mps cuda"""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
