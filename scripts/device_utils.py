"""Utility helpers for selecting the compute device."""

import torch


def get_device() -> torch.device:
    """
    Return the preferred compute device.

    Deep learning note:
    Training/inference must move both model and tensors to the same device.

    Returns:
        torch.device: ``cuda:0`` when CUDA is available, otherwise ``cpu``.
    """

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
