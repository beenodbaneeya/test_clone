"""Utility helpers for selecting the compute device."""

import torch


def get_device() -> torch.device:
    """
    Return the preferred compute device.
    """

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")