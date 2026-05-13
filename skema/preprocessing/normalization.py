"""
skema.preprocessing.normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mean/std normalization helpers for HWC and tiled inputs.
"""

import numpy as np


def normalize_hwc(image: np.ndarray, mean: list, std: list, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize an (H, W, C) float32 image using per-channel mean and std.

    NaNs are replaced with 0 before normalization.
    """
    image = np.nan_to_num(image).astype(np.float32)
    mean_arr = np.array(mean, dtype=np.float32)[np.newaxis, np.newaxis, :]
    std_arr  = np.array(std,  dtype=np.float32)[np.newaxis, np.newaxis, :]
    return (image - mean_arr) / (std_arr + epsilon)


def normalize_tile(tile: np.ndarray, mean: list, std: list, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize an (H, W, C) tile — thin alias kept for backward compatibility.
    Identical behaviour to normalize_hwc.
    """
    return normalize_hwc(tile, mean, std, epsilon)