"""
skema.preprocessing.indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Spectral index calculations used by both the training dataset and inference pipeline.

All functions accept an HWC float32 numpy array and return a 2-D array (H, W).
Channel layout (S2-only / full-model shared bands):
    0: B2 (blue), 1: B3 (green), 2: B4 (red), 3: B8 (NIR), 4: B5 (red-edge)
"""

import numpy as np

_EPS = 1e-10


def ndvi(img: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    nir, red = img[..., 3], img[..., 2]
    return (nir - red) / (nir + red + _EPS)


def ndwi(img: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index."""
    green, nir = img[..., 1], img[..., 3]
    return (green - nir) / (green + nir + _EPS)


def gndvi(img: np.ndarray) -> np.ndarray:
    """Green NDVI."""
    nir, green = img[..., 3], img[..., 1]
    return (nir - green) / (nir + green + _EPS)


def chlorophyll_index_green(img: np.ndarray) -> np.ndarray:
    """Chlorophyll Index Green."""
    nir, green = img[..., 3], img[..., 1]
    return np.where(green < 1e-4, 20.0, nir / (green + _EPS) - 1)


def ndvi_re(img: np.ndarray) -> np.ndarray:
    """Red-edge NDVI."""
    re, red = img[..., 4], img[..., 2]
    return (re - red) / (re + red + _EPS)


# Registry used by SatelliteDataset and DatasetInference
INDEX_CALCULATORS = {
    "ndvi":    ndvi,
    "ndwi":    ndwi,
    "gndvi":   gndvi,
    "clgreen": chlorophyll_index_green,
    "ndvire":  ndvi_re,
}