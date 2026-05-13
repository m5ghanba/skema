# tests/conftest.py
"""
Shared pytest fixtures and configuration for the SKeMa test suite.

All tests in this directory are pure-unit tests: they require no GPU,
no network access, and no real Sentinel-2 or static raster files.
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Seeded numpy random generator, shared across the whole test session."""
    return np.random.default_rng(42)


@pytest.fixture
def s2_only_image():
    """
    Synthetic (16, 16, 10) HWC image representing a model_s2bandsandindices_only scene.
    Channels 0-4: S2 bands (B2,B3,B4,B8,B5).
    Channels 5-9: placeholder zeros (filled by _compute_indices).
    """
    img = np.zeros((16, 16, 10), dtype=np.float32)
    img[..., 0] = 100   # blue
    img[..., 1] = 200   # green
    img[..., 2] = 150   # red
    img[..., 3] = 800   # NIR
    img[..., 4] = 300   # red-edge
    return img


@pytest.fixture
def full_model_image():
    """
    Synthetic (16, 16, 13) HWC image representing a model_full scene.
    Channels 0-4: S2 bands, 5: substrate, 6: bathymetry, 7: slope.
    Channels 8-12: placeholder zeros (filled by _compute_indices).
    """
    img = np.zeros((16, 16, 13), dtype=np.float32)
    img[..., 0] = 100
    img[..., 1] = 200
    img[..., 2] = 150
    img[..., 3] = 800
    img[..., 4] = 300
    img[..., 5] = 2      # substrate class
    img[..., 6] = -25.0  # bathymetry (m) — within valid depth zone
    img[..., 7] = 3.0    # slope (degrees)
    return img