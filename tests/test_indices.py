"""Tests for skema.preprocessing.indices"""

import numpy as np
import pytest

from skema.preprocessing.indices import (
    INDEX_CALCULATORS,
    chlorophyll_index_green,
    gndvi,
    ndvi,
    ndvi_re,
    ndwi,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image():
    """
    Minimal HWC image (4, 4, 5) with controlled band values.
    Channel order: B2(blue)=0, B3(green)=1, B4(red)=2, B8(NIR)=3, B5(red-edge)=4
    """
    img = np.zeros((4, 4, 5), dtype=np.float32)
    img[..., 0] = 100   # blue
    img[..., 1] = 200   # green
    img[..., 2] = 150   # red
    img[..., 3] = 800   # NIR
    img[..., 4] = 300   # red-edge
    return img


@pytest.fixture
def uniform_image():
    """Image where all channels share the same value → ratios predictable."""
    img = np.full((3, 3, 5), 500.0, dtype=np.float32)
    return img


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_ndvi_shape(self, synthetic_image):
        out = ndvi(synthetic_image)
        assert out.shape == (4, 4)

    def test_ndwi_shape(self, synthetic_image):
        out = ndwi(synthetic_image)
        assert out.shape == (4, 4)

    def test_gndvi_shape(self, synthetic_image):
        out = gndvi(synthetic_image)
        assert out.shape == (4, 4)

    def test_cig_shape(self, synthetic_image):
        out = chlorophyll_index_green(synthetic_image)
        assert out.shape == (4, 4)

    def test_ndvire_shape(self, synthetic_image):
        out = ndvi_re(synthetic_image)
        assert out.shape == (4, 4)


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------

class TestIndexValues:
    def test_ndvi_range(self, synthetic_image):
        """NDVI ∈ (−1, 1)."""
        out = ndvi(synthetic_image)
        assert np.all(out > -1) and np.all(out < 1)

    def test_ndvi_positive_when_nir_gt_red(self, synthetic_image):
        """NIR (800) > red (150) → NDVI > 0."""
        out = ndvi(synthetic_image)
        assert np.all(out > 0)

    def test_ndwi_negative_when_nir_gt_green(self, synthetic_image):
        """NIR (800) > green (200) → NDWI < 0."""
        out = ndwi(synthetic_image)
        assert np.all(out < 0)

    def test_ndvi_known_value(self):
        """NDVI = (NIR − red) / (NIR + red); exact float check."""
        img = np.zeros((1, 1, 5), dtype=np.float32)
        img[..., 2] = 100   # red
        img[..., 3] = 300   # NIR
        expected = (300 - 100) / (300 + 100 + 1e-10)
        out = ndvi(img)
        assert abs(float(out[0, 0]) - expected) < 1e-5

    def test_ndwi_known_value(self):
        img = np.zeros((1, 1, 5), dtype=np.float32)
        img[..., 1] = 200   # green
        img[..., 3] = 600   # NIR
        expected = (200 - 600) / (200 + 600 + 1e-10)
        out = ndwi(img)
        assert abs(float(out[0, 0]) - expected) < 1e-5

    def test_uniform_image_ndvi_near_zero(self, uniform_image):
        """All channels equal → NDVI ≈ 0."""
        out = ndvi(uniform_image)
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_cig_clamp_low_green(self):
        """Green < 1e-4 → chlorophyll index clamped to 20."""
        img = np.zeros((2, 2, 5), dtype=np.float32)
        img[..., 1] = 0.0   # green ≈ 0
        img[..., 3] = 500   # NIR
        out = chlorophyll_index_green(img)
        assert np.all(out == 20.0)

    def test_ndvi_re_positive(self, synthetic_image):
        """Red-edge (300) > red (150) → NDVIre > 0."""
        out = ndvi_re(synthetic_image)
        assert np.all(out > 0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestIndexRegistry:
    def test_all_keys_present(self):
        expected = {"ndvi", "ndwi", "gndvi", "clgreen", "ndvire"}
        assert set(INDEX_CALCULATORS.keys()) == expected

    def test_all_callables(self):
        for name, fn in INDEX_CALCULATORS.items():
            assert callable(fn), f"{name} is not callable"

    def test_registry_produces_correct_shape(self, synthetic_image):
        for name, fn in INDEX_CALCULATORS.items():
            out = fn(synthetic_image)
            assert out.shape == (4, 4), f"{name} returned wrong shape"