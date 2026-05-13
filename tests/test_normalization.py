"""Tests for skema.preprocessing.normalization"""

import numpy as np
import pytest

from skema.preprocessing.normalization import normalize_hwc, normalize_tile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_image():
    """3-channel 2×2 image with known values."""
    img = np.array(
        [[[10.0, 20.0, 30.0],
          [10.0, 20.0, 30.0]],
         [[10.0, 20.0, 30.0],
          [10.0, 20.0, 30.0]]],
        dtype=np.float32,
    )
    return img  # shape (2, 2, 3)


@pytest.fixture
def mean_std():
    return [10.0, 20.0, 30.0], [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

class TestShape:
    def test_output_shape_matches_input(self, simple_image, mean_std):
        mean, std = mean_std
        out = normalize_hwc(simple_image, mean, std)
        assert out.shape == simple_image.shape

    def test_normalize_tile_same_as_hwc(self, simple_image, mean_std):
        mean, std = mean_std
        out_hwc  = normalize_hwc(simple_image, mean, std)
        out_tile = normalize_tile(simple_image, mean, std)
        np.testing.assert_array_equal(out_hwc, out_tile)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestNormalizationValues:
    def test_zero_mean_result(self, simple_image, mean_std):
        """When pixel value == channel mean, normalised result ≈ 0."""
        mean, std = mean_std
        out = normalize_hwc(simple_image, mean, std)
        # All pixels equal the mean → all outputs ≈ 0
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_known_value(self):
        """Scalar verification: (x − μ) / (σ + ε)."""
        img  = np.ones((1, 1, 1), dtype=np.float32) * 5.0
        mean = [3.0]
        std  = [2.0]
        out  = normalize_hwc(img, mean, std)
        expected = (5.0 - 3.0) / (2.0 + 1e-8)
        np.testing.assert_allclose(float(out[0, 0, 0]), expected, rtol=1e-5)

    def test_per_channel_scaling(self):
        """Each channel is scaled by its own std."""
        img    = np.zeros((1, 1, 3), dtype=np.float32)
        img[0, 0, :] = [6.0, 12.0, 18.0]
        mean   = [0.0,  0.0,  0.0]
        std    = [2.0,  4.0,  6.0]
        out    = normalize_hwc(img, mean, std)
        expected = np.array([[[3.0, 3.0, 3.0]]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_output_dtype_is_float32(self, simple_image, mean_std):
        mean, std = mean_std
        out = normalize_hwc(simple_image, mean, std)
        assert out.dtype == np.float32

    def test_nan_replaced_before_normalizing(self):
        """NaN inputs must be handled (replaced with 0) without raising."""
        img = np.full((2, 2, 2), np.nan, dtype=np.float32)
        out = normalize_hwc(img, [0.0, 0.0], [1.0, 1.0])
        assert not np.any(np.isnan(out))

    def test_zero_std_does_not_divide_by_zero(self):
        """When std = 0, the epsilon guard prevents ZeroDivisionError."""
        img = np.ones((1, 1, 1), dtype=np.float32)
        out = normalize_hwc(img, [0.0], [0.0])
        assert np.isfinite(out).all()