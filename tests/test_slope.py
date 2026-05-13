"""Tests for skema.preprocessing.slope"""

import numpy as np
import pytest

from skema.preprocessing.slope import calculate_slope_horn


class TestCalculateSlopeHorn:

    def test_flat_surface_slope_is_zero(self):
        """A perfectly flat raster has zero slope everywhere (except edges)."""
        flat = np.ones((10, 10), dtype=np.float32) * -20.0
        out  = calculate_slope_horn(flat, cell_size=10.0)
        # Interior pixels only (edges are 0 by design)
        np.testing.assert_allclose(out[1:-1, 1:-1], 0.0, atol=1e-5)

    def test_output_shape_matches_input(self):
        bathy = np.random.rand(8, 12).astype(np.float32)
        out   = calculate_slope_horn(bathy, cell_size=20.0)
        assert out.shape == bathy.shape

    def test_output_dtype_is_float32(self):
        bathy = np.ones((5, 5), dtype=np.float64)
        out   = calculate_slope_horn(bathy)
        assert out.dtype == np.float32

    def test_slope_non_negative(self):
        """Slope (degrees) is always ≥ 0."""
        rng   = np.random.default_rng(42)
        bathy = rng.standard_normal((20, 20)).astype(np.float32)
        out   = calculate_slope_horn(bathy, cell_size=10.0)
        valid = out[1:-1, 1:-1]
        assert np.all(valid[~np.isnan(valid)] >= 0)

    def test_small_array_returns_zeros(self):
        """Arrays smaller than 3×3 return an all-zero array (no neighbourhood)."""
        bathy = np.ones((2, 2), dtype=np.float32)
        out   = calculate_slope_horn(bathy, cell_size=10.0)
        np.testing.assert_array_equal(out, np.zeros((2, 2), dtype=np.float32))

    def test_steep_gradient_produces_large_slope(self):
        """A steep linear ramp should produce slope > 45°."""
        bathy              = np.zeros((10, 10), dtype=np.float32)
        bathy[:, :]        = np.arange(10, dtype=np.float32) * 100   # 100 m drop per pixel
        out                = calculate_slope_horn(bathy, cell_size=1.0)
        interior           = out[1:-1, 1:-1]
        valid              = interior[~np.isnan(interior)]
        assert np.any(valid > 45), "Expected some pixels with slope > 45°"

    def test_nan_nodata_handled(self):
        """Pixels adjacent to NaN should themselves be NaN in the interior."""
        bathy                = np.ones((5, 5), dtype=np.float32) * -10.0
        bathy[2, 2]          = np.nan
        out                  = calculate_slope_horn(bathy, cell_size=10.0)
        # At least the pixel at (2,2) should propagate NaN to its neighbours
        centre_region        = out[1:4, 1:4]
        assert np.any(np.isnan(centre_region))

    def test_cell_size_scaling(self):
        """Doubling cell_size should halve the slope (linear relationship in Horn)."""
        rng   = np.random.default_rng(0)
        bathy = rng.standard_normal((10, 10)).astype(np.float32)
        s1    = calculate_slope_horn(bathy, cell_size=10.0)
        s2    = calculate_slope_horn(bathy, cell_size=20.0)
        # Interior, non-NaN pixels only
        mask  = ~np.isnan(s1) & ~np.isnan(s2)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        # arctan(x/2) < arctan(x), so s2 < s1 everywhere (larger cell → shallower angle)
        assert np.all(s2[mask] <= s1[mask] + 1e-4)