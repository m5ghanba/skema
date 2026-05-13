"""
Tests for skema.inference.engine
Only exercises pure-numpy/pure-Python logic that requires no GPU,
no model weights, and no real raster files.
"""

import numpy as np
import pytest

from skema.inference.engine import DatasetInference, create_weight_map


# ---------------------------------------------------------------------------
# create_weight_map
# ---------------------------------------------------------------------------

class TestCreateWeightMap:
    def test_shape(self):
        wm = create_weight_map(tile_size=64, halo_size=16)
        assert wm.shape == (64, 64)

    def test_dtype(self):
        wm = create_weight_map(tile_size=32, halo_size=8)
        assert wm.dtype == np.float32

    def test_centre_is_one(self):
        """The central region beyond the halo should be exactly 1.0."""
        ts, hs = 64, 16
        wm = create_weight_map(ts, hs)
        centre = wm[hs:ts - hs, hs:ts - hs]
        np.testing.assert_array_equal(centre, 1.0)

    def test_corners_are_lowest(self):
        """Corner pixels should have the minimum weight."""
        wm = create_weight_map(tile_size=64, halo_size=16)
        corner_val = wm[0, 0]
        assert corner_val < 1.0
        assert corner_val == wm.min()

    def test_values_in_0_1(self):
        wm = create_weight_map(tile_size=64, halo_size=16)
        assert wm.min() > 0 and wm.max() <= 1.0

    def test_symmetry(self):
        """Weight map should be symmetric both horizontally and vertically."""
        wm = create_weight_map(tile_size=64, halo_size=16)
        np.testing.assert_array_equal(wm, wm[::-1, :])   # vertical
        np.testing.assert_array_equal(wm, wm[:, ::-1])   # horizontal


# ---------------------------------------------------------------------------
# DatasetInference._compute_indices  (via a minimal subclass)
# ---------------------------------------------------------------------------

class _MockInference(DatasetInference):
    """Bypass __init__ entirely; only test internal numpy methods."""

    def __init__(self, model_type: str, image: np.ndarray):
        # Populate only the attributes used by _compute_indices
        self.model_type = model_type
        self.image = image.copy()


class TestComputeIndices:
    @pytest.fixture
    def s2_image(self):
        """(8, 8, 10) array: channels 0-4 are S2 bands, 5-9 will be filled."""
        img = np.zeros((8, 8, 10), dtype=np.float32)
        img[..., 0] = 100   # blue
        img[..., 1] = 200   # green
        img[..., 2] = 150   # red
        img[..., 3] = 800   # NIR
        img[..., 4] = 300   # red-edge
        return img

    @pytest.fixture
    def full_image(self):
        """(8, 8, 13) array; channels 5-7 are static layers."""
        img = np.zeros((8, 8, 13), dtype=np.float32)
        img[..., 0] = 100
        img[..., 1] = 200
        img[..., 2] = 150
        img[..., 3] = 800
        img[..., 4] = 300
        img[..., 5] = 2     # substrate
        img[..., 6] = -30   # bathymetry
        img[..., 7] = 5     # slope
        return img

    def test_s2_only_indices_written(self, s2_image):
        obj = _MockInference("model_s2bandsandindices_only", s2_image)
        obj._compute_indices()
        # Channels 5-9 must no longer be zero
        assert not np.all(obj.image[..., 5:10] == 0)

    def test_full_model_indices_written(self, full_image):
        obj = _MockInference("model_full", full_image)
        obj._compute_indices()
        # Channels 8-12 must no longer be zero
        assert not np.all(obj.image[..., 8:13] == 0)

    def test_s2_ndvi_channel_5(self, s2_image):
        """Channel 5 in S2-only mode should be NDVI."""
        obj = _MockInference("model_s2bandsandindices_only", s2_image)
        obj._compute_indices()
        green, red, nir = 200.0, 150.0, 800.0
        expected_ndvi = (nir - red) / (nir + red + 1e-10)
        np.testing.assert_allclose(obj.image[0, 0, 5], expected_ndvi, rtol=1e-5)

    def test_static_channels_unchanged_in_full_model(self, full_image):
        """Substrate, bathymetry, slope (ch 5-7) must not be overwritten."""
        obj = _MockInference("model_full", full_image)
        obj._compute_indices()
        assert obj.image[0, 0, 5] == 2    # substrate
        assert obj.image[0, 0, 6] == -30  # bathymetry
        assert obj.image[0, 0, 7] == 5    # slope


# ---------------------------------------------------------------------------
# DatasetInference.generate_tiles  (via _MockInference)
# ---------------------------------------------------------------------------

class _MockInferenceForTiles(_MockInference):
    """Add tile-generation attributes to the minimal mock."""

    def __init__(self, image: np.ndarray, tile_size: int, overlap: float):
        self.model_type       = "model_s2bandsandindices_only"
        self.image            = image.copy()
        self.tile_size        = tile_size
        self.overlap          = overlap
        self.mean_per_channel = None
        self.std_per_channel  = None


class TestGenerateTiles:
    @pytest.fixture
    def small_image(self):
        """(100, 100, 10) zeros image."""
        return np.zeros((100, 100, 10), dtype=np.float32)

    def test_tile_size_correct(self, small_image):
        obj   = _MockInferenceForTiles(small_image, tile_size=64, overlap=0.0)
        tiles = list(obj.generate_tiles(obj.image))
        for tile, _ in tiles:
            assert tile.shape == (64, 64, 10)

    def test_at_least_one_tile(self, small_image):
        obj   = _MockInferenceForTiles(small_image, tile_size=64, overlap=0.0)
        tiles = list(obj.generate_tiles(obj.image))
        assert len(tiles) >= 1

    def test_image_larger_than_tile(self):
        """A 300×300 image with 64-px tiles should produce many tiles."""
        img   = np.zeros((300, 300, 10), dtype=np.float32)
        obj   = _MockInferenceForTiles(img, tile_size=64, overlap=0.5)
        tiles = list(obj.generate_tiles(obj.image))
        assert len(tiles) > 4

    def test_coords_within_bounds(self, small_image):
        obj = _MockInferenceForTiles(small_image, tile_size=64, overlap=0.0)
        h, w = small_image.shape[:2]
        for _, (i, j) in obj.generate_tiles(obj.image):
            assert 0 <= i < h
            assert 0 <= j < w

    def test_single_tile_for_small_image(self):
        """Image smaller than tile_size → exactly one tile."""
        img   = np.zeros((32, 32, 10), dtype=np.float32)
        obj   = _MockInferenceForTiles(img, tile_size=64, overlap=0.5)
        tiles = list(obj.generate_tiles(obj.image))
        assert len(tiles) == 1