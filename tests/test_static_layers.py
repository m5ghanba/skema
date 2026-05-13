"""
Tests for skema.preprocessing.static_layers

Only the pure-numpy helpers that don't require the static package data
(bathymetry / substrate TIFFs) are tested here.  The warp / merge functions
that hit the filesystem are covered by integration tests that require the
actual package data to be installed.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from skema.preprocessing.static_layers import fill_nodata_fixed_value


# ---------------------------------------------------------------------------
# Helper: write a tiny single-band GeoTIFF with optional nodata
# ---------------------------------------------------------------------------

def _write_geotiff(path: str, data: np.ndarray, nodata=None):
    transform = from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0])
    profile = {
        "driver": "GTiff",
        "dtype": str(data.dtype),
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# fill_nodata_fixed_value
# ---------------------------------------------------------------------------

class TestFillNodataFixedValue:
    def test_nodata_pixels_replaced(self, tmp_path):
        data = np.array([[1, -9999, 3], [4, 5, -9999]], dtype=np.float32)
        inp  = str(tmp_path / "input.tif")
        out  = str(tmp_path / "output.tif")
        _write_geotiff(inp, data, nodata=-9999)

        fill_nodata_fixed_value(inp, out, fill_value=0)

        with rasterio.open(out) as src:
            result = src.read(1)

        assert result[0, 1] == 0, "NoData pixel should be replaced with 0"
        assert result[1, 2] == 0, "NoData pixel should be replaced with 0"

    def test_valid_pixels_unchanged(self, tmp_path):
        data = np.array([[10, -9999, 30]], dtype=np.float32)
        inp  = str(tmp_path / "input.tif")
        out  = str(tmp_path / "output.tif")
        _write_geotiff(inp, data, nodata=-9999)

        fill_nodata_fixed_value(inp, out, fill_value=0)

        with rasterio.open(out) as src:
            result = src.read(1)

        assert result[0, 0] == 10
        assert result[0, 2] == 30

    def test_nodata_flag_cleared(self, tmp_path):
        data = np.array([[1, -9999]], dtype=np.float32)
        inp  = str(tmp_path / "input.tif")
        out  = str(tmp_path / "output.tif")
        _write_geotiff(inp, data, nodata=-9999)

        fill_nodata_fixed_value(inp, out, fill_value=0)

        with rasterio.open(out) as src:
            assert src.nodata is None, "NoData flag should be cleared in output"

    def test_input_file_deleted_after(self, tmp_path):
        data = np.array([[1, 2]], dtype=np.float32)
        inp  = str(tmp_path / "input.tif")
        out  = str(tmp_path / "output.tif")
        _write_geotiff(inp, data, nodata=None)

        fill_nodata_fixed_value(inp, out, fill_value=-1)

        assert not os.path.exists(inp), "Input file should be removed after processing"

    def test_no_nodata_leaves_data_unchanged(self, tmp_path):
        """If the raster has no nodata attribute, data should pass through untouched."""
        data = np.array([[5, 10, 15]], dtype=np.float32)
        inp  = str(tmp_path / "input.tif")
        out  = str(tmp_path / "output.tif")
        _write_geotiff(inp, data, nodata=None)  # no nodata set

        fill_nodata_fixed_value(inp, out, fill_value=-999)

        with rasterio.open(out) as src:
            result = src.read(1)

        np.testing.assert_array_equal(result, data)

    def test_custom_fill_value(self, tmp_path):
        data = np.array([[-9999, 100]], dtype=np.float32)
        inp  = str(tmp_path / "input.tif")
        out  = str(tmp_path / "output.tif")
        _write_geotiff(inp, data, nodata=-9999)

        fill_nodata_fixed_value(inp, out, fill_value=-2000)

        with rasterio.open(out) as src:
            result = src.read(1)

        assert result[0, 0] == -2000