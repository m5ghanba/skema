"""skema.preprocessing — band extraction, normalisation, index computation, slope, static layers."""

from skema.preprocessing.band_extraction import extract_bands_to_geotiffs
from skema.preprocessing.indices import INDEX_CALCULATORS
from skema.preprocessing.normalization import normalize_hwc, normalize_tile
from skema.preprocessing.slope import calculate_slope_for_raster, calculate_slope_horn
from skema.preprocessing.static_layers import (
    apply_fill_nodata_single,
    check_required_static_files,
    merge_substrate_files_single,
    warp_bathy_and_subs,
)

__all__ = [
    "extract_bands_to_geotiffs",
    "INDEX_CALCULATORS",
    "normalize_hwc",
    "normalize_tile",
    "calculate_slope_horn",
    "calculate_slope_for_raster",
    "check_required_static_files",
    "warp_bathy_and_subs",
    "merge_substrate_files_single",
    "apply_fill_nodata_single",
]