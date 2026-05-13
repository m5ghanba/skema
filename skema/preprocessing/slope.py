"""
skema.preprocessing.slope
~~~~~~~~~~~~~~~~~~~~~~~~~~
Slope calculation from bathymetry rasters using Horn's (1981) finite-difference
method, processed in memory-efficient windows.
"""

import os

import numpy as np
import rasterio
from rasterio.windows import Window
from rich.console import Console


def calculate_slope_horn(bathymetry: np.ndarray, cell_size: float = 20.0) -> np.ndarray:
    """
    Vectorized Horn slope calculation for a 2-D float32 array.

    Parameters
    ----------
    bathymetry : np.ndarray
        2-D array (H, W) of depth values; NaN marks no-data cells.
    cell_size : float
        Spatial resolution in the same units as *bathymetry* values.

    Returns
    -------
    np.ndarray
        Slope in degrees, shape (H, W).  Edge rows/cols are left as 0.
    """
    bathy = bathymetry.astype(np.float32)
    H, W  = bathy.shape
    slope = np.zeros((H, W), dtype=np.float32)

    if H < 3 or W < 3:
        return slope

    a, b, c = bathy[:-2, :-2], bathy[:-2, 1:-1], bathy[:-2, 2:]
    d,      f = bathy[1:-1, :-2], bathy[1:-1, 2:]
    g, h, i   = bathy[2:, :-2],  bathy[2:, 1:-1], bathy[2:, 2:]

    invalid = (
        np.isnan(a) | np.isnan(b) | np.isnan(c)
        | np.isnan(d) | np.isnan(f)
        | np.isnan(g) | np.isnan(h) | np.isnan(i)
    )

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cell_size)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cell_size)

    slope_inner = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope_inner[invalid] = np.nan

    slope[1:-1, 1:-1] = slope_inner
    return slope


def calculate_slope_for_raster(input_tiff: str, output_tiff: str, block_size: int = 2048) -> None:
    """
    Compute per-pixel slope from *input_tiff* (bathymetry) and write the result
    to *output_tiff*, processing the raster in overlapping blocks to keep RAM low.
    """
    console = Console()
    console.print(f"[cyan]Starting slope calculation for {os.path.basename(input_tiff)}...[/cyan]")

    with rasterio.open(input_tiff) as src:
        profile   = src.profile.copy()
        nodata_val = src.nodata
        cell_size  = abs(src.transform[0])

        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        with rasterio.open(output_tiff, "w", **profile) as dst:
            for row_idx in range(0, src.height, block_size):
                for col_idx in range(0, src.width, block_size):
                    window = Window(
                        col_off=col_idx,
                        row_off=row_idx,
                        width=min(block_size, src.width - col_idx),
                        height=min(block_size, src.height - row_idx),
                    )
                    # Expand by 1 pixel on all sides for neighbourhood calculation
                    r0 = max(0, window.row_off - 1)
                    r1 = min(src.height, window.row_off + window.height + 1)
                    c0 = max(0, window.col_off - 1)
                    c1 = min(src.width, window.col_off + window.width + 1)

                    buf_window = Window.from_slices((r0, r1), (c0, c1))
                    bathy_data = src.read(1, window=buf_window).astype(np.float32)

                    if nodata_val is not None:
                        bathy_data = np.where(bathy_data == nodata_val, np.nan, bathy_data)

                    slope_data = calculate_slope_horn(bathy_data, cell_size=cell_size)

                    # Crop back to the original (non-buffered) window
                    cr0 = window.row_off - r0
                    cc0 = window.col_off - c0
                    final = slope_data[cr0:cr0 + window.height, cc0:cc0 + window.width]
                    dst.write(final.astype(rasterio.float32), 1, window=window)

                if row_idx % (block_size * 4) == 0:
                    console.print(
                        f"[cyan]Processed up to row "
                        f"{min(row_idx + block_size, src.height)} / {src.height}[/cyan]"
                    )

    console.print(f"[green]✓[/green] Slope calculation complete: {output_tiff}")