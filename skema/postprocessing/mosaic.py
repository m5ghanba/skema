"""
skema.postprocessing.mosaic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maximum-value mosaic creation from per-scene kelp prediction GeoTIFFs,
reprojected to BC Albers (EPSG:3005) at a fixed resolution.
"""

import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as rasterio_from_bounds
from rasterio.warp import reproject, transform_bounds
from rich.console import Console

# Approximate BC Albers extent used for sanity-checking scene footprints
_BC_ALBERS_EXTENT = {"min_x": 200_000, "max_x": 1_900_000, "min_y": 300_000, "max_y": 1_750_000}


def _reproject_tile(src, mosaic_transform, height: int, width: int, target_crs: str) -> np.ndarray:
    tile = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=rasterio.band(src, 1),
        destination=tile,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=mosaic_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    return tile


def create_mosaic(
    tif_paths: list[str],
    output_path: str,
    target_resolution_meters: int = 10,
    soft_substrate_masking: bool  = False,
    eelgrass_masking: bool        = False,
) -> None:
    """
    Create a maximum-value mosaic from *tif_paths* and write it to *output_path*.

    If *soft_substrate_masking* or *eelgrass_masking* is True, a second mosaic
    (``<stem>_masked.tif``) is created from the corresponding ``*_masked.tif``
    per-scene files.

    Parameters
    ----------
    tif_paths : list[str]
        Paths to per-scene prediction GeoTIFFs.
    output_path : str
        Destination path for the mosaic file.
    target_resolution_meters : int
        Output pixel size in metres.
    soft_substrate_masking : bool
        Include soft-substrate-masked scenes in the second mosaic.
    eelgrass_masking : bool
        Include eelgrass-masked scenes in the second mosaic.
    """
    console    = Console()
    target_crs = "EPSG:3005"

    valid_paths = [p for p in tif_paths if os.path.exists(p)]
    if not valid_paths:
        console.print("[red]No valid prediction TIFFs found – mosaic skipped.[/red]")
        return

    # ── 1. Collect bounds ─────────────────────────────────────────────
    all_bounds  = []
    tiles_in_bc = False
    for p in valid_paths:
        with rasterio.open(p) as src:
            bounds = src.bounds if str(src.crs) == target_crs \
                     else transform_bounds(src.crs, target_crs, *src.bounds)
            all_bounds.append(bounds)
            e = _BC_ALBERS_EXTENT
            if bounds[0] < e["max_x"] and bounds[2] > e["min_x"] \
                    and bounds[1] < e["max_y"] and bounds[3] > e["min_y"]:
                tiles_in_bc = True

    if not tiles_in_bc:
        console.print(
            "[yellow]Warning: tiles do not appear to be in British Columbia. "
            "Mosaic creation skipped.[/yellow]"
        )
        return

    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)

    # ── 2. Build canvas ───────────────────────────────────────────────
    r = target_resolution_meters
    width  = int(np.ceil((max_x - min_x) / r))
    height = int(np.ceil((max_y - min_y) / r))
    mosaic_transform = rasterio_from_bounds(min_x, min_y, max_x, max_y, width, height)
    mosaic = np.zeros((height, width), dtype=np.uint8)

    # ── 3. Accumulate (max) ───────────────────────────────────────────
    console.print(f"[cyan]Building mosaic from {len(valid_paths)} scene(s)...[/cyan]")
    for p in valid_paths:
        with rasterio.open(p) as src:
            tile = _reproject_tile(src, mosaic_transform, height, width, target_crs)
            mosaic = np.maximum(mosaic, tile)

    # ── 4. Save ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=height, width=width,
        count=1, dtype=np.uint8,
        crs=target_crs, transform=mosaic_transform,
        compress="lzw",
    ) as dst:
        dst.write(mosaic, 1)
    console.print(f"[green]✓[/green] Mosaic saved to [bold]{output_path}[/bold].")

    # ── 5. Optional masked mosaic ─────────────────────────────────────
    if soft_substrate_masking or eelgrass_masking:
        stem         = os.path.splitext(os.path.basename(output_path))[0]
        masked_dir   = os.path.dirname(output_path)
        masked_paths = []
        for p in valid_paths:
            sc_stem    = os.path.splitext(os.path.basename(p))[0]
            candidate  = os.path.join(os.path.dirname(p), f"{sc_stem}_masked.tif")
            if os.path.exists(candidate):
                masked_paths.append(candidate)
            else:
                console.print(
                    f"[yellow]Warning: masked file not found for "
                    f"{os.path.basename(p)}, skipping in masked mosaic.[/yellow]"
                )

        if masked_paths:
            masked_mosaic     = np.zeros((height, width), dtype=np.uint8)
            masked_mosaic_path = os.path.join(masked_dir, f"{stem}_masked.tif")
            console.print(
                f"[cyan]Building masked mosaic from {len(masked_paths)} scene(s)...[/cyan]"
            )
            for p in masked_paths:
                with rasterio.open(p) as src:
                    tile = _reproject_tile(src, mosaic_transform, height, width, target_crs)
                    masked_mosaic = np.maximum(masked_mosaic, tile)
            with rasterio.open(
                masked_mosaic_path, "w",
                driver="GTiff", height=height, width=width,
                count=1, dtype=np.uint8,
                crs=target_crs, transform=mosaic_transform,
                compress="lzw",
            ) as dst:
                dst.write(masked_mosaic, 1)
            console.print(
                f"[green]✓[/green] Masked mosaic saved to [bold]{masked_mosaic_path}[/bold]."
            )
        else:
            console.print("[yellow]No masked scene files found; masked mosaic skipped.[/yellow]")