"""
skema.preprocessing.static_layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Warps bathymetry, slope, and substrate static rasters to match a Sentinel-2
scene, merges regional substrate tiles, and fills NoData values.
"""

import os
import shutil
from importlib.resources import files

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as rasterio_from_bounds
from rasterio.warp import reproject
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from skema.preprocessing.slope import calculate_slope_for_raster

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bops_input_map() -> dict:
    return {
        "Bathymetry.tif": "_Bathy.tif",
        "Slope.tif":      "_Slope.tif",
        "BoPs_HG_10m.tif":     "_SubsHG.tif",
        "BoPs_NCC_10m.tif":    "_SubsNCC.tif",
        "BoPs_QCSSOG_10m.tif": "_SubsQCSSOG.tif",
        "BoPs_WCVI_10m.tif":   "_SubsWCVI.tif",
    }


def _rf_input_map() -> dict:
    return {
        "Bathymetry.tif":       "_Bathy.tif",
        "Slope.tif":            "_Slope.tif",
        "NCC_substrate_20m.tif":  "_SubsNCC.tif",
        "SOG_substrate_20m.tif":  "_SubsSOG.tif",
        "WCVI_substrate_20m.tif": "_SubsWCVI.tif",
        "QCS_substrate_20m.tif":  "_SubsQCS.tif",
        "HG_substrate_20m.tif":   "_SubsHG.tif",
    }


def _ensure_slope(console: Console) -> None:
    """Generate Slope.tif from Bathymetry.tif if it does not yet exist."""
    try:
        slope_path = str(files("skema.static.bathy_substrate").joinpath("Slope.tif"))
    except Exception:
        return
    if os.path.exists(slope_path):
        return

    console.print("[yellow]Slope.tif not found. Generating from Bathymetry.tif...[/yellow]")
    try:
        bathy_path = str(files("skema.static.bathy_substrate").joinpath("Bathymetry.tif"))
        if os.path.exists(bathy_path):
            calculate_slope_for_raster(bathy_path, slope_path)
        else:
            console.print("[red]Bathymetry.tif not found, cannot generate slope.[/red]")
    except Exception as exc:
        console.print(f"[red]Error generating Slope.tif: {exc}[/red]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_required_static_files(use_bops_substrate: bool) -> list[str]:
    """
    Return a list of static filenames that are missing from the package data.
    An empty list means everything is present.
    """
    if use_bops_substrate:
        required = [
            "Bathymetry.tif", "Slope.tif",
            "BoPs_HG_10m.tif", "BoPs_NCC_10m.tif",
            "BoPs_QCSSOG_10m.tif", "BoPs_WCVI_10m.tif",
        ]
    else:
        required = [
            "Bathymetry.tif", "Slope.tif",
            "NCC_substrate_20m.tif", "SOG_substrate_20m.tif",
            "WCVI_substrate_20m.tif", "QCS_substrate_20m.tif",
            "HG_substrate_20m.tif",
        ]
    missing = []
    for fname in required:
        try:
            p = str(files("skema.static.bathy_substrate").joinpath(fname))
            if not os.path.exists(p):
                missing.append(fname)
        except Exception:
            missing.append(fname)
    return missing


def warp_bathy_and_subs(safe_folder_root: str, basename: str, use_bops_substrate: bool = False) -> None:
    """
    Reproject and resample bathymetry, slope, and substrate rasters to 10 m,
    aligned to the reference Sentinel-2 image (*_B2B3B4B8.tif).
    """
    console = Console()
    _ensure_slope(console)

    input_files = _bops_input_map() if use_bops_substrate else _rf_input_map()

    for folder_name in os.listdir(safe_folder_root):
        folder_path = os.path.join(safe_folder_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        tif_file = next(
            (f for f in os.listdir(folder_path) if f == f"{basename}_B2B3B4B8.tif"), None
        )
        if not tif_file:
            continue

        reference_tif = os.path.join(folder_path, tif_file)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Aligning bathymetry, slope, and substrate files with Sentinel-2 image...",
                total=len(input_files),
            )

            for file_name, suffix in input_files.items():
                try:
                    static_path = str(files("skema.static.bathy_substrate").joinpath(file_name))
                except Exception as exc:
                    console.print(f"[red]Failed to access static file {file_name}: {exc}[/red]")
                    progress.advance(task)
                    continue

                if not os.path.exists(static_path):
                    console.print(f"[red]Static file not found: {static_path}[/red]")
                    progress.advance(task)
                    continue

                output_file = os.path.join(folder_path, folder_name + suffix)

                with rasterio.open(reference_tif) as ref:
                    bounds = ref.bounds
                    crs = ref.crs.to_string()

                width  = int((bounds.right - bounds.left) / 10)
                height = int((bounds.top  - bounds.bottom) / 10)
                transform = rasterio_from_bounds(
                    bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
                )

                with rasterio.open(static_path) as src:
                    out_data = np.empty((height, width), dtype=src.dtypes[0])
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=out_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.bilinear,
                    )
                    profile = src.profile.copy()
                    profile.update({"crs": crs, "transform": transform, "width": width, "height": height})
                    with rasterio.open(output_file, "w", **profile) as dst:
                        dst.write(out_data, 1)

                progress.advance(task)

        console.print(f"[green]✓[/green] Alignment complete.")


def merge_substrate_files_single(safe_output_dir: str, use_bops_substrate: bool = False) -> str | None:
    """
    Merge the per-region warped substrate tiles in *safe_output_dir* into a
    single *_Subs.tif*, then delete the individual tiles.

    Returns the path to the merged file, or None if inputs are missing.
    """
    console = Console()

    b2348_file = next(
        (f for f in os.listdir(safe_output_dir) if f.endswith("_B2B3B4B8.tif")), None
    )
    if not b2348_file:
        return None

    base_name = b2348_file.replace("_B2B3B4B8.tif", "")

    if use_bops_substrate:
        suffixes       = ["_SubsHG.tif", "_SubsNCC.tif", "_SubsQCSSOG.tif", "_SubsWCVI.tif"]
        valid_values   = {1, 2, 3}
        expected_count = 4
    else:
        suffixes       = ["_SubsNCC.tif", "_SubsSOG.tif", "_SubsWCVI.tif", "_SubsQCS.tif", "_SubsHG.tif"]
        valid_values   = {1, 2, 3, 4}
        expected_count = 5

    input_files = [
        os.path.join(safe_output_dir, f)
        for f in os.listdir(safe_output_dir)
        if any(f.endswith(s) for s in suffixes)
    ]
    if len(input_files) != expected_count:
        console.print(
            f"[yellow]Not all substrate files found in {safe_output_dir} "
            f"(found {len(input_files)}, expected {expected_count}), skipping merge.[/yellow]"
        )
        return None

    output_file = os.path.join(safe_output_dir, f"{base_name}_Subs.tif")

    with rasterio.open(input_files[0]) as src:
        meta   = src.meta.copy()
        height, width = src.shape

    merged = np.zeros((height, width), dtype=meta["dtype"])
    for fpath in input_files:
        with rasterio.open(fpath) as src:
            data = src.read(1)
            mask = np.isin(data, list(valid_values))
            merged[mask] = data[mask]

    meta.update(dtype=rasterio.uint8, nodata=0, compress="LZW")
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(merged, 1)

    for fpath in input_files:
        try:
            os.remove(fpath)
        except Exception as exc:
            console.print(f"[red]Error deleting {fpath}: {exc}[/red]")

    return output_file


def fill_nodata_fixed_value(input_file: str, output_file: str, fill_value: float) -> None:
    """Replace NoData pixels with *fill_value* and clear the NoData flag."""
    with rasterio.open(input_file) as src:
        data        = src.read(1)
        nodata_val  = src.nodata
        profile     = src.profile.copy()
        if nodata_val is not None:
            data[data == nodata_val] = fill_value
        profile.update(nodata=None)
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(data, 1)
    os.remove(input_file)


def apply_fill_nodata_single(
    safe_output_dir: str,
    fill_value_subs:  float = 0,
    fill_value_bathy: float = -2000,
    fill_value_slope: float = 85,
) -> tuple:
    """
    Fill NoData in substrate, bathymetry, and slope rasters and rename them
    to their final output filenames.

    Returns (subs_path, bathy_path, slope_path) — each may be None.
    """
    subs_file  = next((f for f in os.listdir(safe_output_dir) if f.endswith("_Subs.tif")),  None)
    bathy_file = next((f for f in os.listdir(safe_output_dir) if f.endswith("_Bathy.tif")), None)
    slope_file = next((f for f in os.listdir(safe_output_dir) if f.endswith("_Slope.tif")), None)

    subs_out = bathy_out = slope_out = None

    if subs_file:
        base = subs_file.replace("_Subs.tif", "")
        subs_out = os.path.join(safe_output_dir, f"{base}_Substrate.tif")
        fill_nodata_fixed_value(os.path.join(safe_output_dir, subs_file), subs_out, fill_value_subs)

    if bathy_file:
        base = bathy_file.replace("_Bathy.tif", "")
        bathy_out = os.path.join(safe_output_dir, f"{base}_Bathymetry.tif")
        fill_nodata_fixed_value(os.path.join(safe_output_dir, bathy_file), bathy_out, fill_value_bathy)

    if slope_file:
        slope_path = os.path.join(safe_output_dir, slope_file)
        temp_out   = os.path.join(safe_output_dir, f"temp_{slope_file}")
        fill_nodata_fixed_value(slope_path, temp_out, fill_value_slope)
        shutil.move(temp_out, slope_path)
        slope_out = slope_path

    return subs_out, bathy_out, slope_out