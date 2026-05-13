"""
skema.preprocessing.band_extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extracts Sentinel-2 bands from .SAFE format into multi-band GeoTIFF files,
applying processing-baseline-dependent offset correction.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn


def _find_band_files(bands: list, root_dir: str, console: Console) -> list:
    """Walk *root_dir* and return one JP2 path per band (None if missing)."""
    band_files: dict = {}
    for root, _dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".jp2"):
                for band in bands:
                    if band in fname and band not in band_files:
                        band_files[band] = os.path.join(root, fname)
    for band in bands:
        if band not in band_files:
            console.print(f"[yellow]Warning: {band}.jp2 not found in {root_dir}[/yellow]")
    return [band_files.get(b) for b in bands]


def _get_processing_baseline(safe_dir: str, console: Console) -> float | None:
    """Parse PROCESSING_BASELINE from the MTD_MSI*.xml metadata file."""
    xml_path = None
    for root, _dirs, files in os.walk(safe_dir):
        for f in files:
            if f.startswith("MTD_MSI") and f.endswith(".xml"):
                xml_path = os.path.join(root, f)
                break
        if xml_path:
            break
    if not xml_path:
        console.print(f"[yellow]No MTD_MSIL2A.xml found in {safe_dir}[/yellow]")
        return None
    tree = ET.parse(xml_path)
    root_elem = tree.getroot()
    pb = root_elem.findtext(".//PROCESSING_BASELINE")
    return float(pb) if pb else None


def _write_multiband_geotiff(output_path: str, band_paths: list, shift: int, description: str) -> None:
    """Stack *band_paths* into a single GeoTIFF, subtracting *shift* from each band."""
    with rasterio.open(band_paths[0]) as src:
        meta = src.meta.copy()
        meta.update({"count": len(band_paths), "dtype": "uint16", "driver": "GTiff"})

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(f"[cyan]{description}", total=len(band_paths))
        with rasterio.open(output_path, "w", **meta) as dst:
            for i, bp in enumerate(band_paths):
                with rasterio.open(bp) as bsrc:
                    arr = bsrc.read(1).astype(np.int32) - shift
                    arr = np.clip(arr, 0, None).astype(np.uint16)
                    dst.write(arr, i + 1)
                progress.advance(task)


def extract_bands_to_geotiffs(safe_dir: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Extract 10m (B2/B3/B4/B8) and 20m (B5/B6/B7/B8A/B11/B12) bands from a .SAFE
    directory and write them as two multi-band GeoTIFF files.

    Returns
    -------
    (path_10m, path_20m) — either may be None on failure.
    """
    console = Console()
    product_id = os.path.basename(safe_dir).replace(".SAFE", "")

    bands_10m = ["B02", "B03", "B04", "B08"]
    bands_20m = ["B05", "B06", "B07", "B8A", "B11", "B12"]

    with console.status("[cyan]Locating Sentinel-2 band files..."):
        paths_10m = _find_band_files(bands_10m, safe_dir, console)
        paths_20m = _find_band_files(bands_20m, safe_dir, console)

    if None in paths_10m:
        console.print(f"[red]Missing some 10m bands in {safe_dir}, skipping...[/red]")
        return None, None

    pb = _get_processing_baseline(safe_dir, console)
    shift = 1000 if pb and pb >= 4.0 else 0

    out_10m = os.path.join(output_dir, f"{product_id}_B2B3B4B8.tif")
    _write_multiband_geotiff(out_10m, paths_10m, shift, "Extracting and stacking 10m bands...")

    out_20m = None
    if None not in paths_20m:
        out_20m = os.path.join(output_dir, f"{product_id}_B5B6B7B8A_B11B12.tif")
        _write_multiband_geotiff(out_20m, paths_20m, shift, "Extracting and stacking 20m bands...")

    console.print(f"[green]✓[/green] Sentinel-2 band extraction complete.")
    return out_10m, out_20m