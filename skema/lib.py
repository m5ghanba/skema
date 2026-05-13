"""
skema.lib
~~~~~~~~~
High-level orchestration layer.  Provides the public ``segment()`` and
``create_mosaic()`` entry points that are called by the CLI.

This module is deliberately thin: all domain logic lives in the
sub-packages (preprocessing, model, inference, masking, postprocessing).
"""

from __future__ import annotations

import os

import rasterio
from rich.console import Console

from skema.inference.engine import DatasetInference
from skema.masking import apply_eelgrass_mask
from skema.model.loader import load_model
from skema.postprocessing.mosaic import create_mosaic  # re-export for CLI
from skema.preprocessing.static_layers import (
    apply_fill_nodata_single,
    check_required_static_files,
    merge_substrate_files_single,
    warp_bathy_and_subs,
)
from skema.preprocessing.band_extraction import extract_bands_to_geotiffs

__all__ = ["segment", "create_mosaic"]


def segment(
    input_dir: str,
    output_filename: str,
    mean_per_channel: list,
    std_per_channel: list,
    model_type: str = "model_full",
    soft_substrate_masking: bool = False,
    use_bops_substrate: bool = False,
    eelgrass_masking: bool = False,
) -> None:
    """
    Perform semantic segmentation inference on a Sentinel-2 scene.

    Parameters
    ----------
    input_dir : str
        Path to a .SAFE folder, **or** a preprocessed output directory that
        already contains the required band TIFFs.
    output_filename : str
        Filename for the saved prediction GeoTIFF.
    mean_per_channel, std_per_channel : list[float]
        Per-channel normalisation statistics matching the training set.
    model_type : str
        ``"model_full"``, ``"model_s2bandsandindices_only"``, or ``"model_ensemble"``.
    soft_substrate_masking : bool
        If True, zero out kelp on substrate classes 3/4 and save a
        ``*_masked.tif`` alongside the main output.
    use_bops_substrate : bool
        If True, use BoPs substrate files / weights; otherwise RF substrate.
    eelgrass_masking : bool
        If True, zero out kelp within BCMCA eelgrass polygons and save a
        ``*_masked.tif``.  Compatible with all model types.
    """
    console = Console()

    # ── 1. Load model ─────────────────────────────────────────────────
    model = load_model(model_type, use_bops_substrate=use_bops_substrate)

    # ── 2. Preprocess .SAFE directory (if needed) ────────────────────
    if input_dir.upper().endswith(".SAFE") and os.path.isdir(input_dir):
        safe_basename = os.path.basename(input_dir).replace(".SAFE", "")
        parent_dir    = os.path.dirname(input_dir)
        output_folder = os.path.join(parent_dir, safe_basename)
        os.makedirs(output_folder, exist_ok=True)

        # 2a. Extract bands
        b2348_file    = os.path.join(output_folder, f"{safe_basename}_B2B3B4B8.tif")
        b5678a1112    = os.path.join(output_folder, f"{safe_basename}_B5B6B7B8A_B11B12.tif")

        if os.path.exists(b2348_file) and os.path.exists(b5678a1112):
            console.print("[yellow]Band TIFFs already exist, skipping extraction.[/yellow]")
        else:
            b2348_file, b5678a1112 = extract_bands_to_geotiffs(input_dir, output_folder)
            if not b2348_file:
                raise RuntimeError(f"Failed to extract bands for {input_dir}")

        # 2b. Static layers (substrate, bathy, slope) for model_full / ensemble
        needs_static = (
            model_type in ("model_full", "model_ensemble")
            or soft_substrate_masking
        )
        if needs_static:
            if soft_substrate_masking and model_type not in ("model_full", "model_ensemble"):
                console.print(
                    "[yellow]Note:[/yellow] Soft substrate masking requires deriving "
                    "bathymetry, substrate, and slope layers even though only substrate "
                    "is ultimately used for masking."
                )

            missing = check_required_static_files(use_bops_substrate)
            if missing:
                raise FileNotFoundError(
                    "model_full requires bathymetry/substrate/slope static files, "
                    "but these are missing:\n"
                    + "\n".join(f"  - {f}" for f in missing)
                    + "\n\nPlace them in the static folder or use "
                    "--model-type model_s2bandsandindices_only."
                )

            warp_bathy_and_subs(parent_dir, safe_basename, use_bops_substrate=use_bops_substrate)
            merge_substrate_files_single(output_folder, use_bops_substrate=use_bops_substrate)
            apply_fill_nodata_single(output_folder)

        input_dir = output_folder

    # ── 3. Run inference ─────────────────────────────────────────────
    dataset = DatasetInference(
        main_directory=input_dir,
        model=model,
        model_type=model_type,
        mean_per_channel=mean_per_channel,
        std_per_channel=std_per_channel,
        tile_size=512,
        overlap=0.5,
        halo_size=64,
        padding_mode="reflect",
    )

    predictions = dataset.run_model_on_tiles(batch_size=8)
    output_path = os.path.join(input_dir, output_filename)
    dataset.save_output(predictions, output_path)
    console.print(
        f"[green]✓[/green] Kelp classification map saved to [bold]{output_path}[/bold]."
    )

    # ── 4. Optional masking ──────────────────────────────────────────
    if soft_substrate_masking or eelgrass_masking:
        masked = predictions.copy()

        if soft_substrate_masking:
            subs_candidates = [f for f in os.listdir(input_dir) if f.endswith("_Substrate.tif")]
            if not subs_candidates:
                console.print(
                    "[yellow]Warning: _Substrate.tif not found; soft substrate masking skipped.[/yellow]"
                )
            else:
                subs_path = os.path.join(input_dir, subs_candidates[0])
                with rasterio.open(subs_path) as src:
                    substrate = src.read(1)
                masked[(masked == 1) & ((substrate == 3) | (substrate == 4))] = 0

        if eelgrass_masking:
            with rasterio.open(output_path) as pred_src:
                pred_crs       = pred_src.crs
                pred_transform = pred_src.transform
                pred_shape     = (pred_src.height, pred_src.width)
            masked = apply_eelgrass_mask(masked, pred_crs, pred_transform, pred_shape)

        stem = output_filename[:-4] if output_filename.lower().endswith(".tif") else output_filename
        masked_path = os.path.join(input_dir, f"{stem}_masked.tif")
        dataset.save_output(masked, masked_path)
        console.print(
            f"[green]✓[/green] Masked kelp map saved to [bold]{masked_path}[/bold]."
        )