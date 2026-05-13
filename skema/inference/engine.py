"""
skema.inference.engine
~~~~~~~~~~~~~~~~~~~~~~~
DatasetInference — loads a preprocessed scene directory, splits it into
overlapping tiles, runs the segmentation model(s), and stitches results
back using a halo-weighted averaging scheme.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from skema.masking import apply_depth_mask, apply_exclusion_zones
from skema.preprocessing.normalization import normalize_tile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_weight_map(tile_size: int, halo_size: int) -> np.ndarray:
    """
    Return a (tile_size, tile_size) float32 weight map that fades linearly
    from 0 at the tile edge to 1 at the halo boundary.
    """
    wm = np.ones((tile_size, tile_size), dtype=np.float32)
    for i in range(halo_size):
        fade = (i + 1) / halo_size
        wm[i, :]              = fade
        wm[tile_size - 1 - i, :] = fade
        wm[:, i]              = np.minimum(wm[:, i], fade)
        wm[:, tile_size - 1 - i] = np.minimum(wm[:, tile_size - 1 - i], fade)
    return wm


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DatasetInference:
    """
    Tile-based inference over a single preprocessed Sentinel-2 scene directory.

    The directory must contain:
      * ``*_B2B3B4B8.tif``     — 10 m bands (B2/B3/B4/B8)
      * ``*_B5B6B7B8A_B11B12.tif`` — 20 m bands (B5-B12 subset)
      * ``*_Substrate.tif``, ``*_Bathymetry.tif``, ``*_Slope.tif``
        — only required for ``model_full`` / ``model_ensemble``

    Parameters
    ----------
    main_directory : str
        Path to the preprocessed scene folder.
    model : SegModel | tuple[SegModel, SegModel]
        Loaded model(s).  A tuple is expected for ``model_ensemble``.
    model_type : str
        ``"model_full"``, ``"model_s2bandsandindices_only"``, or ``"model_ensemble"``.
    mean_per_channel, std_per_channel : list[float]
        Per-channel normalisation statistics.
    tile_size : int
        Spatial size of each inference tile (pixels).
    overlap : float
        Fractional overlap between adjacent tiles [0, 1).
    halo_size : int
        Width of the fade zone used by the weight map.
    padding_mode : str
        Padding strategy for border tiles (passed to ``np.pad``).
    """

    VALID_MODEL_TYPES = {"model_full", "model_s2bandsandindices_only", "model_ensemble"}

    def __init__(
        self,
        main_directory: str,
        model,
        model_type: str = "model_full",
        mean_per_channel: list | None = None,
        std_per_channel:  list | None = None,
        tile_size: int  = 512,
        overlap: float  = 0.5,
        halo_size: int  = 64,
        padding_mode: str = "reflect",
    ):
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Must be one of {self.VALID_MODEL_TYPES}."
            )

        self.main_directory   = main_directory
        self.tile_size        = tile_size
        self.overlap          = overlap
        self.model_type       = model_type
        self.mean_per_channel = mean_per_channel
        self.std_per_channel  = std_per_channel
        self.halo_size        = halo_size
        self.padding_mode     = padding_mode

        if model_type == "model_ensemble":
            self.model_full = model[0].to(DEVICE)
            self.model_s2   = model[1].to(DEVICE)
            self.model      = None
        else:
            self.model = model.to(DEVICE)

        self.weight_map = create_weight_map(tile_size, halo_size)

        self.image, self.metadata = self._load_image()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _get_file_paths(self) -> tuple:
        d = self.main_directory
        if self.model_type in ("model_full", "model_ensemble"):
            patterns = [
                "*_B2B3B4B8.tif",
                "*_B5B6B7B8A_B11B12.tif",
                "*_Substrate.tif",
                "*_Bathymetry.tif",
                "*_Slope.tif",
            ]
        else:
            patterns = ["*_B2B3B4B8.tif", "*_B5B6B7B8A_B11B12.tif"]

        paths = []
        for pat in patterns:
            matches = glob.glob(os.path.join(d, pat))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one file matching '{pat}' in {d}, "
                    f"found {len(matches)}."
                )
            paths.append(matches[0])
        return tuple(paths)

    def _load_image(self) -> tuple[np.ndarray, dict]:
        """Load bands and compute spectral indices into a single HWC array."""
        paths = self._get_file_paths()
        path1 = paths[0]
        path2 = paths[1]

        with rasterio.open(path1) as src1:
            image1   = src1.read([1, 2, 3, 4])
            image1   = np.transpose(image1, (1, 2, 0)).astype(np.float32)
            metadata = src1.meta

        with rasterio.open(path2) as src2:
            image2 = src2.read(
                indexes=[1],
                out_shape=(1, image1.shape[0], image1.shape[1]),
                resampling=Resampling.nearest,
            )
            image2 = np.transpose(image2, (1, 2, 0)).astype(np.float32)

        if self.model_type in ("model_full", "model_ensemble"):
            path_subs  = paths[2]
            path_bathy = paths[3]
            path_slope = paths[4]

            with rasterio.open(path_subs) as src:
                substrate = src.read(1).astype(np.float32)[:, :, np.newaxis]
            with rasterio.open(path_bathy) as src:
                bathymetry = src.read(1).astype(np.float32)[:, :, np.newaxis]
            with rasterio.open(path_slope) as src:
                slope = src.read(1).astype(np.float32)[:, :, np.newaxis]

            image = np.empty((image1.shape[0], image1.shape[1], 13), dtype=np.float32)
            image[:, :, 0:4] = image1
            image[:, :, 4]   = image2[:, :, 0]
            image[:, :, 5]   = substrate[:, :, 0]
            image[:, :, 6]   = bathymetry[:, :, 0]
            image[:, :, 7]   = slope[:, :, 0]
        else:
            image = np.empty((image1.shape[0], image1.shape[1], 10), dtype=np.float32)
            image[:, :, 0:4] = image1
            image[:, :, 4]   = image2[:, :, 0]

        self.image = image  # temporarily assign so _compute_indices can write into it
        self._compute_indices()
        return self.image, metadata

    def _compute_indices(self) -> None:
        """Write spectral indices directly into self.image."""
        green = self.image[:, :, 1]
        red   = self.image[:, :, 2]
        nir   = self.image[:, :, 3]
        re    = self.image[:, :, 4]
        eps   = 1e-10

        if self.model_type in ("model_full", "model_ensemble"):
            # Channels 8-12
            self.image[:, :, 8]  = (nir - red)   / (nir + red   + eps)        # NDVI
            self.image[:, :, 9]  = (green - nir)  / (green + nir + eps)        # NDWI
            self.image[:, :, 10] = (nir - green)  / (nir + green + eps)        # GNDVI
            self.image[:, :, 11] = np.where(green < 1e-4, 20.0, nir / (green + eps) - 1)  # CIG
            self.image[:, :, 12] = (re - red)     / (re  + red   + eps)        # NDVIre
        else:
            # Channels 5-9
            self.image[:, :, 5] = (nir - red)    / (nir + red   + eps)
            self.image[:, :, 6] = (green - nir)  / (green + nir + eps)
            self.image[:, :, 7] = (nir - green)  / (nir + green + eps)
            self.image[:, :, 8] = np.where(green < 1e-4, 20.0, nir / (green + eps) - 1)
            self.image[:, :, 9] = (re - red)     / (re  + red   + eps)

    # ------------------------------------------------------------------
    # Tiling
    # ------------------------------------------------------------------

    def generate_tiles(self, image: np.ndarray):
        """Yield (tile, (row_start, col_start)) for all grid positions."""
        h, w, _ = image.shape
        step    = int(self.tile_size * (1 - self.overlap))

        tiles_y = 1 if h <= self.tile_size else (h - self.tile_size + step - 1) // step + 1
        tiles_x = 1 if w <= self.tile_size else (w - self.tile_size + step - 1) // step + 1

        for y in range(tiles_y):
            for x in range(tiles_x):
                i = y * step
                j = x * step
                tile = image[i:min(i + self.tile_size, h), j:min(j + self.tile_size, w)]

                ah, aw = tile.shape[:2]
                if ah < self.tile_size or aw < self.tile_size:
                    tile = np.pad(
                        tile,
                        ((0, self.tile_size - ah), (0, self.tile_size - aw), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                if self.mean_per_channel and self.std_per_channel:
                    tile = normalize_tile(tile, self.mean_per_channel, self.std_per_channel)

                yield tile, (i, j)

    # ------------------------------------------------------------------
    # Batch processing (weighted halo)
    # ------------------------------------------------------------------

    def _process_batch(self, tiles, coords, predictions, weight_accumulator) -> None:
        batch = torch.cat(tiles, dim=0).to(DEVICE)
        out   = self.model(batch)
        probs = out.squeeze(1).sigmoid().cpu().numpy() if out.shape[1] == 1 \
                else torch.softmax(out, dim=1).cpu().numpy()

        for pred, (i, j) in zip(probs, coords):
            eh = min(self.tile_size, predictions.shape[0] - i)
            ew = min(self.tile_size, predictions.shape[1] - j)
            predictions[i:i+eh, j:j+ew]       += pred[:eh, :ew] * self.weight_map[:eh, :ew]
            weight_accumulator[i:i+eh, j:j+ew] += self.weight_map[:eh, :ew]

    def _process_batch_ensemble(self, tiles_full, tiles_s2, coords, predictions) -> None:
        logits_full = self.model_full(torch.cat(tiles_full, dim=0).to(DEVICE))
        logits_s2   = self.model_s2(torch.cat(tiles_s2,   dim=0).to(DEVICE))
        out = ((logits_full + logits_s2) / 2.0).squeeze(1)

        for pred, (i, j) in zip(out.cpu().numpy(), coords):
            eh = min(self.tile_size, predictions.shape[0] - i)
            ew = min(self.tile_size, predictions.shape[1] - j)
            predictions[i:i+eh, j:j+ew] = np.maximum(
                predictions[i:i+eh, j:j+ew],
                (pred > 0.5).astype(np.uint8)[:eh, :ew],
            )

    # ------------------------------------------------------------------
    # Main inference loop
    # ------------------------------------------------------------------

    def run_model_on_tiles(self, batch_size: int = 8) -> np.ndarray:
        """
        Run inference over the full scene using halo-weighted tile stitching.

        Returns
        -------
        np.ndarray
            Binary uint8 prediction array, shape (H, W).
        """
        console = Console()

        if self.model_type == "model_ensemble":
            self.model_full.eval()
            self.model_s2.eval()
        else:
            self.model.eval()

        predictions        = np.zeros(self.image.shape[:2], dtype=np.float32)
        weight_accumulator = np.zeros(self.image.shape[:2], dtype=np.float32)

        tile_count    = sum(1 for _ in self.generate_tiles(self.image))
        tile_gen      = self.generate_tiles(self.image)
        batch_tiles   = []
        batch_coords  = []
        tiles_s2_buf  = []
        tiles_processed = 0

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(), TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing ...", total=tile_count)

            with torch.no_grad():
                for tile, (i, j) in tile_gen:
                    t = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float()
                    batch_tiles.append(t)
                    batch_coords.append((i, j))

                    if self.model_type == "model_ensemble":
                        tile_s2 = np.concatenate([tile[:, :, :5], tile[:, :, 8:13]], axis=2)
                        tiles_s2_buf.append(
                            torch.tensor(tile_s2).permute(2, 0, 1).unsqueeze(0).float()
                        )

                    if len(batch_tiles) == batch_size:
                        if self.model_type == "model_ensemble":
                            self._process_batch_ensemble(batch_tiles, tiles_s2_buf, batch_coords, predictions)
                            tiles_s2_buf.clear()
                        else:
                            self._process_batch(batch_tiles, batch_coords, predictions, weight_accumulator)
                        tiles_processed += len(batch_tiles)
                        progress.update(task, completed=tiles_processed)
                        batch_tiles.clear()
                        batch_coords.clear()

                if batch_tiles:
                    if self.model_type == "model_ensemble":
                        self._process_batch_ensemble(batch_tiles, tiles_s2_buf, batch_coords, predictions)
                    else:
                        self._process_batch(batch_tiles, batch_coords, predictions, weight_accumulator)
                    tiles_processed += len(batch_tiles)
                    progress.update(task, completed=tiles_processed)

        # Finalise non-ensemble (weighted average → binary)
        if self.model_type != "model_ensemble":
            weight_accumulator = np.where(weight_accumulator == 0, 1, weight_accumulator)
            predictions        = (predictions / weight_accumulator > 0.5).astype(np.uint8)

        # Bathymetry filter (model_full already has bathy channel)
        if self.model_type in ("model_full", "model_ensemble"):
            bathy = self.image[:, :, 6]
            predictions[(bathy < -100) | (bathy > 20)] = 0

        # Post-inference spatial filters
        predictions = apply_exclusion_zones(predictions, self.metadata)
        if self.model_type == "model_s2bandsandindices_only":
            predictions = apply_depth_mask(predictions, self.metadata)

        console.print(f"[green]✓[/green] Processing complete.")
        return predictions

    # ------------------------------------------------------------------
    # Output saving
    # ------------------------------------------------------------------

    def save_output(self, predictions: np.ndarray, output_path: str) -> None:
        """Save *predictions* as a single-band uint8 GeoTIFF."""
        meta = self.metadata.copy()
        meta.update({"driver": "GTiff", "dtype": "uint8", "count": 1, "compress": "lzw"})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(predictions.astype(np.uint8), 1)