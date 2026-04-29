__version__ = "0.3.2"

import rasterio

import numpy as np 
import glob
import torch


from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from sklearn.model_selection import train_test_split

import os
import albumentations as A


from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


# to save the results
import urllib.request


# preprocessing
import xml.etree.ElementTree as ET
from importlib.resources import files

from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds as rasterio_from_bounds
from rich.console import Console


from rasterio.enums import Resampling

from rasterio.warp import transform as warp_transform
from rasterio.features import rasterize
from rasterio.crs import CRS

import geopandas as gpd
from shapely.geometry import box

# print(torch.__version__)
# print(torch.version.cuda)  # Should print the version of CUDA PyTorch is using
# print("cuDNN version:", torch.backends.cudnn.version())
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(DEVICE)


# Exclusion zones in BC Albers (EPSG:3005) coordinates.
# Pixels inside these polygons will be forced to 0 (no kelp)
# after model inference and any other filters have been applied.
EXCLUSION_ZONES_BC_ALBERS = [
    # Polygon 0
    [(1183501.336164263, 387958.1805347335), (1179689.5750786413, 387604.3649198598),
    (1175844.429108302, 397510.62208001973), (1176353.789889322, 403446.69221630145),
    (1179007.6559680663, 406428.1457283077), (1188805.981274077, 404611.72869927227),
    (1188676.1992809451, 394397.6341887866), (1183501.336164263, 387958.1805347335)],

    # Polygon 1
    [(1170668.8573045777, 373853.681240315), (1169897.4363307026, 374170.5885593728),
    (1169483.3480854142, 374223.54157530714), (1169178.7020553653, 373979.87205611233),
    (1168599.1976390644, 373836.4753850734), (1167731.8887724597, 375914.5188937718),
    (1171716.9212878437, 378593.325610006), (1175580.7775909472, 378466.1798169448),
    (1177273.351640993, 376738.88923632784), (1176868.1381737296, 374623.5971600997),
    (1172437.1490685777, 374493.5770180205), (1172357.1044212494, 374476.10430309654),
    (1170668.8573045777, 373853.681240315)],

    # Polygon 2
    [(1228964.7309115161, 451092.07903579326), (1223152.6305408822, 443516.78776073956),
    (1215084.7209121727, 442168.60036443715), (1206831.4246042732, 446616.18640549347),
    (1208095.815024607, 452000.8818810173), (1213258.1648014348, 455942.50566811836),
    (1224093.4862050398, 457939.7434576139), (1229258.5946341362, 457528.7563443975),
    (1229286.8141567863, 457503.3116592518), (1228964.7309115161, 451092.07903579326)],

    # Polygon 3
    [(856802.0185995427, 621245.5029450883), (842039.2872862323, 632055.1333733171),
    (831194.157211808, 639466.6577294703), (834947.626676494, 643360.9361425375),
    (840781.0485671109, 645479.2795775908), (840922.7823112048, 645856.9368079947),
    (841206.9523380621, 645946.465068656), (841564.4506481199, 645715.2591275071),
    (854592.3776990275, 650254.1053568409), (870072.1398371714, 647770.477831984),
    (882235.8444173654, 642481.8691032815), (892208.603557862, 634089.8377137645),
    (915632.1675251884, 612077.1380058321), (1021449.5475117693, 579810.5249577082),
    (1046491.5169638512, 556207.0959516298), (1106542.3911403832, 466339.40439404844),
    (1170098.0275926304, 433232.85046937276), (1176280.6478652935, 417669.26819404116),
    (1173814.837629517, 387240.68444629863), (1131095.4551650733, 391333.4889558901),
    (1079117.6423609706, 412720.8862061742), (1089661.7785248803, 439298.23354545),
    (1036748.6601571619, 460769.06152324163), (1028725.9454698666, 470091.30529292807),
    (1013542.2214699964, 486675.5784639747), (986028.0000043461, 496156.6883248521),
    (977694.9343830012, 527456.324560537), (956031.3854997907, 535983.8113357764),
    (945799.313417558, 551892.2785007206), (922284.9782529981, 571677.3165105806),
    (892621.4460918504, 580031.5481154523), (877880.0213475373, 592268.2741653069),
    (892906.24241468, 611785.2963167346), (897494.0864983617, 620955.4374759638),
    (877324.856407201, 626648.2668424319), (873020.3601312948, 619752.525972377),
    (856802.0185995427, 621245.5029450883)],

    # Polygon 4
    [(1014845.7124855223, 455034.97863539215), (1016227.9140126681, 455780.1345114845),
    (1018076.8755909825, 454482.08266086684), (1019439.2577476983, 453868.1919296341),
    (1019004.3765252775, 452885.3760781294), (1017050.4963084502, 453414.38220678456),
    (1014845.7124855223, 455034.97863539215)],

    # Polygon 5
    [(1006657.0918689288, 461025.20757650543), (1006521.1043602177, 460311.2606926271),
    (1005981.2035429816, 459889.3528698707), (1005664.8964668913, 459980.4002225398),
    (1005295.3413862621, 460295.3016608447), (1004679.4809445102, 461313.0248246574),
    (1005944.6172149541, 461543.08656319656), (1006615.2680217146, 461454.77175142535),
    (1006657.0918689288, 461025.20757650543)],

    # Polygon 6
    [(1004892.7655447604, 463969.19314196403), (1003895.3580308687, 464664.58439990046),
    (1003881.16533169, 465258.2192997604), (1004466.1992165355, 465625.6773580631),
    (1004940.1315956784, 465351.0860325925), (1004892.7655447604, 463969.19314196403)],

    # Polygon 7
    [(1229187.3451448006, 335531.3397770725), (1231184.1040226896, 318941.63944669964),
    (1218371.197868512, 283884.1994365883), (1192815.241561105, 279626.32953011926),
    (1150832.4703476252, 272056.7688337726), (1136525.0551379852, 303033.03016037337),
    (1135274.8086104176, 325431.8620125285), (1142117.1914919897, 336050.1841111155),
    (1148986.5278308126, 346106.4383125802), (1162729.1674537219, 348155.351903564),
    (1229187.3451448006, 335531.3397770725)],
]


def apply_exclusion_zones(predictions, metadata):
    """
    Burn zeros into predictions for all polygons in EXCLUSION_ZONES_BC_ALBERS.

    Reprojects BC Albers (EPSG:3005) polygon coordinates to the image CRS
    using rasterio.warp.transform, then rasterizes using rasterio.features.
    No shapely or pyproj required.

    Args:
        predictions (np.ndarray): 2-D uint8 array of shape (H, W).
        metadata (dict): Rasterio metadata dict with 'crs' and 'transform'.

    Returns:
        np.ndarray: predictions with exclusion zone pixels set to 0.
    """
    if not EXCLUSION_ZONES_BC_ALBERS:
        return predictions

    src_crs = CRS.from_epsg(3005)
    dst_crs = metadata['crs']
    img_transform = metadata['transform']
    h, w = predictions.shape

    # Build a single burn mask for all polygons at once, then apply in one step
    shapes = []
    for coords in EXCLUSION_ZONES_BC_ALBERS:
        if dst_crs.to_epsg() == 3005:
            # Already in BC Albers — use coords directly
            projected = coords
        else:
            # Reproject: split into xs and ys, transform, zip back
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            xs_proj, ys_proj = warp_transform(src_crs, dst_crs, xs, ys)
            projected = list(zip(xs_proj, ys_proj))

        # GeoJSON-style geometry — rasterize accepts this natively
        geom = {
            'type': 'Polygon',
            'coordinates': [projected]   # outer ring only; no holes
        }
        shapes.append((geom, 1))

    # Rasterize all polygons onto a single mask in one call (fast C path)
    exclusion_mask = rasterize(
        shapes=shapes,
        out_shape=(h, w),
        transform=img_transform,
        fill=0,
        dtype=np.uint8,
    )

    # Zero out excluded pixels
    predictions[exclusion_mask == 1] = 0

    return predictions



def _load_valid_depth_zone():
    """Load the valid-depth-zone GeoPackage shipped with the package."""
    try:
        path = str(files("skema.static.masks").joinpath("valid_depth_zone.gpkg"))
        gdf = gpd.read_file(path)
        if gdf.crs is None or gdf.empty:
            print("[WARNING] valid_depth_zone.gpkg loaded but is empty or has no CRS.")
            return None
        # print(f"[INFO] Loaded depth zone mask: {len(gdf)} feature(s), CRS={gdf.crs}")
        return gdf
    except Exception as e:
        print(f"[WARNING] Could not load depth zone mask: {e}")
        return None

_VALID_DEPTH_GDF = _load_valid_depth_zone()

_BATHY_EXTENT_BC_ALBERS = box(
    474232.8234999999986030,
    314708.7365999999456108,
    1304032.8234999999403954,
    1250588.7365999999456108,
)

def apply_depth_mask(predictions, metadata):
    """
    Zero out predictions that fall outside the valid bathymetry depth zone
    (-100m to 20m), BUT only within the extent where bathymetry data exists.
    Pixels outside the bathymetry extent are left untouched.
    """
    if _VALID_DEPTH_GDF is None:
        return predictions

    scene_crs = metadata["crs"]
    img_transform = metadata["transform"]
    height, width = predictions.shape

    # ── 1. Build scene footprint in BC Albers ────────────────────────────────
    src_crs = CRS.from_user_input(scene_crs)
    albers  = CRS.from_epsg(3005)

    corners_img = [
        img_transform * (0, 0),
        img_transform * (width, 0),
        img_transform * (0, height),
        img_transform * (width, height),
    ]
    xs_scene = [c[0] for c in corners_img]
    ys_scene = [c[1] for c in corners_img]

    if src_crs.to_epsg() != 3005:
        xs_albers, ys_albers = warp_transform(src_crs, albers, xs_scene, ys_scene)
    else:
        xs_albers, ys_albers = xs_scene, ys_scene

    scene_bbox_albers = box(
        min(xs_albers), min(ys_albers),
        max(xs_albers), max(ys_albers),
    )

    # ── 2. Check overlap with bathymetry extent ───────────────────────────────
    if not scene_bbox_albers.intersects(_BATHY_EXTENT_BC_ALBERS):
        return predictions  # scene outside bathy coverage — leave untouched

    # ── 3. Clip valid-zone GDF to scene bbox ─────────────────────────────────
    scene_bbox_gdf = gpd.GeoDataFrame(geometry=[scene_bbox_albers], crs="EPSG:3005")
    gdf_clipped = gpd.clip(_VALID_DEPTH_GDF, scene_bbox_gdf)

    if gdf_clipped.empty:
        # Inside bathy extent but no valid-depth polygons — zero everything
        predictions[:] = 0
        return predictions

    # ── 4. Reproject clipped GDF to scene CRS ────────────────────────────────
    gdf_reprojected = gdf_clipped.to_crs(scene_crs)

    # ── 5. Rasterize bathy extent (which pixels are "in scope") ──────────────
    extent_gdf = gpd.GeoDataFrame(
        geometry=[_BATHY_EXTENT_BC_ALBERS], crs="EPSG:3005"
    ).to_crs(scene_crs)

    extent_mask = rasterize(
        [(geom, 1) for geom in extent_gdf.geometry],
        out_shape=(height, width),
        transform=img_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    # ── 6. Rasterize valid depth zone ─────────────────────────────────────────
    valid_mask = rasterize(
        [(geom, 1) for geom in gdf_reprojected.geometry],
        out_shape=(height, width),
        transform=img_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    # ── 7. Zero pixels inside bathy extent but outside valid depth zone ───────
    predictions[(extent_mask == 1) & (valid_mask == 0)] = 0

    return predictions




def extract_bands_to_geotiffs(safe_dir, output_dir):
    """
    Extracts and stacks Sentinel-2 bands from .SAFE format into multi-band GeoTIFF files, 
    applying baseline-dependent offset correction (10m and 20m resolution bands separately).
    """
    
    console = Console()
    product_id = os.path.basename(safe_dir).replace(".SAFE", "")

    bands_10m = ['B02', 'B03', 'B04', 'B08']
    bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

    def find_band_files(bands, root_dir):
        band_files = {}
        for root, dirs, files_ in os.walk(root_dir):
            for file in files_:
                if file.endswith(".jp2"):
                    for band in bands:
                        if band in file and band not in band_files:
                            band_files[band] = os.path.join(root, file)
        for band in bands:
            if band not in band_files:
                console.print(f"[yellow]Warning: {band}.jp2 not found in {safe_dir}[/yellow]")
        return [band_files.get(b) for b in bands]

    with console.status("[cyan]Locating Sentinel-2 band files..."):
        band_paths_10m = find_band_files(bands_10m, safe_dir)
        band_paths_20m = find_band_files(bands_20m, safe_dir)

    if None in band_paths_10m:
        console.print(f"[red]Missing some 10m bands in {safe_dir}, skipping...[/red]")
        return None, None

    def get_processing_baseline(safe_dir):
        xml_path = None
        for root, dirs, files_ in os.walk(safe_dir):
            for f in files_:
                if f.startswith("MTD_MSI") and f.endswith(".xml"):
                    xml_path = os.path.join(root, f)
                    break
            if xml_path:
                break
        if not xml_path:
            console.print(f"[yellow]No MTD_MSIL2A.xml found in {safe_dir}[/yellow]")
            return None
        tree = ET.parse(xml_path)
        root = tree.getroot()
        pb = root.findtext(".//PROCESSING_BASELINE")
        return float(pb) if pb else None

    pb = get_processing_baseline(safe_dir)
    shift = 1000 if pb and pb >= 4.0 else 0

    def write_multiband_geotiff(output_path, band_paths, description):
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

    output_10m = os.path.join(output_dir, f"{product_id}_B2B3B4B8.tif")
    write_multiband_geotiff(output_10m, band_paths_10m, "Extracting and stacking 10m bands...")

    output_20m = None
    if None not in band_paths_20m:
        output_20m = os.path.join(output_dir, f"{product_id}_B5B6B7B8A_B11B12.tif")
        write_multiband_geotiff(output_20m, band_paths_20m, "Extracting and stacking 20m bands...")

    console.print(f"[green]✓[/green] Sentinel-2 band extraction complete.")

    return output_10m, output_20m


def calculate_slope_horn(bathymetry, cell_size=20.0):
    """Vectorized Horn slope calculation for a 2D numpy array."""
    bathy = bathymetry.astype(np.float32)
    H, W = bathy.shape
    slope = np.zeros((H, W), dtype=np.float32)

    # If the chunk is too small (e.g., edge cases), just return zeros
    if H < 3 or W < 3:
        return slope

    # Extract shifted views (a-i)
    a, b, c = bathy[:-2, :-2], bathy[:-2, 1:-1], bathy[:-2, 2:]
    d, f = bathy[1:-1, :-2], bathy[1:-1, 2:]
    g, h, i = bathy[2:, :-2], bathy[2:, 1:-1], bathy[2:, 2:]

    # Mask invalid data (NaNs)
    invalid = np.isnan(a) | np.isnan(b) | np.isnan(c) | \
              np.isnan(d) | np.isnan(f) | \
              np.isnan(g) | np.isnan(h) | np.isnan(i)

    # Horn's formulas for dz/dx and dz/dy
    dz_dx = ((c + 2*f + i) - (a + 2*d + g)) / (8.0 * cell_size)
    dz_dy = ((g + 2*h + i) - (a + 2*b + c)) / (8.0 * cell_size)

    # Calculate slope in degrees
    slope_inner = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope_inner[invalid] = np.nan

    slope[1:-1, 1:-1] = slope_inner
    return slope


def calculate_slope_for_raster(input_tiff, output_tiff, block_size=2048):
    """Calculate slope from bathymetry raster using windowed processing."""
    console = Console()
    console.print(f"[cyan]Starting slope calculation for {os.path.basename(input_tiff)}...[/cyan]")
    
    from rasterio.windows import Window
    
    with rasterio.open(input_tiff) as src:
        profile = src.profile
        
        # Update profile for output: Float32, keep nodata from source
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512
        )
        
        nodata_val = src.nodata
        
        # Get cell size from the transform
        cell_size = abs(src.transform[0])  # pixel width in georeferenced units

        with rasterio.open(output_tiff, 'w', **profile) as dst:
            
            # Iterate over the grid in chunks
            for row_idx in range(0, src.height, block_size):
                for col_idx in range(0, src.width, block_size):
                    
                    # 1. Define the core window we want to write to
                    window = Window(
                        col_off=col_idx, 
                        row_off=row_idx, 
                        width=min(block_size, src.width - col_idx), 
                        height=min(block_size, src.height - row_idx)
                    )
                    
                    # 2. Define the Buffered Window (expand by 1 pixel on all sides, staying within bounds)
                    row_start = max(0, window.row_off - 1)
                    row_stop = min(src.height, window.row_off + window.height + 1)
                    col_start = max(0, window.col_off - 1)
                    col_stop = min(src.width, window.col_off + window.width + 1)
                    
                    buf_window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
                    
                    # 3. Read the buffered data
                    bathy_data = src.read(1, window=buf_window)
                    
                    if nodata_val is not None:
                        bathy_data = np.where(bathy_data == nodata_val, np.nan, bathy_data)
                    
                    # 4. Calculate the slope
                    slope_data = calculate_slope_horn(bathy_data, cell_size=cell_size)
                    
                    # 5. Crop the 1-pixel buffer back out to match our original target window
                    crop_row_start = window.row_off - row_start
                    crop_row_stop = crop_row_start + window.height
                    crop_col_start = window.col_off - col_start
                    crop_col_stop = crop_col_start + window.width
                    
                    final_slope = slope_data[crop_row_start:crop_row_stop, crop_col_start:crop_col_stop]
                    
                    # 6. Write the processed chunk directly to the output disk
                    dst.write(final_slope.astype(rasterio.float32), 1, window=window)
                
                # Optional: Print progress
                if row_idx % (block_size * 4) == 0:  # Print every 4 chunks
                    console.print(f"[cyan]Processed up to row {min(row_idx + block_size, src.height)} / {src.height}[/cyan]")
    
    console.print(f"[green]✓[/green] Slope calculation complete: {output_tiff}")


def warp_bathy_and_subs(safe_folder_root, basename, use_bops_substrate=False):
    """
    Aligns bathymetry, slope, and substrate rasters to match the CRS, resolution (10m), and extent 
    of the reference Sentinel-2 image using bilinear resampling.
    
    Uses substrate files determined by use_bops_substrate:
    - use_bops_substrate=True:  uses 4 BoPs substrate files at 10m resolution
    - use_bops_substrate=False: uses 5 regional RF substrate files at 20m resolution
    """
    console = Console()
    
    # Substrate source is set explicitly by the caller
    is_bops_scene = use_bops_substrate
    
    # Look for _B2B3B4B8.tif inside each subfolder
    for folder_name in os.listdir(safe_folder_root):
        folder_path = os.path.join(safe_folder_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        tif_file = next((f for f in os.listdir(folder_path) if f == f"{basename}_B2B3B4B8.tif"), None)
        if not tif_file:
            # console.print(f"[yellow]No reference image found in {folder_path}, skipping...[/yellow]")
            continue

        reference_tif = os.path.join(folder_path, tif_file)

        # Define static files to warp based on scene type
        if is_bops_scene:
            # BoPs scenes use 4 substrate files at 10m resolution
            input_files = {
                "Bathymetry.tif": "_Bathy.tif",
                "Slope.tif": "_Slope.tif",
                "BoPs_HG_10m.tif": "_SubsHG.tif",
                "BoPs_NCC_10m.tif": "_SubsNCC.tif",
                "BoPs_QCSSOG_10m.tif": "_SubsQCSSOG.tif",
                "BoPs_WCVI_10m.tif": "_SubsWCVI.tif",
            }
            # console.print(f"[cyan]Detected BoPs scene ({basename}). Using 10m BoPs substrate files.[/cyan]")
        else:
            # Regular scenes use 5 substrate files at 20m resolution
            input_files = {
                "Bathymetry.tif": "_Bathy.tif",
                "Slope.tif": "_Slope.tif",
                "NCC_substrate_20m.tif": "_SubsNCC.tif",
                "SOG_substrate_20m.tif": "_SubsSOG.tif",
                "WCVI_substrate_20m.tif": "_SubsWCVI.tif",
                "QCS_substrate_20m.tif": "_SubsQCS.tif",
                "HG_substrate_20m.tif": "_SubsHG.tif",
            }
            # console.print(f"[cyan]Detected regular scene ({basename}). Using 20m regional substrate files.[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]Aligning bathymetry, slope, and substrate files with Sentinel-2 image...", total=len(input_files))
            
            # Load bathy/substrate files from static directory
            for file_name, suffix in input_files.items():
                try:
                    static_path = files("skema.static.bathy_substrate").joinpath(file_name)
                    input_file_path = str(static_path)
                except Exception as e:
                    console.print(f"[red]Failed to access static file {file_name}: {e}[/red]")
                    progress.advance(task)
                    continue

                if not os.path.exists(input_file_path):
                    console.print(f"[red]Static file not found: {input_file_path}[/red]")
                    progress.advance(task)
                    continue

                output_file_path = os.path.join(folder_path, folder_name + suffix)

                with rasterio.open(reference_tif) as ref:
                    bounds = ref.bounds
                    crs = ref.crs.to_string()

                width = int((bounds.right - bounds.left) / 10)
                height = int((bounds.top - bounds.bottom) / 10)
                transform = rasterio_from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, width, height)

                with rasterio.open(input_file_path) as src:
                    out_data = np.empty((height, width), dtype=src.dtypes[0])
                    
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=out_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.bilinear
                    )
                    
                    profile = src.profile.copy()
                    profile.update({
                        'crs': crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })
                    
                    with rasterio.open(output_file_path, 'w', **profile) as dst:
                        dst.write(out_data, 1)

                progress.advance(task)
        
        console.print(f"[green]✓[/green] Alignment complete.")


def fill_nodata_fixed_value(input_file, output_file, fill_value):
    """
    Replaces NoData values in a raster with a specified fill value and removes the NoData flag.
    """

    with rasterio.open(input_file) as src:
        data = src.read(1)
        nodata_value = src.nodata
        profile = src.profile.copy()
        
        if nodata_value is not None:
            data[data == nodata_value] = fill_value
        
        profile.update(nodata=None)
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data, 1)

    # print(f"Processed and saved: {output_file}")
    os.remove(input_file)



def merge_substrate_files_single(safe_output_dir, use_bops_substrate=False):
    """
    Merges substrate rasters in a single SAFE output folder into _Subs.tif.
    
    Uses substrate files determined by use_bops_substrate:
    - use_bops_substrate=True:  merges 4 BoPs substrate files (valid values: 1-3)
    - use_bops_substrate=False: merges 5 regional RF substrate files (valid values: 1-4)
    
    For RF scenes, remaps substrate value 4 to 3 to ensure a consistent 3-class scheme.
    """
    
    console = Console()
    
    # Get the base name from B2B3B4B8.tif file
    b2348_file = next((f for f in os.listdir(safe_output_dir) if f.endswith("_B2B3B4B8.tif")), None)
    if not b2348_file:
        # console.print(f"[yellow]No reference image in {safe_output_dir}, skipping merge.[/yellow]")
        return None
    
    base_name = b2348_file.replace("_B2B3B4B8.tif", "")
    
    # Substrate source is set explicitly by the caller
    is_bops_scene = use_bops_substrate
    
    if is_bops_scene:
        # BoPs scenes use 4 substrate files
        suffixes = ["_SubsHG.tif", "_SubsNCC.tif", "_SubsQCSSOG.tif", "_SubsWCVI.tif"]
        valid_values = {1, 2, 3}
        expected_count = 4
        # console.print(f"[cyan]Merging BoPs substrate (4 files, valid values: 1-3)[/cyan]")
    else:
        # Regular scenes use 5 substrate files
        suffixes = ["_SubsNCC.tif", "_SubsSOG.tif", "_SubsWCVI.tif", "_SubsQCS.tif", "_SubsHG.tif"]
        valid_values = {1, 2, 3, 4}
        expected_count = 5
        # console.print(f"[cyan]Merging regional substrate (5 files, valid values: 1-4)[/cyan]")

    # Collect input rasters
    input_files = [os.path.join(safe_output_dir, f) for f in os.listdir(safe_output_dir) if any(f.endswith(s) for s in suffixes)]
    if len(input_files) != expected_count:
        console.print(f"[yellow]Not all substrate files found in {safe_output_dir} (found {len(input_files)}, expected {expected_count}), skipping merge.[/yellow]")
        return None

    output_file = os.path.join(safe_output_dir, f"{base_name}_Subs.tif")

    with rasterio.open(input_files[0]) as src:
        meta = src.meta.copy()
        height, width = src.shape

    merged_data = np.zeros((height, width), dtype=meta["dtype"])
    for file in input_files:
        with rasterio.open(file) as src:
            data = src.read(1)
            mask = np.isin(data, list(valid_values))
            merged_data[mask] = data[mask]
    

    meta.update(dtype=rasterio.uint8, nodata=0, compress="LZW")
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(merged_data, 1)

    # delete originals
    for file in input_files:
        try:
            os.remove(file)
        except Exception as e:
            console.print(f"[red]Error deleting {file}: {e}[/red]")

    return output_file


def apply_fill_nodata_single(safe_output_dir, fill_value_subs=0, fill_value_bathy=-2000, fill_value_slope=85):
    """
    Applies NoData filling to substrate (default: 0), bathymetry (default: -2000), and slope (default: 85) rasters 
    and renames them to final output files.
    
    Note: Slope files are already named correctly (_Slope.tif), so they are processed in-place.
    """
    subs_file = next((f for f in os.listdir(safe_output_dir) if f.endswith("_Subs.tif")), None)
    bathy_file = next((f for f in os.listdir(safe_output_dir) if f.endswith("_Bathy.tif")), None)
    slope_file = next((f for f in os.listdir(safe_output_dir) if f.endswith("_Slope.tif")), None)

    if subs_file:
        base_name = subs_file.replace("_Subs.tif", "")
        output_file = os.path.join(safe_output_dir, f"{base_name}_Substrate.tif")
        fill_nodata_fixed_value(os.path.join(safe_output_dir, subs_file), output_file, fill_value_subs)
        subs_file = output_file

    if bathy_file:
        base_name = bathy_file.replace("_Bathy.tif", "")
        output_file = os.path.join(safe_output_dir, f"{base_name}_Bathymetry.tif")
        fill_nodata_fixed_value(os.path.join(safe_output_dir, bathy_file), output_file, fill_value_bathy)
        bathy_file = output_file

    if slope_file:
        # Slope file is already correctly named, process in-place using temp file
        slope_path = os.path.join(safe_output_dir, slope_file)
        temp_output = os.path.join(safe_output_dir, f"temp_{slope_file}")
        fill_nodata_fixed_value(slope_path, temp_output, fill_value_slope)
        # Replace original with processed version
        import shutil
        shutil.move(temp_output, slope_path)
        slope_file = slope_path

    return subs_file, bathy_file, slope_file


def create_mosaic(tif_paths, output_path, target_resolution_meters=10, soft_substrate_masking=False):
    """
    Create a maximum-value mosaic from a list of kelp prediction GeoTIFFs,
    reprojected to BC Albers (EPSG:3005) at a fixed resolution.

    Overlapping pixels keep the maximum value (i.e. a pixel is kelp=1 if
    ANY contributing scene predicted kelp there).

    Args:
        tif_paths (list[str]): Paths to the per-scene prediction TIFFs.
        output_path (str): Where to save mosaic_kelp_map.tif.
        target_resolution_meters (float): Output pixel size in metres (default 10).
        soft_substrate_masking (bool): if True, also creates a mosaic from the
            substrate-masked per-scene TIFFs, saved as mosaic_kelp_map_substrate_masked.tif.
    """

    console = Console()

    # ── 1. Collect bounds of every file reprojected to BC Albers ──────────
    target_crs = 'EPSG:3005'
    all_bounds = []

    valid_paths = [p for p in tif_paths if os.path.exists(p)]
    if not valid_paths:
        console.print("[red]No valid prediction TIFFs found – mosaic skipped.[/red]")
        return

    # ── Check if tiles are in BC coast (BC Albers extent) ─────────────────
    # BC Albers approximate extent (EPSG:3005)
    bc_albers_extent = {
        'min_x': 200000,
        'max_x': 1900000,
        'min_y': 300000,
        'max_y': 1750000
    }
    
    tiles_in_bc = False
    for p in valid_paths:
        with rasterio.open(p) as src:
            if str(src.crs) == target_crs:
                bounds = src.bounds
            else:
                bounds = transform_bounds(src.crs, target_crs, *src.bounds)
            all_bounds.append(bounds)
            
            # Check if this tile overlaps with BC coast
            if (bounds[0] < bc_albers_extent['max_x'] and bounds[2] > bc_albers_extent['min_x'] and
                bounds[1] < bc_albers_extent['max_y'] and bounds[3] > bc_albers_extent['min_y']):
                tiles_in_bc = True
    
    if not tiles_in_bc:
        console.print("[yellow]Warning: Mosaic creation is designed for BC coast tiles. "
                     "The provided scenes do not appear to be in British Columbia. Mosaic creation skipped.[/yellow]")
        return

    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)

    # ── 2. Build output canvas ─────────────────────────────────────────────
    width  = int(np.ceil((max_x - min_x) / target_resolution_meters))
    height = int(np.ceil((max_y - min_y) / target_resolution_meters))
    mosaic_transform = rasterio_from_bounds(min_x, min_y, max_x, max_y, width, height)
    mosaic = np.zeros((height, width), dtype=np.uint8)

    # ── 3. Reproject & accumulate with maximum ─────────────────────────────
    console.print(f"[cyan]Building mosaic from {len(valid_paths)} scene(s)...[/cyan]")
    for p in valid_paths:
        with rasterio.open(p) as src:
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
            mosaic = np.maximum(mosaic, tile)

    # ── 4. Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=np.uint8,
        crs=target_crs,
        transform=mosaic_transform,
        compress='lzw',
    ) as dst:
        dst.write(mosaic, 1)

    console.print(f"[green]✓[/green] Mosaic saved to [bold]{output_path}[/bold].")

    # ── Soft substrate masking mosaic (optional second mosaic output) ─────
    if soft_substrate_masking:
        stem = os.path.splitext(os.path.basename(output_path))[0]
        masked_dir = os.path.dirname(output_path)
        masked_tif_paths = []
        for p in valid_paths:
            scene_stem = os.path.splitext(os.path.basename(p))[0]
            masked_candidate = os.path.join(os.path.dirname(p), f"{scene_stem}_substrate_masked.tif")
            if os.path.exists(masked_candidate):
                masked_tif_paths.append(masked_candidate)
            else:
                console.print(f"[yellow]Warning: substrate-masked file not found for {os.path.basename(p)}, skipping in masked mosaic.[/yellow]")
        if masked_tif_paths:
            masked_mosaic_path = os.path.join(masked_dir, f"{stem}_substrate_masked.tif")
            console.print(f"[cyan]Building substrate-masked mosaic from {len(masked_tif_paths)} scene(s)...[/cyan]")
            masked_mosaic = np.zeros((height, width), dtype=np.uint8)
            for p in masked_tif_paths:
                with rasterio.open(p) as src:
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
                    masked_mosaic = np.maximum(masked_mosaic, tile)
            with rasterio.open(
                masked_mosaic_path, "w",
                driver="GTiff",
                height=height, width=width,
                count=1, dtype=np.uint8,
                crs=target_crs,
                transform=mosaic_transform,
                compress="lzw",
            ) as dst:
                dst.write(masked_mosaic, 1)
            console.print(f"[green]✓[/green] Substrate-masked mosaic saved to [bold]{masked_mosaic_path}[/bold].")
        else:
            console.print("[yellow]No substrate-masked scene files found; masked mosaic skipped.[/yellow]")


def normalize_input_mean_std(image_hwc, mean_per_channel, std_per_channel, epsilon=1e-8):
    """Applies mean and std normalization to the input image (H, W, C) at once."""
    image_hwc = np.nan_to_num(image_hwc).astype(np.float32)  # Handle NaNs and ensure float type
    mean = np.array(mean_per_channel, dtype=np.float32)[np.newaxis, np.newaxis, :]
    std = np.array(std_per_channel, dtype=np.float32)[np.newaxis, np.newaxis, :]
    normalized_image = (image_hwc - mean) / (std + epsilon)
    return normalized_image

class SatelliteDataset(BaseDataset):
    CLASSES = ["water", "kelp", "land"]

    def __init__(self, image_paths, mask_paths, classes=None, augmentation=None, mean=None, std=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.mean = mean
        self.std = std
        self.calculated_mean = None
        self.calculated_std = None

        if classes is None:
            classes = self.CLASSES
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.index_calculators = {
            "ndvi": self.calculate_ndvi,
            "ndwi": self.calculate_ndwi,
            "gndvi": self.calculate_gndvi,
            "clgreen": self.calculate_chlorophyll_index_green,
            "ndvire": self.calculate_ndvi_re, #Normalized Difference of Red and Blue
            #"ndrb": self.calculate_ndrb, #Normalized Difference of Red and Blue
            #"mgvi": self.calculate_mgvi, #Modified Green Red Vegetation Index (MGVI)
            #"mpri": self.calculate_mpri, #Modified Photochemical Reflectance Index (MPRI)
            #"rgbvi": self.calculate_rgbvi, #Red Green Blue Vegetation Index (RGBVI)
            #"gli": self.calculate_gli, #Green Leaf Index (GLI)
            #"gi": self.calculate_gi, #Greenness Index (GI)
            #"br": self.calculate_blue_red, #Blue/Red
            #"exg": self.calculate_exg, #Excess of Green (ExG)
            #"vari": self.calculate_vari, #Visible Atmospherically Resistant Index (VARI)
            #"tvi": self.calculate_tvi, #Triangular Vegetation Index (TVI)
            #"rdvi": self.calculate_rdvi, #Renormalized Difference Vegetation Index (RDVI)
            #"ndreb": self.calculate_ndreb, #Normalized Difference Red-edge Blue (NDREB)
            #"evi": self.calculate_evi, #Enhanced Vegetation Index (EVI)
            #"cig": self.calculate_cig,  #Green Chlorophyll Index (CIG)
            #"blue_rededge": self.calculate_blue_rededge, #Blue/Red-edge
            #"bnir": self.calculate_blue_nir, #Blue/NIR
            #"rb": self.calculate_red_minus_blue, #R-B
            #"bndvi": self.calculate_bndvi, #Blue NDVI
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # --- 1. Read Image ---
        img_path = self.image_paths[index]
        with rasterio.open(img_path) as src_img:
            image = src_img.read([1, 2, 3, 4, 5, 11, 12]) # -> (C, H, W) = (6, H, W) 1-4:10m bands, 5-10: 20m bands resampled to 10m, 11:substrate 12: bathymetry

            # --- FIXED PREPROCESSING ---
            # # Modify bathymetry (channel index 5 in CHW format)
            # bathy_mask_gt = image[6, :, :] > 10
            # bathy_mask_lt = image[6, :, :] < -100
            # image[6, :, :][bathy_mask_gt | bathy_mask_lt] = -2000 # Combine conditions

            # # Modify substrate (channel index 4 in CHW format)
            # subs_mask = image[5, :, :] != 1
            # image[5, :, :][subs_mask] = 0
            # # --- END FIX ---

            image_hwc = np.transpose(image, (1, 2, 0)).astype(np.float32) # -> (H, W, 6)

        # --- 2. Calculate Indices ---
        indices_list = []
        for _, calculator in self.index_calculators.items():
            idx = calculator(image_hwc)
            indices_list.append(idx[..., np.newaxis])

        image_with_indices = np.concatenate([image_hwc] + indices_list, axis=-1)

        # --- 3. Read Mask ---
        mask_path = self.mask_paths[index]
        mask = self.read_and_process_mask(mask_path) # -> (H, W, num_classes)

        # --- 4. Apply Normalization ---
        if self.mean is not None and self.std is not None:
            image_with_indices = normalize_input_mean_std(image_with_indices, mean_per_channel=self.mean, std_per_channel=self.std)

        # --- 5. Apply Augmentation ---
        if self.augmentation:  # Proceeds to call whatever is stored in self.augmentation
            # Python expects self.augmentation to be a callable object (something that can be called using parentheses ()), which a function is. 
            # The image and mask variables are passed as arguments to this callable. In the case of albumentation, self.augmentation holds the
            # A.Compose object. When you call it with (image=image_with_indices, mask=mask), the A.Compose object's __call__ method  (which is what makes an
            # object callable) is executed. This method internally applies the defined horizontal and vertical flips (with their respective probabilities)
            # to the provided image and mask and returns a dictionary like {'image': augmented_image, 'mask': augmented_mask}.
            sample = self.augmentation(image=image_with_indices, mask=mask) 
            image_with_indices = sample['image']
            mask = sample['mask']

        # --- 6. Final Transpose ---
        image_final = np.transpose(image_with_indices, (2, 0, 1))
        mask_final = np.transpose(mask, (2, 0, 1))

        return image_final.astype(np.float32), mask_final.astype(np.float32)


    def read_and_process_mask(self, mask_path):
        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype(int) # -> (H, W)
            masks = [(mask == v) for v in self.class_values]
            return np.stack(masks, axis=-1).astype("float") #-> (H, W, num_classes)

    # --- Index Calculation Methods ---
    def calculate_ndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        return (nir - red) / (nir + red + 1e-10)

    def calculate_ndwi(self, image_hwc):
        green = image_hwc[..., 1]
        nir = image_hwc[..., 3]
        return (green - nir) / (green + nir + 1e-10)

    def calculate_gndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        green = image_hwc[..., 1]
        return (nir - green) / (nir + green + 1e-10)

    def calculate_chlorophyll_index_green(self, image_hwc):
        nir = image_hwc[..., 3]
        green = image_hwc[..., 1]
        return np.where(green < 1e-4, 20.0, nir / (green + 1e-10) - 1)

    def calculate_ndvi_re(self, image_hwc):
        re = image_hwc[..., 4]
        red = image_hwc[..., 2]
        return (re - red) / (re + red + 1e-10)

    def calculate_evi(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        blue = image_hwc[..., 0]
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)

    def calculate_sr(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        return nir / (red + 1e-10)

    def calculate_ndrb(self, image_hwc):
        return (image_hwc[..., 2] - image_hwc[..., 0]) / (image_hwc[..., 2] + image_hwc[..., 0] + 1e-10)

    def calculate_mgvi(self, image_hwc):
        return (image_hwc[..., 1]**2 - image_hwc[..., 2]**2) / (image_hwc[..., 1]**2 + image_hwc[..., 2]**2 + 1e-10)

    def calculate_mpri(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 2]) / (image_hwc[..., 1] + image_hwc[..., 2] + 1e-10)

    def calculate_rgbvi(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 0] * image_hwc[..., 2]) / (image_hwc[..., 1]**2 + image_hwc[..., 0] * image_hwc[..., 2] + 1e-10)

    def calculate_gli(self, image_hwc):
        return (2 * image_hwc[..., 1] - image_hwc[..., 2] - image_hwc[..., 0]) / (2 * image_hwc[..., 1] + image_hwc[..., 2] + image_hwc[..., 0] + 1e-10)

    def calculate_gi(self, image_hwc):
        return image_hwc[..., 1] / (image_hwc[..., 2] + 1e-10)

    def calculate_blue_red(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 2] + 1e-10)

    def calculate_red_minus_blue(self, image_hwc):
        return image_hwc[..., 2] - image_hwc[..., 0]

    def calculate_exg(self, image_hwc):
        return 2 * image_hwc[..., 1] - image_hwc[..., 2] - image_hwc[..., 0]

    def calculate_vari(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 2]) / (image_hwc[..., 1] + image_hwc[..., 2] - image_hwc[..., 0] + 1e-10)

    def calculate_tvi(self, image_hwc):
        return (120 * (image_hwc[..., 4] - image_hwc[..., 1]) - 200 * (image_hwc[..., 2] - image_hwc[..., 1])) / 2

    def calculate_rdvi(self, image_hwc):
        return (image_hwc[..., 3] - image_hwc[..., 2]) / np.sqrt(image_hwc[..., 3] + image_hwc[..., 2] + 1e-10)

    def calculate_ndreb(self, image_hwc):
        return (image_hwc[..., 4] - image_hwc[..., 0]) / (image_hwc[..., 4] + image_hwc[..., 0] + 1e-10)

    def calculate_cig(self, image_hwc):
        return (image_hwc[..., 3] / (image_hwc[..., 1] + 1e-10)) - 1

    def calculate_blue_rededge(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 4] + 1e-10)

    def calculate_blue_nir(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 3] + 1e-10)

    def calculate_bndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        blue = image_hwc[..., 0]
        return (nir - blue) / (nir + blue + 1e-10)




OUT_CLASSES = 1

class segModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights=None,
            **kwargs,
        )
        # preprocessing parameteres for image (Cuurently no normalization, so the next few lines do not do anything--Normalizing data is most beneficial when input features have a high variance or differ significantly from what the model was trained on, which could lead to poor learning and performance.)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std //no normalization
        mask = self.model(image)
        return mask

    # How shared_step and shared_epoch_end Work Together
    # shared_step is called for each batch and returns the loss and segmentation statistics (tp, fp, fn, tn).
    # The results from multiple shared_step calls are collected into a list (e.g., self.training_step_outputs).
    # At the end of an epoch, shared_epoch_end aggregates all statistics and computes IoU metrics.
    def shared_step(self, batch, stage):
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # def shared_epoch_end(self, outputs, stage):
    #     # aggregate step metics
    #     tp = torch.cat([x["tp"] for x in outputs])
    #     fp = torch.cat([x["fp"] for x in outputs])
    #     fn = torch.cat([x["fn"] for x in outputs])
    #     tn = torch.cat([x["tn"] for x in outputs])

    #     # per image IoU means that we first calculate IoU score for each image
    #     # and then compute mean over these scores
    #     per_image_iou = smp.metrics.iou_score(
    #         tp, fp, fn, tn, reduction="micro-imagewise"
    #     )

    #     # dataset IoU means that we aggregate intersection and union over whole dataset
    #     # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
    #     # in this particular case will not be much, however for dataset
    #     # with "empty" images (images without target class) a large gap could be observed.
    #     # Empty images influence a lot on per_image_iou and much less on dataset_iou. 
    #     # When we say that empty images influence per_image_iou a lot, it means they tend to increase it. 
    #     # This is because empty images typically have high IoU, and when computing per_image_iou (i.e., averaging IoU across all images),
    #     # empty images dominate the average because they tend to have high IoU values.
    #     dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    #     metrics = {
    #         f"{stage}_per_image_iou": per_image_iou,
    #         f"{stage}_dataset_iou": dataset_iou,
    #     }

    #     self.log_dict(metrics, prog_bar=True)

    # Mohsen Ghanbari Feb 2025 - Revised this fucntion to output other metrics as well
    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
    
        # Compute IoU:
        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou. 
        # When we say that empty images influence per_image_iou a lot, it means they tend to increase it. 
        # This is because empty images typically have high IoU, and when computing per_image_iou (i.e., averaging IoU across all images),
        # empty images dominate the average because they tend to have high IoU values.
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    
        # Compute additional metrics
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
        # Log metrics
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_f1_score": f1_score,
            # f"{stage}_tp": tp.sum().item(),
            # f"{stage}_fp": fp.sum().item(),
            # f"{stage}_fn": fn.sum().item(),
            # f"{stage}_tn": tn.sum().item(),
        }
    
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return


def load_model(model_type='model_full', use_bops_substrate=False):
    """Load the appropriate model(s) based on model_type.

    use_bops_substrate selects the weights trained with BoPs substrate (True)
    or RF substrate (False, default). Only applies to model_full and model_ensemble.
    """
    if model_type == 'model_full':
        in_channels = 13
        if use_bops_substrate:
            MODEL_URL = "https://huggingface.co/m5ghanba/SKeMa/resolve/main/model_full_bops_subs.pth"
            model_filename = f"model_full_bops_subs_v{__version__}.pth"
        else:
            MODEL_URL = "https://huggingface.co/m5ghanba/SKeMa/resolve/main/model_full_rf_subs.pth"
            model_filename = f"model_full_rf_subs_v{__version__}.pth"
        
        # Create model
        model = segModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=in_channels, out_classes=OUT_CLASSES)
        
        # Download model if needed
        LOCAL_PATH = os.path.join(os.path.expanduser("~"), ".skema", model_filename)
        os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
        
        if not os.path.exists(LOCAL_PATH):
            print(f"Downloading model from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, LOCAL_PATH)
            print("Download complete.")
        
        # Load weights
        model.load_state_dict(torch.load(LOCAL_PATH, map_location="cpu"))
        return model
        
    elif model_type == 'model_s2bandsandindices_only':
        in_channels = 10
        MODEL_URL = "https://huggingface.co/m5ghanba/SKeMa/resolve/main/modelS2Only.pth" # "https://github.com/m5ghanba/skema/releases/download/v0.2.0/modelS2Only.pth"  https://huggingface.co/m5ghanba/SKeMa/blob/main/modelS2Only.pth
        model_filename = f"modelS2Only_v{__version__}.pth"
        
        # Create model
        model = segModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=in_channels, out_classes=OUT_CLASSES)
        
        # Download model if needed
        LOCAL_PATH = os.path.join(os.path.expanduser("~"), ".skema", model_filename)
        os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)
        
        if not os.path.exists(LOCAL_PATH):
            print(f"Downloading model from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, LOCAL_PATH)
            print("Download complete.")
        
        # Load weights
        model.load_state_dict(torch.load(LOCAL_PATH, map_location="cpu"))
        return model
        
    elif model_type == 'model_ensemble':
        # Load both models for ensemble
        print("Loading ensemble models...")
        
        # Load model_full
        in_channels_full = 13
        if use_bops_substrate:
            MODEL_URL_full = "https://huggingface.co/m5ghanba/SKeMa/resolve/main/model_full_bops_subs.pth"
            model_filename_full = f"model_full_bops_subs_v{__version__}.pth"
        else:
            MODEL_URL_full = "https://huggingface.co/m5ghanba/SKeMa/resolve/main/model_full_rf_subs.pth"
            model_filename_full = f"model_full_rf_subs_v{__version__}.pth"
        LOCAL_PATH_full = os.path.join(os.path.expanduser("~"), ".skema", model_filename_full)
        
        os.makedirs(os.path.dirname(LOCAL_PATH_full), exist_ok=True)
        if not os.path.exists(LOCAL_PATH_full):
            print(f"Downloading model_full from {MODEL_URL_full}...")
            urllib.request.urlretrieve(MODEL_URL_full, LOCAL_PATH_full)
            print("Download complete.")
        
        model_full = segModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=in_channels_full, out_classes=OUT_CLASSES)
        model_full.load_state_dict(torch.load(LOCAL_PATH_full, map_location="cpu"))
        
        # Load model_s2bandsandindices_only
        in_channels_s2 = 10
        MODEL_URL_s2 = "https://huggingface.co/m5ghanba/SKeMa/resolve/main/modelS2Only.pth"
        model_filename_s2 = f"modelS2Only_v{__version__}.pth"
        LOCAL_PATH_s2 = os.path.join(os.path.expanduser("~"), ".skema", model_filename_s2)
        
        if not os.path.exists(LOCAL_PATH_s2):
            print(f"Downloading model_s2bandsandindices_only from {MODEL_URL_s2}...")
            urllib.request.urlretrieve(MODEL_URL_s2, LOCAL_PATH_s2)
            print("Download complete.")
        
        model_s2 = segModel("Unet", "tu-maxvit_tiny_tf_512", in_channels=in_channels_s2, out_classes=OUT_CLASSES)
        model_s2.load_state_dict(torch.load(LOCAL_PATH_s2, map_location="cpu"))
        
        print("Both models loaded successfully.")
        return (model_full, model_s2)  # Return tuple of both models
        
    else:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'model_full', 'model_s2bandsandindices_only', or 'model_ensemble'.")




# # Set model to evaluation mode
# model.eval()


# supports having both 10m and 20m bands as well as substrate and bathymetry channels.
# also normalization is added here. Set mean and std to None if it was off during training. Otherwise, it is assumed mean_per_channel and 
# std_per_channel are provided. 




def normalize_tile_mean_std(tile, mean_per_channel, std_per_channel):
    """Applies mean and std normalization to each channel of the input tile (H, W, C) using vectorization."""
    tile = np.nan_to_num(tile).astype(np.float32)  # Handle NaNs and ensure float type
    mean = np.array(mean_per_channel, dtype=np.float32)
    std = np.array(std_per_channel, dtype=np.float32)
    # Reshape mean and std to match tile's shape for broadcasting
    mean = mean[np.newaxis, np.newaxis, :]
    std = std[np.newaxis, np.newaxis, :]
    normalized_tile = (tile - mean) / (std + 1e-8)
    return normalized_tile

def create_weight_map(tile_size, halo_size):
    """Create a weight map that gives less weight to edge pixels and more to center."""
    weight_map = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Create a linear fade from edge to center
    for i in range(halo_size):
        fade_weight = (i + 1) / halo_size
        # Top and bottom edges
        weight_map[i, :] = fade_weight
        weight_map[tile_size-1-i, :] = fade_weight
        # Left and right edges
        weight_map[:, i] = np.minimum(weight_map[:, i], fade_weight)
        weight_map[:, tile_size-1-i] = np.minimum(weight_map[:, tile_size-1-i], fade_weight)
    
    return weight_map
#  July 2025 - Same as above but with these modifications: 
# 1.padding (zero-padding was failing in the edge tiles, so using mirror padding)
# 2.Handling the overlap differently now using a weighted average method: we give 
# weight 1 topredictions from a square sized halo_size by halo_size in the centre 
# of each tile and the rest of the tile get weight values faded from  1 to 0 going
# from the edge of the square in the centre to the edge of the tile. Specifically, 
# (a) Accumulation Process:Each tile adds its weighted prediction to the accumulator:
# predictions += tile_pred * tile_weights. Then, each tile adds its weights to the 
# weight accumulator: weight_accumulator += tile_weights 
# (b) Final Normalization: At the end, we divide by total weights: 
# predictions = predictions / weight_accumulator (Converts probabilities > 0.5 to 1
# probabilities ≤ 0.5 to 0)
# This gives us the weighted average of all predictions that contributed to each pixel
class DatasetInference(SatelliteDataset):
    def __init__(self, main_directory, model, dataset, model_type='model_full', tile_size=512, 
                 overlap=0.7, mean_per_channel=None, std_per_channel=None, halo_size=64, padding_mode='reflect'):
        
        # Validate model_type
        if model_type not in ['model_full', 'model_s2bandsandindices_only', 'model_ensemble']:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'model_full', 'model_s2bandsandindices_only', or 'model_ensemble'.")
        
        self.main_directory = main_directory
        self.tile_size = tile_size
        self.overlap = overlap
        self.model_type = model_type
        self.mean_per_channel = mean_per_channel
        self.std_per_channel = std_per_channel
        self.halo_size = halo_size
        self.padding_mode = padding_mode
        self.dataset = dataset
        
        # Handle ensemble mode (model is a tuple of two models)
        if model_type == 'model_ensemble':
            self.model_full = model[0].to(DEVICE)
            self.model_s2 = model[1].to(DEVICE)
            self.model = None  # Not used in ensemble mode
        else:
            self.model = model.to(DEVICE)
        
        self.weight_map = create_weight_map(tile_size, halo_size)
        
        # Get file paths based on model type
        if self.model_type == 'model_full' or self.model_type == 'model_ensemble':
            self.image_path1, self.image_path2, self.substrate_path, self.bathymetry_path, self.slope_path = self.get_file_paths(main_directory)
        else:  # model_s2bandsandindices_only
            self.image_path1, self.image_path2 = self.get_file_paths(main_directory)
        
        # Load and process image
        self.image, self.metadata = self.load_image()

    def get_file_paths(self, main_directory):
        """Retrieve file paths based on model type."""
        if self.model_type == 'model_full' or self.model_type == 'model_ensemble':
            file_patterns = ["*_B2B3B4B8.tif", "*_B5B6B7B8A_B11B12.tif", "*_Substrate.tif", "*_Bathymetry.tif", "*_Slope.tif"]
        else:  # model_s2bandsandindices_only
            file_patterns = ["*_B2B3B4B8.tif", "*_B5B6B7B8A_B11B12.tif"]
        
        file_paths = []
        for pattern in file_patterns:
            matching_files = glob.glob(os.path.join(main_directory, pattern))
            if len(matching_files) != 1:
                raise ValueError(f"Expected one file for pattern {pattern}, found {len(matching_files)}.")
            file_paths.append(matching_files[0])
        
        return tuple(file_paths)

    def load_image(self):
        """Load all image bands and compute indices directly into self.image."""
        # Load 10m bands
        with rasterio.open(self.image_path1) as src1:
            image1 = src1.read([1, 2, 3, 4])
            image1 = np.transpose(image1, (1, 2, 0)).astype(np.float32)
            metadata = src1.meta
        
        # Load 20m band
        with rasterio.open(self.image_path2) as src2:
            image2 = src2.read(indexes=[1], out_shape=(
                1, image1.shape[0], image1.shape[1]
            ), resampling=Resampling.nearest)
            image2 = np.transpose(image2, (1, 2, 0)).astype(np.float32)
        
        if self.model_type == 'model_full' or self.model_type == 'model_ensemble':
            # Load substrate, bathymetry, and slope
            with rasterio.open(self.substrate_path) as src3:
                substrate = src3.read(1).astype(np.float32)[:, :, np.newaxis]
            
            with rasterio.open(self.bathymetry_path) as src4:
                bathymetry = src4.read(1).astype(np.float32)[:, :, np.newaxis]
            
            with rasterio.open(self.slope_path) as src5:
                slope = src5.read(1).astype(np.float32)[:, :, np.newaxis]
            
            # Allocate image array with 13 channels (5 S2 bands + substrate + bathymetry + slope + 5 indices)
            self.image = np.empty((image1.shape[0], image1.shape[1], 13), dtype=np.float32)
            self.image[:, :, 0:4] = image1
            self.image[:, :, 4] = image2[:, :, 0]
            self.image[:, :, 5] = substrate[:, :, 0]
            self.image[:, :, 6] = bathymetry[:, :, 0]
            self.image[:, :, 7] = slope[:, :, 0]
        else:  # model_s2bandsandindices_only
            # Allocate image array with 10 channels (5 base + 5 indices)
            self.image = np.empty((image1.shape[0], image1.shape[1], 10), dtype=np.float32)
            self.image[:, :, 0:4] = image1
            self.image[:, :, 4] = image2[:, :, 0]
        
        # Compute indices directly into self.image
        self._compute_slope_and_all_indices()
        
        return self.image, metadata

    def _compute_slope_and_all_indices(self):
        """Compute all spectral indices directly into self.image."""
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]
        nir = self.image[:, :, 3]
        re = self.image[:, :, 4]

        eps = 1e-10
        
        if self.model_type == 'model_full' or self.model_type == 'model_ensemble':
            # Slope is already loaded at channel 7, no need to calculate
            # Indices start at channel 8
            self.image[:, :, 8] = (nir - red) / (nir + red + eps)  # NDVI
            self.image[:, :, 9] = (green - nir) / (green + nir + eps)  # NDWI
            self.image[:, :, 10] = (nir - green) / (nir + green + eps)  # GNDVI
            self.image[:, :, 11] = np.where(green < 1e-4, 20.0, nir / (green + eps) - 1)  # Chlorophyll Index
            self.image[:, :, 12] = (re - red) / (re + red + eps)  # NDVI-RE
        else:  # model_s2bandsandindices_only
            # Indices start at channel 5
            self.image[:, :, 5] = (nir - red) / (nir + red + eps)  # NDVI
            self.image[:, :, 6] = (green - nir) / (green + nir + eps)  # NDWI
            self.image[:, :, 7] = (nir - green) / (nir + green + eps)  # GNDVI
            self.image[:, :, 8] = np.where(green < 1e-4, 20.0, nir / (green + eps) - 1)  # Chlorophyll Index
            self.image[:, :, 9] = (re - red) / (re + red + eps)  # NDVI-RE
            self.image[:, :, 9] = (re - red) / (re + red + eps)  # NDVI-RE


    def generate_tiles(self, image):
        """Generator that yields one tile and its coordinates at a time."""
        h, w, c = image.shape
        tile_size = self.tile_size
        overlap = self.overlap
        step_size = int(tile_size * (1 - overlap))

        # Integer ceil division: (a + b - 1) // b  does the same as math.ceil(a / b)
        if h <= tile_size:
            tiles_y = 1
        else:
            tiles_y = ((h - tile_size) + step_size - 1) // step_size + 1

        if w <= tile_size:
            tiles_x = 1
        else:
            tiles_x = ((w - tile_size) + step_size - 1) // step_size + 1

        # Generate ALL tiles with consistent grid alignment
        for y in range(tiles_y):
            for x in range(tiles_x):
                i = y * step_size          # top-left row (can be >= h for the last tiles)
                j = x * step_size          # top-left col (can be >= w for the last tiles)

                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)

                tile = image[i:i_end, j:j_end]

                # Pad to full tile_size with zeros if we are on the border
                actual_h, actual_w = tile.shape[:2]
                if actual_h < tile_size or actual_w < tile_size:
                    pad_bottom = tile_size - actual_h
                    pad_right = tile_size - actual_w
                    tile = np.pad(tile,
                                ((0, pad_bottom), (0, pad_right), (0, 0)),
                                mode='constant',
                                constant_values=0)

                # Optional normalization
                if self.mean_per_channel is not None and self.std_per_channel is not None:
                    tile = normalize_tile_mean_std(tile, self.mean_per_channel, self.std_per_channel)

                yield tile, (i, j)


    def _process_batch_not_weighted(self, tiles, coords, predictions):
        """Not weighted - Run inference on a batch of tiles and write results into full image."""
        batch_tensor = torch.cat(tiles, dim=0).to(DEVICE)  # shape: (B, C, H, W)
        outputs = self.model(batch_tensor)  # shape: (B, 1, H, W) or (B, H, W)
    
        # Handle binary output (apply sigmoid then threshold)
        if outputs.shape[1] == 1:
            outputs = (outputs.squeeze(1).sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
        else:
            outputs = outputs.cpu().numpy().astype(np.uint8)
    
        # Write predictions into full image
        for pred, (i, j) in zip(outputs, coords):
            effective_tile_height = min(self.tile_size, predictions.shape[0] - i)
            effective_tile_width = min(self.tile_size, predictions.shape[1] - j)
    
            predictions[i:i + effective_tile_height, j:j + effective_tile_width] = np.maximum(
                predictions[i:i + effective_tile_height, j:j + effective_tile_width],
                pred[:effective_tile_height, :effective_tile_width]
            )

    def _process_batch_not_weighted_ensemble(self, tiles_full, tiles_s2, coords, predictions):
        """Not weighted ensemble mode - Run both models, average logits, then threshold."""
        # Process with model_full (13 channels)
        batch_tensor_full = torch.cat(tiles_full, dim=0).to(DEVICE)
        logits_full = self.model_full(batch_tensor_full)  # shape: (B, 1, H, W)

        # Process with model_s2 (10 channels)
        batch_tensor_s2 = torch.cat(tiles_s2, dim=0).to(DEVICE)
        logits_s2 = self.model_s2(batch_tensor_s2)  # shape: (B, 1, H, W)

        # Average logits from both models, then apply sigmoid and threshold
        averaged_logits = (logits_full + logits_s2) / 2.0
        outputs = (averaged_logits.squeeze(1).sigmoid() > 0.5).cpu().numpy().astype(np.uint8)

        # Write predictions into full image
        for pred, (i, j) in zip(outputs, coords):
            effective_tile_height = min(self.tile_size, predictions.shape[0] - i)
            effective_tile_width = min(self.tile_size, predictions.shape[1] - j)

            predictions[i:i + effective_tile_height, j:j + effective_tile_width] = np.maximum(
                predictions[i:i + effective_tile_height, j:j + effective_tile_width],
                pred[:effective_tile_height, :effective_tile_width]
            )

    def _process_batch_ensemble(self, tiles_full, tiles_s2, coords, predictions):
        """Ensemble mode - Run both models and average their logits before thresholding."""
        # Process with model_full (13 channels)
        batch_tensor_full = torch.cat(tiles_full, dim=0).to(DEVICE)
        logits_full = self.model_full(batch_tensor_full)  # shape: (B, 1, H, W)
        
        # Process with model_s2 (10 channels)
        batch_tensor_s2 = torch.cat(tiles_s2, dim=0).to(DEVICE)
        logits_s2 = self.model_s2(batch_tensor_s2)  # shape: (B, 1, H, W)
        
        # Average logits from both models
        averaged_logits = (logits_full + logits_s2) / 2.0
        
        # Apply threshold to averaged logits
        outputs = (averaged_logits.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
        
        # Write predictions into full image
        for pred, (i, j) in zip(outputs, coords):
            effective_tile_height = min(self.tile_size, predictions.shape[0] - i)
            effective_tile_width = min(self.tile_size, predictions.shape[1] - j)
            
            predictions[i:i + effective_tile_height, j:j + effective_tile_width] = np.maximum(
                predictions[i:i + effective_tile_height, j:j + effective_tile_width],
                pred[:effective_tile_height, :effective_tile_width]
            )

    def _process_batch(self, tiles, coords, predictions, weight_accumulator):
        """Process batch using weighted averaging with halo method."""
        batch_tensor = torch.cat(tiles, dim=0).to(DEVICE)
        outputs = self.model(batch_tensor)
    
        # Handle binary output (thresholding)
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1).sigmoid().cpu().numpy()  # Keep as probabilities for averaging
        else:
            outputs = torch.softmax(outputs, dim=1).cpu().numpy()  # Multi-class probabilities
    
        # Process each prediction in the batch
        for pred, (i, j) in zip(outputs, coords):
            # Calculate the region bounds in the full image
            end_i = min(i + self.tile_size, predictions.shape[0])
            end_j = min(j + self.tile_size, predictions.shape[1])
            
            # Calculate effective tile size (in case of edge tiles)
            effective_h = end_i - i
            effective_w = end_j - j
            
            # Get the corresponding portion of the prediction and weight map
            tile_pred = pred[:effective_h, :effective_w]
            tile_weights = self.weight_map[:effective_h, :effective_w]
            
            # Weighted accumulation
            predictions[i:end_i, j:end_j] += tile_pred * tile_weights
            weight_accumulator[i:end_i, j:end_j] += tile_weights


    def run_model_on_tiles(self, batch_size=8):
        """
        Run the model on tiles with improved edge handling.
        
        Args:
            batch_size (int): Number of tiles to process in each batch
        """
        console = Console()
        
        # Set model(s) to eval mode
        if self.model_type == 'model_ensemble':
            self.model_full.eval()
            self.model_s2.eval()
        else:
            self.model.eval()
        
        predictions = np.zeros_like(self.image[:, :, 0], dtype=np.float32)
        weight_accumulator = np.zeros_like(self.image[:, :, 0], dtype=np.float32)

        # First pass: count tiles
        tile_count = sum(1 for _ in self.generate_tiles(self.image))
        
        # Second pass: process tiles
        tile_generator = self.generate_tiles(self.image)
        batch_tiles = []
        batch_coords = []
        tiles_processed = 0
        
        # For ensemble mode, we need separate tile lists
        if self.model_type == 'model_ensemble':
            batch_tiles_s2 = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing ...", total=tile_count)
            
            with torch.no_grad():
                for tile, (i, j) in tile_generator:
                    # Preprocess and add tile to batch
                    tile_tensor = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float()
                    batch_tiles.append(tile_tensor)
                    batch_coords.append((i, j))
                    
                    # For ensemble, also create S2-only tiles (first 10 channels)
                    if self.model_type == 'model_ensemble':
                        # Extract first 5 S2 bands + 5 indices (skip substrate, bathy, slope at channels 5-7)
                        # Channels: 0-4 (S2 bands), 8-12 (indices) -> remap to 0-9
                        tile_s2 = np.concatenate([tile[:, :, :5], tile[:, :, 8:13]], axis=2)
                        tile_s2_tensor = torch.tensor(tile_s2).permute(2, 0, 1).unsqueeze(0).float()
                        batch_tiles_s2.append(tile_s2_tensor)

                    # Run batch when full
                    if len(batch_tiles) == batch_size:
                        if self.model_type == 'model_ensemble':
                            self._process_batch_ensemble(batch_tiles, batch_tiles_s2, batch_coords, predictions)
                            batch_tiles_s2.clear()
                        else:
                            self._process_batch(batch_tiles, batch_coords, predictions, weight_accumulator)
                        tiles_processed += len(batch_tiles)
                        progress.update(task, completed=tiles_processed)
                        batch_tiles.clear()
                        batch_coords.clear()

                # Handle remaining tiles
                if batch_tiles:
                    if self.model_type == 'model_ensemble':
                        self._process_batch_ensemble(batch_tiles, batch_tiles_s2, batch_coords, predictions)
                    else:
                        self._process_batch(batch_tiles, batch_coords, predictions, weight_accumulator)
                    tiles_processed += len(batch_tiles)
                    progress.update(task, completed=tiles_processed)

        # Finalize predictions (only for non-ensemble modes)
        if self.model_type != 'model_ensemble':
            weight_accumulator = np.where(weight_accumulator == 0, 1, weight_accumulator)
            predictions = predictions / weight_accumulator
                
            # Convert back to binary/class predictions
            if predictions.ndim == 2:  # Binary case
                predictions = (predictions > 0.5).astype(np.uint8)
            else:  # Multi-class case
                predictions = np.argmax(predictions, axis=-1).astype(np.uint8)

        # Apply filters for model_full and model_ensemble
        if self.model_type == 'model_full' or self.model_type == 'model_ensemble':
            predictions[(self.image[:, :, 6] < -100) | (self.image[:, :, 6] > 20)] = 0

        # Apply exclusion zones (defined in BC Albers EPSG:3005, reprojected to image CRS)
        predictions = apply_exclusion_zones(predictions, self.metadata)
        if self.model_type == 'model_s2bandsandindices_only': #this is already applied in the model_full case
            predictions = apply_depth_mask(predictions, self.metadata)  


        console.print(f"[green]✓[/green] Processing complete.")

        return predictions


    def run_model_on_tiles_not_weighted(self, batch_size=8):
        """Run the model on tiles in batches with GPU acceleration and low RAM usage."""
        # Set model(s) to eval mode
        if self.model_type == 'model_ensemble':
            self.model_full.eval()
            self.model_s2.eval()
        else:
            self.model.eval()

        predictions = np.zeros_like(self.image[:, :, 0], dtype=np.uint8)

        # First pass: count tiles
        tile_count = sum(1 for _ in self.generate_tiles(self.image))
        
        # Second pass: process tiles
        tile_generator = self.generate_tiles(self.image)
        batch_tiles = []
        batch_coords = []
        tiles_processed = 0

        # For ensemble mode, we need separate tile lists
        if self.model_type == 'model_ensemble':
            batch_tiles_s2 = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing ...", total=tile_count)
            
            with torch.no_grad():
                for tile, (i, j) in tile_generator:
                    # Preprocess and add tile to batch
                    tile_tensor = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float()
                    batch_tiles.append(tile_tensor)
                    batch_coords.append((i, j))

                    # For ensemble, also create S2-only tiles (first 10 channels)
                    if self.model_type == 'model_ensemble':
                        # Extract first 5 S2 bands + 5 indices (skip substrate, bathy, slope at channels 5-7)
                        # Channels: 0-4 (S2 bands), 8-12 (indices) -> remap to 0-9
                        tile_s2 = np.concatenate([tile[:, :, :5], tile[:, :, 8:13]], axis=2)
                        tile_s2_tensor = torch.tensor(tile_s2).permute(2, 0, 1).unsqueeze(0).float()
                        batch_tiles_s2.append(tile_s2_tensor)

                    # Run batch when full
                    if len(batch_tiles) == batch_size:
                        if self.model_type == 'model_ensemble':
                            self._process_batch_not_weighted_ensemble(batch_tiles, batch_tiles_s2, batch_coords, predictions)
                            batch_tiles_s2.clear()
                        else:
                            self._process_batch_not_weighted(batch_tiles, batch_coords, predictions)
                        tiles_processed += len(batch_tiles)
                        progress.update(task, completed=tiles_processed)
                        batch_tiles.clear()
                        batch_coords.clear()

                # Handle remaining tiles
                if batch_tiles:
                    if self.model_type == 'model_ensemble':
                        self._process_batch_not_weighted_ensemble(batch_tiles, batch_tiles_s2, batch_coords, predictions)
                    else:
                        self._process_batch_not_weighted(batch_tiles, batch_coords, predictions)
                    tiles_processed += len(batch_tiles)
                    progress.update(task, completed=tiles_processed)

        # Apply filters for model_full and model_ensemble
        if self.model_type == 'model_full' or self.model_type == 'model_ensemble':
            predictions[(self.image[:, :, 6] < -100) | (self.image[:, :, 6] > 20)] = 0

        # Apply exclusion zones (defined in BC Albers EPSG:3005, reprojected to image CRS)
        predictions = apply_exclusion_zones(predictions, self.metadata)
        if self.model_type == 'model_s2bandsandindices_only': #this is already applied in the model_full case
            predictions = apply_depth_mask(predictions, self.metadata)  


        return predictions


    def __getitem__(self, idx):
        """Return the tile and its corresponding coordinates."""
        return self.tiles[idx], self.tile_coords[idx]

    def __len__(self):
        """Return the number of tiles."""
        return len(self.tiles)

    def save_output(self, predictions, output_path):
        """Save the reconstructed output as a GeoTIFF."""
        console = Console()
        # Update predictions to uint8 if needed before saving
        predictions = predictions.astype(np.uint8)

        # Update the metadata, ensuring it matches the predictions
        updated_metadata = self.metadata.copy()
        updated_metadata.update({
            'driver': 'GTiff',
            'dtype': 'uint8',      # Make sure this matches predictions' dtype
            'count': 1,          # Single-band output
            'compress': 'lzw'      # Optional compression to reduce file size
        })

        # Save the file with rasterio, ensuring that spatial metadata is preserved
        with rasterio.open(output_path, 'w', **updated_metadata) as dst:
            dst.write(predictions, 1)  # Write data to the first band
        
        
        console.print(f"[green]✓[/green] Kelp classification map saved to [bold]{output_path}[bold].")

# Set the main directory
# main_directory = r"C:\Alena\results\20220806T191919_20220806T192707_T09UXQ"
# output_filename = "output_cli.tif"
# mean_per_channel = [ 9.73005488e+02  7.08909146e+02  4.49016997e+02  7.64114558e+02
#   5.04806707e+02  1.55057075e+00 -3.07090868e+01 -4.03823853e-02
#   2.47833866e-01 -2.47833866e-01 -3.45288439e-02  1.30275308e-02]
# std_per_channel = [3.14151787e+02 2.99688583e+02 3.10539248e+02 9.25681716e+02
#  3.78586231e+02 1.27053181e+00 8.72741729e+01 3.88193209e-01
#  4.20784913e-01 4.20784913e-01 1.12317310e+00 1.60626435e-01]
# data = SatelliteDataset(image_paths=X_val_paths, mask_paths=y_val_paths, augmentation=None, classes=["kelp"])  # solely to access to the methods of the class for index calculation
# dataset = DatasetInference(main_directory=main_directory, model=model, dataset=data, mean_per_channel=mean_per_channel, std_per_channel=std_per_channel)
# 
# # Run model and save predictions
# predictions = dataset.run_model_on_tiles()
# 
# output_path = os.path.join(main_directory, output_filename)
# dataset.save_output(predictions, output_path)


def segment(input_dir, output_filename, mean_per_channel, std_per_channel, model_type='model_full', soft_substrate_masking=False, use_bops_substrate=False): 
    """
    Perform semantic segmentation inference on a Sentinel-2 scene.
    
    Parameters:
    - input_dir: path to a .SAFE folder OR directory containing the expected input TIFF files
    - output_filename: output TIFF filename to save prediction
    - mean_per_channel, std_per_channel: normalization stats used during training
    - model_type: 'model_full' (with substrate/bathymetry) or 'model_s2bandsandindices_only'
    - soft_substrate_masking: if True, also saves a substrate-masked prediction where kelp
      pixels overlapping substrate classes 3 or 4 are set to 0 (no kelp).
    - use_bops_substrate: if True, use BoPs substrate files and the BoPs-trained model weights;
      if False (default), use RF substrate files and RF-trained model weights.
    """
    
    # Load appropriate model
    model = load_model(model_type, use_bops_substrate=use_bops_substrate)
    
    # Preprocessing if input_dir is a SAFE folder
    if input_dir.endswith(".SAFE") and os.path.isdir(input_dir):
        safe_basename = os.path.basename(input_dir).replace(".SAFE", "")
        parent_dir = os.path.dirname(input_dir)
        output_folder = os.path.join(parent_dir, safe_basename)
        os.makedirs(output_folder, exist_ok=True)

        # Step 1: Extract S2 bands (skip if already extracted)
        b2348_file = os.path.join(output_folder, f"{safe_basename}_B2B3B4B8.tif")
        b5678a1112_file = os.path.join(output_folder, f"{safe_basename}_B5B6B7B8A_B11B12.tif")

        if os.path.exists(b2348_file) and os.path.exists(b5678a1112_file):
            console = Console()
            console.print("[yellow]Band TIFFs already exist, skipping extraction.[/yellow]")
        else:
            b2348_file, b5678a1112_file = extract_bands_to_geotiffs(input_dir, output_folder)
            if not b2348_file:
                raise RuntimeError(f"Failed to extract bands for {input_dir}")

        # Steps 2-4: Only for model_full and model_ensemble (skip if bathymetry, substrate, and slope already exist)
        if model_type == 'model_full' or model_type == 'model_ensemble':
            # Substrate source is set explicitly by the caller via use_bops_substrate
            if use_bops_substrate:
                # BoPs substrate: 4 files at 10m
                required_static = ["Bathymetry.tif", 
                                    "BoPs_HG_10m.tif", "BoPs_NCC_10m.tif",
                                    "BoPs_QCSSOG_10m.tif", "BoPs_WCVI_10m.tif"]
            else:
                # RF substrate: 5 regional files at 20m
                required_static = ["Bathymetry.tif", "NCC_substrate_20m.tif",
                                    "SOG_substrate_20m.tif", "WCVI_substrate_20m.tif",
                                    "QCS_substrate_20m.tif", "HG_substrate_20m.tif"]
            
            # Check if Slope.tif exists, if not, we need to generate it
            slope_path = None
            try:
                slope_path = str(files("skema.static.bathy_substrate").joinpath("Slope.tif"))
            except Exception:
                pass
            
            if slope_path and not os.path.exists(slope_path):
                # Generate Slope.tif from Bathymetry.tif
                console = Console()
                console.print("[yellow]Slope.tif not found. Generating from Bathymetry.tif...[/yellow]")
                try:
                    bathy_path = str(files("skema.static.bathy_substrate").joinpath("Bathymetry.tif"))
                    if os.path.exists(bathy_path):
                        calculate_slope_for_raster(bathy_path, slope_path)
                    else:
                        console.print("[red]Bathymetry.tif not found, cannot generate slope.[/red]")
                        required_static.append("Slope.tif")
                except Exception as e:
                    console.print(f"[red]Error generating Slope.tif: {e}[/red]")
                    required_static.append("Slope.tif")
            else:
                # Slope.tif already exists, add it to required files
                required_static.append("Slope.tif")
            
            missing = []
            for fname in required_static:
                try:
                    p = str(files("skema.static.bathy_substrate").joinpath(fname))
                    if not os.path.exists(p):
                        missing.append(fname)
                except Exception:
                    missing.append(fname)

            if missing:
                raise FileNotFoundError(
                    f"model_full requires bathymetry/substrate/slope static files, but these are missing:\n"
                    + "\n".join(f"  - {f}" for f in missing)
                    + "\n\nPlace these files in the static folder or use --model-type model_s2bandsandindices_only."
                )

            warp_bathy_and_subs(parent_dir, safe_basename, use_bops_substrate=use_bops_substrate)
            merge_substrate_files_single(output_folder, use_bops_substrate=use_bops_substrate)
            apply_fill_nodata_single(output_folder)

        input_dir = output_folder

    # Create dataset and run inference
    data = SatelliteDataset(image_paths=None, mask_paths=None, augmentation=None, classes=["kelp"])
    dataset = DatasetInference(
        main_directory=input_dir,
        model=model,
        dataset=data,
        model_type=model_type,
        mean_per_channel=mean_per_channel,
        std_per_channel=std_per_channel,
        tile_size=512,
        overlap=0.5,  
        halo_size=64,
        padding_mode='reflect'
    )

    predictions = dataset.run_model_on_tiles(batch_size=8) # not weighted option for stiching the tiles: run_model_on_tiles_not_weighted(batch_size=8)
    output_path = os.path.join(input_dir, output_filename)
    dataset.save_output(predictions, output_path)

        # ── Soft substrate masking (optional second output) ────────────────────────
    if soft_substrate_masking:
        console = Console()
        # Locate the merged substrate file for this scene
        subs_candidates = [f for f in os.listdir(input_dir) if f.endswith('_Substrate.tif')]
        if not subs_candidates:
            console.print('[yellow]Warning: _Substrate.tif not found in output folder; substrate-masked output skipped.[/yellow]')
        else:
            subs_path = os.path.join(input_dir, subs_candidates[0])
            with rasterio.open(subs_path) as subs_src:
                substrate = subs_src.read(1)
            masked_predictions = predictions.copy()
            masked_predictions[(predictions == 1) & ((substrate == 3) | (substrate == 4))] = 0
            stem = output_filename[:-4] if output_filename.lower().endswith('.tif') else output_filename
            masked_output_path = os.path.join(input_dir, f'{stem}_substrate_masked.tif')
            dataset.save_output(masked_predictions, masked_output_path)
            console.print(f'[green]✓[/green] Substrate-masked kelp map saved to [bold]{masked_output_path}[bold].')