"""
skema.masking
~~~~~~~~~~~~~
Post-inference spatial filters:
  * Hard exclusion zones (BC Albers polygons burned to 0)
  * Valid-depth-zone mask (−100 m – 20 m) from a packaged GeoPackage
  * Eelgrass mask from the BCMCA eelgrass shapefile
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from importlib.resources import files
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform as warp_transform
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Hard-coded exclusion zones (BC Albers, EPSG:3005)
# ---------------------------------------------------------------------------

EXCLUSION_ZONES_BC_ALBERS: list[list[tuple[float, float]]] = [
    [(1183501.336164263, 387958.1805347335), (1179689.5750786413, 387604.3649198598),
     (1175844.429108302, 397510.62208001973), (1176353.789889322, 403446.69221630145),
     (1179007.6559680663, 406428.1457283077), (1188805.981274077, 404611.72869927227),
     (1188676.1992809451, 394397.6341887866), (1183501.336164263, 387958.1805347335)],
    [(1170668.8573045777, 373853.681240315), (1169897.4363307026, 374170.5885593728),
     (1169483.3480854142, 374223.54157530714), (1169178.7020553653, 373979.87205611233),
     (1168599.1976390644, 373836.4753850734), (1167731.8887724597, 375914.5188937718),
     (1171716.9212878437, 378593.325610006), (1175580.7775909472, 378466.1798169448),
     (1177273.351640993, 376738.88923632784), (1176868.1381737296, 374623.5971600997),
     (1172437.1490685777, 374493.5770180205), (1172357.1044212494, 374476.10430309654),
     (1170668.8573045777, 373853.681240315)],
    [(1228964.7309115161, 451092.07903579326), (1223152.6305408822, 443516.78776073956),
     (1215084.7209121727, 442168.60036443715), (1206831.4246042732, 446616.18640549347),
     (1208095.815024607, 452000.8818810173), (1199256.525435853, 457319.93044620747),
     (1199256.525435853, 476200.20035552874), (1228251.2256537392, 475188.7573246722),
     (1229258.5946341362, 457528.7563443975), (1229286.8141567863, 457503.3116592518),
     (1228964.7309115161, 451092.07903579326)],
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
    [(1014845.7124855223, 455034.97863539215), (1016227.9140126681, 455780.1345114845),
     (1018076.8755909825, 454482.08266086684), (1019439.2577476983, 453868.1919296341),
     (1019004.3765252775, 452885.3760781294), (1017050.4963084502, 453414.38220678456),
     (1014845.7124855223, 455034.97863539215)],
    [(1006657.0918689288, 461025.20757650543), (1006521.1043602177, 460311.2606926271),
     (1005981.2035429816, 459889.3528698707), (1005664.8964668913, 459980.4002225398),
     (1005295.3413862621, 460295.3016608447), (1004679.4809445102, 461313.0248246574),
     (1005944.6172149541, 461543.08656319656), (1006615.2680217146, 461454.77175142535),
     (1006657.0918689288, 461025.20757650543)],
    [(1004892.7655447604, 463969.19314196403), (1003895.3580308687, 464664.58439990046),
     (1003881.16533169, 465258.2192997604), (1004466.1992165355, 465625.6773580631),
     (1004940.1315956784, 465351.0860325925), (1004892.7655447604, 463969.19314196403)],
    [(1229187.3451448006, 335531.3397770725), (1231184.1040226896, 318941.63944669964),
     (1218371.197868512, 283884.1994365883), (1192815.241561105, 279626.32953011926),
     (1150832.4703476252, 272056.7688337726), (1136525.0551379852, 303033.03016037337),
     (1135274.8086104176, 325431.8620125285), (1142117.1914919897, 336050.1841111155),
     (1148986.5278308126, 346106.4383125802), (1162729.1674537219, 348155.351903564),
     (1229187.3451448006, 335531.3397770725)],
    [(1209353.8349639473, 479280.6238570563), (1209947.355353582, 478997.9114821121),
     (1210050.958025145, 478928.55037148914), (1210176.5104152607, 479042.68890795717),
     (1210228.7507454131, 478922.84344466554), (1209217.3077145568, 477964.07973833283),
     (1208086.4582147796, 479323.20631104626), (1208588.6677752398, 479667.37789793493),
     (1209353.8349639473, 479280.6238570563)],
]

# Bathymetry data extent in BC Albers
_BATHY_EXTENT_BC_ALBERS = box(
    474232.8234999999986030,
    314708.7365999999456108,
    1304032.8234999999403954,
    1250588.7365999999456108,
)

# ---------------------------------------------------------------------------
# Lazy-loaded static geodataframes
# ---------------------------------------------------------------------------

def _load_valid_depth_zone() -> gpd.GeoDataFrame | None:
    try:
        path = str(files("skema.static.masks").joinpath("valid_depth_zone.gpkg"))
        gdf  = gpd.read_file(path)
        if gdf.crs is None or gdf.empty:
            print("[WARNING] valid_depth_zone.gpkg is empty or has no CRS.")
            return None
        return gdf
    except Exception as exc:
        print(f"[WARNING] Could not load depth zone mask: {exc}")
        return None


def _load_eelgrass_polygons() -> gpd.GeoDataFrame | None:
    try:
        path = str(
            files("skema.static.masks").joinpath(
                "BCMCA_ECO_VascPlants_Eelgrass_Polygons_DATA.shp"
            )
        )
        gdf = gpd.read_file(path)
        if gdf.crs is None or gdf.empty:
            print("[WARNING] Eelgrass shapefile is empty or has no CRS.")
            return None
        return gdf
    except Exception as exc:
        print(f"[WARNING] Could not load eelgrass polygons: {exc}")
        return None


_VALID_DEPTH_GDF  = _load_valid_depth_zone()
_EELGRASS_GDF     = _load_eelgrass_polygons()

# ---------------------------------------------------------------------------
# Public masking functions
# ---------------------------------------------------------------------------

def apply_exclusion_zones(predictions: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Zero-out predictions inside the hard-coded BC Albers exclusion polygons.

    Polygons are reprojected to the scene CRS on the fly using rasterio.warp.
    """
    if not EXCLUSION_ZONES_BC_ALBERS:
        return predictions

    src_crs       = CRS.from_epsg(3005)
    dst_crs       = metadata["crs"]
    img_transform = metadata["transform"]
    h, w          = predictions.shape

    shapes = []
    for coords in EXCLUSION_ZONES_BC_ALBERS:
        if dst_crs.to_epsg() == 3005:
            projected = coords
        else:
            xs, ys       = zip(*coords)
            xs_p, ys_p   = warp_transform(src_crs, dst_crs, list(xs), list(ys))
            projected    = list(zip(xs_p, ys_p))
        shapes.append(({"type": "Polygon", "coordinates": [projected]}, 1))

    exclusion_mask = rasterize(
        shapes=shapes,
        out_shape=(h, w),
        transform=img_transform,
        fill=0,
        dtype=np.uint8,
    )
    predictions[exclusion_mask == 1] = 0
    return predictions


def apply_depth_mask(predictions: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Zero out predictions outside the valid bathymetry depth zone (−100 m – 20 m),
    but only within the spatial extent where bathymetry data exists.
    """
    if _VALID_DEPTH_GDF is None:
        return predictions

    scene_crs     = metadata["crs"]
    img_transform = metadata["transform"]
    height, width = predictions.shape

    src_crs = CRS.from_user_input(scene_crs)
    albers  = CRS.from_epsg(3005)

    corners = [
        img_transform * (0,     0),
        img_transform * (width, 0),
        img_transform * (0,     height),
        img_transform * (width, height),
    ]
    xs_s = [c[0] for c in corners]
    ys_s = [c[1] for c in corners]

    if src_crs.to_epsg() != 3005:
        xs_a, ys_a = warp_transform(src_crs, albers, xs_s, ys_s)
    else:
        xs_a, ys_a = xs_s, ys_s

    scene_bbox_albers = box(min(xs_a), min(ys_a), max(xs_a), max(ys_a))

    if not scene_bbox_albers.intersects(_BATHY_EXTENT_BC_ALBERS):
        return predictions

    scene_bbox_gdf = gpd.GeoDataFrame(geometry=[scene_bbox_albers], crs="EPSG:3005")
    gdf_clipped    = gpd.clip(_VALID_DEPTH_GDF, scene_bbox_gdf)

    if gdf_clipped.empty:
        predictions[:] = 0
        return predictions

    gdf_reprojected = gdf_clipped.to_crs(scene_crs)

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
    valid_mask = rasterize(
        [(geom, 1) for geom in gdf_reprojected.geometry],
        out_shape=(height, width),
        transform=img_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    predictions[(extent_mask == 1) & (valid_mask == 0)] = 0
    return predictions


def apply_eelgrass_mask(
    predictions: np.ndarray,
    scene_crs,
    img_transform,
    pred_shape: tuple,
) -> np.ndarray:
    """
    Zero out kelp pixels that overlap eelgrass polygons (BCMCA dataset).
    """
    if _EELGRASS_GDF is None:
        return predictions

    height, width = pred_shape
    try:
        eelgrass_gdf = _EELGRASS_GDF.to_crs(scene_crs)
    except Exception as exc:
        print(f"[WARNING] Failed to reproject eelgrass polygons: {exc}")
        return predictions

    if eelgrass_gdf.empty:
        return predictions

    eelgrass_mask = rasterize(
        [(geom, 1) for geom in eelgrass_gdf.geometry],
        out_shape=(height, width),
        transform=img_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    predictions[(predictions == 1) & (eelgrass_mask == 1)] = 0
    return predictions