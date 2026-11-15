"""Inspect and window a local GeoTIFF/VRT without changing its pixel resolution."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds

from .common import provenance, sha256_file, write_json

def validate_raster(src, bands):
    if not src.crs:
        raise ValueError("Imagery must have a CRS. Assign the correct source CRS before processing.")
    if len(bands) != 3 or len(set(bands)) != 3 or any(b < 1 or b > src.count for b in bands):
        raise ValueError("Select three distinct existing RGB bands (1-based).")
    if any(src.dtypes[b - 1] != "uint8" for b in bands):
        raise ValueError("Expected 8-bit RGB. Explicitly rescale other radiometry before tiling; no silent stretch is applied.")
    if src.transform.b != 0 or src.transform.d != 0 or src.transform.a <= 0 or src.transform.e >= 0:
        raise ValueError("Use north-up imagery with positive x and negative y pixel size. Warp rotated imagery first.")

def _window_starts(size, tile_size, step):
    if size <= tile_size:
        return [0]
    # The last window sits flush with the raster edge instead of leaving a thin
    # sliver, so it may overlap its neighbor by more than the requested overlap.
    return [*range(0, size - tile_size, step), size - tile_size]

def windows(width, height, tile_size, overlap):
    """Full-size windows overlapping by at least `overlap` pixels; smaller only if the raster is."""
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size.")
    step = tile_size - overlap
    rows = _window_starts(height, tile_size, step)
    cols = _window_starts(width, tile_size, step)
    for row in rows:
        for col in cols:
            yield Window(col, row, min(tile_size, width - col), min(tile_size, height - row))

def read_areas(path, raster_crs, layer=None, id_field="area_id"):
    """Area ids and bounding boxes, in the imagery's CRS, from a polygon layer."""
    import geopandas as gpd

    frame = gpd.read_file(path, **({"layer": layer} if layer else {}))
    if frame.crs is None:
        raise ValueError(f"Areas need a CRS: {path}")
    if id_field not in frame.columns:
        raise ValueError(f"Areas have no '{id_field}' field; name it with --area-id-field.")
    ids = [str(value) for value in frame[id_field]]
    if frame.empty or len(set(ids)) != len(ids) or not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids):
        raise ValueError("Area ids must be unique and use only letters, digits, '-' and '_'.")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError("Every area needs a geometry.")
    return list(zip(ids, (tuple(bounds) for bounds in frame.to_crs(raster_crs).geometry.bounds.to_numpy())))

def area_windows(src, areas, margin, tile_size, overlap):
    """Tile windows over each area's bounding box plus a margin, starting at the area's own corner.

    A whole-raster grid would cut an area wherever its lines happen to fall. Starting at the
    area's corner gives, for example, exactly nine 1024 px tiles at a 512 px step in a 2048 px square.
    """
    if margin < 0:
        raise ValueError("The margin must be 0 or more pixels.")
    transform = src.transform
    for area_id, (west, south, east, north) in areas:
        col = max(round((west - transform.c) / transform.a) - margin, 0)
        row = max(round((transform.f - north) / -transform.e) - margin, 0)
        width = min(round((east - transform.c) / transform.a) + margin, src.width) - col
        height = min(round((transform.f - south) / -transform.e) + margin, src.height) - row
        if width <= 0 or height <= 0:
            raise ValueError(f"Area {area_id} lies outside the imagery.")
        for window in windows(width, height, tile_size, overlap):
            yield area_id, Window(window.col_off + col, window.row_off + row, window.width, window.height)

def inspect(path, bands=(1, 2, 3)):
    with rasterio.open(path) as src:
        validate_raster(src, bands)
        result = {"path": str(Path(path).resolve()), "width": src.width, "height": src.height, "crs": src.crs.to_string(), "bounds": list(src.bounds), "resolution_in_crs_units": list(src.res), "bands": list(bands), "dtypes": src.dtypes, "nodata": src.nodata}
    return result

def tile(source, output, tile_size=1024, overlap=128, bands=(1, 2, 3), min_valid_fraction=0.01, hash_source=False,
         areas=None, area_margin=0):
    """Write overlapping tiles and a manifest. areas, as read_areas returns them, tiles each area on its own."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1.")
    metadata = inspect(source, bands)
    # Validate tiling parameters even before creating output directories.
    if areas is None:
        all_windows = [(None, window) for window in windows(metadata["width"], metadata["height"], tile_size, overlap)]
    else:
        with rasterio.open(source) as src:
            all_windows = list(area_windows(src, areas, area_margin, tile_size, overlap))
    output.mkdir(parents=True, exist_ok=False)
    (output / "tiles").mkdir()
    metadata.update({"bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns, "sha256": sha256_file(source) if hash_source else None})
    manifest = {"schema_version": 1, "source": metadata, "tile_size": tile_size, "overlap": overlap, "provenance": provenance(), "tiles": [], "skipped_nodata_windows": 0, "status": "running"}
    if areas is not None:
        manifest["areas"] = {"ids": [area_id for area_id, _ in areas], "margin_px": area_margin}
    try:
        with rasterio.open(source) as src:
            for area_id, window in all_windows:
                valid = src.dataset_mask(window=window) > 0
                fraction = float(valid.mean())
                if not valid.any() or fraction < min_valid_fraction:
                    manifest["skipped_nodata_windows"] += 1
                    continue
                tile_id = f"r{int(window.row_off):08d}_c{int(window.col_off):08d}"
                if area_id is not None:
                    tile_id = f"{area_id}_{tile_id}"
                name = f"tiles/{tile_id}.tif"
                transform = src.window_transform(window)
                # 256 px blocks with a horizontal predictor make city tiles about a third smaller
                # than row strips; level 1 is as small as the default here and threads keep it fast.
                profile = dict(driver="GTiff", width=int(window.width), height=int(window.height), count=3, dtype="uint8", crs=src.crs, transform=transform,
                               compress="deflate", predictor=2, zlevel=1, tiled=True, blockxsize=256, blockysize=256, num_threads="ALL_CPUS")
                rgb = src.read(list(bands), window=window)
                rgb[:, ~valid] = 0
                with rasterio.open(output / name, "w", **profile) as dst:
                    dst.write(rgb)
                    dst.write_mask(valid.astype(np.uint8) * 255)
                manifest["tiles"].append({"id": tile_id, "path": name, "row_off": int(window.row_off), "col_off": int(window.col_off), "width": int(window.width), "height": int(window.height), "crs": src.crs.to_string(), "bounds": list(window_bounds(window, src.transform)), "transform": list(transform)[:6], "valid_fraction": fraction, "sha256": sha256_file(output / name),
                                          **({"area_id": area_id} if area_id is not None else {})})
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest

def inspect_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--bands", nargs=3, type=int, default=[1, 2, 3])
    args = p.parse_args(argv)
    print(json.dumps(inspect(args.source, args.bands), indent=2))

def tile_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=1024, help="Tile width and height in pixels.")
    p.add_argument("--overlap", type=int, default=128, help="Minimum overlap between neighboring tiles, in pixels.")
    p.add_argument("--bands", nargs=3, type=int, default=[1, 2, 3])
    p.add_argument("--min-valid-fraction", type=float, default=0.01)
    p.add_argument("--hash-source", action="store_true", help="Hash the full source file; a VRT hash does not hash its source rasters.")
    p.add_argument("--areas", type=Path, help="Polygon layer: tile each area on its own, starting at its corner, into one manifest.")
    p.add_argument("--areas-layer")
    p.add_argument("--area-id-field", default="area_id")
    p.add_argument("--area-margin-px", type=int, default=0, help="Extra pixels around each area, so buildings on its edge are seen whole.")
    a = p.parse_args(argv)
    areas = None
    if a.areas is not None:
        with rasterio.open(a.source) as src:
            areas = read_areas(a.areas, src.crs, a.areas_layer, a.area_id_field)
    result = tile(a.source, a.output, a.tile_size, a.overlap, a.bands, a.min_valid_fraction, a.hash_source,
                  areas=areas, area_margin=a.area_margin_px)
    print(f"Prepared {len(result['tiles'])} tiles: {a.output / 'manifest.json'}")
