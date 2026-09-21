"""Inspect and window a local GeoTIFF/VRT without changing its pixel resolution."""
import argparse
import json
from pathlib import Path

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

def windows(width, height, tile_size, overlap):
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size.")
    step = tile_size - overlap
    # Stop when the last window reaches the edge; avoid a redundant sliver tile.
    rows = [0] if height <= tile_size else list(range(0, height - overlap, step))
    cols = [0] if width <= tile_size else list(range(0, width - overlap, step))
    for row in rows:
        for col in cols:
            yield Window(col, row, min(tile_size, width - col), min(tile_size, height - row))

def inspect(path, bands=(1, 2, 3)):
    with rasterio.open(path) as src:
        validate_raster(src, bands)
        result = {"path": str(Path(path).resolve()), "width": src.width, "height": src.height, "crs": src.crs.to_string(), "bounds": list(src.bounds), "resolution_in_crs_units": list(src.res), "bands": list(bands), "dtypes": src.dtypes, "nodata": src.nodata}
    return result

def tile(source, output, tile_size=1024, overlap=128, bands=(1, 2, 3), min_valid_fraction=0.01, hash_source=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1.")
    metadata = inspect(source, bands)
    # Validate tiling parameters even before creating output directories.
    all_windows = list(windows(metadata["width"], metadata["height"], tile_size, overlap))
    output.mkdir(parents=True, exist_ok=False)
    (output / "tiles").mkdir()
    metadata.update({"bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns, "sha256": sha256_file(source) if hash_source else None})
    manifest = {"schema_version": 1, "source": metadata, "tile_size": tile_size, "overlap": overlap, "provenance": provenance(), "tiles": [], "skipped_nodata_windows": 0, "status": "running"}
    try:
        with rasterio.open(source) as src:
            for window in all_windows:
                valid = src.dataset_mask(window=window) > 0
                fraction = float(valid.mean())
                if not valid.any() or fraction < min_valid_fraction:
                    manifest["skipped_nodata_windows"] += 1
                    continue
                tile_id = f"r{int(window.row_off):08d}_c{int(window.col_off):08d}"
                name = f"tiles/{tile_id}.tif"
                transform = src.window_transform(window)
                profile = dict(driver="GTiff", width=int(window.width), height=int(window.height), count=3, dtype="uint8", crs=src.crs, transform=transform, compress="deflate")
                rgb = src.read(list(bands), window=window)
                rgb[:, ~valid] = 0
                with rasterio.open(output / name, "w", **profile) as dst:
                    dst.write(rgb)
                    dst.write_mask(valid.astype(np.uint8) * 255)
                manifest["tiles"].append({"id": tile_id, "path": name, "row_off": int(window.row_off), "col_off": int(window.col_off), "width": int(window.width), "height": int(window.height), "crs": src.crs.to_string(), "bounds": list(window_bounds(window, src.transform)), "transform": list(transform)[:6], "valid_fraction": fraction, "sha256": sha256_file(output / name)})
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
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--bands", nargs=3, type=int, default=[1, 2, 3])
    p.add_argument("--min-valid-fraction", type=float, default=0.01)
    p.add_argument("--hash-source", action="store_true", help="Hash the full source file; a VRT hash does not hash its source rasters.")
    a = p.parse_args(argv)
    result = tile(a.source, a.output, a.tile_size, a.overlap, a.bands, a.min_valid_fraction, a.hash_source)
    print(f"Prepared {len(result['tiles'])} tiles: {a.output / 'manifest.json'}")
