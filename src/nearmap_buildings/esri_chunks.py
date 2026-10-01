"""Run Esri's building model over a whole city in overlapping, resumable chunks.

plan writes one small VRT per chunk of the tiled source raster: a core that no other
chunk's core overlaps, plus an overlap on every side. run restarts ArcGIS Pro's
Python every few chunks, because its GPU memory grows with each call. merge keeps
each detection only in the chunk whose core holds it, so overlaps never count twice.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape

from .common import read_json, write_json

PLAN_SCHEMA = "nbf-esri-chunks-v1"
ESRI_SCRIPT = Path(__file__).with_name("esri.py")
ARCGIS_PYTHON = Path(r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe")


def plan_chunks(manifest_path, output, *, core=10240, overlap=512):
    """Write one VRT per chunk that holds imagery, and chunks.json with each core's bounds.

    Chunks are sized in source pixels and cover only manifest tiles with imagery.
    """
    import rasterio

    if core < 1 or overlap < 0:
        raise ValueError("core must be positive and overlap nonnegative, in pixels")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete" or not manifest.get("tiles"):
        raise ValueError("use a completed schema_version=1 tile manifest from nbf tile")
    source = Path(manifest["source"]["path"])
    bands = manifest["source"].get("bands", [1, 2, 3])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with rasterio.open(source) as src:
        width, height, transform, crs = src.width, src.height, src.transform, src.crs
    if (width, height) != (manifest["source"]["width"], manifest["source"]["height"]):
        raise ValueError(f"{source} no longer matches the manifest's raster size")
    srs = escape(crs.to_wkt())
    chunks = []
    for row in range(0, height, core):
        for column in range(0, width, core):
            row_end, column_end = min(row + core, height), min(column + core, width)
            hits = [tile for tile in manifest["tiles"]
                    if tile["row_off"] < row_end and tile["row_off"] + tile["height"] > row
                    and tile["col_off"] < column_end and tile["col_off"] + tile["width"] > column]
            if not hits:
                continue
            # Shrink to the tiles with imagery, never beyond the core plus the overlap.
            top = max(row - overlap, 0, min(tile["row_off"] for tile in hits))
            left = max(column - overlap, 0, min(tile["col_off"] for tile in hits))
            bottom = min(row_end + overlap, height, max(tile["row_off"] + tile["height"] for tile in hits))
            right = min(column_end + overlap, width, max(tile["col_off"] + tile["width"] for tile in hits))
            name, w, h = f"chunk_r{row // core:03d}_c{column // core:03d}", right - left, bottom - top
            vrt_bands = "".join(
                f'<VRTRasterBand dataType="Byte" band="{band}"><NoDataValue>0</NoDataValue><SimpleSource>'
                f'<SourceFilename relativeToVRT="0">{escape(str(source))}</SourceFilename>'
                f'<SourceBand>{source_band}</SourceBand>'
                f'<SrcRect xOff="{left}" yOff="{top}" xSize="{w}" ySize="{h}"/>'
                f'<DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/></SimpleSource></VRTRasterBand>'
                for band, source_band in enumerate(bands, 1))
            x, y = transform @ (left, top)
            (output / f"{name}.vrt").write_text(
                f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}"><SRS>{srs}</SRS>'
                f"<GeoTransform>{x!r}, {transform.a!r}, 0, {y!r}, 0, {transform.e!r}</GeoTransform>"
                f"{vrt_bands}</VRTDataset>", encoding="utf-8")
            (x0, y1), (x1, y0) = transform @ (column, row), transform @ (column_end, row_end)
            chunks.append({"name": name, "window": [left, top, w, h], "pixels": w * h,
                           "core_bounds": [x0, y0, x1, y1]})
    plan = {"schema": PLAN_SCHEMA, "manifest": str(Path(manifest_path).resolve()), "source": str(source),
            "crs": crs.to_wkt(), "core": core, "overlap": overlap, "chunks": chunks}
    write_json(output / "chunks.json", plan)
    return plan


def _read_plan(chunks):
    plan = read_json(Path(chunks) / "chunks.json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("chunks.json was not written by nbf esri-chunks plan")
    return plan


def _arcgis_environment():
    # nbf removes inherited PROJ/GDAL settings for its own wheels; ArcGIS Pro may need them.
    from .cli import REMOVED_GIS_VARIABLES

    return {**os.environ, **REMOVED_GIS_VARIABLES}


def run_chunks(chunks, gdb, done, model, *, arcgis_python=ARCGIS_PYTHON, per_process=8, max_failures=3,
               threshold=0.5, batch_size=4, padding=128, processor="GPU", gpu_id="0", execute=False,
               runner=subprocess.run):
    """Restart ArcGIS Pro's Python every per_process chunks until every chunk is done.

    Without execute this returns the command. A process that fails counts toward
    max_failures in a row; one that finishes its budget resets the count.
    """
    plan = _read_plan(chunks)
    if per_process < 1 or max_failures < 1:
        raise ValueError("per_process and max_failures must be at least 1")
    command = [str(arcgis_python), str(ESRI_SCRIPT), "--chunks", str(Path(chunks).resolve()),
               "--gdb", str(Path(gdb).resolve()), "--done", str(Path(done).resolve()),
               "--model", str(Path(model).resolve()), "--budget", str(per_process),
               "--threshold", str(threshold), "--batch-size", str(batch_size), "--padding", str(padding),
               "--processor", processor, "--gpu-id", str(gpu_id), "--execute"]
    def finished():
        return sum((Path(done) / f"{chunk['name']}.json").exists() for chunk in plan["chunks"])

    summary = {"chunks": len(plan["chunks"]), "command": command}
    if not execute:
        return {"dry_run": True, "done": finished(), **summary}
    failures = processes = 0
    while True:
        result = runner(command, capture_output=True, text=True, env=_arcgis_environment())
        processes += 1
        status = None
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                print(line, flush=True)
                status = json.loads(line).get("status", status)
        if result.returncode:
            failures += 1
            last = (result.stderr.strip().splitlines() or [f"exit code {result.returncode}"])[-1]
            print(f"ArcGIS process failed ({failures} in a row): {last}", file=sys.stderr, flush=True)
            if failures >= max_failures:
                raise RuntimeError(f"Esri chunk run stopped after {failures} failed processes in a row: {last}")
        elif status == "complete":
            return {"done": finished(), "processes": processes, **summary}
        elif status == "budget":
            failures = 0
        else:
            raise RuntimeError("ArcGIS process ended without reporting its status")


def _owned(points, bounds):
    """Half-open, so a point on a shared core edge belongs to exactly one chunk."""
    x0, y0, x1, y1 = bounds
    return (points.x >= x0) & (points.x < x1) & (points.y >= y0) & (points.y < y1)


def merge_chunks(gdb, chunks, done, crs, output, *, min_score=None):
    """Import every chunk with the Esri preset, keeping detections centred in the chunk's core.

    Writes one raw GeoPackage layer for nbf clean. Empty outlines are dropped and counted.
    """
    import geopandas as gpd
    import pandas as pd

    from .import_vectors import import_vectors

    plan = _read_plan(chunks)
    output = Path(output)
    if output.suffix.lower() != ".gpkg" or output.exists():
        raise ValueError("output must be a new .gpkg file")
    if min_score is not None and not (math.isfinite(min_score) and 0 <= min_score <= 1):
        raise ValueError("min_score must be in [0, 1]")
    layers = set(gpd.list_layers(gdb).name)
    parts = []
    counts = {"chunks": len(plan["chunks"]), "detected": 0, "empty_outlines": 0, "kept_by_core": 0}
    with tempfile.TemporaryDirectory() as temporary:
        for chunk in plan["chunks"]:
            name = chunk["name"]
            if not (Path(done) / f"{name}.json").exists() or name not in layers:
                raise ValueError(f"{name} has not finished; run nbf esri-chunks run again")
            source = gpd.read_file(gdb, layer=name, fid_as_index=True)
            empty = source.geometry.isna() | source.geometry.is_empty
            counts["empty_outlines"] += int(empty.sum())
            source = source[~empty].rename_axis("source_fid").reset_index()
            counts["detected"] += len(source)
            if source.empty:
                continue
            if source.crs is None:
                raise ValueError(f"{name} has no CRS")
            # Ownership is decided in the plan's CRS, where the cores tile the raster exactly.
            centres = source.geometry.buffer(0).representative_point().to_crs(plan["crs"])
            mine = source[_owned(centres, chunk["core_bounds"]).to_numpy()]
            if mine.empty:
                continue
            staged, imported = Path(temporary) / f"{name}_in.gpkg", Path(temporary) / f"{name}.gpkg"
            mine.to_file(staged, layer="detections")
            import_vectors(staged, imported, crs, layer="detections", preset="esri", id_field="source_fid")
            frame = gpd.read_file(imported)
            parts.append(frame.assign(source_id=name + ":" + frame.source_id.astype(str), chunk=name))
            counts["kept_by_core"] += len(frame)
    if not parts:
        raise ValueError("no detections in any chunk core")
    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)
    if min_score is not None:
        merged = merged[merged.score >= min_score].reset_index(drop=True)
        counts["at_min_score"] = len(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output, layer="buildings", driver="GPKG", index=False)
    return {"output": str(output.resolve()), "min_score": min_score, **counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write chunk VRTs and chunks.json from a tile manifest")
    plan.add_argument("--manifest", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path, help="new folder for the chunk plan")
    plan.add_argument("--core", type=int, default=10240, help="core size in source pixels")
    plan.add_argument("--overlap", type=int, default=512, help="overlap on every side, in source pixels")
    run = commands.add_parser("run", help="run every chunk in ArcGIS Pro's Python; --execute starts it")
    run.add_argument("--chunks", required=True, type=Path)
    run.add_argument("--gdb", required=True, type=Path, help="file geodatabase for the chunk outputs")
    run.add_argument("--done", required=True, type=Path, help="folder of done-markers")
    run.add_argument("--model", required=True, type=Path, help="downloaded .dlpk")
    run.add_argument("--arcgis-python", type=Path, default=ARCGIS_PYTHON)
    run.add_argument("--per-process", type=int, default=8)
    run.add_argument("--max-failures", type=int, default=3)
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--padding", type=int, default=128)
    run.add_argument("--processor", choices=["GPU", "CPU"], default="GPU")
    run.add_argument("--gpu-id", default="0")
    run.add_argument("--execute", action="store_true")
    merge = commands.add_parser("merge", help="keep each detection in its own chunk's core; write one GeoPackage")
    merge.add_argument("--gdb", required=True, type=Path)
    merge.add_argument("--chunks", required=True, type=Path)
    merge.add_argument("--done", required=True, type=Path)
    merge.add_argument("--crs", required=True, help="output CRS, e.g. EPSG:26914")
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--min-score", type=float, help="drop detections below this 0-1 score")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_chunks(args.manifest, args.output, core=args.core, overlap=args.overlap)
        result = {"output": str(args.output.resolve()), "chunks": len(result["chunks"]),
                  "pixels": sum(chunk["pixels"] for chunk in result["chunks"])}
    elif args.command == "run":
        result = run_chunks(args.chunks, args.gdb, args.done, args.model, arcgis_python=args.arcgis_python,
                            per_process=args.per_process, max_failures=args.max_failures,
                            threshold=args.threshold, batch_size=args.batch_size, padding=args.padding,
                            processor=args.processor, gpu_id=args.gpu_id, execute=args.execute)
    else:
        result = merge_chunks(args.gdb, args.chunks, args.done, args.crs, args.output, min_score=args.min_score)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
