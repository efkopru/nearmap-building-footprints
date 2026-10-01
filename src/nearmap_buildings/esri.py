"""Run the downloaded Esri Building Footprint Extraction USA package in ArcGIS Pro.

This file runs in ArcGIS Pro's own Python, so it imports only the standard library
and ArcPy. --chunks runs part of a city-wide plan from nbf esri-chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

CHUNK_NAME = re.compile(r"[A-Za-z0-9_]+")


def run_esri(raster, model, output, *, threshold=0.5, batch_size=4, padding=128,
             processor="GPU", gpu_id="0", execute=False):
    raster, model, output = Path(raster).resolve(), Path(model).resolve(), Path(output).resolve()
    if not raster.is_file() or not model.is_file():
        raise FileNotFoundError("Provide a local imagery file and an already downloaded .dlpk")
    if model.suffix.lower() != ".dlpk":
        raise ValueError("model must be the downloaded Building Footprint Extraction USA .dlpk")
    if not 0 <= threshold <= 1 or not math.isfinite(threshold):
        raise ValueError("threshold must be in [0, 1]")
    if batch_size < 1 or math.isqrt(batch_size) ** 2 != batch_size:
        raise ValueError("batch_size must be a positive perfect square")
    if padding < 0:
        raise ValueError("padding must be nonnegative and no more than half the model tile size")
    if processor not in ("CPU", "GPU"):
        raise ValueError("processor must be CPU or GPU")
    # Esri appends to existing outputs. Never append silently to a benchmark.
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    arguments = f"padding {padding};batch_size {batch_size};threshold {threshold};return_bboxes False"
    plan = {"tool": "arcpy.ia.DetectObjectsUsingDeepLearning", "in_raster": str(raster),
            "out_detected_objects": str(output), "in_model_definition": str(model),
            "arguments": arguments, "run_nms": "NO_NMS",
            "processing_mode": "PROCESS_AS_MOSAICKED_IMAGE", "processorType": processor,
            "gpuId": str(gpu_id), "execute": execute}
    if not execute:
        return plan

    try:
        import arcpy
    except ImportError as exc:
        raise RuntimeError("Run this module with your ArcGIS Pro cloned environment and its deep learning libraries") from exc
    if arcpy.Exists(str(output)):
        raise FileExistsError(f"Output feature class exists: {output}")
    if not arcpy.Exists(str(output.parent)):
        raise ValueError("Create the output folder or file geodatabase before running")
    # Describe only reports pixelType per band for multiband rasters; a Raster object
    # reports band count, pixel type and coordinate system for the whole dataset.
    image = arcpy.Raster(str(raster))
    if image.bandCount != 3 or image.pixelType != "U8":
        raise ValueError("Esri USA model expects 3-band unsigned 8-bit orthorectified RGB imagery")
    if getattr(image.spatialReference, "name", "Unknown") == "Unknown":
        raise ValueError("Input raster must have a known coordinate system")
    if arcpy.CheckExtension("ImageAnalyst") != "Available":
        raise RuntimeError("An available ArcGIS Image Analyst license is required")
    arcpy.CheckOutExtension("ImageAnalyst")
    try:
        with arcpy.EnvManager(overwriteOutput=False, processorType=processor, gpuId=str(gpu_id)):
            arcpy.ia.DetectObjectsUsingDeepLearning(
                in_raster=str(raster), out_detected_objects=str(output),
                in_model_definition=str(model), arguments=arguments, run_nms="NO_NMS",
                processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
            )
        if arcpy.Describe(str(output)).shapeType != "Polygon":
            raise RuntimeError("Model returned non-polygon output; verify the chosen .dlpk and return_bboxes")
        arcpy.management.AddField(str(output), "method", "TEXT", field_length=40)
        arcpy.management.CalculateField(str(output), "method", "'esri_building_usa'", "PYTHON3")
    finally:
        arcpy.CheckInExtension("ImageAnalyst")
    return plan


def plan_digest(chunks):
    """SHA-256 of a plan's chunks.json; done-markers record it so another plan's markers are never trusted."""
    return hashlib.sha256((Path(chunks) / "chunks.json").read_bytes()).hexdigest()


def check_marker(marker, digest):
    if json.loads(marker.read_text(encoding="utf-8")).get("plan_sha256") != digest:
        raise ValueError(f"{marker} belongs to another chunk plan; use a new --done folder and --gdb for this plan")


def run_chunk_batch(chunks, gdb, done, model, *, budget, **options):
    """Run up to budget unfinished chunks of a plan, each into its own feature class.

    A finished chunk writes a done-marker, so the next process resumes after it. A
    chunk without a marker was interrupted: its partial feature class is replaced.
    Returns "complete" once every chunk has a marker, otherwise "budget".
    """
    import arcpy

    chunks, gdb, done = Path(chunks).resolve(), Path(gdb).resolve(), Path(done).resolve()
    if budget < 1:
        raise ValueError("budget must be at least one chunk")
    if gdb.suffix.lower() != ".gdb":
        raise ValueError("chunk outputs go into a file geodatabase (.gdb)")
    plan = json.loads((chunks / "chunks.json").read_text(encoding="utf-8"))
    if plan.get("schema") != "nbf-esri-chunks-v1":
        raise ValueError("chunks.json was not written by nbf esri-chunks plan")
    if any(not CHUNK_NAME.fullmatch(chunk["name"]) for chunk in plan["chunks"]):
        raise ValueError("chunk names must be letters, digits and underscores")
    digest = plan_digest(chunks)
    done.mkdir(parents=True, exist_ok=True)
    if not arcpy.Exists(str(gdb)):
        arcpy.management.CreateFileGDB(str(gdb.parent), gdb.name)
    for chunk in plan["chunks"]:
        marker = done / f"{chunk['name']}.json"
        if marker.exists():
            check_marker(marker, digest)
            continue
        if budget == 0:
            return "budget"
        budget -= 1
        output = gdb / chunk["name"]
        if arcpy.Exists(str(output)):
            arcpy.management.Delete(str(output))
        start = time.perf_counter()
        run_esri(chunks / f"{chunk['name']}.vrt", model, output, execute=True, **options)
        result = {"chunk": chunk["name"], "seconds": round(time.perf_counter() - start, 1),
                  "features": int(arcpy.management.GetCount(str(output))[0]), "plan_sha256": digest}
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        os.replace(temporary, marker)
        print(json.dumps(result), flush=True)
    return "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New polygon feature class or shapefile")
    parser.add_argument("--chunks", type=Path, help="folder from nbf esri-chunks plan; runs its chunks instead of --raster")
    parser.add_argument("--gdb", type=Path, help="with --chunks: file geodatabase for one feature class per chunk")
    parser.add_argument("--done", type=Path, help="with --chunks: folder of done-markers")
    parser.add_argument("--budget", type=int, default=8,
                        help="with --chunks: chunks to run before exiting, so a fresh process releases GPU memory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--padding", type=int, default=128)
    parser.add_argument("--processor", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--execute", action="store_true", help="Run ArcPy inference; otherwise only print the plan")
    args = parser.parse_args(argv)
    options = dict(threshold=args.threshold, batch_size=args.batch_size, padding=args.padding,
                   processor=args.processor, gpu_id=args.gpu_id)
    if args.chunks is not None:
        if args.raster or args.output or not (args.gdb and args.done and args.execute):
            parser.error("--chunks needs --gdb, --done and --execute, and no --raster or --output")
        try:
            status = run_chunk_batch(args.chunks, args.gdb, args.done, args.model, budget=args.budget, **options)
        except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
            parser.error(str(exc))
        # The driver in nbf esri-chunks run reads this last line.
        print(json.dumps({"status": status}), flush=True)
        return
    if not (args.raster and args.output):
        parser.error("--raster and --output are required without --chunks")
    try:
        plan = run_esri(args.raster, args.model, args.output, execute=args.execute, **options)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
