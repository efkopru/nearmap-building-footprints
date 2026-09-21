"""Run the downloaded Esri Building Footprint Extraction USA package in ArcGIS Pro."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
    description = arcpy.Describe(str(raster))
    if getattr(description, "bandCount", None) != 3 or getattr(description, "pixelType", None) != "U8":
        raise ValueError("Esri USA model expects 3-band unsigned 8-bit orthorectified RGB imagery")
    if getattr(description.spatialReference, "name", "Unknown") == "Unknown":
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New polygon feature class or shapefile")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--padding", type=int, default=128)
    parser.add_argument("--processor", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--execute", action="store_true", help="Run ArcPy inference; otherwise only print the plan")
    args = parser.parse_args(argv)
    try:
        plan = run_esri(args.raster, args.model, args.output, threshold=args.threshold,
                        batch_size=args.batch_size, padding=args.padding, processor=args.processor,
                        gpu_id=args.gpu_id, execute=args.execute)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
