"""Normalize locally exported building polygons without contacting any service."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# Settings for known sources. Explicit command-line options override them.
PRESETS = {
    # Output of esri.py or Esri's Detect Objects Using Deep Learning. arcgis.learn's
    # Mask R-CNN inference writes Confidence as score * 100 (arcgis 2.4.3,
    # _maskrcnn_inferencing.py). Raw outlines can self-intersect; nbf clean repairs
    # and audits them exactly as it does for SAM 3 masks.
    "esri": {"method": "esri_building_usa", "score_field": "Confidence", "score_scale": 100.0,
             "allow_invalid": True},
}


def _scores(values, field, scale):
    import numpy as np

    try:
        scores = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not np.isfinite(scores).all():
        raise ValueError(f"{field} must be finite")
    if scale != 1 and (scores <= 1).all():
        # Dividing 0-1 scores again would silently squash them toward zero.
        raise ValueError(f"Every {field} value is at most 1, so it already looks like a 0-1 score; "
                         f"use --score-scale 1 instead of {scale:g}")
    scores = scores / scale
    if ((scores < 0) | (scores > 1)).any():
        hint = " If it is a percentage, add --score-scale 100." if scale == 1 and (scores <= 100).all() else ""
        raise ValueError(f"{field} must be a score in [0, 1] after scaling.{hint}")
    return scores


def import_vectors(source, output, target_crs, *, layer=None, preset=None, method=None,
                   id_field=None, score_field=None, score_scale=None, allow_invalid=None):
    """Write one polygon per building with source_id, method and an optional 0-1 score.

    The method label comes from --method, the preset, or the input's own method
    field (esri.py writes one); they must agree, and nearmap_ai is the fallback.
    """
    import geopandas as gpd
    from pyproj import CRS

    if preset is not None and preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    settings = PRESETS.get(preset, {})
    method = method if method is not None else settings.get("method")
    score_field = score_field or settings.get("score_field")
    score_scale = score_scale if score_scale is not None else settings.get("score_scale", 1.0)
    allow_invalid = allow_invalid if allow_invalid is not None else settings.get("allow_invalid", False)
    source, output = Path(source), Path(output)
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    if output.suffix.lower() != ".gpkg":
        raise ValueError("Output must be a .gpkg file")
    if method is not None and not method.strip():
        raise ValueError("method must not be empty")
    if not math.isfinite(score_scale) or score_scale <= 0:
        raise ValueError("score_scale must be a positive number")
    if layer is None:
        # A file geodatabase usually holds several feature classes; never guess.
        layers = gpd.list_layers(source)
        if len(layers) > 1:
            raise ValueError(f"{source.name} has {len(layers)} layers; choose one with --layer: "
                             + ", ".join(layers["name"]))
    # Feature IDs (OBJECTID in a file geodatabase) keep each polygon traceable to its source.
    frame = gpd.read_file(source, fid_as_index=True, **({"layer": layer} if layer else {}))
    if frame.crs is None:
        raise ValueError("Input vectors have no CRS; assign their known CRS before import")
    if frame.empty:
        raise ValueError("Input contains no building polygons")
    geometry = frame.geometry
    if (geometry.isna().any() or geometry.is_empty.any()
            or not geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Input must contain nonempty Polygon/MultiPolygon buildings only")
    invalid = int((~geometry.is_valid).sum())
    if invalid and not allow_invalid:
        raise ValueError(f"{invalid} polygons are invalid. Repair them first, or pass --allow-invalid "
                         "to keep them for nbf clean to repair and audit")
    declared = None
    if "method" in frame.columns:
        labels = {str(value).strip() for value in frame["method"].dropna()} - {""}
        if len(labels) > 1:
            raise ValueError(f"The input's method field mixes {sorted(labels)}; import each method separately")
        declared = next(iter(labels), None)
    if method is not None and declared is not None and method != declared:
        raise ValueError(f"Method {method!r} contradicts the input's method field ({declared!r})")
    method = method or declared or "nearmap_ai"
    for field in (id_field, score_field):
        if field and field not in frame.columns:
            raise ValueError(f"Missing input field: {field}")
    ids = frame[id_field].astype(str) if id_field else [str(fid) for fid in frame.index]
    if id_field and (frame[id_field].isna().any() or frame[id_field].duplicated().any()):
        raise ValueError("Source IDs must be non-null and unique")
    if len(set(ids)) != len(frame):
        raise ValueError("Source IDs must remain unique when represented as text")
    result = gpd.GeoDataFrame(
        {"source_id": list(ids), "method": [method] * len(frame)},
        geometry=geometry.to_list(), crs=frame.crs,
    ).to_crs(CRS.from_user_input(target_crs))
    if score_field:
        result["score"] = _scores(frame[score_field], score_field, score_scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output, layer="buildings", driver="GPKG", index=False)
    return {"output": str(output), "features": len(result), "crs": result.crs.to_string(),
            "method": method, "source_id": id_field or "source feature ID",
            "score_field": score_field, "score_scale": score_scale if score_field else None,
            "invalid_polygons_kept": invalid}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="GeoPackage, GeoJSON, shapefile or file geodatabase")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crs", required=True, help="Target CRS, for example EPSG:26914")
    parser.add_argument("--layer", help="Input layer; required when the source has several")
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="esri: method esri_building_usa, score from Confidence (0-100), invalid outlines kept for nbf clean")
    parser.add_argument("--method", help="Method label; default: the preset's, then the input's method field, then nearmap_ai")
    parser.add_argument("--id-field", help="Unique source ID field; default: the source feature ID")
    parser.add_argument("--score-field", help="Optional existing confidence field")
    parser.add_argument("--score-scale", type=float, help="Divide scores by this first, e.g. 100 for a percentage")
    parser.add_argument("--allow-invalid", action="store_true", default=None,
                        help="Keep invalid polygons for nbf clean to repair and audit")
    args = parser.parse_args(argv)
    try:
        result = import_vectors(args.input, args.output, args.crs, layer=args.layer, preset=args.preset,
                                method=args.method, id_field=args.id_field, score_field=args.score_field,
                                score_scale=args.score_scale, allow_invalid=args.allow_invalid)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
