"""Normalize locally exported building polygons without contacting any service."""

from __future__ import annotations

import argparse
from pathlib import Path


def import_vectors(source, output, target_crs, *, layer=None, method="nearmap_ai",
                   id_field=None, score_field=None):
    import geopandas as gpd
    import numpy as np
    from pyproj import CRS

    source, output = Path(source), Path(output)
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    if output.suffix.lower() != ".gpkg":
        raise ValueError("Output must be a .gpkg file")
    if not method.strip():
        raise ValueError("method must not be empty")
    frame = gpd.read_file(source, **({"layer": layer} if layer else {}))
    if frame.crs is None:
        raise ValueError("Input vectors have no CRS; assign their known CRS before import")
    if frame.empty:
        raise ValueError("Input contains no building polygons")
    if (frame.geometry.isna().any() or frame.geometry.is_empty.any()
            or not frame.geometry.is_valid.all()
            or not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Input must contain valid, nonempty Polygon/MultiPolygon buildings only")
    for field in (id_field, score_field):
        if field and field not in frame.columns:
            raise ValueError(f"Missing input field: {field}")
    ids = frame[id_field].astype(str) if id_field else [str(i) for i in range(1, len(frame) + 1)]
    if id_field and (frame[id_field].isna().any() or frame[id_field].duplicated().any()):
        raise ValueError("Source IDs must be non-null and unique")
    if len(set(ids)) != len(frame):
        raise ValueError("Source IDs must remain unique when represented as text")
    result = gpd.GeoDataFrame(
        {"source_id": list(ids), "method": [method] * len(frame)},
        geometry=frame.geometry.to_list(), crs=frame.crs,
    ).to_crs(CRS.from_user_input(target_crs))
    if score_field:
        try:
            scores = np.asarray(frame[score_field], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Scores must be numeric") from exc
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Scores must be finite probabilities in [0, 1]; convert explicitly first")
        result["score"] = scores
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output, layer="buildings", driver="GPKG", index=False)
    return {"output": str(output), "features": len(result), "crs": result.crs.to_string(),
            "method": method}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crs", required=True, help="Target CRS, for example EPSG:26914")
    parser.add_argument("--layer", help="Input layer for a multi-layer dataset")
    parser.add_argument("--method", default="nearmap_ai")
    parser.add_argument("--id-field")
    parser.add_argument("--score-field", help="Optional existing confidence in [0, 1]")
    args = parser.parse_args(argv)
    try:
        result = import_vectors(args.input, args.output, args.crs, layer=args.layer,
                                method=args.method, id_field=args.id_field,
                                score_field=args.score_field)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(result)


if __name__ == "__main__":
    main()
