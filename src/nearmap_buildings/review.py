"""Check a reviewed reference layer against the inventory it was reviewed from.

A person reviews the old outlines against the imagery: adds new buildings, removes
demolished ones and corrects changed ones. This reports what the review changed
(added, removed, modified, unchanged), flags outlines nbf evaluate would refuse or
that overlap each other, and writes a provenance record. It never edits labels:
reviewed outlines must come from a person, not from a model, to stay independent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.strtree import STRtree

from .common import write_json
from .evaluation import _within_aoi, geometry_fingerprint, maximum_cardinality_matches
from .postprocess import metric_crs, write_layers


def _problem(geometry):
    """Why nbf evaluate would refuse this outline, or None."""
    if geometry is None or geometry.is_empty:
        return "null_or_empty"
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return "not_a_polygon"
    if not geometry.is_valid:
        return "invalid: " + shapely.is_valid_reason(geometry)
    if geometry.area <= 0:
        return "zero_area"
    return None


def _overlaps(frame, min_overlap_m2):
    """Pairs of outlines whose shared area exceeds min_overlap_m2."""
    geometries = list(frame.geometry)
    tree = STRtree(geometries)
    rows = []
    for first, geometry in enumerate(geometries):
        for second in tree.query(geometry, predicate="intersects"):
            second = int(second)
            if second <= first:
                continue
            shared = geometry.intersection(geometries[second])
            if shared.area > min_overlap_m2:
                rows.append({"review_id": int(frame.review_id.iloc[first]),
                             "other_review_id": int(frame.review_id.iloc[second]),
                             "overlap_m2": float(shared.area), "geometry": shared})
    return gpd.GeoDataFrame(rows, columns=["review_id", "other_review_id", "overlap_m2", "geometry"],
                            geometry="geometry", crs=frame.crs)


def review_check(original: gpd.GeoDataFrame, reviewed: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, *, crs: str,
                 match_iou: float = 0.5, unchanged_iou: float = 0.95, min_overlap_m2: float = 0.5,
                 provenance: dict | None = None):
    """Return (summary, layers) describing what a review changed and whether it is ready to score.

    Outlines are compared inside the AOI with the edge policy nbf evaluate uses by
    default: outlines crossing the AOI boundary are left out and counted. Original
    and reviewed outlines are paired one to one, as in evaluation, at match_iou;
    a pair at unchanged_iou or above counts as unchanged, below it as modified.
    """
    if not 0 < match_iou <= unchanged_iou <= 1:
        raise ValueError("need 0 < match IoU <= unchanged IoU <= 1")
    if min_overlap_m2 < 0:
        raise ValueError("min_overlap_m2 must be nonnegative")
    target = metric_crs(crs)
    frames = {}
    for label, frame in (("original", original), ("reviewed", reviewed), ("aoi", aoi)):
        if frame.crs is None:
            raise ValueError(f"{label}: missing CRS")
        frames[label] = frame.to_crs(target).reset_index(drop=True)
    area = frames["aoi"]
    if area.empty or any(_problem(geometry) for geometry in area.geometry):
        raise ValueError("AOI must hold valid polygons")
    region = shapely.union_all(area.geometry.to_numpy())

    checked = {}
    problems = []
    for label in ("original", "reviewed"):
        frame = frames[label].copy()
        frame["review_id"] = np.arange(len(frame), dtype=np.int64)
        reasons = [_problem(geometry) for geometry in frame.geometry]
        bad = np.array([reason is not None for reason in reasons], dtype=bool)
        for index in np.flatnonzero(bad):
            problems.append({"layer": label, "review_id": int(index), "problem": reasons[index],
                             "geometry": frame.geometry.iloc[index] if reasons[index] != "null_or_empty" else None})
        inside, excluded = _within_aoi(frame.loc[~bad].reset_index(drop=True), region, "exclude")
        checked[label] = (inside, excluded)
    original_inside, original_excluded = checked["original"]
    reviewed_inside, reviewed_excluded = checked["reviewed"]
    overlaps = _overlaps(reviewed_inside, min_overlap_m2)

    matches = maximum_cardinality_matches(list(original_inside.geometry), list(reviewed_inside.geometry), match_iou)
    original_inside["reviewed_id"], original_inside["review_iou"] = -1, np.nan
    reviewed_inside["original_id"], reviewed_inside["review_iou"] = -1, np.nan
    for original_row, reviewed_row, iou in matches:
        original_inside.loc[original_row, ["reviewed_id", "review_iou"]] = [int(reviewed_inside.review_id.iloc[reviewed_row]), iou]
        reviewed_inside.loc[reviewed_row, ["original_id", "review_iou"]] = [int(original_inside.review_id.iloc[original_row]), iou]
    matched = reviewed_inside.original_id >= 0
    layers = {
        "added": reviewed_inside.loc[~matched].copy(),
        "removed": original_inside.loc[original_inside.reviewed_id < 0].copy(),
        "modified": reviewed_inside.loc[matched & (reviewed_inside.review_iou < unchanged_iou)].copy(),
        "unchanged": reviewed_inside.loc[matched & (reviewed_inside.review_iou >= unchanged_iou)].copy(),
        "problems": gpd.GeoDataFrame(problems, columns=["layer", "review_id", "problem", "geometry"],
                                     geometry="geometry", crs=target),
        "overlaps": overlaps,
    }
    reviewed_problems = sum(problem["layer"] == "reviewed" for problem in problems)
    summary = {
        "schema_version": 1,
        "purpose": "What a manual review of an existing inventory changed. The reviewed layer is an independent reference only if a person drew or checked every outline against the imagery; this check cannot establish that.",
        "metric_crs": target.to_string(), "match_iou": match_iou, "unchanged_iou": unchanged_iou,
        "min_overlap_m2": min_overlap_m2,
        # Equal to reference_fingerprint in nbf evaluate reports made from this reviewed file.
        "reviewed_fingerprint": geometry_fingerprint(reviewed) if not reviewed_problems else None,
        "original_fingerprint": geometry_fingerprint(original) if len(problems) == reviewed_problems else None,
        "aoi_fingerprint": geometry_fingerprint(aoi),
        "counts": {
            "original_in_aoi": len(original_inside), "reviewed_in_aoi": len(reviewed_inside),
            "unchanged": len(layers["unchanged"]), "modified": len(layers["modified"]),
            "added": len(layers["added"]), "removed": len(layers["removed"]),
            "crossing_aoi_edge": {"original": int((original_excluded.aoi_action == "crossing_excluded").sum()),
                                  "reviewed": int((reviewed_excluded.aoi_action == "crossing_excluded").sum())},
            "reviewed_problems": reviewed_problems,
            "original_problems": len(problems) - reviewed_problems,
            "reviewed_overlaps": len(overlaps),
        },
        # Overlaps are allowed (e.g. a carport under a roof) but worth a second look; problems are not.
        "ready_for_evaluation": reviewed_problems == 0,
        "provenance": {"checked_utc": datetime.now(timezone.utc).isoformat(), **(provenance or {})},
    }
    return summary, layers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="the inventory the review started from")
    parser.add_argument("--original-layer")
    parser.add_argument("--reviewed", required=True, help="the reviewed outlines")
    parser.add_argument("--reviewed-layer")
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--aoi-layer")
    parser.add_argument("--metric-crs", required=True)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--unchanged-iou", type=float, default=0.95)
    parser.add_argument("--min-overlap-m2", type=float, default=0.5)
    parser.add_argument("--reviewer", help="recorded in the provenance")
    parser.add_argument("--imagery-date", help="capture date the outlines were reviewed against, e.g. 2026-05")
    parser.add_argument("--label-convention", help="e.g. 'roof outline; sheds and carports included from 4 m2'")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-gpkg", type=Path, help="added, removed, modified, unchanged, problems, overlaps")
    args = parser.parse_args(argv)
    inputs = {Path(value).resolve() for value in (args.original, args.reviewed, args.aoi)}
    outputs = [path.resolve() for path in (args.output_json, args.output_gpkg) if path]
    if len(set(outputs)) != len(outputs) or any(path.exists() or path in inputs for path in outputs):
        parser.error("outputs must be new, distinct files")
    read = lambda path, layer: gpd.read_file(path, **({"layer": layer} if layer else {}))  # noqa: E731
    provenance = {key: value for key, value in (("reviewer", args.reviewer), ("imagery_date", args.imagery_date),
                                                ("label_convention", args.label_convention)) if value}
    summary, layers = review_check(read(args.original, args.original_layer), read(args.reviewed, args.reviewed_layer),
                                   read(args.aoi, args.aoi_layer), crs=args.metric_crs, match_iou=args.match_iou,
                                   unchanged_iou=args.unchanged_iou, min_overlap_m2=args.min_overlap_m2,
                                   provenance=provenance)
    summary["sources"] = {role: {"path": str(Path(path).resolve()), "layer": layer} for role, path, layer in
                          (("original", args.original, args.original_layer), ("reviewed", args.reviewed, args.reviewed_layer),
                           ("aoi", args.aoi, args.aoi_layer))}
    if args.output_gpkg:
        write_layers(args.output_gpkg, layers)
    write_json(args.output_json, summary)
    print(json.dumps({"ready_for_evaluation": summary["ready_for_evaluation"], **summary["counts"]}))


if __name__ == "__main__":
    main()
