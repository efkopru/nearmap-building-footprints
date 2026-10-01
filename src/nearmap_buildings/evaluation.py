"""Evaluate predictions against independently annotated, completely labeled AOIs.

No pseudo-labels are accepted as validation evidence. Source geometries must be
valid polygons. All distances/areas are computed in the supplied metre-based
projected CRS. AOI-crossing objects are excluded by default and counted explicitly;
--edge-policy clip clips both datasets and retains each original instance identity.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import math
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.strtree import STRtree

from .common import FINGERPRINT_METHOD
from .postprocess import metric_crs, polygon_parts, write_layers


ROW_ORDER_FINGERPRINT_METHOD = "sha256-normalized-wkb-source-crs-row-order-v1"
# What the reference is. Only an independent, completely labeled holdout measures accuracy.
REFERENCE_REQUIREMENTS = {
    "independent_holdout": "Independent annotated holdout; AOI completely labeled. Independence is asserted by the operator, not inferred by this program. No pseudo-label validation.",
    "agreement_only": "An existing inventory the operator marks as possibly outdated or incomplete. The metrics measure agreement with it, not accuracy: an unmatched prediction may be a real building missing from the inventory.",
}


def _normalized_wkb(geometry) -> bytes:
    return shapely.to_wkb(shapely.normalize(geometry), hex=False, byte_order=1,
                          output_dimension=3, include_srid=False)


def _geometry_digest(frame: gpd.GeoDataFrame, method: str, geometries) -> str:
    if frame.crs is None:
        raise ValueError("fingerprinting requires an explicit source CRS")
    authority = frame.crs.to_authority(min_confidence=100)
    crs_key = ":".join(authority) if authority else frame.crs.to_wkt(version="WKT2_2019", pretty=False)
    digest = hashlib.sha256()
    digest.update(json.dumps({"method": method, "crs": crs_key,
                              "feature_count": len(frame)}, sort_keys=True).encode("utf-8"))
    for geometry in geometries:
        digest.update(len(geometry).to_bytes(8, "big"))
        digest.update(geometry)
    return digest.hexdigest()


def geometry_fingerprint(frame: gpd.GeoDataFrame) -> str:
    """Hash source geometry and CRS, independent of row/ring/part ordering.

    Exact coordinates and duplicate geometries are retained. Feature attributes
    are excluded because this evaluator treats every feature as one building.
    Polygon decomposition and any coordinate change intentionally change the hash.
    This proves matching input geometry, not annotation quality or independence.
    """
    return _geometry_digest(frame, FINGERPRINT_METHOD,
                            sorted(_normalized_wkb(geometry) for geometry in frame.geometry))


def row_order_fingerprint(frame: gpd.GeoDataFrame) -> str:
    """Like geometry_fingerprint, but also fixes which row holds which geometry.

    Reports refer to reference buildings by row position, so joining reports
    building by building needs the reference rows in the same order each time.
    """
    return _geometry_digest(frame, ROW_ORDER_FINGERPRINT_METHOD,
                            [_normalized_wkb(geometry) for geometry in frame.geometry])


def size_bin_edges(edges) -> list[float]:
    """Validate size-class edges in square metres: positive, finite and increasing."""
    edges = [float(edge) for edge in edges]
    if not edges or any(not math.isfinite(edge) or edge <= 0 for edge in edges) \
            or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("size bins must be positive, finite, strictly increasing areas in m2")
    return edges


def _edge_text(value: float) -> str:
    return f"{value:g}"


def size_class_labels(edges) -> list[str]:
    """Labels for the classes the edges make, e.g. [20, 50] gives <20, 20-50 and >=50."""
    text = [_edge_text(edge) for edge in edges]
    return [f"<{text[0]}", *(f"{a}-{b}" for a, b in zip(text, text[1:])), f">={text[-1]}"]


def size_class(area: float, edges, labels) -> str:
    # bisect_right puts an area exactly on an edge in the class above it.
    return labels[bisect_right(edges, area)]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _subset_metrics(reference_selected, reference_matched, prediction_selected, prediction_matched):
    """Recall over the selected references, precision over the selected predictions."""
    references, predictions = int(reference_selected.sum()), int(prediction_selected.sum())
    matched_references = int((reference_selected & reference_matched).sum())
    matched_predictions = int((prediction_selected & prediction_matched).sum())
    precision, recall = _ratio(matched_predictions, predictions), _ratio(matched_references, references)
    if precision is None or recall is None:
        # A pair can span two classes, so one side of a class may be empty while the
        # other side all matched. F1 is zero only when the defined ratio is zero.
        defined = recall if precision is None else precision
        f1 = 0.0 if defined == 0 else None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"reference": references, "matched_reference": matched_references,
            "predictions": predictions, "matched_predictions": matched_predictions,
            "precision": precision, "recall": recall, "f1": f1}


def size_metrics(reference_areas, reference_matched, prediction_areas, prediction_matched, edges):
    """Per-class and at-or-above-edge metrics from one global matching.

    A reference counts in the class of its own area, and so does a prediction, so
    recall in a class is over its references and precision over its predictions.
    Matching is not redone per class: a matched pair may span two classes.
    """
    labels = size_class_labels(edges)
    reference_areas, prediction_areas = np.asarray(reference_areas, float), np.asarray(prediction_areas, float)
    reference_matched, prediction_matched = np.asarray(reference_matched, bool), np.asarray(prediction_matched, bool)
    bounds = [0.0, *edges, math.inf]
    classes = []
    for label, low, high in zip(labels, bounds, bounds[1:]):
        classes.append({"size_class": label, "min_m2": low, "max_m2": None if math.isinf(high) else high,
                        **_subset_metrics((reference_areas >= low) & (reference_areas < high), reference_matched,
                                          (prediction_areas >= low) & (prediction_areas < high), prediction_matched)})
    thresholds = [{"size_class": f">={_edge_text(edge)}", "min_m2": edge, **_subset_metrics(reference_areas >= edge, reference_matched,
                                                     prediction_areas >= edge, prediction_matched)}
                  for edge in edges]
    return classes, thresholds


@dataclass
class EvaluationResult:
    report: dict
    layers: dict[str, gpd.GeoDataFrame]


def _validate(frame: gpd.GeoDataFrame, label: str, target, extra_reserved=()):
    if frame.crs is None:
        raise ValueError(f"{label}: missing CRS")
    frame = frame.to_crs(target).copy().reset_index(drop=True)
    for index, geometry in enumerate(frame.geometry):
        if geometry is None or geometry.is_empty:
            raise ValueError(f"{label}: null/empty geometry at row {index}; correct the input rather than silently drop it")
        if not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.is_valid or geometry.area <= 0:
            raise ValueError(f"{label}: invalid/nonpolygon geometry at row {index}; repair and review before evaluation")
    reserved = {"eval_id", "aoi_action", "aoi_boundary_touch", "match_id", "match_iou",
                "area_error_m2", "relative_area_error", "boundary_hausdorff_m", *extra_reserved}
    if reserved.intersection(frame.columns):
        raise ValueError(f"{label}: input contains reserved evaluation fields")
    return frame


def _within_aoi(frame, aoi, policy):
    data = frame.copy()
    data["eval_id"] = np.arange(len(data), dtype=np.int64)
    data["aoi_action"] = ""
    data["aoi_boundary_touch"] = False
    selected, excluded = [], []
    for index, geometry in enumerate(data.geometry):
        data.at[index, "aoi_boundary_touch"] = bool(geometry.intersects(aoi.boundary))
        if aoi.covers(geometry):
            data.at[index, "aoi_action"] = "inside"
            selected.append(index)
        elif geometry.intersection(aoi).area <= 0:
            data.at[index, "aoi_action"] = "outside"
            excluded.append(index)
        elif policy == "exclude":
            data.at[index, "aoi_action"] = "crossing_excluded"
            excluded.append(index)
        else:
            clipped = polygon_parts(geometry.intersection(aoi))
            if clipped is None or not clipped.is_valid or clipped.area <= 0:
                raise ValueError("AOI clipping produced an invalid geometry; review source geometry")
            data.at[index, data.geometry.name] = clipped
            data.at[index, "aoi_action"] = "clipped"
            selected.append(index)
    return data.loc[selected].copy().reset_index(drop=True), data.loc[excluded].copy()


def maximum_cardinality_matches(predictions, references, threshold: float = 0.5):
    """Maximum number of one-to-one IoU-qualified matches; then maximize total IoU.

    Assignment runs per connected overlap component, avoiding a full city-sized
    dense matrix. A component with k potential matches uses a k+1 bonus per valid
    edge, so one additional valid match always outweighs all IoU tie-break scores.
    """
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if len(predictions) == 0 or len(references) == 0:
        return []
    tree = STRtree(list(references))
    edges = []
    for prediction_index, prediction in enumerate(predictions):
        for reference_index in tree.query(prediction, predicate="intersects"):
            reference_index = int(reference_index)
            reference = references[reference_index]
            intersection = prediction.intersection(reference).area
            iou = intersection / (prediction.area + reference.area - intersection)
            if iou >= threshold:
                edges.append((prediction_index, reference_index, float(iou)))
    if not edges:
        return []
    n_pred, n_ref = len(predictions), len(references)
    graph = coo_matrix((np.ones(len(edges)),
                        ([edge[0] for edge in edges], [n_pred + edge[1] for edge in edges])),
                       shape=(n_pred + n_ref, n_pred + n_ref)).tocsr()
    _, labels = connected_components(graph, directed=False)
    components = {}
    for edge in edges:
        components.setdefault(int(labels[edge[0]]), []).append(edge)
    matches = []
    for component in components.values():
        prediction_ids = sorted({edge[0] for edge in component})
        reference_ids = sorted({edge[1] for edge in component})
        prediction_lookup = {value: index for index, value in enumerate(prediction_ids)}
        reference_lookup = {value: index for index, value in enumerate(reference_ids)}
        bonus = min(len(prediction_ids), len(reference_ids)) + 1
        weights = np.zeros((len(prediction_ids), len(reference_ids) + len(prediction_ids)))
        ious = {}
        for prediction_id, reference_id, iou in component:
            weights[prediction_lookup[prediction_id], reference_lookup[reference_id]] = bonus + iou
            ious[(prediction_id, reference_id)] = iou
        rows, columns = linear_sum_assignment(weights, maximize=True)
        for row, column in zip(rows, columns):
            if column < len(reference_ids):
                pair = (prediction_ids[int(row)], reference_ids[int(column)])
                if pair in ious:
                    matches.append((*pair, ious[pair]))
    return sorted(matches)


def _summary(values):
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def evaluate(predictions: gpd.GeoDataFrame, reference: gpd.GeoDataFrame,
             aoi: gpd.GeoDataFrame, *, crs: str, iou_threshold: float = 0.5,
             edge_policy: str = "exclude", reference_status: str = "independent_holdout",
             size_bins=None) -> EvaluationResult:
    """Evaluate independently labeled objects, never model-generated pseudo-labels.

    This API cannot prove label independence; the caller must establish provenance.
    CLI users must explicitly acknowledge an independent, completely labeled AOI, or
    declare the reference an existing inventory so the report records agreement only.
    Empty dataframes are valid; malformed rows are errors. Undefined ratios are null.
    size_bins, edges in square metres such as [20, 50, 100], adds per-size-class metrics.
    """
    edges = size_bin_edges(size_bins) if size_bins is not None else None
    if not 0 < iou_threshold <= 1 or edge_policy not in {"exclude", "clip"}:
        raise ValueError("IoU must be in (0, 1] and edge policy must be exclude or clip")
    if reference_status not in REFERENCE_REQUIREMENTS:
        raise ValueError(f"reference_status must be one of {sorted(REFERENCE_REQUIREMENTS)}")
    target = metric_crs(crs)
    # size_class is only written, and so only reserved, when size bins are requested.
    extra = ("size_class",) if edges is not None else ()
    pred = _validate(predictions, "predictions", target, extra)
    ref = _validate(reference, "reference", target, extra)
    area = _validate(aoi, "AOI", target)
    if area.empty:
        raise ValueError("AOI must contain at least one polygon")
    region = shapely.union_all(area.geometry.to_numpy())
    pred, excluded_pred = _within_aoi(pred, region, edge_policy)
    ref, excluded_ref = _within_aoi(ref, region, edge_policy)
    matches = maximum_cardinality_matches(list(pred.geometry), list(ref.geometry), iou_threshold)
    for frame in (pred, ref):
        frame["match_id"] = -1
        for field in ("match_iou", "area_error_m2", "relative_area_error", "boundary_hausdorff_m"):
            frame[field] = np.nan
    match_rows = []
    for prediction_index, reference_index, iou in matches:
        prediction, actual = pred.geometry.iloc[prediction_index], ref.geometry.iloc[reference_index]
        area_error = float(prediction.area - actual.area)
        relative_error = area_error / actual.area
        distance = float(shapely.hausdorff_distance(prediction.boundary, actual.boundary, densify=0.25))
        for frame, index, other in ((pred, prediction_index, int(ref.eval_id.iloc[reference_index])),
                                    (ref, reference_index, int(pred.eval_id.iloc[prediction_index]))):
            frame.loc[index, ["match_id", "match_iou", "area_error_m2", "relative_area_error", "boundary_hausdorff_m"]] = [other, iou, area_error, relative_error, distance]
        match_rows.append({"prediction_id": int(pred.eval_id.iloc[prediction_index]),
                           "reference_id": int(ref.eval_id.iloc[reference_index]), "iou": iou,
                           "area_error_m2": area_error, "relative_area_error": relative_error,
                           "boundary_hausdorff_m": distance})
    tp = len(matches)
    fp, fn = len(pred) - tp, len(ref) - tp
    if edges is not None:
        labels = size_class_labels(edges)
        for frame in (pred, ref):
            frame["size_class"] = [size_class(geometry.area, edges, labels) for geometry in frame.geometry]
    report = {
        "schema_version": 1,
        "fingerprint_method": FINGERPRINT_METHOD,
        # Identifies which prediction set produced this report; not a compatibility check.
        "predictions_fingerprint": geometry_fingerprint(predictions),
        "reference_fingerprint": geometry_fingerprint(reference),
        "reference_order_fingerprint": row_order_fingerprint(reference),
        # Lets nbf agree link prediction_id values back to rows of the same predictions file.
        "predictions_order_fingerprint": row_order_fingerprint(predictions),
        "aoi_fingerprint": geometry_fingerprint(aoi),
        "fingerprint_definition": "SHA-256 of source CRS and sorted normalized WKB geometries, including duplicate features. Independent of row order, ring start/direction, and multipart order; exact coordinates retained. Attributes excluded. This is input identity, not proof of annotation independence. reference_order_fingerprint hashes the same geometries in row order, so reference_id values keep pointing at the same buildings.",
        "reference_status": reference_status,
        "reference_requirement": REFERENCE_REQUIREMENTS[reference_status],
        "metric_crs": target.to_string(), "iou_threshold": iou_threshold,
        "edge_policy": edge_policy,
        "edge_policy_definition": "exclude: retain whole objects covered by AOI, including boundary-touching objects; exclude crossings. clip: intersect both datasets with AOI and retain one identity per original object. Zero-area intersections are outside.",
        "matching": "One-to-one maximum-cardinality assignment among IoU-qualified pairs, then maximum total IoU; no score threshold is applied by evaluation.",
        "input_counts": {"predictions": len(predictions), "reference": len(reference)},
        "evaluated_counts": {"predictions": len(pred), "reference": len(ref)},
        "excluded_counts": {"predictions": {str(k): int(v) for k, v in excluded_pred.aoi_action.value_counts().items()},
                            "reference": {str(k): int(v) for k, v in excluded_ref.aoi_action.value_counts().items()}},
        "clipped_counts": {"predictions": int((pred.aoi_action == "clipped").sum()),
                           "reference": int((ref.aoi_action == "clipped").sum())},
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "undefined_metric_policy": "A ratio with zero denominator is null; F1 is 0 when only one dataset is empty and null when both are empty.",
        "matched_only": {
            "iou": _summary([row["iou"] for row in match_rows]),
            "signed_area_error_m2": _summary([row["area_error_m2"] for row in match_rows]),
            "absolute_area_error_m2": _summary([abs(row["area_error_m2"]) for row in match_rows]),
            "absolute_relative_area_error": _summary([abs(row["relative_area_error"]) for row in match_rows]),
            "boundary_hausdorff_m": _summary([row["boundary_hausdorff_m"] for row in match_rows]),
        },
        "metric_definitions": {
            "iou": "intersection area / union area, dimensionless",
            "area_error_m2": "prediction area minus reference area, square metres",
            "relative_area_error": "(prediction area minus reference area) / reference area, dimensionless",
            "boundary_hausdorff_m": "Symmetric discrete Hausdorff distance between polygon boundaries in metres, with Shapely densify=0.25; includes interior rings. Not average boundary error.",
        },
        # Row positions in the reference source; matches[].reference_id uses the same numbering.
        "evaluated_reference_ids": [int(value) for value in ref.eval_id],
        # Row positions in the predictions source; matches[].prediction_id uses the same numbering.
        "evaluated_prediction_ids": [int(value) for value in pred.eval_id],
        "size_bins_m2": edges,
        "matches": match_rows,
    }
    if edges is not None:
        classes, thresholds = size_metrics([g.area for g in ref.geometry], ref.match_id >= 0,
                                           [g.area for g in pred.geometry], pred.match_id >= 0, edges)
        report.update({
            "size_class_definition": "Each object is classed by its own area in the metric CRS, after AOI clipping. Recall in a class is over its reference objects and precision over its predictions; matching is the global one-to-one matching, not redone per class, so a matched pair may span two classes. A class with only references or only predictions has an undefined (null) F1 unless its one defined ratio is 0, when F1 is 0. size_thresholds give the same metrics for objects at or above each edge.",
            "size_classes": classes,
            "size_thresholds": thresholds,
            # Aligned with evaluated_reference_ids.
            "evaluated_reference_size_classes": list(ref.size_class),
        })
    return EvaluationResult(report, {
        "matched_predictions": pred.loc[pred.match_id >= 0].copy(),
        "matched_reference": ref.loc[ref.match_id >= 0].copy(),
        "unmatched_predictions": pred.loc[pred.match_id < 0].copy(),
        "unmatched_reference": ref.loc[ref.match_id < 0].copy(),
        "excluded_predictions": excluded_pred, "excluded_reference": excluded_ref,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--predictions-layer")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-layer")
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--aoi-layer")
    parser.add_argument("--metric-crs", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-gpkg", help="optional matched, unmatched, and AOI-excluded objects")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--edge-policy", choices=["exclude", "clip"], default="exclude")
    parser.add_argument("--size-bins-m2", type=float, nargs="+", metavar="AREA",
                        help="also report metrics by building size, e.g. 20 50 100 for <20, 20-50, 50-100 and >=100 m2")
    status = parser.add_mutually_exclusive_group(required=True)
    status.add_argument("--independent-holdout", action="store_true",
                        help="acknowledge independent annotations and complete labels within AOI, not pseudo-labels")
    status.add_argument("--agreement-only", action="store_true",
                        help="the reference is an existing inventory that may be outdated or incomplete; report agreement, not accuracy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    sources = [Path(value).resolve() for value in (args.predictions, args.reference, args.aoi)]
    destinations = [Path(value).resolve() for value in (args.output_json, args.output_gpkg) if value]
    if len(set(destinations)) != len(destinations):
        parser.error("JSON and GPKG outputs must be different files")
    for output in destinations:
        if any(output == source or (output.exists() and output.samefile(source)) for source in sources):
            parser.error("output must not overwrite an input file")
        if output.exists() and not args.overwrite:
            parser.error(f"output already exists: {output}")
    frames = []
    for filename, layer in ((args.predictions, args.predictions_layer), (args.reference, args.reference_layer), (args.aoi, args.aoi_layer)):
        frames.append(gpd.read_file(filename, layer=layer) if layer else gpd.read_file(filename))
    result = evaluate(*frames, crs=args.metric_crs, iou_threshold=args.iou_threshold, edge_policy=args.edge_policy,
                      reference_status="agreement_only" if args.agreement_only else "independent_holdout",
                      size_bins=args.size_bins_m2)
    result.report["sources"] = {role: {"path": str(path), "layer": layer} for role, path, layer in
                                zip(("predictions", "reference", "aoi"), sources,
                                    (args.predictions_layer, args.reference_layer, args.aoi_layer))}
    if args.output_gpkg:
        write_layers(args.output_gpkg, result.layers, overwrite=args.overwrite)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result.report[key] for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")}))


if __name__ == "__main__":
    main()
