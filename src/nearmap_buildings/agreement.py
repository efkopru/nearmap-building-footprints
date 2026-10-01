"""Match two methods' outlines to each other, without reference labels.

Uses the same one-to-one matching as nbf evaluate. Where the reference is outdated,
an outline two independent methods agree on but the reference lacks is a likely new
or changed building: pass each method's evaluation report to list those candidates.
Agreement between two models is not accuracy; both can be wrong in the same way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .common import FINGERPRINT_METHOD, read_json, write_json
from .compare import FINGERPRINT_PATTERN, _field_prefix
from .evaluation import (_summary, _validate, _within_aoi, geometry_fingerprint, maximum_cardinality_matches,
                         row_order_fingerprint)
from .postprocess import metric_crs, write_layers


def _reference_matches(report, frame, label):
    """Which rows of frame the report evaluated, and which of those matched the reference."""
    if not isinstance(report, dict):
        raise ValueError(f"{label}: evaluation report must be a JSON object")
    order = report.get("predictions_order_fingerprint")
    if not isinstance(order, str) or not FINGERPRINT_PATTERN.fullmatch(order) \
            or not isinstance(report.get("evaluated_prediction_ids"), list):
        raise ValueError(f"{label}: report predates nbf agree; re-run nbf evaluate on the same predictions file")
    if order != row_order_fingerprint(frame):
        raise ValueError(f"{label}: the report was made from different predictions, or the same ones in another row order")
    evaluated = set(report["evaluated_prediction_ids"])
    matched = {match["prediction_id"] for match in report.get("matches", [])}
    if not matched <= evaluated:
        raise ValueError(f"{label}: report matches refer to predictions it did not evaluate")
    return evaluated, matched


def agree(first: gpd.GeoDataFrame, second: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, *, crs: str,
          labels=("a", "b"), iou_threshold: float = 0.5, edge_policy: str = "exclude", reports=(None, None)):
    """Return (summary, layers): pairs both methods drew, and outlines only one drew.

    reports holds an optional nbf evaluate report for each method, made from the same
    file in the same row order. With at least one, the candidate_new layer lists
    agreed pairs that no supplied report matched to its reference.
    """
    if edge_policy not in {"exclude", "clip"}:
        raise ValueError("edge policy must be exclude or clip")
    labels = [str(label).strip() for label in labels]
    prefixes = [_field_prefix(label) for label in labels]
    if len(labels) != 2 or not all(labels) or prefixes[0] == prefixes[1]:
        raise ValueError("give two distinct labels")
    generated = {"agreement_iou", "reference_match", *(f"{prefix}_{field}" for prefix in prefixes
                                                      for field in ("match_id", "reference_match"))}
    for frame, label in zip((first, second), labels):
        if generated.intersection(frame.columns):
            raise ValueError(f"{label}: input already has agreement fields {sorted(generated.intersection(frame.columns))}")
    target = metric_crs(crs)
    sources = (first, second)
    frames = [_validate(frame, label, target) for frame, label in zip(sources, labels)]
    area = _validate(aoi, "AOI", target)
    if area.empty:
        raise ValueError("AOI must contain at least one polygon")
    region = shapely.union_all(area.geometry.to_numpy())
    selected = [_within_aoi(frame, region, edge_policy)[0] for frame in frames]
    matches = maximum_cardinality_matches(list(selected[0].geometry), list(selected[1].geometry), iou_threshold)
    reference = [_reference_matches(report, source, label) if report is not None else None
                 for report, source, label in zip(reports, sources, labels)]
    for index, (frame, other) in enumerate(zip(selected, reversed(prefixes))):
        frame[f"{other}_match_id"] = -1
        frame["agreement_iou"] = np.nan
        if reference[index] is not None:
            evaluated, matched = reference[index]
            # Null where the evaluation left the outline out, e.g. it crossed that report's AOI edge.
            frame["reference_match"] = pd.array([(int(row) in matched) if int(row) in evaluated else None
                                                 for row in frame.eval_id], dtype="boolean")
    for row_a, row_b, iou in matches:
        selected[0].loc[row_a, [f"{prefixes[1]}_match_id", "agreement_iou"]] = [int(selected[1].eval_id.iloc[row_b]), iou]
        selected[1].loc[row_b, [f"{prefixes[0]}_match_id", "agreement_iou"]] = [int(selected[0].eval_id.iloc[row_a]), iou]
    a, b = selected
    both = a.loc[a[f"{prefixes[1]}_match_id"] >= 0].copy()
    layers = {"both": both,
              f"only_{prefixes[0]}": a.loc[a[f"{prefixes[1]}_match_id"] < 0].copy(),
              f"only_{prefixes[1]}": b.loc[b[f"{prefixes[0]}_match_id"] < 0].copy()}
    summary = {
        "schema_version": 1,
        "labels": labels,
        "fingerprint_method": FINGERPRINT_METHOD,
        "fingerprints": {label: geometry_fingerprint(source) for label, source in zip(labels, sources)},
        "aoi_fingerprint": geometry_fingerprint(aoi),
        "metric_crs": target.to_string(), "iou_threshold": iou_threshold, "edge_policy": edge_policy,
        "matching": "One-to-one maximum-cardinality assignment among IoU-qualified pairs, then maximum total IoU, as in nbf evaluate.",
        "interpretation": "Agreement between two methods, not accuracy against labels.",
        "evaluated_counts": {label: len(frame) for label, frame in zip(labels, selected)},
        "pairs": len(matches),
        "share_matched": {label: (len(matches) / len(frame) if len(frame) else None)
                          for label, frame in zip(labels, selected)},
        "pair_iou": _summary([iou for _, _, iou in matches]),
    }
    supplied = [index for index, value in enumerate(reference) if value is not None]
    if supplied:
        # The both layer holds the first method's outlines; carry the second's reference result over.
        if reference[1] is not None:
            lookup = dict(zip(b.eval_id, b.reference_match))
            both[f"{prefixes[1]}_reference_match"] = pd.array([lookup[value] for value in both[f"{prefixes[1]}_match_id"]],
                                                              dtype="boolean")
        if reference[0] is not None:
            both = both.rename(columns={"reference_match": f"{prefixes[0]}_reference_match"})
        layers["both"] = both
        unmatched = np.ones(len(both), dtype=bool)
        for index in supplied:
            # False only: an outline the evaluation left out is not evidence either way.
            unmatched &= both[f"{prefixes[index]}_reference_match"].eq(False).fillna(False).to_numpy(dtype=bool)
        layers["candidate_new"] = both.loc[unmatched].copy()
        summary["candidate_new"] = int(unmatched.sum())
        summary["candidate_new_definition"] = (
            "Pairs both methods drew where no supplied evaluation report matched the outline to its reference "
            f"(reports supplied for: {', '.join(labels[index] for index in supplied)}). "
            "Likely new or changed since the reference was made; review them before using them as labels.")
        for index in supplied:
            frame, (_, matched) = selected[index], reference[index]
            no_reference = frame.loc[frame.reference_match.eq(False).fillna(False).to_numpy(dtype=bool)]
            summary.setdefault("unmatched_to_reference", {})[labels[index]] = {
                "outlines": len(no_reference),
                "matched_by_other": int((no_reference[f"{prefixes[1 - index]}_match_id"] >= 0).sum())}
    return summary, layers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs=2, required=True, metavar="FILE", help="the two methods' outlines")
    parser.add_argument("--layers", nargs=2, metavar="LAYER", help="layer names, one per file")
    parser.add_argument("--labels", nargs=2, default=["a", "b"])
    parser.add_argument("--reports", nargs=2, metavar="REPORT",
                        help="each method's nbf evaluate report, or - for none; enables candidate_new")
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--aoi-layer")
    parser.add_argument("--metric-crs", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--edge-policy", choices=["exclude", "clip"], default="exclude")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-gpkg", type=Path, help="both, only_<label> and candidate_new layers")
    args = parser.parse_args(argv)
    inputs = [Path(value).resolve() for value in (*args.predictions, args.aoi)]
    outputs = [path.resolve() for path in (args.output_json, args.output_gpkg) if path]
    for output in outputs:
        if output.exists() or output in inputs:
            parser.error(f"output must be a new file: {output}")
    if len(set(outputs)) != len(outputs):
        parser.error("JSON and GPKG outputs must be different files")
    layers = args.layers or [None, None]
    frames = [gpd.read_file(path, **({"layer": layer} if layer else {})) for path, layer in zip(args.predictions, layers)]
    aoi = gpd.read_file(args.aoi, **({"layer": args.aoi_layer} if args.aoi_layer else {}))
    reports = [None if value in (None, "-") else read_json(value) for value in (args.reports or [None, None])]
    summary, result = agree(*frames, aoi, crs=args.metric_crs, labels=args.labels, iou_threshold=args.iou_threshold,
                            edge_policy=args.edge_policy, reports=reports)
    summary["sources"] = {label: {"path": str(Path(path).resolve()), "layer": layer}
                          for label, path, layer in zip(summary["labels"], args.predictions, layers)}
    if args.output_gpkg:
        write_layers(args.output_gpkg, result)
    write_json(args.output_json, summary)
    print(json.dumps({key: summary[key] for key in ("pairs", "share_matched", "candidate_new") if key in summary}))


if __name__ == "__main__":
    main()
