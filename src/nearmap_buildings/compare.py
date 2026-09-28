"""Compare evaluation JSON reports only when the holdout and settings match.

--per-building also maps every evaluated reference building with the methods
that found it, e.g. to see where Esri and SAM 3 disagree against the same labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

from .common import FINGERPRINT_METHOD


COMPATIBILITY_FIELDS = ("schema_version", "fingerprint_method", "reference_fingerprint",
                        "aoi_fingerprint", "metric_crs", "iou_threshold", "edge_policy",
                        "matching", "metric_definitions")
CSV_FIELDS = ("label", "report", "true_positives", "false_positives", "false_negatives",
              "precision", "recall", "f1", "matched_iou_mean", "matched_iou_median",
              "matched_absolute_area_error_m2_mean", "matched_absolute_relative_area_error_mean",
              "matched_boundary_hausdorff_m_mean", "matched_boundary_hausdorff_m_p95",
              "evaluated_predictions", "evaluated_reference", "excluded_predictions",
              "excluded_reference", "clipped_predictions", "clipped_reference",
              "predictions_fingerprint", "reference_fingerprint", "aoi_fingerprint", "metric_crs",
              "iou_threshold", "edge_policy", "reference_status")
FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
REFERENCE_STATUSES = ("independent_holdout", "agreement_only")


def _reference_status(report):
    # Reports written before --agreement-only existed could only be independent holdouts.
    return report.get("reference_status", "independent_holdout")
PER_BUILDING_FIELDS = ("eval_id", "found_by", "found_count", "outcome")


def _number(value, field, *, count=False, ratio=False, nullable=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid numeric field {field}")
    if count and (not isinstance(value, int) or value < 0):
        raise ValueError(f"{field} must be a nonnegative integer")
    if ratio and not 0 <= value <= 1:
        raise ValueError(f"{field} must be in [0, 1]")


def _validate_report(report, path):
    if not isinstance(report, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    for field in COMPATIBILITY_FIELDS:
        if field not in report:
            raise ValueError(f"{path}: missing {field}; regenerate old evaluation reports with holdout fingerprints")
    if report["schema_version"] != 1 or report["fingerprint_method"] != FINGERPRINT_METHOD:
        raise ValueError(f"{path}: unsupported evaluation/fingerprint schema")
    for field in ("reference_fingerprint", "aoi_fingerprint"):
        if not isinstance(report[field], str) or not FINGERPRINT_PATTERN.fullmatch(report[field]):
            raise ValueError(f"{path}: invalid {field}")
    # Optional: reports written before predictions were fingerprinted lack it.
    predictions = report.get("predictions_fingerprint")
    if predictions is not None and (not isinstance(predictions, str) or not FINGERPRINT_PATTERN.fullmatch(predictions)):
        raise ValueError(f"{path}: invalid predictions_fingerprint")
    if _reference_status(report) not in REFERENCE_STATUSES:
        raise ValueError(f"{path}: reference_status must be one of {REFERENCE_STATUSES}")
    if not isinstance(report["metric_crs"], str) or not report["metric_crs"]:
        raise ValueError(f"{path}: missing metric CRS")
    _number(report["iou_threshold"], "iou_threshold", ratio=True)
    if report["iou_threshold"] <= 0 or report["edge_policy"] not in {"exclude", "clip"}:
        raise ValueError(f"{path}: invalid evaluation settings")
    for field in ("true_positives", "false_positives", "false_negatives"):
        _number(report[field], field, count=True)
    for field in ("precision", "recall", "f1"):
        _number(report[field], field, ratio=True, nullable=True)


def _load_reports(paths):
    """Read and validate reports; every report must share the first one's holdout and settings."""
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        try:
            _validate_report(report, path)
        except KeyError as error:
            raise ValueError(f"{path}: incomplete evaluation report, missing {error.args[0]}") from error
        reports.append(report)
    for path, report in zip(paths[1:], reports[1:]):
        mismatches = [field for field in COMPATIBILITY_FIELDS if report[field] != reports[0][field]]
        if _reference_status(report) != _reference_status(reports[0]):
            mismatches.append("reference_status")
        if mismatches:
            raise ValueError(f"{path}: incompatible evaluation report ({', '.join(mismatches)}); reports must use the same holdout and settings")
    return reports


def compare_reports(paths, labels=None):
    """Return comparable CSV rows; refuse old or mismatched holdout reports.

    This does not pool metrics across runs. Every row remains a separate method or
    experiment on identical reference/AOI geometry and compatible evaluator settings.
    """
    paths = [Path(path).resolve() for path in paths]
    if not paths:
        raise ValueError("at least one evaluation report is required")
    if len(set(paths)) != len(paths):
        raise ValueError("the same report was supplied more than once")
    if labels is None:
        labels = [path.parent.name or path.stem for path in paths]
        if len(set(labels)) != len(labels):
            labels = [f"{path.parent.name}/{path.stem}" for path in paths]
    if len(labels) != len(paths) or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("provide exactly one nonempty label per report")
    labels = [label.strip() for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("report labels are ambiguous; supply unique --labels")
    reports = _load_reports(paths)
    rows = []
    for path, label, report in zip(paths, labels, reports):
        try:
            metrics = report["matched_only"]
            row = {"label": label, "report": str(path),
                   **{field: report[field] for field in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")},
                   "matched_iou_mean": metrics["iou"]["mean"],
                   "matched_iou_median": metrics["iou"]["median"],
                   "matched_absolute_area_error_m2_mean": metrics["absolute_area_error_m2"]["mean"],
                   "matched_absolute_relative_area_error_mean": metrics["absolute_relative_area_error"]["mean"],
                   "matched_boundary_hausdorff_m_mean": metrics["boundary_hausdorff_m"]["mean"],
                   "matched_boundary_hausdorff_m_p95": metrics["boundary_hausdorff_m"]["p95"],
                   "evaluated_predictions": report["evaluated_counts"]["predictions"],
                   "evaluated_reference": report["evaluated_counts"]["reference"],
                   "excluded_predictions": sum(report["excluded_counts"]["predictions"].values()),
                   "excluded_reference": sum(report["excluded_counts"]["reference"].values()),
                   "clipped_predictions": report["clipped_counts"]["predictions"],
                   "clipped_reference": report["clipped_counts"]["reference"],
                   "predictions_fingerprint": report.get("predictions_fingerprint"),
                   **{field: report[field] for field in ("reference_fingerprint", "aoi_fingerprint", "metric_crs", "iou_threshold", "edge_policy")},
                   "reference_status": _reference_status(report)}
            for field, value in row.items():
                if field.startswith("matched_"):
                    _number(value, field, nullable=True)
                elif field.startswith(("evaluated_", "excluded_", "clipped_")):
                    _number(value, field, count=True)
        except KeyError as error:
            raise ValueError(f"{path}: incomplete evaluation report, missing {error.args[0]}") from error
        rows.append(row)
    return rows


def _field_prefix(label):
    return re.sub(r"[^0-9a-z]+", "_", label.lower()).strip("_") or "method"


def _outcome(found, labels):
    if len(labels) == 1:
        return "found" if found else "missed"
    everyone = "both" if len(labels) == 2 else "all"
    if not found:
        return f"missed by {everyone}"
    if len(found) == len(labels):
        return f"found by {everyone}"
    if len(found) == 1:
        return f"only {found[0]}"
    return f"found by {len(found)} of {len(labels)}"


def per_building(paths, labels, reference=None, reference_layer=None):
    """One row per evaluated reference building, recording which methods matched it.

    Reports identify reference buildings by row position, so the reference file must
    hold the same rows in the same order as when the reports were made. Its row-order
    fingerprint is checked before joining. match_id is the matched prediction's row
    in that method's evaluated predictions (-1 when missed), as in nbf evaluate layers.
    """
    import geopandas as gpd

    from .evaluation import row_order_fingerprint

    paths = [Path(path).resolve() for path in paths]
    reports = _load_reports(paths)
    for path, report in zip(paths, reports):
        if (not FINGERPRINT_PATTERN.fullmatch(str(report.get("reference_order_fingerprint", "")))
                or not isinstance(report.get("evaluated_reference_ids"), list)):
            raise ValueError(f"{path}: report predates per-building comparison; re-run nbf evaluate")
    if len({report["reference_order_fingerprint"] for report in reports}) > 1:
        raise ValueError("Reports were made from reference files with rows in different orders; "
                         "re-run nbf evaluate with one reference file")
    ids = reports[0]["evaluated_reference_ids"]
    if any(report["evaluated_reference_ids"] != ids for report in reports[1:]):
        raise ValueError("Reports evaluated different reference buildings; check they share the AOI and edge policy")
    if reference is None:
        recorded = reports[0].get("sources", {}).get("reference")
        if not recorded or not Path(recorded["path"]).exists():
            raise ValueError("Pass --reference: the reports do not record a reference file that exists here")
        reference, reference_layer = recorded["path"], recorded.get("layer")
    frame = gpd.read_file(reference, **({"layer": reference_layer} if reference_layer else {}))
    if frame.crs is None or row_order_fingerprint(frame) != reports[0]["reference_order_fingerprint"]:
        raise ValueError(f"{reference} does not hold the reference rows the reports were made from "
                         "(its content or row order has changed)")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(frame) for value in ids):
        raise ValueError("evaluated_reference_ids must be row positions in the reference file")
    prefixes = [_field_prefix(label) for label in labels]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"Labels {labels} do not give distinct field names; choose clearly different --labels")
    generated = [*PER_BUILDING_FIELDS, *(f"{prefix}_{field}" for prefix in prefixes for field in ("match_id", "match_iou"))]
    clashes = sorted(set(generated) & set(frame.columns))
    if clashes:
        raise ValueError(f"Reference fields clash with comparison fields: {clashes}")
    buildings = frame.iloc[ids].reset_index(drop=True)
    buildings["eval_id"] = ids
    position = {value: index for index, value in enumerate(ids)}
    found = [[] for _ in ids]
    for path, label, prefix, report in zip(paths, labels, prefixes, reports):
        match_id, match_iou = [-1] * len(ids), [math.nan] * len(ids)
        for match in report["matches"]:
            if match["reference_id"] not in position:
                raise ValueError(f"{path}: a match refers to reference row {match['reference_id']}, which was not evaluated")
            index = position[match["reference_id"]]
            match_id[index], match_iou[index] = match["prediction_id"], match["iou"]
            found[index].append(label)
        buildings[f"{prefix}_match_id"] = match_id
        buildings[f"{prefix}_match_iou"] = match_iou
    buildings["found_by"] = [";".join(names) for names in found]
    buildings["found_count"] = [len(names) for names in found]
    buildings["outcome"] = [_outcome(names, labels) for names in found]
    return buildings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", help="one unique label per report; default is its parent folder")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-building", type=Path,
                        help="also write a GeoPackage of every evaluated reference building and which methods found it")
    parser.add_argument("--reference", type=Path,
                        help="reference file for --per-building; default: the one recorded in the reports")
    parser.add_argument("--reference-layer")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".csv":
        parser.error("output must have a .csv extension")
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    if args.per_building is not None and (args.per_building.suffix.lower() != ".gpkg" or args.per_building.exists()):
        parser.error("--per-building must be a new .gpkg file")
    if args.reference is not None and args.per_building is None:
        parser.error("--reference is only used with --per-building")
    rows = compare_reports(args.reports, args.labels)
    # Build the per-building layer before writing anything, so a failure leaves no partial output.
    buildings = None
    if args.per_building is not None:
        buildings = per_building(args.reports, [row["label"] for row in rows], args.reference, args.reference_layer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create closes the exists-check race and never overwrites a report.
    with args.output.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"reports": len(rows), "output": str(args.output.resolve()),
               "same_holdout_and_settings": True, "undefined_metrics": "empty CSV cells"}
    if buildings is not None:
        from .postprocess import write_layers

        write_layers(args.per_building, {"buildings": buildings})
        summary["per_building"] = str(args.per_building.resolve())
        summary["outcomes"] = {str(name): int(count) for name, count in buildings["outcome"].value_counts().items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
