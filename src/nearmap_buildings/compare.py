"""Compare evaluation JSON reports only when the holdout and settings match."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


FINGERPRINT_METHOD = "sha256-normalized-wkb-source-crs-v1"
COMPATIBILITY_FIELDS = ("schema_version", "fingerprint_method", "reference_fingerprint",
                        "aoi_fingerprint", "metric_crs", "iou_threshold", "edge_policy",
                        "matching", "metric_definitions")
CSV_FIELDS = ("label", "report", "true_positives", "false_positives", "false_negatives",
              "precision", "recall", "f1", "matched_iou_mean", "matched_iou_median",
              "matched_absolute_area_error_m2_mean", "matched_absolute_relative_area_error_mean",
              "matched_boundary_hausdorff_m_mean", "matched_boundary_hausdorff_m_p95",
              "evaluated_predictions", "evaluated_reference", "excluded_predictions",
              "excluded_reference", "clipped_predictions", "clipped_reference",
              "reference_fingerprint", "aoi_fingerprint", "metric_crs", "iou_threshold", "edge_policy")


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
        if not isinstance(report[field], str) or not re.fullmatch(r"[0-9a-f]{64}", report[field]):
            raise ValueError(f"{path}: invalid {field}")
    if not isinstance(report["metric_crs"], str) or not report["metric_crs"]:
        raise ValueError(f"{path}: missing metric CRS")
    _number(report["iou_threshold"], "iou_threshold", ratio=True)
    if report["iou_threshold"] <= 0 or report["edge_policy"] not in {"exclude", "clip"}:
        raise ValueError(f"{path}: invalid evaluation settings")
    for field in ("true_positives", "false_positives", "false_negatives"):
        _number(report[field], field, count=True)
    for field in ("precision", "recall", "f1"):
        _number(report[field], field, ratio=True, nullable=True)


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
        if mismatches:
            raise ValueError(f"{path}: incompatible evaluation report ({', '.join(mismatches)}); reports must use the same holdout and settings")
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
                   **{field: report[field] for field in ("reference_fingerprint", "aoi_fingerprint", "metric_crs", "iou_threshold", "edge_policy")}}
            for field, value in row.items():
                if field.startswith("matched_"):
                    _number(value, field, nullable=True)
                elif field.startswith(("evaluated_", "excluded_", "clipped_")):
                    _number(value, field, count=True)
        except KeyError as error:
            raise ValueError(f"{path}: incomplete evaluation report, missing {error.args[0]}") from error
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", help="one unique label per report; default is its parent folder")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".csv":
        parser.error("output must have a .csv extension")
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    rows = compare_reports(args.reports, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create closes the exists-check race and never overwrites a report.
    with args.output.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"reports": len(rows), "output": str(args.output.resolve()),
                      "same_holdout_and_settings": True, "undefined_metrics": "empty CSV cells"}))


if __name__ == "__main__":
    main()
