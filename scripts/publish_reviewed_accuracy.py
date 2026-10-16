"""Publish aggregate results from the completed Lewisville reviewed test run.

Run from the repository root. This reads existing evidence only; it never runs
inference or evaluation. Existing exports require --check (byte comparison).
Private reports, feature IDs, geometry and source paths are never copied.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path


METHODS = {
    "sam3": "SAM 3",
    "esri_city": "Esri, city run",
    "esri_local": "Esri, local raw run",
}
PREDICTIONS = {
    "sam3": "outputs/lewisville/city/sam3_building_1536/cleaned.gpkg",
    "esri_city": "outputs/lewisville/esri_city_202605/cleaned.gpkg",
    "esri_local": "outputs/lewisville/city/esri_local/cleaned.gpkg",
}
HISTORICAL = {
    key: f"outputs/lewisville/agreement_2015/{key}_city/evaluation.json"
    for key in METHODS
}
HISTORICAL["esri_city"] = "outputs/lewisville/agreement_2015/esri_city/evaluation.json"
COHORTS = {"all_sizes": None, "at_least_20m2": 20.0}
SCORE_FIELDS = (
    "cohort method method_key predictions reference tp fp fn precision recall f1 "
    "median_iou mean_iou min_area_m2 min_area_filter_stage"
).split()
SIZE_FIELDS = (
    "cohort method method_key label kind size_class min_m2 max_m2 reference "
    "matched_reference predictions matched_predictions precision recall f1"
).split()
COUNT_FIELDS = (
    "original_in_aoi reviewed_in_aoi unchanged modified added removed "
    "reviewed_problems original_problems reviewed_overlaps"
).split()
ID_COUNT_FIELDS = (
    "added_features_without_original_id removed_original_id_count retained_original_id_count"
).split()
LABEL_CONVENTION = (
    "Visible roof outlines; permanent roofed buildings including detached garages and "
    "sheds around 4 m2 or larger; exclude open carports, canopies, pergolas, tents, "
    "trailers and vehicles; retain the existing splitting of attached buildings and "
    "draw one polygon per separately roofed added building; review every building "
    "touching the AOI, with boundary crossings excluded only when scoring."
)
DEFAULT_SNAPSHOT = (
    "data/reference/review_snapshots/"
    "review_reported_2026-10-15_20261016T030027445168Z"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def count(value):
    require(type(value) is int and value >= 0, "Invalid aggregate count")
    return value


def fingerprint(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Invalid fingerprint")
    return value


def equal_metric(observed, expected, name):
    require(observed is None if expected is None else
            isinstance(observed, (int, float)) and math.isfinite(observed)
            and math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"Inconsistent {name}")


def validate_report(report, receipt, receipt_hash, cohort, historical):
    """Bind aggregate metrics to the reviewed holdout and frozen predictions."""
    require(report.get("schema_version") == 1, "Unsupported evaluation schema")
    for key, expected in {
        "reference_status": "independent_holdout", "metric_crs": "EPSG:26914",
        "iou_threshold": 0.5, "edge_policy": "exclude",
        "size_bins_m2": [20.0, 50.0, 100.0],
        "reference_fingerprint": receipt["reviewed_fingerprint"],
        "aoi_fingerprint": receipt["aoi_fingerprint"],
        "min_area_m2": COHORTS[cohort],
        "min_area_filter_stage": None if cohort == "all_sizes" else "after_aoi_before_matching",
        "predictions_fingerprint": historical["predictions_fingerprint"],
    }.items():
        require(report.get(key) == expected, f"Report binding mismatch: {key}")
    embedded = report.get("review_receipt", {})
    require(embedded.get("sha256") == receipt_hash, "Review receipt hash mismatch")
    for key in ("reviewed_fingerprint", "aoi_fingerprint", "provenance"):
        require(embedded.get(key) == receipt[key], f"Embedded receipt mismatch: {key}")
    tp, fp, fn = (count(report[k]) for k in
                  ("true_positives", "false_positives", "false_negatives"))
    pred = count(report["evaluated_counts"]["predictions"])
    ref = count(report["evaluated_counts"]["reference"])
    require(tp + fp == pred and tp + fn == ref, "Metric count identities failed")
    require(ref == (600 if cohort == "all_sizes" else 468), "Unexpected reference denominator")
    for side in ("predictions", "reference"):
        excluded = sum(count(v) for v in report["excluded_counts"][side].values())
        area = count(report["area_excluded_counts"][side])
        require(count(report["input_counts"][side]) ==
                count(report["evaluated_counts"][side]) + excluded + area,
                "Input/exclusion count identity failed")
        require(report["clipped_counts"][side] == 0, "Unexpected clipped geometry")
        if cohort == "all_sizes":
            require(area == 0, "All-size report contains area exclusions")
    require(report["input_counts"]["reference"] == 640
            and report["excluded_counts"]["reference"] == {"crossing_excluded": 40},
            "Reference boundary policy mismatch")
    equal_metric(report["precision"], tp / pred if pred else None, "precision")
    equal_metric(report["recall"], tp / ref if ref else None, "recall")
    equal_metric(report["f1"], 2 * tp / (pred + ref) if pred + ref else None, "F1")
    for value in (report["matched_only"]["iou"]["mean"], report["matched_only"]["iou"]["median"]):
        require(value is None if tp == 0 else isinstance(value, (int, float))
                and math.isfinite(value) and 0.5 <= value <= 1, "Invalid matched IoU")


def validate_size_classes(report):
    rows = report["size_classes"]
    require(len(rows) == 4, "Expected four diagnostic size classes")
    for row, bounds in zip(rows, (("<20", 0.0, 20.0), ("20-50", 20.0, 50.0),
                                  ("50-100", 50.0, 100.0), (">=100", 100.0, None)), strict=True):
        require(tuple(row[k] for k in ("size_class", "min_m2", "max_m2")) == bounds,
                "Unexpected diagnostic size class")
        ref, pred, matched_ref, matched_pred = (count(row[k]) for k in
            ("reference", "predictions", "matched_reference", "matched_predictions"))
        require(matched_ref <= ref and matched_pred <= pred, "Invalid size-class match count")
        precision = matched_pred / pred if pred else None
        recall = matched_ref / ref if ref else None
        equal_metric(row["precision"], precision, "size precision")
        equal_metric(row["recall"], recall, "size recall")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else 0.0 if pred or ref else None)
        equal_metric(row["f1"], f1, "size F1")
    for field, expected in (("reference", report["evaluated_counts"]["reference"]),
                            ("predictions", report["evaluated_counts"]["predictions"]),
                            ("matched_reference", report["true_positives"]),
                            ("matched_predictions", report["true_positives"])):
        require(sum(row[field] for row in rows) == expected, "Size-class totals mismatch")


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_export(root, snapshot, input_audit=None):
    root, snapshot = Path(root).resolve(), Path(snapshot).resolve()
    accuracy = root / "outputs/accuracy"
    input_audit = Path(input_audit) if input_audit else accuracy / "input_audit.json"
    audit = read_json(input_audit)
    require(set(audit["methods"]) == set(METHODS), "Unexpected audited methods")
    completion_path = snapshot / "review_completion.json"
    completion = read_json(completion_path)
    require(completion["reviewer"] == "Esad"
            and completion["review_date"] == "2026-10-15"
            and completion["imagery_date"] == "2026-05"
            and completion["label_convention"] == LABEL_CONVENTION,
            "Unexpected human review metadata")
    require(completion["coverage_complete"] is True
            and completion["test_area_independent_review"] is True
            and completion["test_area_prior_model_exposure"] is False
            and completion["exceptions"] == [] and completion["unresolved_cases"] == [],
            "Human review attestations incomplete")
    require(completion["snapshot_manifest_sha256"] == digest(snapshot / "manifest.json"),
            "Snapshot manifest hash mismatch")
    receipt_path = accuracy / "review_check.json"
    receipt, receipt_hash = read_json(receipt_path), digest(receipt_path)
    require(receipt["ready_for_evaluation"] is True and receipt["edge_policy"] == "exclude",
            "Review receipt is not ready")
    for key in ("reviewer", "review_date", "imagery_date", "label_convention", "coverage_complete"):
        require(receipt["provenance"][key] == completion[key], f"Review metadata mismatch: {key}")
    require(receipt["provenance"]["independent_review"] is True, "Independent review not attested")
    config_path, run_path = root / "configs/accuracy.example.json", accuracy / "run_record.json"
    config, run = read_json(config_path), read_json(run_path)
    require(run["status"] == "completed" and run["settings_changed"] is False
            and run["model_inference_rerun"] is False, "Run is incomplete or settings changed")
    require(run["config_sha256"] == digest(config_path), "Completed recipe hash mismatch")
    require(len(run["steps"]) == 8 and [s["command"] for s in run["steps"]] == config["steps"]
            and all(s["return_code"] == 0 for s in run["steps"]), "Completed steps mismatch")
    require([s[0] for s in config["steps"]] ==
            ["evaluate"] * 3 + ["compare"] + ["evaluate"] * 3 + ["compare"],
            "Unexpected recipe commands")
    frozen_path = root / "outputs/lewisville/tuning/frozen_settings.json"
    frozen = read_json(frozen_path)
    require(audit["frozen_settings_sha256"] == digest(frozen_path), "Audited frozen settings mismatch")
    settings = {
        "sam3": {"prompt": "building", "tile_px": 1536.0, "threshold": 0.7, "regularize": "off"},
        "esri_city": {"prompt": "", "tile_px": None, "threshold": None, "regularize": "off"},
        "esri_local": {"prompt": "", "tile_px": 512.0, "threshold": 0.9, "regularize": "off"},
    }
    for key, frozen_name in {"sam3": "SAM 3", "esri_city": "Esri city run", "esri_local": "Esri local"}.items():
        require(all(frozen[frozen_name][k] == v for k, v in settings[key].items()),
                "Frozen setting mismatch")
    areas = {}
    for area_name in ("old_town", "test_area"):
        area = completion["areas"][area_name]
        area_receipt_path = accuracy / ("review_check.json" if area_name == "test_area"
                                       else "old_town_review_check.json")
        area_receipt = read_json(area_receipt_path)
        require(digest(area_receipt_path) == area["receipt_sha256"], "Area receipt hash mismatch")
        require(area_receipt["counts"] == area["counts"], "Area review count mismatch")
        for key in ("reviewed_fingerprint", "aoi_fingerprint"):
            require(area_receipt[key] == area[key], "Area fingerprint mismatch")
        reviewed_path = root / "data/reference/test_area_reviewed.gpkg" if area_name == "test_area" else snapshot / area_name / "reviewed.gpkg"
        aoi_path = root / "data/reference/test_area_aoi.gpkg" if area_name == "test_area" else snapshot / area_name / "aoi.gpkg"
        require(digest(reviewed_path) == area["reviewed_file_sha256"]
                and digest(aoi_path) == area["aoi_file_sha256"], "Reviewed source file hash mismatch")
        areas[area_name] = {
            "total_features": count(area["total_features"]),
            "reviewed_fingerprint": fingerprint(area["reviewed_fingerprint"]),
            "aoi_fingerprint": fingerprint(area["aoi_fingerprint"]),
            "reviewed_file_sha256": fingerprint(area["reviewed_file_sha256"]),
            "aoi_file_sha256": fingerprint(area["aoi_file_sha256"]),
            "receipt_sha256": fingerprint(area["receipt_sha256"]),
            "counts": {key: count(area["counts"][key]) for key in COUNT_FIELDS},
            "crossing_aoi_edge": {key: count(area["counts"]["crossing_aoi_edge"][key])
                                  for key in ("original", "reviewed")},
            "id_based_edit_counts": {key: count(area["id_audit"][key]) for key in ID_COUNT_FIELDS},
        }
    scores, by_size, report_hashes, prediction_evidence = [], [], {}, {}
    reference_order = None
    for cohort in COHORTS:
        with (accuracy / cohort / "comparison_by_size.csv").open(newline="", encoding="utf-8-sig") as stream:
            size_rows = [r for r in csv.DictReader(stream) if r["kind"] == "class"]
        require(len(size_rows) == 12 and {r["label"] for r in size_rows} == set(METHODS),
                "Unexpected size-comparison methods or rows")
        for index, (key, method) in enumerate(METHODS.items()):
            report_path = accuracy / cohort / key / "evaluation.json"
            report, historical = read_json(report_path), read_json(root / HISTORICAL[key])
            validate_report(report, receipt, receipt_hash, cohort, historical)
            if reference_order is None:
                reference_order = fingerprint(report["reference_order_fingerprint"])
            require(report["reference_order_fingerprint"] == reference_order,
                    "Reference row-order binding mismatch")
            command = config["steps"][(0 if cohort == "all_sizes" else 4) + index]
            def argument(flag):
                require(command.count(flag) == 1, f"Missing or repeated recipe flag: {flag}")
                return command[command.index(flag) + 1]
            require(argument("--predictions") == PREDICTIONS[key]
                    and argument("--output-json") == f"outputs/accuracy/{cohort}/{key}/evaluation.json"
                    and argument("--review-receipt") == "outputs/accuracy/review_check.json"
                    and argument("--reference") == "data/reference/test_area_reviewed.gpkg"
                    and argument("--aoi") == "data/reference/test_area_aoi.gpkg"
                    and argument("--predictions-layer") == "cleaned"
                    and argument("--metric-crs") == "EPSG:26914"
                    and command.count("--independent-holdout") == 1,
                    "Recipe input/output binding mismatch")
            require(("--min-area-m2" not in command) if cohort == "all_sizes"
                    else argument("--min-area-m2") == "20", "Recipe filter mismatch")
            for name, relative in (("predictions", PREDICTIONS[key]),
                                   ("reference", "data/reference/test_area_reviewed.gpkg"),
                                   ("aoi", "data/reference/test_area_aoi.gpkg")):
                require(Path(report["sources"][name]["path"]).resolve() == root / relative,
                        "Report source path mismatch")
            report_hashes[f"{cohort}/{key}"] = digest(report_path)
            scores.append({
                "cohort": cohort, "method": method, "method_key": key,
                "predictions": report["evaluated_counts"]["predictions"],
                "reference": report["evaluated_counts"]["reference"],
                "tp": report["true_positives"], "fp": report["false_positives"],
                "fn": report["false_negatives"],
                **{name: report[name] for name in ("precision", "recall", "f1", "min_area_m2", "min_area_filter_stage")},
                "median_iou": report["matched_only"]["iou"]["median"],
                "mean_iou": report["matched_only"]["iou"]["mean"],
            })
            if cohort == "all_sizes":
                prediction_evidence[key] = {
                    "sha256": digest(root / PREDICTIONS[key]),
                    "geometry_fingerprint": fingerprint(report["predictions_fingerprint"]),
                    "input_features": count(report["input_counts"]["predictions"]),
                    "historical_report_sha256": digest(root / HISTORICAL[key]),
                    "same_geometry_as_historical_agreement": True,
                    "frozen_settings": settings[key],
                }
                observed = prediction_evidence[key]
                for field, audit_field in (("sha256", "sha256"), ("geometry_fingerprint", "predictions_fingerprint"),
                                           ("input_features", "feature_count"),
                                           ("historical_report_sha256", "historical_report_sha256")):
                    require(observed[field] == audit["methods"][key][audit_field],
                            f"Frozen input audit mismatch: {key}/{field}")
                validate_size_classes(report)
                method_rows = [r for r in size_rows if r["label"] == key]
                require(len(method_rows) == 4, "Missing or duplicate size classes")
                for source, expected in zip(method_rows, report["size_classes"], strict=True):
                    for field, value in expected.items():
                        require(source[field] == ("" if value is None else str(value)),
                                f"Size comparison does not match report: {field}")
                    by_size.append({"cohort": cohort, "method": method, "method_key": key,
                                    **{field: source[field] for field in SIZE_FIELDS
                                       if field not in ("cohort", "method", "method_key")}})
    provenance = {
        "schema_version": 1, "purpose": "Aggregate accuracy against independently reviewed test-area roof outlines.",
        "review": {
            "reviewer": "Esad", "review_date": "2026-10-15", "timezone": "America/Chicago",
            "imagery_provider": "Nearmap", "imagery_date": "2026-05",
            "label_convention": LABEL_CONVENTION, "coverage_complete": True,
            "test_area_independent_review": True, "test_area_prior_model_exposure": False,
            "review_sources": ["May 2026 aerial imagery", "historical building outlines"],
            "test_predictions_viewed_before_or_during_review": False,
            "exceptions": [], "unresolved_cases": [], "areas": areas,
            "count_interpretation": "Geometric edit counts exclude AOI crossings. ID-based edit counts cover the whole package and are not construction or demolition counts.",
            "approximate_minimum_label_rule": "Around 4 m2, not a strict 4 m2 cutoff: the test reference includes three outlines below 4 m2 and none below 3 m2; minimum area is approximately 3.499959 m2. No reviewed outlines were altered for scoring.",
        },
        "evaluation": {
            "reference_status": "independent_holdout", "metric_crs": "EPSG:26914",
            "iou_threshold": 0.5, "edge_policy": "exclude", "size_bins_m2": [20, 50, 100],
            "cohorts": {"all_sizes": {"reference": 600, "min_area_m2": None},
                        "at_least_20m2": {"reference": 468, "min_area_m2": 20.0,
                                           "min_area_filter_stage": "after_aoi_before_matching"}},
            "matching": "One-to-one maximum-cardinality matching at IoU >= 0.5, then maximum total IoU.",
            "area_filter": "For the primary cohort, filter both inputs by metric area >= 20 m2 after AOI exclusions, then match anew.",
            "size_class_interpretation": "Diagnostic classes from all-size matching. Reference recall and prediction precision use each object's own area; cross-class matches can give different matched-reference and matched-prediction counts. These rows are not the prefiltered primary cohort.",
            "scope": "One held-out test area; no city-wide accuracy claim or dated change inference.",
        },
        "execution": {
            "review_date_local": "2026-10-15", "timezone": "America/Chicago",
            "started_utc": datetime.fromisoformat(run["started_utc"]).isoformat(),
            "completed_utc": datetime.fromisoformat(run["completed_utc"]).isoformat(),
            "completed_evaluations": 6, "completed_comparisons": 2,
            "model_inference_rerun": False, "settings_changed": False, "retuned_on_reviewed_labels": False,
        },
        "evidence_sha256": {
            "completed_recipe": digest(config_path), "run_record": digest(run_path),
            "review_completion": digest(completion_path), "snapshot_manifest": digest(snapshot / "manifest.json"),
            "frozen_settings": digest(frozen_path), "review_receipt": receipt_hash,
            "sam3_run_receipt": digest(root / "outputs/lewisville/city/sam3_building_1536/run.json"),
            "esri_local_merge_receipt": digest(root / "outputs/lewisville/city/esri_local/merge.json"),
            "raw_evaluation_reports": report_hashes,
        },
        "predictions": prediction_evidence,
        "privacy": "Only allowlisted aggregate counts, settings, statements and hashes are published. Geometry, coordinates, feature IDs, imagery and private source paths are excluded.",
    }
    require(provenance["evidence_sha256"]["sam3_run_receipt"] == audit["sam_run_receipt_sha256"]
            and provenance["evidence_sha256"]["esri_local_merge_receipt"] == audit["esri_local_merge_receipt_sha256"],
            "Inference receipt audit mismatch")
    provenance["evidence_sha256"]["frozen_input_audit"] = digest(input_audit)
    return {
        "test_area_scores.csv": csv_text(scores, SCORE_FIELDS),
        "test_area_by_size.csv": csv_text(by_size, SIZE_FIELDS),
        "provenance.json": json.dumps(provenance, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
    }


def publish(files, output, check=False):
    output = Path(output)
    if check:
        for name, content in files.items():
            require((output / name).read_bytes() == content.encode("utf-8"), f"Export differs: {name}")
        return
    require(not any((output / name).exists() for name in files), "Export exists; use --check to verify it")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        with (output / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-audit", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    files = build_export(root, args.snapshot or root / DEFAULT_SNAPSHOT, args.input_audit)
    publish(files, args.output or root / "results/lewisville/reviewed_2026", args.check)
    print("Verified 6 score rows, 12 diagnostic size rows and aggregate provenance.")


if __name__ == "__main__":
    main()
