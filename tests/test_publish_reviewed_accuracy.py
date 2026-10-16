"""Synthetic publication checks; no private GIS inputs or model runs required."""

import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "publish_reviewed_accuracy", Path(__file__).parents[1] / "scripts/publish_reviewed_accuracy.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root, snapshot = tmp_path.resolve(), (tmp_path / "snapshot").resolve()
    objects = {}

    def sha(path):
        return hashlib.sha256(str(Path(path).relative_to(root)).encode()).hexdigest()

    def put(path, value):
        objects[str(root / path)] = value
        return value

    review = {
        "reviewer": "Esad", "review_date": "2026-10-15", "imagery_date": "2026-05",
        "label_convention": publisher.LABEL_CONVENTION, "coverage_complete": True,
        "independent_review": True,
    }
    counts = {key: 0 for key in publisher.COUNT_FIELDS}
    counts.update(original_in_aoi=600, reviewed_in_aoi=600, unchanged=600,
                  crossing_aoi_edge={"original": 40, "reviewed": 40})
    receipt = put("outputs/accuracy/review_check.json", {
        "schema_version": 1, "ready_for_evaluation": True, "edge_policy": "exclude",
        "reviewed_fingerprint": "a" * 64, "aoi_fingerprint": "b" * 64,
        "provenance": review, "counts": counts,
        "private_unknown": "PRIVATE_SENTINEL",
    })
    put("outputs/accuracy/old_town_review_check.json", copy.deepcopy(receipt))
    completion = put("snapshot/review_completion.json", {
        **review, "test_area_independent_review": True, "test_area_prior_model_exposure": False,
        "exceptions": [], "unresolved_cases": [], "snapshot_manifest_sha256": sha(snapshot / "manifest.json"),
        "human_statements": ["PRIVATE_SENTINEL"], "areas": {},
    })
    for area in ("test_area", "old_town"):
        is_test = area == "test_area"
        receipt_path = "outputs/accuracy/" + ("review_check.json" if is_test else "old_town_review_check.json")
        reference_path = root / "data/reference/test_area_reviewed.gpkg" if is_test else snapshot / area / "reviewed.gpkg"
        aoi_path = root / "data/reference/test_area_aoi.gpkg" if is_test else snapshot / area / "aoi.gpkg"
        completion["areas"][area] = {
            "reviewed_fingerprint": "a" * 64, "aoi_fingerprint": "b" * 64,
            "receipt_sha256": sha(root / receipt_path),
            "reviewed_file_sha256": sha(reference_path), "aoi_file_sha256": sha(aoi_path),
            "total_features": 640, "counts": copy.deepcopy(counts),
            "id_audit": {**{key: 0 for key in publisher.ID_COUNT_FIELDS},
                         "removed_original_ids": [812345], "unexpected_original_ids": [812346]},
        }
    frozen = put("outputs/lewisville/tuning/frozen_settings.json", {})
    for name, tile, threshold, prompt in (("SAM 3", 1536.0, 0.7, "building"),
                                         ("Esri city run", None, None, ""),
                                         ("Esri local", 512.0, 0.9, "")):
        frozen[name] = {"tile_px": tile, "threshold": threshold, "prompt": prompt,
                        "regularize": "off", "private": "PRIVATE_SENTINEL"}
    config = put("configs/accuracy.example.json", {"steps": []})
    audit = put("outputs/accuracy/input_audit.json", {
        "methods": {},
        "frozen_settings_sha256": sha(root / "outputs/lewisville/tuning/frozen_settings.json"),
        "sam_run_receipt_sha256": sha(root / "outputs/lewisville/city/sam3_building_1536/run.json"),
        "esri_local_merge_receipt_sha256": sha(root / "outputs/lewisville/city/esri_local/merge.json"),
    })
    for cohort in publisher.COHORTS:
        size_rows = []
        for key in publisher.METHODS:
            is_all = cohort == "all_sizes"
            ref = 600 if is_all else 468
            report_path = f"outputs/accuracy/{cohort}/{key}/evaluation.json"
            historical = put(publisher.HISTORICAL[key], {"predictions_fingerprint": "c" * 64})
            report = put(report_path, {
                "schema_version": 1, "reference_status": "independent_holdout", "metric_crs": "EPSG:26914",
                "iou_threshold": 0.5, "edge_policy": "exclude", "size_bins_m2": [20.0, 50.0, 100.0],
                "reference_fingerprint": "a" * 64, "reference_order_fingerprint": "d" * 64,
                "aoi_fingerprint": "b" * 64, **historical,
                "min_area_m2": None if is_all else 20.0,
                "min_area_filter_stage": None if is_all else "after_aoi_before_matching",
                "review_receipt": {**receipt, "sha256": sha(root / "outputs/accuracy/review_check.json")},
                "input_counts": {"predictions": 1000, "reference": 640},
                "evaluated_counts": {"predictions": 500, "reference": ref},
                "excluded_counts": {"predictions": {"outside": 500}, "reference": {"crossing_excluded": 40}},
                "area_excluded_counts": {"predictions": 0, "reference": 0 if is_all else 132},
                "clipped_counts": {"predictions": 0, "reference": 0},
                "true_positives": 400, "false_positives": 100, "false_negatives": ref - 400,
                "precision": 0.8, "recall": 400 / ref, "f1": 800 / (500 + ref),
                "matched_only": {"iou": {"mean": 0.7555555555555555, "median": 0.8}},
                "size_classes": [], "matches": [{"reference_id": 812345, "geometry": "PRIVATE_SENTINEL"}],
                "sources": {name: {"path": str(root / relative)} for name, relative in (
                    ("predictions", publisher.PREDICTIONS[key]),
                    ("reference", "data/reference/test_area_reviewed.gpkg"),
                    ("aoi", "data/reference/test_area_aoi.gpkg"))},
            })
            for label, lo, hi, nref, npred, matched in (("<20", 0.0, 20.0, 132, 0, 0),
                    ("20-50", 20.0, 50.0, 0, 0, 0), ("50-100", 50.0, 100.0, 0, 0, 0),
                    (">=100", 100.0, None, 468, 500, 400)):
                row = {"size_class": label, "min_m2": lo, "max_m2": hi,
                       "reference": nref, "predictions": npred, "matched_reference": matched,
                       "matched_predictions": matched, "precision": matched / npred if npred else None,
                       "recall": matched / nref if nref else None,
                       "f1": 2 * matched / (nref + npred) if nref + npred else None}
                report["size_classes"].append(row)
                size_rows.append({"label": key, "kind": "class", **row})
            command = ["evaluate", "--predictions", publisher.PREDICTIONS[key],
                       "--predictions-layer", "cleaned", "--metric-crs", "EPSG:26914",
                       "--independent-holdout", "--reference", "data/reference/test_area_reviewed.gpkg",
                       "--aoi", "data/reference/test_area_aoi.gpkg", "--review-receipt",
                       "outputs/accuracy/review_check.json", "--output-json", report_path]
            if not is_all:
                command += ["--min-area-m2", "20"]
            config["steps"].append(command)
            audit["methods"][key] = {
                "sha256": sha(root / publisher.PREDICTIONS[key]), "predictions_fingerprint": "c" * 64,
                "feature_count": 1000, "historical_report_sha256": sha(root / publisher.HISTORICAL[key]),
            }
        config["steps"].append(["compare"])
        output = root / f"outputs/accuracy/{cohort}/comparison_by_size.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(publisher.csv_text(size_rows, list(size_rows[0])), encoding="utf-8")
    put("outputs/accuracy/run_record.json", {
        "status": "completed", "settings_changed": False, "model_inference_rerun": False,
        "config_sha256": sha(root / "configs/accuracy.example.json"),
        "steps": [{"command": command, "return_code": 0} for command in config["steps"]],
        "started_utc": "2026-10-16T03:10:21+00:00", "completed_utc": "2026-10-16T03:11:31+00:00",
    })
    monkeypatch.setattr(publisher, "digest", sha)
    monkeypatch.setattr(publisher, "read_json", lambda path: objects[str(Path(path))])
    return root, snapshot, objects


def test_export_preserves_precision_and_excludes_private_fields(evidence):
    root, snapshot, _ = evidence
    files = publisher.build_export(root, snapshot)
    scores = list(csv.DictReader(io.StringIO(files["test_area_scores.csv"])))
    sizes = list(csv.DictReader(io.StringIO(files["test_area_by_size.csv"])))
    assert len(scores) == 6 and len(sizes) == 12
    assert scores[0]["mean_iou"] == "0.7555555555555555"
    assert scores[0]["min_area_m2"] == "" and scores[3]["min_area_m2"] == "20.0"
    assert all(row["kind"] == "class" and row["cohort"] == "all_sizes" for row in sizes)
    combined = "".join(files.values())
    for secret in ("PRIVATE_SENTINEL", "812345", "812346", str(root), "removed_original_ids", '"sources"'):
        assert secret not in combined
    provenance = json.loads(files["provenance.json"])
    assert provenance["review"]["areas"]["test_area"]["id_based_edit_counts"]


@pytest.mark.parametrize("field,value", [
    ("reference_fingerprint", "f" * 64), ("aoi_fingerprint", "f" * 64),
    ("predictions_fingerprint", "f" * 64), ("reference_order_fingerprint", "f" * 64),
    ("iou_threshold", 0.4), ("edge_policy", "clip"), ("min_area_m2", 20.0),
    ("min_area_filter_stage", "after_matching"), ("false_positives", 101), ("precision", 0.99),
])
def test_export_rejects_report_binding_or_count_changes(evidence, field, value):
    root, snapshot, objects = evidence
    objects[str(root / "outputs/accuracy/all_sizes/esri_city/evaluation.json")][field] = value
    with pytest.raises(ValueError):
        publisher.build_export(root, snapshot)


@pytest.mark.parametrize("change", ["receipt", "input", "completed_recipe", "failed_step", "exposure", "method"])
def test_export_rejects_stale_or_incomplete_evidence(evidence, change):
    root, snapshot, objects = evidence
    if change == "receipt":
        objects[str(root / "outputs/accuracy/all_sizes/sam3/evaluation.json")]["review_receipt"]["sha256"] = "f" * 64
    elif change == "input":
        objects[str(root / "outputs/accuracy/input_audit.json")]["methods"]["sam3"]["sha256"] = "f" * 64
    elif change == "completed_recipe":
        objects[str(root / "outputs/accuracy/run_record.json")]["config_sha256"] = "f" * 64
    elif change == "failed_step":
        objects[str(root / "outputs/accuracy/run_record.json")]["steps"][0]["return_code"] = 1
    elif change == "exposure":
        objects[str(snapshot / "review_completion.json")]["test_area_prior_model_exposure"] = True
    else:
        objects[str(root / "outputs/accuracy/input_audit.json")]["methods"]["unexpected"] = {}
    with pytest.raises(ValueError):
        publisher.build_export(root, snapshot)


def test_existing_targets_are_not_overwritten_but_readme_is_allowed(tmp_path):
    (tmp_path / "README.md").write_text("Documentation", encoding="utf-8")
    files = {"test_area_scores.csv": "a,b\n1,2\n", "provenance.json": "{}\n"}
    publisher.publish(files, tmp_path)
    publisher.publish(files, tmp_path, check=True)
    with pytest.raises(ValueError, match="Export exists"):
        publisher.publish(files, tmp_path)
    assert (tmp_path / "test_area_scores.csv").read_bytes() == files["test_area_scores.csv"].encode()
    with pytest.raises(ValueError, match="Export differs"):
        publisher.publish({"test_area_scores.csv": "changed"}, tmp_path, check=True)
