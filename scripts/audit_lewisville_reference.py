"""Audit the local Lewisville reference and historical scores without publishing data.

Run in the project's CPU environment from the repository root:
    python scripts/audit_lewisville_reference.py --output outputs/reference_audit.json
    python scripts/audit_lewisville_reference.py --check results/lewisville/reference_provenance.json

The original local data and outputs are required. This script does not download
them, alter them, or rerun a model. Output contains aggregate counts, hashes and
repository-relative artifact names, never geometries, feature rows or user paths.
Existing output files are refused. --check compares only the audit content, so an
environment version change does not by itself invalidate an identical snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/lewisville")
OUTPUTS = Path("outputs/lewisville")
RESULTS = Path("results/lewisville")
METHODS = {
    "esri_city": ("Esri, city run", OUTPUTS / "esri_city_202605/cleaned.gpkg"),
    "sam3_city": ("SAM 3", OUTPUTS / "city/sam3_building_1536/cleaned.gpkg"),
    "esri_local_city": ("Esri, local raw run", OUTPUTS / "city/esri_local/cleaned.gpkg"),
}
CSVS = ("citywide_scores.csv", "citywide_by_size.csv", "test_area_scores.csv",
        "old_town_scores.csv", "old_town_by_size.csv", "tuning_final.csv")


def digest_values(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def year_counts(frame):
    return dict(sorted(Counter("unknown" if value is None or str(value) == "nan" else str(value)
                               for value in frame.Year_Colle).items()))


def metrics(tp, predictions, references):
    precision = tp / predictions if predictions else None
    recall = tp / references if references else None
    return {"tp": int(tp), "fp": int(predictions - tp), "fn": int(references - tp),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (predictions + references) if predictions + references else None}


def assert_csv_metrics(row, result, names=("tp", "fp", "fn")):
    for key, column in zip(("tp", "fp", "fn"), names):
        assert int(row[column]) == result[key], (column, row[column], result[key])
    for key in ("precision", "recall", "f1"):
        assert abs(float(row[key]) - result[key]) <= 0.000051, (key, row[key], result[key])


def audit():
    # Avoid inherited PostGIS/ArcGIS projection databases in this process only.
    for name in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
        os.environ.pop(name, None)
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely
    from nearmap_buildings.common import FINGERPRINT_METHOD, sha256_file
    from nearmap_buildings.evaluation import (
        ROW_ORDER_FINGERPRINT_METHOD, geometry_fingerprint,
        maximum_cardinality_matches, row_order_fingerprint,
    )

    def load(path, layer=None):
        return gpd.read_file(ROOT / path, layer=layer)

    def within(frame, region):
        """Equivalent exclude-edge selection, without computing every boundary flag."""
        shapely.prepare(region)
        inside = shapely.covers(region, frame.geometry.to_numpy())
        kept, excluded = frame.loc[inside].copy(), frame.loc[~inside].copy()
        kept["eval_id"] = np.flatnonzero(inside)
        excluded["eval_id"] = np.flatnonzero(~inside)
        overlap = shapely.area(shapely.intersection(excluded.geometry.to_numpy(), region))
        excluded["aoi_action"] = np.where(overlap > 0, "crossing_excluded", "outside")
        return kept, excluded

    def identity(path, frame):
        return {"filename": path.name, "sha256": sha256_file(ROOT / path), "count": len(frame),
                "crs": frame.crs.to_string(), "geometry_fingerprint": geometry_fingerprint(frame),
                "row_order_fingerprint": row_order_fingerprint(frame),
                "invalid_geometries": int((~frame.is_valid).sum()),
                "year_colle_counts": year_counts(frame),
                "objectid_order_sha256": digest_values([int(value) for value in frame.OBJECTID])}

    source_geojson = DATA / "reference_2015_outlines_source.geojson"
    raw_path = DATA / "reference_2015_outlines_source.gpkg"
    normalized_path = DATA / "reference_2015_outlines.gpkg"
    payload = read_json(source_geojson)
    # The saved export declares its actual coordinate CRS in this custom member.
    # It is not RFC 7946 longitude/latitude: do not silently assume EPSG:4326.
    exported = gpd.GeoDataFrame.from_features(payload["features"], crs=payload["crs_of_coordinates"])
    raw, reference = load(raw_path, "buildings"), load(normalized_path, "buildings")
    assert raw.OBJECTID.is_unique and reference.OBJECTID.is_unique
    assert raw.OBJECTID.tolist() == reference.OBJECTID.tolist() == exported.OBJECTID.tolist()
    assert geometry_fingerprint(raw) == geometry_fingerprint(exported)
    assert row_order_fingerprint(raw) == row_order_fingerprint(exported)
    retained_fields = [name for name in reference.columns if name != reference.geometry.name]
    assert raw[retained_fields].equals(reference[retained_fields])
    raw_wkb = shapely.to_wkb(shapely.normalize(raw.geometry))
    normalized_wkb = shapely.to_wkb(shapely.normalize(reference.geometry))
    changed = raw_wkb != normalized_wkb
    repaired_wkb = shapely.to_wkb(shapely.normalize(shapely.make_valid(raw.geometry)))
    assert np.array_equal(repaired_wkb, normalized_wkb)
    snapshot_id = "lewisville-reference-" + geometry_fingerprint(reference)
    artifacts = []
    evidence = {
        "citywide_scores.csv": "City evaluation report fingerprints and stored matching rows; >=20 counts numerically reconstructed",
        "citywide_by_size.csv": "Associated historical city summary; this audit does not regenerate every size-class row",
        "test_area_scores.csv": "Stored test receipt and fresh geometric matching on the recorded test AOI",
        "old_town_scores.csv": "Published CSV carries the reference geometry fingerprint",
        "old_town_by_size.csv": "Associated documented Old Town inventory comparison; this audit does not regenerate every size-class row",
        "tuning_final.csv": "Matches local tuning receipt after column renaming/rounding; no per-variant reference fingerprints were retained",
    }
    for name in CSVS:
        path = RESULTS / name
        if name == "old_town_scores.csv":
            assert all(row["reference_fingerprint"] == geometry_fingerprint(reference) for row in csv_rows(path))
        artifacts.append({"path": name, "sha256": sha256_file(ROOT / path),
                          "rows": len(csv_rows(path)), "reference_snapshot_id": snapshot_id,
                          "reference_link_evidence": evidence[name]})

    reference_metric = reference.to_crs("EPSG:26914")
    city_aoi = load(DATA / "city_limits.gpkg")
    city_region = shapely.union_all(city_aoi.to_crs("EPSG:26914").geometry.to_numpy())
    included, excluded = within(reference_metric, city_region)
    selected_ids = [int(value) for value in included.eval_id]
    selected_objectids = [int(value) for value in included.OBJECTID]
    area_records = {}
    for label, folder in (("old_town", "sample_old_town"), ("test_area", "test_area")):
        path = DATA / folder / "reference_2015_subset.gpkg"
        subset = load(path, "buildings")
        aoi = load(DATA / folder / "aoi.gpkg")
        region = shapely.union_all(aoi.to_crs("EPSG:26914").geometry.to_numpy())
        kept, _ = within(reference_metric, region)
        touching = reference_metric.loc[reference_metric.intersects(region)]
        assert touching.OBJECTID.tolist() == subset.id_2015.tolist()
        assert geometry_fingerprint(reference.loc[touching.index]) == geometry_fingerprint(subset)
        area_records[label] = {
            "subset_filename": path.name, "subset_sha256": sha256_file(ROOT / path),
            "touching_count": len(subset), "scored_count": len(kept),
            "scored_reference_at_least_20_m2": int((kept.area >= 20).sum()),
            "touching_objectid_order_sha256": digest_values([int(value) for value in subset.id_2015]),
            "scored_objectid_order_sha256": digest_values([int(value) for value in kept.OBJECTID]),
            "aoi_fingerprint": geometry_fingerprint(aoi),
            "scored_year_colle_counts": year_counts(kept),
        }

    city_csv = csv_rows(RESULTS / "citywide_scores.csv")
    test_csv = csv_rows(RESULTS / "test_area_scores.csv")
    city_receipt = read_json(OUTPUTS / "city/city_summary.json")
    test_receipt = read_json(OUTPUTS / "test_area_final/scores.json")
    test_aoi = load(DATA / "test_area/aoi.gpkg").to_crs("EPSG:26914")
    test_region = shapely.union_all(test_aoi.geometry.to_numpy())
    test_reference = reference_metric.loc[test_region.covers(reference_metric.geometry)].reset_index(drop=True)
    scoring = {}
    for name, (label, prediction_path) in METHODS.items():
        report_path = OUTPUTS / "agreement_2015" / name / "evaluation.json"
        report = read_json(report_path)
        assert report["reference_fingerprint"] == geometry_fingerprint(reference)
        assert report["reference_order_fingerprint"] == row_order_fingerprint(reference)
        assert report["aoi_fingerprint"] == geometry_fingerprint(city_aoi)
        assert report["evaluated_reference_ids"] == selected_ids
        assert report["excluded_counts"]["reference"] == dict(excluded.aoi_action.value_counts())
        predictions = load(prediction_path, "cleaned")
        assert report["predictions_fingerprint"] == geometry_fingerprint(predictions)
        predictions = predictions.to_crs("EPSG:26914")
        selected_predictions, _ = within(predictions, city_region)
        pairs = report["matches"]
        pi = np.array([pair["prediction_id"] for pair in pairs])
        ri = np.array([pair["reference_id"] for pair in pairs])
        pg, rg = predictions.geometry.iloc[pi].to_numpy(), reference_metric.geometry.iloc[ri].to_numpy()
        intersection = shapely.area(shapely.intersection(pg, rg))
        actual_iou = intersection / (shapely.area(pg) + shapely.area(rg) - intersection)
        assert np.allclose(actual_iou, [pair["iou"] for pair in pairs], rtol=0, atol=1e-12)
        p_large, r_large = shapely.area(pg) >= 20, shapely.area(rg) >= 20
        historical = metrics(int((p_large & r_large).sum()), int((selected_predictions.area >= 20).sum()),
                             int((included.area >= 20).sum()))
        row = next(row for row in city_csv if row["method"] == label and row["buildings"] == "20 m² and up")
        assert_csv_metrics(row, historical, ("matched", "extra", "missed"))
        assert city_receipt["methods"][name]["main"]["true_positives"] == historical["tp"]

        # Small AOI rerun, without model inference: compare two explicit interpretations.
        test_predictions = predictions.loc[test_region.covers(predictions.geometry)].reset_index(drop=True)
        test_pairs = maximum_cardinality_matches(list(test_predictions.geometry), list(test_reference.geometry), 0.5)
        test_large_tp = sum(test_predictions.geometry.iloc[p].area >= 20 and test_reference.geometry.iloc[r].area >= 20
                            for p, r, _ in test_pairs)
        test_historical = metrics(test_large_tp, int((test_predictions.area >= 20).sum()),
                                  int((test_reference.area >= 20).sum()))
        test_row = next(row for row in test_csv if row["method"] == label)
        assert_csv_metrics(test_row, test_historical)
        receipt_row = next(row for row in test_receipt["rows"] if row["method"] == label)
        assert all(receipt_row[key] == test_historical[key] for key in ("tp", "fp", "fn"))
        filtered_p = test_predictions.loc[test_predictions.area >= 20]
        filtered_r = test_reference.loc[test_reference.area >= 20]
        filtered_pairs = maximum_cardinality_matches(list(filtered_p.geometry), list(filtered_r.geometry), 0.5)
        scoring[name] = {
            "label": label, "evaluation_receipt_sha256": sha256_file(ROOT / report_path),
            "predictions_file_sha256": sha256_file(ROOT / prediction_path),
            "predictions_geometry_fingerprint": report["predictions_fingerprint"],
            "stored_match_ious_verified_against_current_rows": True,
            "city_historical_at_least_20_reconstruction": historical,
            "city_global_matches_by_area": {
                "prediction_and_reference_at_least_20": int((p_large & r_large).sum()),
                "prediction_at_least_20_reference_under_20": int((p_large & ~r_large).sum()),
                "prediction_under_20_reference_at_least_20": int((~p_large & r_large).sum()),
                "both_under_20": int((~p_large & ~r_large).sum()),
            },
            "test_historical_at_least_20_reconstruction": test_historical,
            "test_filter_both_before_matching_comparison": metrics(len(filtered_pairs), len(filtered_p), len(filtered_r)),
        }

    # The public tuning file renamed columns and rounded values.
    local_tuning = pd.read_csv(ROOT / OUTPUTS / "tuning/final_vs_2015.csv")
    public_tuning = pd.read_csv(ROOT / RESULTS / "tuning_final.csv")
    frozen = read_json(OUTPUTS / "tuning/frozen_settings.json")
    comparable = local_tuning.rename(columns={"shed_recall": "recall_under_20_m2", "sheds": "reference_under_20_m2"})
    assert comparable.shape == public_tuning.shape and comparable.columns.tolist() == public_tuning.columns.tolist()
    for column in public_tuning:
        if pd.api.types.is_numeric_dtype(public_tuning[column]):
            assert np.allclose(comparable[column], public_tuning[column], rtol=0, atol=0.000051, equal_nan=True), column
        else:
            assert comparable[column].fillna("").tolist() == public_tuning[column].fillna("").tolist(), column
    inference_receipts = []
    for path in sorted((ROOT / OUTPUTS / "tuning").glob("sam3_*/run.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        plan = receipt["plan"]
        inference_receipts.append({"run_name": path.parent.name, "sha256": sha256_file(path),
                                   "status": receipt["status"], "prompt": plan["text"],
                                   "inference_confidence": plan["confidence"], "tile_count": plan["tile_count"],
                                   "manifest_sha256": plan["manifest_sha256"],
                                   "checkpoint_sha256": plan["checkpoint_sha256"]})
    return {
        "schema": "lewisville-reference-provenance-v1",
        "snapshot_id": snapshot_id,
        "audit_kind": "Local snapshot and stored-result audit; no network acquisition or model inference",
        "fingerprint_method": FINGERPRINT_METHOD,
        "row_order_fingerprint_method": ROW_ORDER_FINGERPRINT_METHOD,
        "id_hash_definition": "SHA-256 of ordered integer IDs encoded as compact ASCII JSON",
        "source": {
            "service_layer_url": payload["source"], "layer_id": 0,
            "export_coordinate_crs": payload["crs_of_coordinates"],
            "retrieved_at": None, "query_parameters": None, "server_filter": None,
            "service_item_revision_at_retrieval": None, "license_at_retrieval": None,
            "authoritative_outline_convention": None,
            "unknown_fields_reason": "Not recorded in the inspected export or adjacent local metadata; not inferred from filenames or the current live service",
            "collection_date_interpretation": "Year_Colle has Pre_2015, 2015 and missing values; the service name's 082020 is not a feature collection date",
        },
        "files": {
            "source_geojson": identity(source_geojson, exported),
            "source_geopackage": identity(raw_path, raw),
            "normalized_geopackage": identity(normalized_path, reference),
        },
        "normalization": {
            "same_objectids_in_same_order": True, "retained_attributes_unchanged": True,
            "retained_fields": retained_fields,
            "removed_fields": [name for name in raw.columns if name not in reference.columns],
            "unchanged_geometries": int((~changed).sum()), "changed_geometries": int(changed.sum()),
            "changed_objectids_sha256": digest_values([int(value) for value in reference.loc[changed, "OBJECTID"]]),
            "normalized_equals_make_valid_of_source": True,
            "historical_transformation_command": None,
            "evidence_limit": "This proves the observed transformation, not which historical command performed it; attribute reduction and one geometry repair are consistent with the inspected files",
        },
        "city_evaluation_reference": {
            "crs_for_area_and_scoring": "EPSG:26914", "edge_policy": "exclude", "iou_threshold": 0.5,
            "source_count": len(reference), "evaluated_count": len(included),
            "excluded_counts": {key: int(value) for key, value in excluded.aoi_action.value_counts().items()},
            "evaluated_year_colle_counts": year_counts(included),
            "evaluated_objectid_order_sha256": digest_values(selected_objectids),
            "evaluated_zero_based_row_ids_sha256": digest_values(selected_ids),
            "aoi_fingerprint": geometry_fingerprint(city_aoi),
            "all_three_historical_reports_reference_and_id_checks_pass": True,
        },
        "review_subsets": area_records,
        "result_csvs": artifacts,
        "historical_scoring_audit": {
            "notebook_baseline": {
                "path": "../../notebooks/04_score_and_compare.ipynb",
                "commit": "09adb7362162888187d7f750412cf057a4cd8512",
                "git_blob_sha256": "bb67c819f6645a0010dc59fca50bb3a9bdc5d49ab077f7d2c9d22c086e13d7f0",
                "finding": "The baseline synthetic score function had no area filter; the later explicit >=20 correction does not establish historical real-data execution",
            },
            "methods": scoring,
            "tuning_receipts": {
                "local_csv_sha256": sha256_file(ROOT / OUTPUTS / "tuning/final_vs_2015.csv"),
                "public_rows": len(public_tuning), "local_rows": len(local_tuning),
                "public_csv_matches_local_after_column_renaming_and_rounding": True,
                "frozen_settings_sha256": sha256_file(ROOT / OUTPUTS / "tuning/frozen_settings.json"),
                "frozen_result_counts": {name: {key: row[key] for key in ("tp", "fp", "fn")} for name, row in frozen.items()},
                "inference_runs": inference_receipts,
            },
            "confirmed": "All six published city/test >=20 rows are reproduced by global all-size matching, retaining matches only when both prediction and reference areas are >=20, then using that TP for both precision and recall denominators",
            "limits": "The original real-data scoring script is not retained in the inspected files. Receipts store results, not the matching/filter call sequence. Agreement with this reconstruction does not prove the historical execution path. At baseline commit 09adb7362162888187d7f750412cf057a4cd8512, the synthetic notebook did not implement its prose's >=20 filter; correcting it later does not prove the historical real-data execution path. The 176 tuning variants were not re-executed by this audit.",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="new JSON file for aggregate audit evidence")
    destination.add_argument("--check", type=Path, help="existing aggregate audit to reproduce and compare")
    args = parser.parse_args(argv)
    result = audit()
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if result != expected:
            raise ValueError("Audit differs from the saved manifest; inspect changed inputs and result files")
        print("Reference snapshot, result hashes and scoring reconstruction verified")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False, ensure_ascii=False)
            stream.write("\n")
        print("Wrote aggregate reference provenance; no geometries or feature rows exported")


if __name__ == "__main__":
    main()
