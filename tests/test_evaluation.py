import json

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from nearmap_buildings.evaluation import evaluate, geometry_fingerprint, main, maximum_cardinality_matches


CRS = "EPSG:26914"


def frame(geometries, **columns):
    return gpd.GeoDataFrame(columns, geometry=geometries, crs=CRS)


def aoi():
    return frame([box(-10, -10, 100, 100)])


def test_maximum_cardinality_beats_greedy_highest_iou():
    # The wide prediction qualifies for either reference and prefers reference 0.
    # The smaller prediction qualifies only for reference 0. Greedy yields one;
    # maximum-cardinality matching correctly assigns both.
    reference = [box(0, 0, 10, 10), box(10, 0, 20, 10)]
    predictions = [box(0, 0, 18, 10), box(0, 0, 6, 10)]
    matches = maximum_cardinality_matches(predictions, reference, threshold=0.4)
    assert {(p, r) for p, r, _ in matches} == {(0, 1), (1, 0)}


def test_matching_maximizes_iou_after_cardinality():
    reference = [box(0, 0, 10, 10), box(4, 0, 14, 10)]
    predictions = [box(0, 0, 10, 10), box(4, 0, 14, 10)]
    assert maximum_cardinality_matches(predictions, reference, 0.3) == [(0, 0, 1.0), (1, 1, 1.0)]


def test_duplicates_are_false_positives_and_unmatched_reference_false_negative():
    predictions = frame([box(0, 0, 10, 10), box(0, 0, 10, 10), box(40, 0, 50, 10)])
    reference = frame([box(0, 0, 10, 10), box(20, 0, 30, 10)])
    result = evaluate(predictions, reference, aoi(), crs=CRS)
    assert result.report["true_positives"] == 1
    assert result.report["false_positives"] == 2
    assert result.report["false_negatives"] == 1
    assert result.report["precision"] == pytest.approx(1 / 3)
    assert result.report["recall"] == 0.5
    assert result.report["f1"] == 0.4
    assert len(result.layers["unmatched_predictions"]) == 2


def test_known_boundary_distance_area_error_and_iou():
    predictions = frame([box(1, 0, 12, 10)])
    reference = frame([box(0, 0, 10, 10)])
    report = evaluate(predictions, reference, aoi(), crs=CRS).report
    match = report["matches"][0]
    assert match["iou"] == pytest.approx(90 / 120)
    assert match["area_error_m2"] == 10
    assert match["relative_area_error"] == 0.1
    assert match["boundary_hausdorff_m"] == 2
    assert "densify=0.25" in report["metric_definitions"]["boundary_hausdorff_m"]


def test_fingerprint_value_is_stable_for_existing_reports():
    # Reports already on disk carry this value; comparisons break if it drifts.
    holes = Polygon(box(20, 0, 30, 10).exterior.coords, [box(22, 2, 24, 4).exterior.coords])
    assert geometry_fingerprint(frame([box(0, 0, 10, 10), holes])) == "9d0e81faec438fa143ce5913a23f72eb54443e0a9698562100cedea3a98c6406"


def test_report_records_evaluated_reference_rows_and_their_order():
    reference = frame([box(0, 0, 10, 10), box(200, 200, 210, 210), box(20, 0, 30, 10)])
    report = evaluate(frame([box(20, 0, 30, 10)]), reference, aoi(), crs=CRS).report
    assert report["evaluated_reference_ids"] == [0, 2]  # Row 1 lies outside the AOI.
    assert report["matches"][0]["reference_id"] == 2
    reordered = evaluate(frame([box(20, 0, 30, 10)]), reference.iloc[::-1], aoi(), crs=CRS).report
    assert reordered["reference_fingerprint"] == report["reference_fingerprint"]
    assert reordered["reference_order_fingerprint"] != report["reference_order_fingerprint"]


def test_predictions_fingerprint_identifies_the_prediction_set():
    reference = frame([box(0, 0, 10, 10)])
    first = evaluate(frame([box(0, 0, 10, 10)]), reference, aoi(), crs=CRS).report
    second = evaluate(frame([box(1, 0, 11, 10)]), reference, aoi(), crs=CRS).report
    assert first["predictions_fingerprint"] != second["predictions_fingerprint"]
    assert first["reference_fingerprint"] == second["reference_fingerprint"]


@pytest.mark.parametrize("predicted,actual,precision,recall,f1,fp,fn", [
    ([], [], None, None, None, 0, 0),
    ([], [box(0, 0, 10, 10)], None, 0, 0, 0, 1),
    ([box(0, 0, 10, 10)], [], 0, None, 0, 1, 0),
])
def test_empty_dataset_policy(predicted, actual, precision, recall, f1, fp, fn):
    report = evaluate(frame(predicted), frame(actual), aoi(), crs=CRS).report
    assert (report["precision"], report["recall"], report["f1"]) == (precision, recall, f1)
    assert (report["false_positives"], report["false_negatives"]) == (fp, fn)
    assert report["matched_only"]["iou"]["mean"] is None
    json.dumps(report, allow_nan=False)


def test_aoi_exclude_policy_counts_crossings_and_outside_and_keeps_boundary_touch():
    predictions = frame([box(0, 0, 2, 2), box(8, 0, 12, 4), box(20, 20, 22, 22)], edge_touch=[True, False, False])
    reference = frame([box(0, 0, 2, 2), box(8, 0, 12, 4)])
    result = evaluate(predictions, reference, frame([box(0, 0, 10, 10)]), crs=CRS)
    assert result.report["evaluated_counts"] == {"predictions": 1, "reference": 1}
    assert result.report["excluded_counts"]["predictions"] == {"crossing_excluded": 1, "outside": 1}
    assert result.layers["matched_predictions"].aoi_boundary_touch.iloc[0]
    assert result.layers["matched_predictions"].edge_touch.iloc[0]


def test_aoi_clip_policy_matches_clipped_shapes_retains_identity():
    predictions = frame([box(8, 0, 12, 4)])
    reference = frame([box(8, 0, 14, 4)])
    result = evaluate(predictions, reference, frame([box(0, 0, 10, 10)]), crs=CRS, edge_policy="clip")
    assert result.report["clipped_counts"] == {"predictions": 1, "reference": 1}
    assert result.report["matches"][0]["iou"] == 1
    assert result.layers["matched_predictions"].area.iloc[0] == 8
    assert predictions.area.iloc[0] == 16


@pytest.mark.parametrize("geometry", [None, Polygon(), Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])])
def test_malformed_rows_are_errors_not_silent_drops(geometry):
    with pytest.raises(ValueError, match="predictions:"):
        evaluate(frame([geometry]), frame([]), aoi(), crs=CRS)


def test_invalid_crs_and_empty_aoi_rejected():
    with pytest.raises(ValueError, match="projected"):
        evaluate(frame([]), frame([]), aoi(), crs="EPSG:4326")
    with pytest.raises(ValueError, match="missing CRS"):
        evaluate(frame([]).set_crs(None, allow_override=True), frame([]), aoi(), crs=CRS)
    with pytest.raises(ValueError, match="at least one"):
        evaluate(frame([]), frame([]), frame([]), crs=CRS)


def test_cli_writes_json_and_all_six_layers_without_altering_inputs(tmp_path):
    source = {"predictions": frame([box(0, 0, 10, 10)], instance_id=["synthetic-1"]),
              "reference": frame([box(0, 0, 10, 10)]), "aoi": aoi()}
    argv = []
    for role, data in source.items():
        path = tmp_path / f"{role}.gpkg"
        data.to_file(path, driver="GPKG")
        argv.extend([f"--{role}", str(path)])
    before = {role: (tmp_path / f"{role}.gpkg").read_bytes() for role in source}
    output_json, output_gpkg = tmp_path / "metrics.json", tmp_path / "matches.gpkg"
    argv.extend(["--metric-crs", CRS, "--output-json", str(output_json), "--output-gpkg", str(output_gpkg), "--independent-holdout"])
    main(argv)
    report = json.loads(output_json.read_text())
    assert report["f1"] == 1
    assert report["predictions_fingerprint"] == geometry_fingerprint(source["predictions"])
    assert report["sources"]["predictions"] == {"path": str((tmp_path / "predictions.gpkg").resolve()), "layer": None}
    assert len(gpd.list_layers(output_gpkg)) == 6
    assert gpd.read_file(output_gpkg, layer="matched_predictions").instance_id.tolist() == ["synthetic-1"]
    assert {role: (tmp_path / f"{role}.gpkg").read_bytes() for role in source} == before


def test_agreement_only_reports_say_so_and_holdout_is_the_default():
    reference = frame([box(0, 0, 10, 10)])
    default = evaluate(frame([box(0, 0, 10, 10)]), reference, aoi(), crs=CRS).report
    agreement = evaluate(frame([box(0, 0, 10, 10)]), reference, aoi(), crs=CRS, reference_status="agreement_only").report
    assert default["reference_status"] == "independent_holdout"
    assert agreement["reference_status"] == "agreement_only"
    assert "not accuracy" in agreement["reference_requirement"]
    with pytest.raises(ValueError, match="reference_status"):
        evaluate(frame([]), frame([]), aoi(), crs=CRS, reference_status="ground_truth")


def test_cli_takes_exactly_one_reference_status(tmp_path):
    with pytest.raises(SystemExit):
        main(["--predictions", "a.gpkg", "--reference", "b.gpkg", "--aoi", "c.gpkg", "--metric-crs", CRS,
              "--output-json", str(tmp_path / "metrics.json"), "--independent-holdout", "--agreement-only"])


def test_cli_requires_independent_holdout_acknowledgement(tmp_path):
    with pytest.raises(SystemExit):
        main(["--predictions", "a.gpkg", "--reference", "b.gpkg", "--aoi", "c.gpkg",
              "--metric-crs", CRS, "--output-json", str(tmp_path / "metrics.json")])


def test_size_classes_split_recall_by_reference_area_and_precision_by_prediction_area():
    # References of 9, 25 and 400 m2; the 25 m2 one is missed and a 16 m2 prediction is extra.
    reference = frame([box(0, 0, 3, 3), box(10, 0, 15, 5), box(30, 0, 50, 20)])
    predictions = frame([box(0, 0, 3, 3), box(30, 0, 50, 20), box(60, 0, 64, 4)])
    result = evaluate(predictions, reference, aoi(), crs=CRS, size_bins=[20, 100])
    report = result.report
    assert report["size_bins_m2"] == [20.0, 100.0]
    classes = {entry["size_class"]: entry for entry in report["size_classes"]}
    assert list(classes) == ["<20", "20-100", ">=100"]
    assert (classes["<20"]["reference"], classes["<20"]["predictions"]) == (1, 2)
    assert (classes["<20"]["recall"], classes["<20"]["precision"]) == (1.0, 0.5)
    assert classes["<20"]["f1"] == pytest.approx(2 / 3)
    # Only a missed reference: recall 0, no predictions, F1 0 as for the overall score.
    assert (classes["20-100"]["recall"], classes["20-100"]["precision"], classes["20-100"]["f1"]) == (0.0, None, 0.0)
    assert (classes[">=100"]["recall"], classes[">=100"]["precision"]) == (1.0, 1.0)
    thresholds = {entry["min_m2"]: entry for entry in report["size_thresholds"]}
    assert (thresholds[20.0]["reference"], thresholds[20.0]["matched_reference"]) == (2, 1)
    assert thresholds[20.0]["precision"] == 1.0
    assert report["evaluated_reference_size_classes"] == ["<20", "20-100", ">=100"]
    assert list(result.layers["unmatched_predictions"].size_class) == ["<20"]


def test_area_on_an_edge_belongs_to_the_class_above():
    reference = frame([box(0, 0, 4, 5)])
    report = evaluate(reference, reference, aoi(), crs=CRS, size_bins=[20]).report
    assert report["evaluated_reference_size_classes"] == [">=20"]


def test_without_size_bins_the_report_has_no_size_classes():
    reference = frame([box(0, 0, 4, 5)])
    report = evaluate(reference, reference, aoi(), crs=CRS).report
    assert report["size_bins_m2"] is None and "size_classes" not in report


@pytest.mark.parametrize("bins", [[], [0], [-5], [50, 20], [20, 20], [float("nan")]])
def test_invalid_size_bins_are_refused(bins):
    reference = frame([box(0, 0, 4, 5)])
    with pytest.raises(ValueError, match="size bins"):
        evaluate(reference, reference, aoi(), crs=CRS, size_bins=bins)


def test_a_pair_spanning_two_classes_leaves_class_f1_undefined_not_zero():
    # A 19 m2 building matched by a 21 m2 outline: perfect overall, but each class has one side only.
    report = evaluate(frame([box(0, 0, 4.2, 5)]), frame([box(0, 0, 3.8, 5)]), aoi(), crs=CRS,
                      size_bins=[20]).report
    assert report["f1"] == 1.0
    classes = {entry["size_class"]: entry for entry in report["size_classes"]}
    assert (classes["<20"]["recall"], classes["<20"]["precision"], classes["<20"]["f1"]) == (1.0, None, None)
    assert (classes[">=20"]["recall"], classes[">=20"]["precision"], classes[">=20"]["f1"]) == (None, 1.0, None)
    assert report["size_thresholds"][0]["size_class"] == ">=20"


def test_an_input_size_class_field_is_only_reserved_when_size_bins_are_used():
    reference = frame([box(0, 0, 4, 5)], size_class=["house"])
    assert evaluate(reference, reference, aoi(), crs=CRS).report["f1"] == 1.0
    with pytest.raises(ValueError, match="reserved"):
        evaluate(reference, reference, aoi(), crs=CRS, size_bins=[20])
