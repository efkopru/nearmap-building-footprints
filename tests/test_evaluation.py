import json

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from nearmap_buildings.evaluation import evaluate, main, maximum_cardinality_matches


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
    assert len(gpd.list_layers(output_gpkg)) == 6
    assert gpd.read_file(output_gpkg, layer="matched_predictions").instance_id.tolist() == ["synthetic-1"]
    assert {role: (tmp_path / f"{role}.gpkg").read_bytes() for role in source} == before


def test_cli_requires_independent_holdout_acknowledgement(tmp_path):
    with pytest.raises(SystemExit):
        main(["--predictions", "a.gpkg", "--reference", "b.gpkg", "--aoi", "c.gpkg",
              "--metric-crs", CRS, "--output-json", str(tmp_path / "metrics.json")])
