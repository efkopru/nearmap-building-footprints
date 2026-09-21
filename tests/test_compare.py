from copy import deepcopy
import csv
import json

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from nearmap_buildings.compare import compare_reports, main
from nearmap_buildings.evaluation import evaluate, geometry_fingerprint


CRS = "EPSG:26914"


def frame(geometries, **attributes):
    return gpd.GeoDataFrame(attributes, geometry=geometries, crs=CRS)


def report(predictions=None):
    reference = frame([box(0, 0, 10, 10), box(20, 0, 30, 10)])
    predictions = reference if predictions is None else predictions
    return evaluate(predictions, reference, frame([box(-10, -10, 100, 100)]), crs=CRS).report


def save(tmp_path, name, value):
    path = tmp_path / name / "evaluation.json"
    path.parent.mkdir()
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def test_fingerprint_stable_to_order_ring_start_direction_and_attributes():
    square = box(0, 0, 10, 10)
    second = box(20, 0, 30, 10)
    original = frame([square, second], label=["a", "b"])
    rotated_ring = Polygon([(10, 10), (10, 0), (0, 0), (0, 10), (10, 10)])
    reordered = frame([second, rotated_ring], label=["different", "attributes"])
    assert geometry_fingerprint(original) == geometry_fingerprint(reordered)
    assert geometry_fingerprint(frame([MultiPolygon([square, second])])) == geometry_fingerprint(frame([MultiPolygon([second, square])]))


def test_fingerprint_changes_for_geometry_crs_or_duplicate_count():
    original = frame([box(0, 0, 10, 10)])
    fingerprint = geometry_fingerprint(original)
    assert fingerprint != geometry_fingerprint(frame([box(0, 0, 10.01, 10)]))
    assert fingerprint != geometry_fingerprint(original.set_crs("EPSG:32614", allow_override=True))
    assert fingerprint != geometry_fingerprint(frame([box(0, 0, 10, 10)] * 2))


def test_evaluation_fingerprints_use_source_geometry_and_are_order_independent():
    source = frame([box(0, 0, 10, 10), box(20, 0, 30, 10)])
    aoi = frame([box(-10, -10, 100, 100)])
    first = evaluate(source, source, aoi, crs=CRS).report
    second = evaluate(source, source.iloc[::-1], aoi, crs=CRS).report
    assert first["reference_fingerprint"] == second["reference_fingerprint"]
    assert first["aoi_fingerprint"] == second["aoi_fingerprint"]


def test_comparison_preserves_separate_metrics_for_same_holdout(tmp_path):
    perfect = save(tmp_path, "text", report())
    empty = save(tmp_path, "box", report(frame([])))
    rows = compare_reports([perfect, empty])
    assert [row["label"] for row in rows] == ["text", "box"]
    assert [row["f1"] for row in rows] == [1, 0]
    assert rows[1]["precision"] is None
    assert rows[1]["matched_iou_mean"] is None


@pytest.mark.parametrize("field,new_value", [
    ("reference_fingerprint", "a" * 64), ("aoi_fingerprint", "b" * 64),
    ("metric_crs", "EPSG:32614"), ("iou_threshold", 0.75), ("edge_policy", "clip"),
    ("matching", "some other algorithm"),
])
def test_rejects_mismatched_holdout_or_settings(tmp_path, field, new_value):
    first = report()
    second = deepcopy(first)
    second[field] = new_value
    paths = [save(tmp_path, "one", first), save(tmp_path, "two", second)]
    with pytest.raises(ValueError, match=field):
        compare_reports(paths)


@pytest.mark.parametrize("missing", ["fingerprint_method", "reference_fingerprint", "aoi_fingerprint"])
def test_rejects_old_reports_without_fingerprints(tmp_path, missing):
    value = report()
    del value[missing]
    path = save(tmp_path, "old", value)
    with pytest.raises(ValueError, match="regenerate old"):
        compare_reports([path])


def test_cli_writes_csv_with_explicit_labels_and_refuses_overwrite(tmp_path):
    first = save(tmp_path, "one", report())
    second = save(tmp_path, "two", report(frame([])))
    output = tmp_path / "comparison.csv"
    args = ["--reports", str(first), str(second), "--labels", "SAM text", "SAM boxes", "--output", str(output)]
    main(args)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["label"] for row in rows] == ["SAM text", "SAM boxes"]
    assert rows[1]["precision"] == ""
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before


def test_failed_comparison_does_not_create_csv(tmp_path):
    first = save(tmp_path, "one", report())
    second_report = report()
    second_report["iou_threshold"] = 0.8
    second = save(tmp_path, "two", second_report)
    output = tmp_path / "comparison.csv"
    with pytest.raises(ValueError, match="iou_threshold"):
        main(["--reports", str(first), str(second), "--output", str(output)])
    assert not output.exists()


def test_labels_must_match_reports_and_be_unique(tmp_path):
    paths = [save(tmp_path, "one", report()), save(tmp_path, "two", report())]
    with pytest.raises(ValueError, match="one nonempty"):
        compare_reports(paths, ["one"])
    with pytest.raises(ValueError, match="unique"):
        compare_reports(paths, ["same", "same"])
