from copy import deepcopy
import csv
import json

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from nearmap_buildings.compare import _outcome, compare_reports, main, per_building
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


def test_rows_identify_predictions_and_older_reports_remain_comparable(tmp_path):
    current = report()
    older = report(frame([]))
    del older["predictions_fingerprint"]
    rows = compare_reports([save(tmp_path, "text", current), save(tmp_path, "box", older)])
    assert rows[0]["predictions_fingerprint"] == current["predictions_fingerprint"]
    assert rows[1]["predictions_fingerprint"] is None


def test_agreement_only_and_holdout_reports_are_never_mixed(tmp_path):
    holdout = report()
    agreement = deepcopy(holdout)
    agreement["reference_status"] = "agreement_only"
    with pytest.raises(ValueError, match="reference_status"):
        compare_reports([save(tmp_path, "one", holdout), save(tmp_path, "two", agreement)])


def test_older_reports_count_as_holdouts_and_the_status_is_in_the_csv(tmp_path):
    older = report()
    del older["reference_status"]
    rows = compare_reports([save(tmp_path, "old", older), save(tmp_path, "new", report())])
    assert [row["reference_status"] for row in rows] == ["independent_holdout", "independent_holdout"]


def test_rejects_malformed_predictions_fingerprint(tmp_path):
    value = report()
    value["predictions_fingerprint"] = "not-a-hash"
    with pytest.raises(ValueError, match="predictions_fingerprint"):
        compare_reports([save(tmp_path, "bad", value)])


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


def two_method_reports(tmp_path):
    """Esri finds buildings a and b; SAM 3 finds only b; nobody finds c."""
    reference = frame([box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)], building_id=["a", "b", "c"])
    reference_path = tmp_path / "reference.gpkg"
    reference.to_file(reference_path, driver="GPKG")
    aoi = frame([box(-10, -10, 100, 100)])
    esri = evaluate(frame([box(0, 0, 10, 10), box(20, 0, 30, 10)]), reference, aoi, crs=CRS).report
    sam = evaluate(frame([box(20.5, 0, 30, 10)]), reference, aoi, crs=CRS).report
    return [save(tmp_path, "esri", esri), save(tmp_path, "sam3", sam)], reference, reference_path


def test_per_building_shows_which_methods_found_each_building(tmp_path):
    paths, _, reference_path = two_method_reports(tmp_path)
    buildings = per_building(paths, ["Esri", "SAM 3 text"], reference=reference_path)
    assert buildings.building_id.tolist() == ["a", "b", "c"]
    assert buildings.outcome.tolist() == ["only Esri", "found by both", "missed by both"]
    assert buildings.found_by.tolist() == ["Esri", "Esri;SAM 3 text", ""]
    assert buildings.esri_match_id.tolist() == [0, 1, -1]
    assert buildings.sam_3_text_match_iou.iloc[1] == pytest.approx(0.95)


def test_per_building_refuses_a_reordered_reference(tmp_path):
    paths, reference, _ = two_method_reports(tmp_path)
    reordered = tmp_path / "reordered.gpkg"
    reference.iloc[::-1].to_file(reordered, driver="GPKG")
    with pytest.raises(ValueError, match="row order"):
        per_building(paths, ["esri", "sam3"], reference=reordered)


def test_per_building_needs_reports_with_reference_row_ids(tmp_path):
    paths, _, reference_path = two_method_reports(tmp_path)
    old = json.loads(paths[1].read_text())
    del old["evaluated_reference_ids"]
    paths[1].write_text(json.dumps(old))
    with pytest.raises(ValueError, match="re-run nbf evaluate"):
        per_building(paths, ["esri", "sam3"], reference=reference_path)


def test_per_building_labels_must_give_distinct_field_names(tmp_path):
    paths, _, reference_path = two_method_reports(tmp_path)
    with pytest.raises(ValueError, match="distinct field names"):
        per_building(paths, ["SAM 3", "sam-3"], reference=reference_path)


def test_outcome_names_for_three_methods():
    labels = ["esri", "sam3_text", "nearmap"]
    assert _outcome(["esri", "sam3_text", "nearmap"], labels) == "found by all"
    assert _outcome(["esri", "nearmap"], labels) == "found by 2 of 3"
    assert _outcome([], labels) == "missed by all"
    assert _outcome(["nearmap"], labels) == "only nearmap"


def test_cli_writes_per_building_layer_and_refuses_to_replace_it(tmp_path, capsys):
    paths, _, reference_path = two_method_reports(tmp_path)
    layer = tmp_path / "per_building.gpkg"
    args = ["--reports", *map(str, paths), "--labels", "esri", "sam3", "--output", str(tmp_path / "comparison.csv"),
            "--per-building", str(layer), "--reference", str(reference_path)]
    main(args)
    assert json.loads(capsys.readouterr().out)["outcomes"] == {"only esri": 1, "found by both": 1, "missed by both": 1}
    assert gpd.read_file(layer, layer="buildings").outcome.tolist() == ["only esri", "found by both", "missed by both"]
    again = ["--reports", *map(str, paths), "--labels", "esri", "sam3", "--output", str(tmp_path / "second.csv"),
             "--per-building", str(layer), "--reference", str(reference_path)]
    with pytest.raises(SystemExit):
        main(again)
    assert not (tmp_path / "second.csv").exists()


def test_labels_must_match_reports_and_be_unique(tmp_path):
    paths = [save(tmp_path, "one", report()), save(tmp_path, "two", report())]
    with pytest.raises(ValueError, match="one nonempty"):
        compare_reports(paths, ["one"])
    with pytest.raises(ValueError, match="unique"):
        compare_reports(paths, ["same", "same"])


def sized_report(predictions=None, bins=(50,)):
    reference = frame([box(0, 0, 5, 5), box(20, 0, 30, 10)])
    predictions = reference if predictions is None else predictions
    return evaluate(predictions, reference, frame([box(-10, -10, 100, 100)]), crs=CRS, size_bins=list(bins)).report


def test_by_size_csv_lists_classes_and_thresholds_per_method(tmp_path):
    first = save(tmp_path, "a", sized_report())
    second = save(tmp_path, "b", sized_report(frame([box(20, 0, 30, 10)])))
    output, by_size = tmp_path / "comparison.csv", tmp_path / "by_size.csv"
    main(["--reports", str(first), str(second), "--labels", "a", "b", "--output", str(output),
          "--by-size", str(by_size)])
    rows = list(csv.DictReader(by_size.open(encoding="utf-8")))
    assert [(row["label"], row["kind"], row["size_class"]) for row in rows] == [
        ("a", "class", "<50"), ("a", "class", ">=50"), ("a", "at_least", ">=50"),
        ("b", "class", "<50"), ("b", "class", ">=50"), ("b", "at_least", ">=50")]
    small_b = rows[3]
    assert (small_b["reference"], small_b["matched_reference"], small_b["recall"]) == ("1", "0", "0.0")


def test_by_size_needs_reports_made_with_size_bins(tmp_path):
    path = save(tmp_path, "a", report())
    with pytest.raises(ValueError, match="--size-bins-m2"):
        main(["--reports", str(path), "--output", str(tmp_path / "c.csv"), "--by-size", str(tmp_path / "s.csv")])
    assert not (tmp_path / "c.csv").exists()


def test_reports_with_different_size_bins_are_not_compared(tmp_path):
    first = save(tmp_path, "a", sized_report(bins=(50,)))
    second = save(tmp_path, "b", sized_report(bins=(20,)))
    with pytest.raises(ValueError, match="size_bins_m2"):
        compare_reports([first, second])
    unsized = save(tmp_path, "c", report())
    with pytest.raises(ValueError, match="size_bins_m2"):
        compare_reports([first, unsized])


def test_per_building_records_each_reference_size_class(tmp_path):
    reference = frame([box(0, 0, 5, 5), box(20, 0, 30, 10)])
    reference_path = tmp_path / "reference.gpkg"
    reference.to_file(reference_path)
    first = save(tmp_path, "a", sized_report())
    buildings = per_building([first], ["a"], reference_path)
    assert list(buildings.size_class) == ["<50", ">=50"]


def test_by_size_outputs_go_into_new_folders(tmp_path):
    first = save(tmp_path, "a", sized_report())
    output, by_size = tmp_path / "out" / "comparison.csv", tmp_path / "sub" / "by_size.csv"
    main(["--reports", str(first), "--output", str(output), "--by-size", str(by_size)])
    assert output.exists() and by_size.exists()


def test_per_building_without_size_bins_keeps_a_reference_size_class_field(tmp_path):
    reference = frame([box(0, 0, 10, 10), box(20, 0, 30, 10)], size_class=["shed", "house"])
    reference_path = tmp_path / "reference.gpkg"
    reference.to_file(reference_path)
    evaluation = evaluate(reference, reference, frame([box(-10, -10, 100, 100)]), crs=CRS).report
    buildings = per_building([save(tmp_path, "a", evaluation)], ["a"], reference_path)
    assert list(buildings.size_class) == ["shed", "house"]
