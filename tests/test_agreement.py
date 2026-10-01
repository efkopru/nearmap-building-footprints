import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from nearmap_buildings.agreement import agree, main
from nearmap_buildings.evaluation import evaluate


CRS = "EPSG:26914"


def frame(geometries, **columns):
    return gpd.GeoDataFrame(columns, geometry=geometries, crs=CRS)


def aoi():
    return frame([box(-10, -10, 200, 200)])


def methods():
    # Both draw houses at x=0 and x=40 (x=40 is missing from the reference);
    # only sam draws x=80, only esri draws x=120.
    sam = frame([box(0, 0, 10, 10), box(40, 0, 50, 10), box(80, 0, 90, 10)])
    esri = frame([box(120, 0, 130, 10), box(41, 0, 50, 10), box(0, 0, 10, 10)])
    reference = frame([box(0, 0, 10, 10), box(80, 0, 90, 10)])
    return sam, esri, reference


def test_pairs_and_one_sided_outlines_without_reference():
    sam, esri, _ = methods()
    summary, layers = agree(sam, esri, aoi(), crs=CRS, labels=("sam3", "esri"))
    assert summary["pairs"] == 2
    assert summary["share_matched"] == {"sam3": pytest.approx(2 / 3), "esri": pytest.approx(2 / 3)}
    assert list(layers["both"].eval_id) == [0, 1]
    assert list(layers["both"].esri_match_id) == [2, 1]
    assert list(layers["only_sam3"].eval_id) == [2]
    assert list(layers["only_esri"].eval_id) == [0]
    assert "candidate_new" not in layers and "candidate_new" not in summary


def test_candidates_are_agreed_outlines_the_reference_lacks():
    sam, esri, reference = methods()
    reports = [evaluate(method, reference, aoi(), crs=CRS, reference_status="agreement_only").report
               for method in (sam, esri)]
    summary, layers = agree(sam, esri, aoi(), crs=CRS, labels=("sam3", "esri"), reports=reports)
    assert list(layers["candidate_new"].eval_id) == [1]
    assert summary["candidate_new"] == 1
    # sam3's x=80 house matches the reference; esri's x=120 extra does not, and nor does x=40.
    assert summary["unmatched_to_reference"] == {"sam3": {"outlines": 1, "matched_by_other": 1},
                                                 "esri": {"outlines": 2, "matched_by_other": 1}}
    assert list(layers["both"].sam3_reference_match) == [True, False]
    assert list(layers["both"].esri_reference_match) == [True, False]
    # One report is enough; it only judges its own method's outlines.
    summary, layers = agree(sam, esri, aoi(), crs=CRS, labels=("sam3", "esri"), reports=[reports[0], None])
    assert list(layers["candidate_new"].eval_id) == [1]


def test_a_report_from_other_or_reordered_predictions_is_refused():
    sam, esri, reference = methods()
    report = evaluate(sam, reference, aoi(), crs=CRS).report
    with pytest.raises(ValueError, match="different predictions"):
        agree(sam.iloc[::-1].reset_index(drop=True), esri, aoi(), crs=CRS, reports=[report, None])
    old = {key: value for key, value in report.items() if key != "predictions_order_fingerprint"}
    with pytest.raises(ValueError, match="predates"):
        agree(sam, esri, aoi(), crs=CRS, reports=[old, None])


def test_labels_must_give_distinct_fields():
    sam, esri, _ = methods()
    with pytest.raises(ValueError, match="distinct"):
        agree(sam, esri, aoi(), crs=CRS, labels=("SAM 3", "sam-3"))


def test_command_writes_summary_and_layers(tmp_path):
    sam, esri, reference = methods()
    paths = {}
    for name, value in (("sam", sam), ("esri", esri), ("reference", reference), ("aoi", aoi())):
        paths[name] = tmp_path / f"{name}.gpkg"
        value.to_file(paths[name])
    report = tmp_path / "sam_eval.json"
    report.write_text(json.dumps(evaluate(sam, reference, aoi(), crs=CRS).report), encoding="utf-8")
    output_json, output_gpkg = tmp_path / "agree.json", tmp_path / "agree.gpkg"
    main(["--predictions", str(paths["sam"]), str(paths["esri"]), "--labels", "sam3", "esri",
          "--reports", str(report), "-", "--aoi", str(paths["aoi"]), "--metric-crs", CRS,
          "--output-json", str(output_json), "--output-gpkg", str(output_gpkg)])
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert (summary["pairs"], summary["candidate_new"]) == (2, 1)
    assert set(gpd.list_layers(output_gpkg).name) == {"both", "only_sam3", "only_esri", "candidate_new"}
    with pytest.raises(SystemExit):
        main(["--predictions", str(paths["sam"]), str(paths["esri"]), "--aoi", str(paths["aoi"]),
              "--metric-crs", CRS, "--output-json", str(output_json)])


def test_an_outline_the_evaluation_left_out_is_not_a_candidate():
    sam, esri, reference = methods()
    # This report's AOI cuts through the x=40 house, so the evaluation excluded it.
    report = evaluate(sam, reference, frame([box(-10, -10, 45, 200)]), crs=CRS).report
    summary, layers = agree(sam, esri, aoi(), crs=CRS, labels=("sam3", "esri"), reports=[report, None])
    assert layers["both"].sam3_reference_match.isna().tolist() == [False, True]
    assert summary["candidate_new"] == 0 and layers["candidate_new"].empty
