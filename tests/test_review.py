import json

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from nearmap_buildings.evaluation import evaluate
from nearmap_buildings.review import main, review_check


CRS = "EPSG:26914"


def frame(geometries, **columns):
    return gpd.GeoDataFrame(columns, geometry=geometries, crs=CRS)


def aoi():
    return frame([box(-10, -10, 200, 200)])


def layers():
    original = frame([box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10), box(195, 0, 205, 10)])
    # Kept as is, extended (an addition built on), demolished, and one new house; one crosses the AOI edge.
    reviewed = frame([box(0, 0, 10, 10), box(20, 0, 30, 14), box(60, 0, 70, 10), box(195, 0, 205, 10)])
    return original, reviewed


def test_counts_what_the_review_changed():
    summary, result = review_check(*layers(), aoi(), crs=CRS, provenance={"reviewer": "EK", "imagery_date": "2026-05"})
    assert summary["counts"] == {
        "original_in_aoi": 3, "reviewed_in_aoi": 3, "unchanged": 1, "modified": 1, "added": 1, "removed": 1,
        "crossing_aoi_edge": {"original": 1, "reviewed": 1}, "reviewed_problems": 0, "original_problems": 0,
        "reviewed_overlaps": 0}
    assert summary["ready_for_evaluation"] and summary["provenance"]["reviewer"] == "EK"
    assert result["modified"].review_iou.iloc[0] == pytest.approx(100 / 140)
    assert list(result["removed"].geometry) == [box(40, 0, 50, 10)]


def test_fingerprint_matches_the_evaluation_reference_fingerprint():
    original, reviewed = layers()
    summary, _ = review_check(original, reviewed, aoi(), crs=CRS)
    report = evaluate(reviewed, reviewed, aoi(), crs=CRS).report
    assert summary["reviewed_fingerprint"] == report["reference_fingerprint"]


def test_invalid_and_overlapping_outlines_are_flagged():
    original, _ = layers()
    bowtie = Polygon([(80, 0), (90, 10), (90, 0), (80, 10), (80, 0)])
    reviewed = frame([box(0, 0, 10, 10), box(5, 0, 15, 10), bowtie, None])
    summary, result = review_check(original, reviewed, aoi(), crs=CRS)
    assert not summary["ready_for_evaluation"] and summary["reviewed_fingerprint"] is None
    assert summary["counts"]["reviewed_problems"] == 2 and summary["counts"]["reviewed_overlaps"] == 1
    assert sorted(result["problems"].problem.str.split(":").str[0]) == ["invalid", "null_or_empty"]
    assert result["overlaps"].overlap_m2.iloc[0] == pytest.approx(50)


def test_command_writes_summary_and_layers(tmp_path):
    original, reviewed = layers()
    paths = {}
    for name, value in (("original", original), ("reviewed", reviewed), ("aoi", aoi())):
        paths[name] = tmp_path / f"{name}.gpkg"
        value.to_file(paths[name])
    output_json, output_gpkg = tmp_path / "review.json", tmp_path / "review.gpkg"
    main(["--original", str(paths["original"]), "--reviewed", str(paths["reviewed"]), "--aoi", str(paths["aoi"]),
          "--metric-crs", CRS, "--reviewer", "EK", "--imagery-date", "2026-05",
          "--output-json", str(output_json), "--output-gpkg", str(output_gpkg)])
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["counts"]["added"] == 1 and summary["provenance"]["imagery_date"] == "2026-05"
    assert set(gpd.list_layers(output_gpkg).name) == {"added", "removed", "modified", "unchanged", "problems", "overlaps"}
    with pytest.raises(SystemExit):
        main(["--original", str(paths["original"]), "--reviewed", str(paths["reviewed"]), "--aoi", str(paths["aoi"]),
              "--metric-crs", CRS, "--output-json", str(output_json)])
