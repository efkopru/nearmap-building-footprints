"""Exercise the shipped accuracy recipe with synthetic, cross-threshold objects."""
import csv
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from nearmap_buildings.cli import main
from nearmap_buildings.review import review_check


CONFIG = Path(__file__).parents[1] / "configs" / "accuracy.example.json"
CRS = "EPSG:26914"


def prepare(tmp_path, *, attested):
    reference = gpd.GeoDataFrame(geometry=[box(0, 0, 19, 1), box(100, 0, 121, 1)], crs=CRS)
    predictions = gpd.GeoDataFrame(geometry=[box(0, 0, 21, 1), box(100, 0, 119, 1)], crs=CRS)
    aoi = gpd.GeoDataFrame(geometry=[box(-10, -10, 200, 10)], crs=CRS)
    folder = tmp_path / "data/reference"
    folder.mkdir(parents=True)
    reference.to_file(folder / "test_area_reviewed.gpkg")
    aoi.to_file(folder / "test_area_aoi.gpkg")
    for method in ("city/sam3_building_1536", "esri_city_202605", "city/esri_local"):
        folder = tmp_path / "outputs/lewisville" / method
        folder.mkdir(parents=True)
        predictions.to_file(folder / "cleaned.gpkg", layer="cleaned")
    receipt, _ = review_check(reference, reference, aoi, crs=CRS, provenance={
        "reviewer": "Synthetic test fixture", "review_date": "2026-10-15",
        "imagery_date": "2026-05", "label_convention": "Synthetic rectangles",
        "coverage_complete": attested, "independent_review": attested,
    })
    folder = tmp_path / "outputs/accuracy"
    folder.mkdir()
    (folder / "review_check.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_accuracy_recipe_separates_primary_scores_from_all_size_diagnostics(tmp_path, monkeypatch):
    prepare(tmp_path, attested=True)
    monkeypatch.chdir(tmp_path)
    for step in json.loads(CONFIG.read_text(encoding="utf-8"))["steps"]:
        assert main(step) == 0
    for method in ("sam3", "esri_city", "esri_local"):
        all_sizes = json.loads(Path(f"outputs/accuracy/all_sizes/{method}/evaluation.json").read_text())
        primary = json.loads(Path(f"outputs/accuracy/at_least_20m2/{method}/evaluation.json").read_text())
        assert all_sizes["f1"] == 1.0
        assert primary["f1"] == 0.0
        assert primary["min_area_m2"] == 20
        assert primary["true_positives"] == 0
        assert primary["false_positives"] == primary["false_negatives"] == 1
    with Path("outputs/accuracy/at_least_20m2/comparison.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and all(float(row["f1"]) == 0.0 for row in rows)


@pytest.mark.parametrize("missing", [False, True])
def test_accuracy_recipe_cannot_attest_to_its_own_review(tmp_path, monkeypatch, missing):
    prepare(tmp_path, attested=False)
    if missing:
        (tmp_path / "outputs/accuracy/review_check.json").unlink()
    monkeypatch.chdir(tmp_path)
    step = json.loads(CONFIG.read_text(encoding="utf-8"))["steps"][0]
    assert main(step) == 2
    assert not Path("outputs/accuracy/all_sizes").exists()
