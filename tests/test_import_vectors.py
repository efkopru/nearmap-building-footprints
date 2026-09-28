import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from nearmap_buildings.compare import per_building
from nearmap_buildings.demo import create_demo
from nearmap_buildings.evaluation import main as evaluate_main
from nearmap_buildings.import_vectors import import_vectors, main
from nearmap_buildings.postprocess import main as clean_main, read_polygon_inputs


CRS = "EPSG:26914"


def esri_gdb(path, confidence=(97.5, 88.0), extra_layer=False):
    """A file geodatabase shaped like esri.py output: Class, a 0-100 Confidence, and method."""
    bowtie = Polygon([(20, 0), (26, 6), (20, 6), (26, 0)])
    frame = gpd.GeoDataFrame({"Class": "Building", "Confidence": list(confidence), "method": "esri_building_usa"},
                             geometry=[box(0, 0, 10, 10), bowtie], crs=CRS)
    frame.to_file(path, layer="esri_buildings", driver="OpenFileGDB")
    if extra_layer:
        frame.iloc[:1].to_file(path, layer="earlier_run", driver="OpenFileGDB")
    return path


def test_esri_preset_rescales_confidence_and_keeps_invalid_outlines(tmp_path):
    source = esri_gdb(tmp_path / "results.gdb")
    result = import_vectors(source, tmp_path / "esri.gpkg", CRS, layer="esri_buildings", preset="esri")
    assert (result["method"], result["invalid_polygons_kept"]) == ("esri_building_usa", 1)
    imported = gpd.read_file(tmp_path / "esri.gpkg")
    np.testing.assert_allclose(imported.score, [0.975, 0.88])
    assert imported.source_id.tolist() == ["1", "2"]  # OBJECTIDs, traceable in ArcGIS Pro.
    assert not imported.geometry.iloc[1].is_valid  # Left for nbf clean to repair and audit.


def test_invalid_outlines_need_explicit_permission(tmp_path):
    source = esri_gdb(tmp_path / "results.gdb")
    with pytest.raises(ValueError, match="--allow-invalid"):
        import_vectors(source, tmp_path / "out.gpkg", CRS, layer="esri_buildings")


def test_percentage_scores_need_an_explicit_scale(tmp_path):
    source = esri_gdb(tmp_path / "results.gdb")
    with pytest.raises(ValueError, match="--score-scale 100"):
        import_vectors(source, tmp_path / "out.gpkg", CRS, layer="esri_buildings",
                       score_field="Confidence", allow_invalid=True)


def test_rescaling_scores_already_between_0_and_1_is_refused(tmp_path):
    source = esri_gdb(tmp_path / "results.gdb", confidence=(0.975, 0.88))
    with pytest.raises(ValueError, match="already looks like a 0-1 score"):
        import_vectors(source, tmp_path / "out.gpkg", CRS, layer="esri_buildings", preset="esri")
    result = import_vectors(source, tmp_path / "out.gpkg", CRS, layer="esri_buildings", preset="esri", score_scale=1)
    assert result["score_scale"] == 1


def test_method_label_comes_from_the_input_and_contradictions_fail(tmp_path):
    source = esri_gdb(tmp_path / "results.gdb")
    result = import_vectors(source, tmp_path / "a.gpkg", CRS, layer="esri_buildings", allow_invalid=True)
    assert result["method"] == "esri_building_usa"
    with pytest.raises(ValueError, match="contradicts"):
        import_vectors(source, tmp_path / "b.gpkg", CRS, layer="esri_buildings", method="nearmap_ai", allow_invalid=True)


def test_several_layers_require_choosing_one(tmp_path):
    source = esri_gdb(tmp_path / "results.gdb", extra_layer=True)
    with pytest.raises(ValueError, match="choose one with --layer"):
        import_vectors(source, tmp_path / "out.gpkg", CRS, preset="esri")


def test_clean_points_file_geodatabases_to_the_importer(tmp_path):
    with pytest.raises(ValueError, match="--preset esri"):
        read_polygon_inputs(esri_gdb(tmp_path / "results.gdb"), "esri_buildings")


def test_cli_reports_the_resolved_preset(tmp_path, capsys):
    source = esri_gdb(tmp_path / "results.gdb")
    main(["--input", str(source), "--layer", "esri_buildings", "--preset", "esri", "--crs", CRS,
          "--output", str(tmp_path / "esri.gpkg")])
    printed = json.loads(capsys.readouterr().out)
    assert (printed["score_field"], printed["score_scale"], printed["source_id"]) == ("Confidence", 100.0, "source feature ID")


def test_demo_esri_output_joins_the_per_building_comparison(tmp_path):
    demo = create_demo(tmp_path / "demo")
    import_vectors(demo / "esri_format_predictions.gdb", tmp_path / "esri.gpkg", "EPSG:32614",
                   layer="esri_buildings", preset="esri")
    clean_main(["--input", str(tmp_path / "esri.gpkg"), "--output", str(tmp_path / "esri_clean.gpkg"),
                "--metric-crs", "EPSG:32614"])
    reports = []
    for name, predictions, layer in (("identity", demo / "perfect_predictions.gpkg", None),
                                     ("esri", tmp_path / "esri_clean.gpkg", "cleaned")):
        report = tmp_path / name / "evaluation.json"
        evaluate_main(["--predictions", str(predictions), *(["--predictions-layer", layer] if layer else []),
                       "--reference", str(demo / "reference.gpkg"), "--aoi", str(demo / "aoi.geojson"),
                       "--metric-crs", "EPSG:32614", "--output-json", str(report), "--independent-holdout"])
        reports.append(report)
    # The reference file is found through the path each report recorded.
    buildings = per_building(reports, ["identity", "esri"])
    assert buildings.outcome.value_counts().to_dict() == {"found by both": 8, "only identity": 1}
    assert buildings.loc[buildings.outcome == "only identity", "building_id"].tolist() == ["synthetic_5"]
