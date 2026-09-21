import math

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from nearmap_buildings.postprocess import (
    clean_polygons, conservative_regularize, main, metric_crs, read_polygon_inputs,
    write_layers,
)


CRS = "EPSG:26914"


def frame(geometries, **columns):
    return gpd.GeoDataFrame(columns, geometry=geometries, crs=CRS)


def test_overlap_dedup_keeps_best_score_audits_and_preserves_touching_neighbor():
    source = frame([box(0, 0, 10, 10), box(0.2, 0, 10.2, 10), box(10, 0, 20, 10)],
                   score=[0.8, 0.95, 0.7], tile_id=["a", "b", "c"],
                   instance_id=[1, 2, 3], edge_touch=[True, False, True], method=["sam3"] * 3)
    before = source.geometry.to_wkb().tolist()
    result = clean_polygons(source, crs=CRS)
    assert result.cleaned.instance_id.tolist() == [2, 3]
    assert result.cleaned.edge_touch.tolist() == [False, True]
    assert result.removed.instance_id.tolist() == [1]
    assert result.removed.removed_reason.tolist() == ["duplicate_overlap"]
    assert result.removed.kept_cleanup_id.tolist() == [1]
    assert result.removed.duplicate_iou.iloc[0] > 0.9
    assert source.geometry.to_wkb().tolist() == before
    assert result.cleaned.geometry.iloc[1].equals(source.geometry.iloc[2])


def test_containment_removes_low_score_fragment_without_union():
    source = frame([box(0, 0, 10, 10), box(1, 1, 3, 3)], score=[0.9, 0.4])
    result = clean_polygons(source, crs=CRS, min_area=0)
    assert len(result.cleaned) == 1
    assert result.cleaned.geometry.iloc[0].area == 100
    assert result.removed.duplicate_containment.iloc[0] == 1
    assert result.removed.duplicate_iou.iloc[0] == 0.04


def test_exact_edge_touch_is_not_duplicate():
    source = frame([box(0, 0, 10, 10), box(10, 0, 20, 10)], score=[1, 1])
    result = clean_polygons(source, crs=CRS)
    assert len(result.cleaned) == 2
    assert result.removed.empty


def test_missing_scores_tie_by_input_order_and_finite_score_wins():
    result = clean_polygons(frame([box(0, 0, 10, 10)] * 3, score=[np.nan, 0.5, np.inf]), crs=CRS)
    assert result.cleaned.cleanup_id.tolist() == [1]
    result = clean_polygons(frame([box(0, 0, 10, 10)] * 2), crs=CRS)
    assert result.cleaned.cleanup_id.tolist() == [0]


def test_repairs_invalid_and_audits_null_empty_small_nonpolygon():
    bowtie = Polygon([(0, 0), (4, 4), (0, 4), (4, 0), (0, 0)])
    source = frame([bowtie, None, Polygon(), box(20, 0, 20.1, 0.1), LineString([(30, 0), (31, 1)])])
    result = clean_polygons(source, crs=CRS, min_area=1)
    assert len(result.cleaned) == 1
    assert result.cleaned.geometry.iloc[0].is_valid
    assert "repaired_invalid" in result.cleaned.cleanup_flags.iloc[0]
    assert result.removed.removed_reason.tolist() == ["null_or_empty", "null_or_empty", "below_min_area", "nonpolygon_or_unrepairable"]


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:2276"])
def test_rejects_degrees_or_feet(crs):
    with pytest.raises(ValueError, match="metre"):
        metric_crs(crs)


def test_requires_source_crs_and_reprojects_valid_inputs():
    source = frame([box(500000, 3600000, 500010, 3600010)])
    geographic = source.to_crs(4326)
    cleaned = clean_polygons(geographic, crs=CRS).cleaned
    assert cleaned.area_m2.iloc[0] == pytest.approx(100, abs=1e-5)
    source = source.set_crs(None, allow_override=True)
    with pytest.raises(ValueError, match="input CRS"):
        clean_polygons(source, crs=CRS)


def test_regularization_is_bounded_and_preserves_nonrectangular_outline():
    crooked = Polygon([(0, 0), (10, 0.08), (10.05, 5), (4.04, 5.07), (4, 10), (0.08, 10), (0, 0)])
    result, flag = conservative_regularize(crooked, max_displacement=0.3, max_area_change=0.05)
    assert flag == "regularized"
    assert result.is_valid
    assert len(result.exterior.coords) == len(crooked.exterior.coords)
    assert result.area < result.minimum_rotated_rectangle.area * 0.8
    assert shapely.hausdorff_distance(result.boundary, crooked.boundary, densify=0.25) <= 0.3
    assert abs(result.area - crooked.area) / crooked.area <= 0.05
    coords = np.asarray(result.exterior.coords)
    edges = np.diff(coords, axis=0)
    for a, b in zip(edges, np.roll(edges, 1, axis=0)):
        assert abs(float(a @ b)) < 1e-7


def test_regularization_rejects_bounds_diagonals_holes_multipart():
    crooked = Polygon([(0, 0), (10, 0), (10.3, 10), (0, 10), (0, 0)])
    result, flag = conservative_regularize(crooked, max_displacement=0.001)
    assert flag == "regularize_rejected_bounds"
    assert result.equals_exact(crooked, 0)
    _, flag = conservative_regularize(crooked, max_displacement=2, max_area_change=0)
    assert flag == "regularize_rejected_bounds"
    trapezoid = Polygon([(0, 0), (10, 0), (5, 8), (0, 8), (0, 0)])
    assert conservative_regularize(trapezoid)[1] == "regularize_skipped_angles"
    hole = Polygon(box(0, 0, 10, 10).exterior.coords, [box(2, 2, 4, 4).exterior.coords])
    assert conservative_regularize(hole)[1] == "regularize_skipped_complex"
    assert conservative_regularize(MultiPolygon([box(0, 0, 2, 2), box(4, 0, 6, 2)]))[1] == "regularize_skipped_complex"


def test_simplification_cannot_bypass_regularization_bound():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (6, 10), (6, 8), (4, 8), (4, 10), (0, 10)])
    result = clean_polygons(frame([polygon]), crs=CRS, simplify=3, regularize=True, max_displacement=0.1)
    assert "combined_change_rejected" in result.cleaned.cleanup_flags.iloc[0]
    assert result.cleaned.geometry.iloc[0].equals(polygon)


@pytest.mark.parametrize("kwargs", [{"simplify": -1}, {"min_area": math.nan}, {"iou_threshold": 0},
                                   {"max_displacement": math.inf}, {"containment_threshold": 1.01}])
def test_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        clean_polygons(frame([box(0, 0, 2, 2)]), crs=CRS, **kwargs)


def test_empty_input_and_empty_audit_are_written(tmp_path):
    result = clean_polygons(frame([]), crs=CRS)
    path = tmp_path / "empty.gpkg"
    write_layers(path, {"cleaned": result.cleaned, "removed_audit": result.removed})
    assert set(gpd.list_layers(path).name) == {"cleaned", "removed_audit"}
    assert gpd.read_file(path, layer="cleaned").empty


def test_cli_directory_provenance_and_preserves_input(tmp_path):
    inputs = tmp_path / "raw"
    inputs.mkdir()
    for name, score in [("a", 0.4), ("b", 0.9)]:
        frame([box(0, 0, 10, 10)], score=[score], tile_id=[name]).to_file(inputs / f"{name}.geojson", driver="GeoJSON")
    before = {path.name: path.read_bytes() for path in inputs.iterdir()}
    output = tmp_path / "cleaned.gpkg"
    main(["--input", str(inputs), "--output", str(output), "--metric-crs", CRS])
    cleaned = gpd.read_file(output, layer="cleaned")
    removed = gpd.read_file(output, layer="removed_audit")
    assert cleaned.tile_id.tolist() == ["b"]
    assert cleaned.input_file.iloc[0].endswith("b.geojson")
    assert removed.tile_id.tolist() == ["a"]
    assert {path.name: path.read_bytes() for path in inputs.iterdir()} == before


def test_cli_refuses_overwriting_raw_even_with_overwrite(tmp_path):
    path = tmp_path / "raw.gpkg"
    frame([box(0, 0, 10, 10)]).to_file(path, driver="GPKG")
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        main(["--input", str(path), "--output", str(path), "--metric-crs", CRS, "--overwrite"])
    assert path.read_bytes() == before


def test_directory_ignores_receipts_and_interrupted_gpkg_writes(tmp_path):
    complete = tmp_path / "tile-1.gpkg"
    frame([box(0, 0, 10, 10)], score=[0.9]).to_file(complete, layer="predictions", driver="GPKG")
    (tmp_path / "tile-1.json").write_text('{"status":"complete","instances":1}', encoding="utf-8")
    (tmp_path / "tile-2.part.gpkg").write_bytes(b"incomplete GDAL output")
    (tmp_path / "tile-3.PART.GPKG").write_bytes(b"incomplete GDAL output")
    (tmp_path / "misleading.geojson").mkdir()
    loaded, files = read_polygon_inputs(tmp_path)
    assert files == [complete]
    assert len(loaded) == 1
    assert loaded.score.iloc[0] == 0.9


def test_explicit_json_geojson_is_still_accepted(tmp_path):
    path = tmp_path / "predictions.json"
    frame([box(0, 0, 10, 10)], score=[0.9]).to_file(path, driver="GeoJSON")
    loaded, files = read_polygon_inputs(path)
    assert files == [path]
    assert len(loaded) == 1
