import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from nearmap_buildings.preprocessing import read_areas, tile, tile_main, windows

@pytest.fixture
def raster(tmp_path):
    path = tmp_path / "source.tif"
    data = np.arange(3 * 48 * 64, dtype=np.uint8).reshape(3, 48, 64)
    with rasterio.open(path, "w", driver="GTiff", width=64, height=48, count=3, dtype="uint8", crs="EPSG:32614", transform=from_origin(500000, 3600000, .25, .25)) as dst:
        dst.write(data)
    return path

def test_overlap_preserves_pixels_crs_and_coverage(raster, tmp_path):
    output = tmp_path / "tiles"
    manifest = tile(raster, output, tile_size=32, overlap=8)
    coverage = np.zeros((48, 64), dtype=int)
    with rasterio.open(raster) as src:
        for item in manifest["tiles"]:
            row, col, h, w = [item[k] for k in ["row_off", "col_off", "height", "width"]]
            with rasterio.open(output / item["path"]) as part:
                assert part.crs == src.crs
                assert part.transform @ (0, 0) == src.transform @ (col, row)
                np.testing.assert_array_equal(part.read(), src.read()[:, row:row+h, col:col+w])
            coverage[row:row+h, col:col+w] += 1
    assert coverage.min() >= 1
    assert coverage.max() > 1
    assert manifest["status"] == "complete"
    with pytest.raises(FileExistsError):
        tile(raster, output)

def test_tiles_are_block_tiled_with_predictor_and_keep_the_mask(tmp_path):
    path = tmp_path / "source.tif"
    data = np.random.default_rng(0).integers(1, 256, (3, 300, 300), dtype=np.uint8)
    valid = np.ones((300, 300), dtype=bool)
    valid[:, :40] = False
    with rasterio.open(path, "w", driver="GTiff", width=300, height=300, count=3, dtype="uint8", crs="EPSG:32614", transform=from_origin(1, 400, 1, 1)) as dst:
        dst.write(data)
        dst.write_mask(valid.astype(np.uint8) * 255)
    manifest = tile(path, tmp_path / "tiles", tile_size=300, overlap=0)
    with rasterio.open(tmp_path / "tiles" / manifest["tiles"][0]["path"]) as part:
        assert part.compression.name == "deflate"
        assert part.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR") == "2"
        assert part.block_shapes[0] == (256, 256)
        np.testing.assert_array_equal(part.dataset_mask() > 0, valid)
        np.testing.assert_array_equal(part.read(), np.where(valid, data, 0))

@pytest.mark.parametrize("width", [1100, 1920, 2000, 5000])
def test_edge_windows_stay_full_size_with_at_least_the_requested_overlap(width):
    tiles = list(windows(width, 1024, 1024, 128))
    assert all((w.width, w.height) == (1024, 1024) for w in tiles)
    starts = [int(w.col_off) for w in tiles]
    assert starts == sorted(set(starts))
    assert starts[0] == 0 and starts[-1] == width - 1024
    assert all(1024 - (right - left) >= 128 for left, right in zip(starts, starts[1:]))

def test_raster_smaller_than_a_tile_is_one_window():
    assert [(w.width, w.height) for w in windows(300, 200, 1024, 128)] == [(300, 200)]

def test_reject_invalid_overlap():
    for size, overlap in [(0, 0), (16, 16), (16, -1)]:
        with pytest.raises(ValueError):
            list(windows(64, 48, size, overlap))

def test_nodata_skipped_and_not_inferred_as_dark_roof(tmp_path):
    path = tmp_path / "source.tif"
    with rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=3, dtype="uint8", crs="EPSG:32614", transform=from_origin(1, 20, 1, 1)) as dst:
        dst.write(np.zeros((3,16,16), dtype=np.uint8))
        dst.write_mask(np.zeros((16,16), dtype=np.uint8))
    result = tile(path, tmp_path / "empty", tile_size=16, overlap=0)
    assert result["tiles"] == []
    assert result["skipped_nodata_windows"] == 1


def area_box(col, row, width, height):
    """Bounds of a pixel rectangle of the 64 x 48 test raster, in its CRS."""
    return (500000 + col * .25, 3600000 - (row + height) * .25, 500000 + (col + width) * .25, 3600000 - row * .25)


def test_areas_are_tiled_from_their_own_corner_into_one_manifest(raster, tmp_path):
    areas = [("north", area_box(8, 4, 32, 32)), ("east-2", area_box(40, 20, 24, 16))]
    manifest = tile(raster, tmp_path / "tiles", tile_size=16, overlap=8, areas=areas)
    north = [item for item in manifest["tiles"] if item["area_id"] == "north"]
    # A 32 px square gives exactly nine 16 px tiles at an 8 px step, all inside it.
    assert len(north) == 9
    assert sorted({item["col_off"] for item in north}) == [8, 16, 24]
    assert sorted({item["row_off"] for item in north}) == [4, 12, 20]
    assert all(item["id"].startswith("north_r") and item["width"] == item["height"] == 16 for item in north)
    east = [item for item in manifest["tiles"] if item["area_id"] == "east-2"]
    assert [(item["col_off"], item["row_off"]) for item in east] == [(40, 20), (48, 20)]
    assert manifest["areas"] == {"ids": ["north", "east-2"], "margin_px": 0}
    assert len({item["id"] for item in manifest["tiles"]}) == len(manifest["tiles"])
    with rasterio.open(raster) as src:
        for item in manifest["tiles"]:
            row, col = item["row_off"], item["col_off"]
            with rasterio.open(tmp_path / "tiles" / item["path"]) as part:
                assert part.transform @ (0, 0) == src.transform @ (col, row)
                np.testing.assert_array_equal(part.read(), src.read()[:, row:row + 16, col:col + 16])


def test_area_margin_widens_the_window_and_stops_at_the_raster_edge(raster, tmp_path):
    manifest = tile(raster, tmp_path / "margin", tile_size=16, overlap=8, areas=[("north", area_box(8, 4, 32, 32))], area_margin=4)
    assert sorted({item["col_off"] for item in manifest["tiles"]}) == [4, 12, 20, 28]
    # Rows would start 4 px above the area, but the raster starts at row 0.
    assert sorted({item["row_off"] for item in manifest["tiles"]}) == [0, 8, 16, 24]
    assert manifest["areas"]["margin_px"] == 4
    with pytest.raises(ValueError, match="outside the imagery"):
        tile(raster, tmp_path / "outside", tile_size=16, overlap=8, areas=[("far", area_box(200, 4, 16, 16))])
    assert not (tmp_path / "outside").exists()
    with pytest.raises(ValueError, match="margin"):
        tile(raster, tmp_path / "negative", tile_size=16, overlap=8, areas=[("north", area_box(8, 4, 32, 32))], area_margin=-1)


def test_areas_file_is_reprojected_and_its_ids_are_checked(raster, tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame({"area_id": ["a", "b"]}, geometry=[box(*area_box(8, 4, 32, 32)), box(*area_box(40, 20, 24, 16))],
                             crs="EPSG:32614")
    frame.to_crs(4326).to_file(tmp_path / "areas.gpkg", layer="squares", driver="GPKG")
    areas = read_areas(tmp_path / "areas.gpkg", "EPSG:32614", layer="squares")
    assert [name for name, _ in areas] == ["a", "b"]
    np.testing.assert_allclose(areas[0][1], area_box(8, 4, 32, 32), atol=0.05)
    for ids, message in ([["a", "a"], "unique"], [["a", "b c"], "letters, digits"]):
        frame.assign(area_id=ids).to_file(tmp_path / "bad.gpkg", driver="GPKG")
        with pytest.raises(ValueError, match=message):
            read_areas(tmp_path / "bad.gpkg", "EPSG:32614")
    with pytest.raises(ValueError, match="no 'name' field"):
        read_areas(tmp_path / "areas.gpkg", "EPSG:32614", id_field="name")


def test_tile_command_takes_an_areas_layer(raster, tmp_path, capsys):
    import json

    import geopandas as gpd
    from shapely.geometry import box

    gpd.GeoDataFrame({"square": ["s1"]}, geometry=[box(*area_box(8, 4, 32, 32))], crs="EPSG:32614").to_file(
        tmp_path / "areas.gpkg", driver="GPKG")
    tile_main([str(raster), "--output", str(tmp_path / "tiles"), "--tile-size", "16", "--overlap", "8",
               "--areas", str(tmp_path / "areas.gpkg"), "--area-id-field", "square", "--area-margin-px", "4"])
    assert "Prepared 16 tiles" in capsys.readouterr().out
    manifest = json.loads((tmp_path / "tiles/manifest.json").read_text())
    assert manifest["areas"] == {"ids": ["s1"], "margin_px": 4}
    assert {item["area_id"] for item in manifest["tiles"]} == {"s1"}
