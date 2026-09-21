import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from nearmap_buildings.preprocessing import tile, windows

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
                assert part.transform * (0, 0) == src.transform * (col, row)
                np.testing.assert_array_equal(part.read(), src.read()[:, row:row+h, col:col+w])
            coverage[row:row+h, col:col+w] += 1
    assert coverage.min() >= 1
    assert coverage.max() > 1
    assert manifest["status"] == "complete"
    with pytest.raises(FileExistsError):
        tile(raster, output)

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
