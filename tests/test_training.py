import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from nearmap_buildings.esri import run_esri
from nearmap_buildings.import_vectors import import_vectors
from nearmap_buildings.training import (detector_state_from_trainer, encode_rle, export_checkpoint,
                                      launch_sam3, prepare_coco, validate_splits)


def decode_rle(rle):
    pixels = []
    for i, count in enumerate(rle["counts"]):
        pixels.extend([i % 2] * count)
    return np.asarray(pixels, dtype=np.uint8).reshape(rle["size"], order="F")


def make_data(tmp_path, size=10):
    tmp_path.mkdir(exist_ok=True)
    tiles = []
    areas = []
    for i, split in enumerate(("train", "val", "test")):
        x = i * size * 3
        path = tmp_path / f"{split}.tif"
        transform = from_origin(x, size, 1, 1)
        with rasterio.open(path, "w", driver="GTiff", width=size, height=size,
                           count=3, dtype="uint8", crs="EPSG:26914", transform=transform) as dst:
            dst.write(np.full((3, size, size), 127, dtype=np.uint8))
        tiles.append({"id": split, "path": path.name, "row_off": 0, "col_off": x,
                      "width": size, "height": size, "crs": "EPSG:26914",
                      "bounds": [x, 0, x + size, size]})
        areas.append(box(x, 0, x + size, size))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tiles": tiles}), encoding="utf-8")
    split_file = tmp_path / "splits.gpkg"
    gpd.GeoDataFrame({"split": ["train", "val", "test"]}, geometry=areas,
                     crs="EPSG:26914").to_file(split_file)
    truth_file = tmp_path / "truth.gpkg"
    donut = Polygon([(1, 1), (5, 1), (5, 5), (1, 5)],
                    holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]])
    gpd.GeoDataFrame(geometry=[donut, box(5, 1, 7, 5), box(size * 3 + 1, 1, size * 3 + 3, 3)],
                     crs="EPSG:26914").to_file(truth_file)
    return manifest, truth_file, split_file


def prepare(inputs, output, **kwargs):
    return prepare_coco(*inputs, output, labels_complete=True, **kwargs)


@pytest.mark.parametrize("mask", [np.ones((2, 3), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8),
                                  np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)])
def test_rle_preserves_column_order_and_integer_runs(mask):
    encoded = encode_rle(mask)
    assert all(isinstance(count, int) for count in encoded["counts"])
    assert sum(encoded["counts"]) == mask.size
    np.testing.assert_array_equal(decode_rle(encoded), mask)


def test_holes_touching_instances_and_negative_test_tile(tmp_path):
    inputs = make_data(tmp_path / "source")
    output = tmp_path / "prepared"
    report = prepare(inputs, output)
    assert report["counts"] == {"train": {"images": 1, "instances": 2},
                                "val": {"images": 1, "instances": 1},
                                "test": {"images": 1, "instances": 0}}
    train = json.loads((output / "train/annotations.json").read_text())
    assert [ann["area"] for ann in train["annotations"]] == [12, 8]
    first = train["annotations"][0]
    mask = decode_rle(first["segmentation"])
    assert mask[6, 2] == 0  # Interior courtyard preserved.
    assert first["bbox"] == [1, 5, 4, 4]
    with rasterio.open(output / "train/instances/tile_000001.tif") as src:
        assert src.dtypes == ("uint32",)
        values = src.read(1)
        assert set(np.unique(values)) == {0, 1, 2}
        assert values[6, 4] == 1 and values[6, 5] == 2  # Touching buildings stay separate.
        assert src.crs.to_epsg() == 26914
    with rasterio.open(output / "train/images/tile_000001.png") as src:
        np.testing.assert_array_equal(src.read(), np.full((3, 10, 10), 127, dtype=np.uint8))


def test_more_than_255_instances_keep_integer_ids(tmp_path):
    inputs = make_data(tmp_path / "source", size=20)
    buildings = [box(x, y, x + 1, y + 1) for y in range(15) for x in range(20)]
    gpd.GeoDataFrame(geometry=buildings, crs="EPSG:26914").to_file(inputs[1])
    prepare(inputs, tmp_path / "out")
    with rasterio.open(tmp_path / "out/train/instances/tile_000001.tif") as src:
        assert src.read(1).max() == 300
        assert len(np.unique(src.read(1))) == 301


def test_overlap_and_distance_rules_are_explicit():
    frame = gpd.GeoDataFrame({"split": ["train", "val", "test"]},
                             geometry=[box(0, 0, 10, 10), box(9, 0, 20, 10), box(30, 0, 40, 10)],
                             crs="EPSG:26914")
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(frame)
    frame.loc[1, "geometry"] = box(11, 0, 20, 10)
    with pytest.raises(ValueError, match="closer"):
        validate_splits(frame, min_distance_m=2)
    assert set(validate_splits(frame, min_distance_m=1)) == {"train", "val", "test"}
    with pytest.raises(ValueError, match="projected"):
        validate_splits(frame.to_crs(4326))


def test_missing_split_and_incomplete_labels_fail(tmp_path):
    inputs = make_data(tmp_path / "source")
    with pytest.raises(ValueError, match="labels-complete"):
        prepare_coco(*inputs, tmp_path / "out")
    split_frame = gpd.read_file(inputs[2])
    split_frame.loc[2, "split"] = "validation"
    split_frame.to_file(inputs[2])
    with pytest.raises(ValueError, match="exactly train, val, test"):
        prepare(inputs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_building_crossing_split_aois_is_rejected(tmp_path):
    inputs = make_data(tmp_path / "source")
    frame = gpd.read_file(inputs[2])
    frame.loc[0, "geometry"] = box(0, 0, 20, 10)
    frame.loc[1, "geometry"] = box(20, 0, 40, 10)
    frame.to_file(inputs[2])
    gpd.GeoDataFrame(geometry=[box(19, 1, 21, 4)], crs="EPSG:26914").to_file(inputs[1])
    with pytest.raises(ValueError, match="crosses split AOIs"):
        prepare(inputs, tmp_path / "out")


def test_boundary_tile_rejected_instead_of_centroid_assignment(tmp_path):
    inputs = make_data(tmp_path / "source")
    frame = gpd.read_file(inputs[2])
    frame.loc[0, "geometry"] = box(0, 0, 9, 10)
    frame.to_file(inputs[2])
    with pytest.raises(ValueError, match="not wholly within"):
        prepare(inputs, tmp_path / "out")
    with pytest.raises(ValueError, match="Each split must contain"):
        prepare(inputs, tmp_path / "out", skip_unassigned=True)


@pytest.mark.parametrize("key,value,message", [("bounds", [0, 0, 11, 10], "bounds disagree"),
                                                ("width", 12, "dimensions disagree"),
                                                ("crs", "EPSG:4326", "CRS missing or disagrees")])
def test_manifest_is_checked_against_actual_raster(tmp_path, key, value, message):
    inputs = make_data(tmp_path / "source")
    manifest = json.loads(inputs[0].read_text())
    manifest["tiles"][0][key] = value
    inputs[0].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        prepare(inputs, tmp_path / "out")


def test_duplicate_tiles_and_overlapping_labels_rejected(tmp_path):
    inputs = make_data(tmp_path / "source")
    manifest = json.loads(inputs[0].read_text())
    manifest["tiles"].append(manifest["tiles"][0].copy())
    inputs[0].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate tile ID"):
        prepare(inputs, tmp_path / "out")
    manifest["tiles"].pop()
    inputs[0].write_text(json.dumps(manifest))
    gpd.GeoDataFrame(geometry=[box(1, 1, 5, 5), box(4, 4, 6, 6)], crs="EPSG:26914").to_file(inputs[1])
    with pytest.raises(ValueError, match="overlap in raster pixels"):
        prepare(inputs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_nodata_imagery_is_not_used_as_exhaustive_supervision(tmp_path):
    inputs = make_data(tmp_path / "source")
    with rasterio.open(inputs[0].parent / "train.tif", "r+") as src:
        valid = np.full((10, 10), 255, dtype=np.uint8)
        valid[0, :] = 0
        src.write_mask(valid)
    with pytest.raises(ValueError, match="nodata pixels present"):
        prepare(inputs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_launcher_uses_real_entry_and_dry_run_never_spawns(tmp_path, monkeypatch):
    root = tmp_path / "sam3"
    directory = root / "sam3/train/configs"
    directory.mkdir(parents=True)
    (directory.parent / "train.py").write_text("raise RuntimeError('must not run')")
    config = directory / "buildings.yaml"
    config.write_text("trainer: {}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    result = launch_sam3(root, config, python="gpu-python")
    assert result["command"][2:4] == ["-c", "configs/buildings.yaml"]
    assert result["execute"] is False
    external = tmp_path / "external.yaml"
    external.write_text("trainer: {}")
    with pytest.raises(ValueError, match="inside checkout"):
        launch_sam3(root, external)


def test_vector_import_reprojects_and_preserves_instances(tmp_path):
    source = tmp_path / "export.gpkg"
    output = tmp_path / "standard.gpkg"
    gpd.GeoDataFrame({"building_id": ["a", "b"], "confidence": [0.8, 0.9]},
                     geometry=[box(-97, 32, -96.999, 32.001), box(-96.998, 32, -96.997, 32.001)],
                     crs=4326).to_file(source)
    import_vectors(source, output, "EPSG:26914", id_field="building_id", score_field="confidence")
    result = gpd.read_file(output)
    assert result.crs.to_epsg() == 26914
    assert result["source_id"].to_list() == ["a", "b"]
    assert result["method"].to_list() == ["nearmap_ai", "nearmap_ai"]
    np.testing.assert_allclose(result["score"], [0.8, 0.9])
    with pytest.raises(FileExistsError):
        import_vectors(source, output, 26914)


def test_esri_dry_run_does_not_require_arcpy(tmp_path):
    image, model = tmp_path / "image.tif", tmp_path / "usa.dlpk"
    image.touch()
    model.touch()
    plan = run_esri(image, model, tmp_path / "footprints.shp")
    assert plan["tool"] == "arcpy.ia.DetectObjectsUsingDeepLearning"
    assert plan["execute"] is False
    assert "return_bboxes False" in plan["arguments"]
    with pytest.raises(ValueError, match="perfect square"):
        run_esri(image, model, tmp_path / "footprints.shp", batch_size=3)


def native_state(factory=lambda: np.array([1.0])):
    return {name: factory() for name in (
        "backbone.vision_backbone.trunk.weight", "backbone.language_backbone.encoder.weight",
        "transformer.encoder.weight", "geometry_encoder.encode.weight",
        "segmentation_head.pixel_decoder.weight", "dot_prod_scoring.projection.weight",
    )}


def test_trainer_state_export_maps_every_tensor_without_mutating_input():
    original = native_state()
    checkpoint = {"model": original, "optimizer": {"state": "must not be exported"}, "epoch": 2}
    converted = detector_state_from_trainer(checkpoint, is_tensor=lambda value: isinstance(value, np.ndarray))
    assert set(converted) == {"detector." + key for key in original}
    assert all(converted["detector." + key] is value for key, value in original.items())
    assert all(not key.startswith("detector.") for key in original)
    # Emulate the pinned model builder's filtering and stripping exactly.
    restored = {key.replace("detector.", ""): value for key, value in converted.items() if "detector" in key}
    assert set(restored) == set(original)


@pytest.mark.parametrize("bad_prefix", ["detector", "tracker", "module", "model", "_orig_mod",
                                        "inst_interactive_predictor", "unknown_architecture"])
def test_checkpoint_export_rejects_prefixed_wrapped_or_unknown_states(bad_prefix):
    state = native_state()
    state[f"{bad_prefix}.weight"] = np.array([1.0])
    with pytest.raises(ValueError, match="unsupported|Unexpected"):
        detector_state_from_trainer({"model": state}, is_tensor=lambda value: isinstance(value, np.ndarray))


@pytest.mark.parametrize("bad", [{}, {"model": {}}, {"model": []}, native_state()])
def test_checkpoint_export_requires_native_trainer_envelope(bad):
    with pytest.raises(ValueError, match="trainer|mapping"):
        detector_state_from_trainer(bad, is_tensor=lambda value: isinstance(value, np.ndarray))


def test_checkpoint_export_rejects_partial_or_nontensor_state():
    state = native_state()
    state.pop("segmentation_head.pixel_decoder.weight")
    with pytest.raises(ValueError, match="Missing native PCS"):
        detector_state_from_trainer({"model": state}, is_tensor=lambda value: isinstance(value, np.ndarray))
    state = native_state()
    state["transformer.encoder.weight"] = "not a tensor"
    with pytest.raises(ValueError, match="not a tensor"):
        detector_state_from_trainer({"model": state}, is_tensor=lambda value: isinstance(value, np.ndarray))
    state = native_state()
    state.pop("backbone.language_backbone.encoder.weight")
    with pytest.raises(ValueError, match="Missing native backbone"):
        detector_state_from_trainer({"model": state}, is_tensor=lambda value: isinstance(value, np.ndarray))


def test_export_file_wrapper_uses_safe_load_and_hash_bound_metadata(tmp_path, monkeypatch):
    class TensorStandIn:
        device = SimpleNamespace(type="cpu")
        layout = "strided"
        is_quantized = False

        def is_floating_point(self):
            return True

        def is_complex(self):
            return False

    calls = {}

    def safe_load(path, **kwargs):
        calls["load"] = kwargs
        return {"model": native_state(TensorStandIn), "optimizer": {"discard": True}}

    def save(payload, handle):
        calls["saved"] = payload
        handle.write(b"mock serialized inference checkpoint")

    fake_torch = SimpleNamespace(load=safe_load, save=save, strided="strided",
                                 is_tensor=lambda value: isinstance(value, TensorStandIn),
                                 isfinite=lambda value: np.array([True]))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    source = tmp_path / "training.pt"
    source.write_bytes(b"mock trainer checkpoint")
    output = tmp_path / "inference.pt"
    result = export_checkpoint(source, output)
    assert calls["load"] == {"weights_only": True, "map_location": "cpu"}
    assert set(calls["saved"]) == {"model", "metadata"}
    metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
    assert metadata["supported_methods"] == ["text", "exemplar"]
    assert metadata["schema"] == "sam3-building-inference-checkpoint-v1"
    assert metadata["format"] == "sam3-detector-prefixed"
    assert metadata["tensor_count"] == 6
    assert len(metadata["checkpoint_sha256"]) == len(metadata["source_checkpoint_sha256"]) == 64
    assert result["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    with pytest.raises(FileExistsError):
        export_checkpoint(source, output)
    assert source.read_bytes() == b"mock trainer checkpoint"
