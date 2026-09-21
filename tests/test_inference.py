from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from nearmap_buildings import inference as inf
from nearmap_buildings.preprocessing import tile

class FakeSam:
    """Exercise adapter/file plumbing only, not model performance."""
    def __init__(self, **kwargs):
        self.calls = []
    def set_image(self, path):
        with rasterio.open(path) as src:
            self.height, self.width = src.height, src.width
    def _masks(self):
        masks = np.zeros((1, 1, self.height, self.width), dtype=bool)
        masks[0, 0, 3:9, 4:12] = True
        return masks
    def generate_masks(self, prompt):
        self.masks, self.scores = self._masks(), np.array([.8])
    def generate_masks_by_boxes(self, boxes, box_labels):
        self.calls.append((boxes, box_labels))
        self.generate_masks("visual")
    def predict_inst(self, **kwargs):
        self.calls.append(kwargs)
        return self._masks()[:, 0], np.array([.8]), None

@pytest.fixture
def manifest(tmp_path):
    source = tmp_path / "image.tif"
    with rasterio.open(source, "w", driver="GTiff", width=24, height=24, count=3, dtype="uint8", crs="EPSG:32614", transform=from_origin(500000, 3600000, .25, .25)) as dst:
        dst.write(np.full((3,24,24), 128, dtype=np.uint8))
    tile(source, tmp_path / "prepared", tile_size=24, overlap=0)
    return tmp_path / "prepared" / "manifest.json"

def test_dry_run_does_not_load_model_or_create_outputs(manifest, tmp_path):
    args = inf.parser().parse_args(["--manifest", str(manifest), "--output", str(tmp_path / "out")])
    inf.run(args, model_factory=lambda **kw: pytest.fail("Model loaded during dry run"))
    assert not args.output.exists()

def test_cpu_fake_adapter_export_and_resume_integrity(manifest, tmp_path):
    args = inf.parser().parse_args(["--manifest", str(manifest), "--output", str(tmp_path / "out"), "--execute"])
    result = inf.run(args, model_factory=FakeSam)
    assert result["tiles"][0]["instances"] == 1
    vectors = next((args.output / "raw").glob("*.gpkg"))
    g = gpd.read_file(vectors)
    assert g.crs.to_epsg() == 32614
    assert g.geometry.iloc[0].area == pytest.approx(48 * .25**2)
    args.resume = True
    assert inf.run(args, model_factory=FakeSam)["status"] == "complete"
    args.confidence = .9
    with pytest.raises(ValueError, match="signature changed"):
        inf.run(args, model_factory=FakeSam)

def test_mask_holes_instance_ids_and_nodata():
    mask = np.ones((8,8), dtype=bool)
    mask[3:5,3:5] = False
    rows = inf.vectorize([(mask,.9)], np.ones_like(mask), from_origin(0,8,1,1), "tile", "text", 1, .5)
    assert len(rows) == 1 and rows[0]["geometry"].area == 60
    assert len(rows[0]["geometry"].interiors) == 1
    assert rows[0]["edge_touch"]
    assert inf.vectorize([(mask,.9)], np.zeros_like(mask), from_origin(0,8,1,1), "tile", "text", 1, .5) == []

def test_prompt_modes_and_groups(manifest, tmp_path):
    path = tmp_path / "points.geojson"
    frame = gpd.GeoDataFrame({"object_id":["a","a"],"label":[1,0]}, geometry=[Point(500002,3599998),Point(500003,3599997)], crs=32614)
    frame.to_file(path, driver="GeoJSON")
    loaded = inf.load_prompts(path,"point")
    with rasterio.open(manifest.parent / "tiles/r00000000_c00000000.tif") as src:
        assert len(inf.prompts_for_tile(loaded,src,"point")) == 2
    for method in ["text","box"]:
        with pytest.raises(ValueError):
            inf.load_prompts(path,method)
    with pytest.raises(ValueError,match="thresholded"):
        inf.normalize_predictions(np.full((1,4,4),.8),[.9],4,4)

@pytest.mark.parametrize("method", ["exemplar","box","point"])
def test_guided_modes_call_distinct_interfaces(manifest,tmp_path,method):
    prompts = tmp_path / "prompts.geojson"
    geom = Point(500002,3599998) if method == "point" else box(500001,3599996,500005,3599999)
    gpd.GeoDataFrame({"object_id":["one"],"label":[1]},geometry=[geom],crs=32614).to_file(prompts,driver="GeoJSON")
    args = inf.parser().parse_args(["--manifest",str(manifest),"--method",method,"--prompts",str(prompts),"--output",str(tmp_path/method),"--execute"])
    result = inf.run(args,model_factory=FakeSam)
    assert result["tiles"][0]["instances"] == 1

def test_exported_checkpoint_metadata_restricts_method_and_hash(tmp_path):
    checkpoint = tmp_path / "exported.pt"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = {"schema":"sam3-building-inference-checkpoint-v1", "supported_methods":["text","exemplar"], "checkpoint_sha256":digest}
    Path(str(checkpoint)+".metadata.json").write_text(json.dumps(metadata),encoding="utf-8")
    assert inf.checkpoint_metadata(checkpoint,"text",digest) == metadata
    with pytest.raises(ValueError,match="does not support"):
        inf.checkpoint_metadata(checkpoint,"point",digest)
    with pytest.raises(ValueError,match="metadata hash"):
        inf.checkpoint_metadata(checkpoint,"text","wrong")

def test_checkpoint_preflight_rejects_native_keys_and_missing_interactive_weights(monkeypatch):
    state = {"model":{"backbone.weight":object()}}
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        return state
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(load=load,is_tensor=lambda v:True))
    with pytest.raises(ValueError,match="Export a trainer checkpoint"):
        inf.check_checkpoint_keys("fixture.pt","text")
    state["model"] = {"detector.backbone.weight":object()}
    inf.check_checkpoint_keys("fixture.pt","text")
    with pytest.raises(ValueError,match="tracker/interactive"):
        inf.check_checkpoint_keys("fixture.pt","box")
    state["model"]["tracker.weight"] = object()
    inf.check_checkpoint_keys("fixture.pt","point")
    assert all(call == {"map_location":"cpu","weights_only":True} for call in calls)
