"""Checkpoint export and loading with real PyTorch, not a stand-in.

Skipped where Torch is not installed; CI runs this file in a CPU-only Torch job.
A tiny module with SAM 3's native component names stands in for the detector, so
the round trip checks that exported weights really reach a model, not their shapes
against Meta's architecture.
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nearmap_buildings import inference  # noqa: E402
from nearmap_buildings.common import sha256_file  # noqa: E402
from nearmap_buildings.training import export_checkpoint  # noqa: E402


class TinyDetector(torch.nn.Module):
    """The native trainer's top-level components, each a small real layer."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.vision_backbone = torch.nn.Linear(4, 4)
        self.backbone.language_backbone = torch.nn.Linear(4, 4)
        self.transformer = torch.nn.Linear(4, 4)
        self.geometry_encoder = torch.nn.Linear(4, 4)
        # BatchNorm adds buffers, including an integer counter, as a real model has.
        self.segmentation_head = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        self.dot_prod_scoring = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = self.backbone.vision_backbone(x) + self.backbone.language_backbone(x)
        x = self.segmentation_head(self.geometry_encoder(self.transformer(x)))
        return self.dot_prod_scoring(x)


def trainer_checkpoint(path, model):
    """Save the way Meta's trainer does: model state plus optimizer state and progress."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    loss = model(torch.randn(8, 4)).pow(2).mean()
    loss.backward()
    optimizer.step()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1, "steps": {"train": 1}}, path)


def meta_inference_state(path):
    """What SAM 3's model builder keeps from a checkpoint: detector keys, prefix stripped."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {key.replace("detector.", ""): value for key, value in checkpoint["model"].items() if "detector" in key}


def test_exported_weights_load_into_the_model_unchanged(tmp_path):
    torch.manual_seed(0)
    trained = TinyDetector()
    source = tmp_path / "checkpoint.pt"
    trainer_checkpoint(source, trained)
    output = tmp_path / "inference.pt"
    result = export_checkpoint(source, output)
    assert result["tensor_count"] == len(trained.state_dict())

    # The same checks nbf infer runs before loading the model.
    inference.check_checkpoint_keys(output, "text")
    metadata = inference.checkpoint_metadata(output, "text", sha256_file(output))
    assert metadata["checkpoint_sha256"] == sha256_file(output)
    with pytest.raises(ValueError, match="tracker/interactive"):
        inference.check_checkpoint_keys(output, "box")

    torch.manual_seed(1)
    fresh = TinyDetector()
    assert not torch.equal(fresh.transformer.weight, trained.transformer.weight)
    fresh.load_state_dict(meta_inference_state(output), strict=True)
    for name, value in trained.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], value), name
    trained.eval()
    fresh.eval()
    sample = torch.randn(3, 4)
    assert torch.equal(fresh(sample), trained(sample))


def test_inference_refuses_a_trainer_checkpoint_that_was_not_exported(tmp_path):
    source = tmp_path / "checkpoint.pt"
    trainer_checkpoint(source, TinyDetector())
    with pytest.raises(ValueError, match="export-checkpoint"):
        inference.check_checkpoint_keys(source, "text")
    # Loaded directly, Meta's filter would keep nothing and leave the model untrained.
    assert meta_inference_state(source) == {}


def test_export_refuses_non_finite_weights(tmp_path):
    model = TinyDetector()
    with torch.no_grad():
        model.transformer.weight[0, 0] = float("nan")
    source = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, source)
    with pytest.raises(ValueError, match="Non-finite"):
        export_checkpoint(source, tmp_path / "inference.pt")
    assert not (tmp_path / "inference.pt").exists()


class Payload:
    """An arbitrary class; the safe loader must refuse to rebuild it."""


def test_export_never_unpickles_arbitrary_objects(tmp_path):
    source = tmp_path / "checkpoint.pt"
    state = TinyDetector().state_dict()
    torch.save({"model": state, "extra": Payload()}, source)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported global"):
        export_checkpoint(source, tmp_path / "inference.pt")
    assert not Path(str(tmp_path / "inference.pt") + ".metadata.json").exists()


def test_metadata_sidecar_matches_the_saved_payload(tmp_path):
    source = tmp_path / "checkpoint.pt"
    trainer_checkpoint(source, TinyDetector())
    output = tmp_path / "inference.pt"
    export_checkpoint(source, output)
    sidecar = json.loads(Path(str(output) + ".metadata.json").read_text(encoding="utf-8"))
    embedded = torch.load(output, map_location="cpu", weights_only=True)["metadata"]
    assert {key: value for key, value in sidecar.items() if key != "checkpoint_sha256"} == embedded
