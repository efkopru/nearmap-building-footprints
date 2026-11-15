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
from nearmap_buildings.training import (FROZEN_BACKBONES, FROZEN_PARTS,  # noqa: E402
                                      build_frozen_backbone_model, export_checkpoint, freeze_modules)


class TinyDetector(torch.nn.Module):
    """The native trainer's top-level components, each a small real layer."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.vision_backbone = torch.nn.Module()
        self.backbone.vision_backbone.trunk = torch.nn.Linear(4, 4)
        # The real image backbone keeps complex position tables next to its weights.
        self.backbone.vision_backbone.trunk.register_buffer("freqs_cis", torch.polar(torch.ones(4), torch.arange(4.0)))
        self.backbone.vision_backbone.convs = torch.nn.Linear(4, 4)
        self.backbone.language_backbone = torch.nn.Linear(4, 4)
        self.transformer = torch.nn.Linear(4, 4)
        self.geometry_encoder = torch.nn.Linear(4, 4)
        self.segmentation_head = torch.nn.Module()
        # BatchNorm adds buffers, including an integer counter, as a real model has.
        self.segmentation_head.pixel_decoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        # Present in the real model and unused by the building recipe's loss.
        self.segmentation_head.semantic_seg_head = torch.nn.Linear(4, 1)
        self.dot_prod_scoring = torch.nn.Linear(4, 1)

    def forward(self, x):
        vision = self.backbone.vision_backbone
        x = vision.convs(vision.trunk(x).float()) + self.backbone.language_backbone(x).float()
        x = self.segmentation_head.pixel_decoder(self.geometry_encoder(self.transformer(x)))
        return self.dot_prod_scoring(x)


class InferenceOnlyTrunk(torch.nn.Module):
    """Refuses gradients the way the image backbone does at the pinned SAM 3 commit."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        return self.drop(self.layer(x))


class TrunkAndHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = InferenceOnlyTrunk()
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(4, 1))

    def forward(self, x):
        return self.head(self.trunk(x))


def test_frozen_part_runs_without_gradients_and_stays_in_evaluation_mode():
    torch.manual_seed(0)
    model = TrunkAndHead()
    sample = torch.randn(8, 4)
    with pytest.raises(ValueError, match="Expected grad to be disabled"):
        model(sample)

    assert freeze_modules(model, ["trunk"]) is model
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()  # what the trainer does at the start of every epoch
    assert model.training and model.head[0].training
    assert not model.trunk.training and not model.trunk.drop.training
    # Dropout in the frozen part is off, so it gives the same features every time.
    assert torch.equal(model.trunk(sample), model.trunk(sample))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(sample).pow(2).mean().backward()
    assert all(parameter.grad is None for parameter in model.trunk.parameters())
    assert all(parameter.grad is not None for parameter in model.head.parameters())
    optimizer.step()
    after = model.state_dict()
    assert all(torch.equal(after[name], before[name]) for name in before if name.startswith("trunk."))
    assert not torch.equal(after["head.1.weight"], before["head.1.weight"])
    # The saved names are unchanged, so the checkpoint still exports and loads as before.
    assert set(after) == set(before)
    with pytest.raises(AttributeError):
        freeze_modules(model, ["no_such_part"])
    with pytest.raises(ValueError, match="Only frozen parts"):
        freeze_modules(model, ["trunk"], half=["head"])


def test_half_precision_keeps_complex_tables_and_whole_numbers():
    """A plain .to(bfloat16) would drop the imaginary part of the position tables."""
    model = TinyDetector()
    trunk = model.backbone.vision_backbone.trunk
    table = trunk.freqs_cis.clone()
    weights = trunk.weight.detach().clone()
    counter = model.segmentation_head.pixel_decoder[1].num_batches_tracked.clone()
    freeze_modules(model, FROZEN_PARTS, half=FROZEN_BACKBONES)
    assert trunk.weight.dtype == trunk.bias.dtype == model.backbone.language_backbone.weight.dtype == torch.bfloat16
    assert trunk.freqs_cis.dtype == torch.complex64 and torch.equal(trunk.freqs_cis, table)
    assert torch.equal(trunk.weight, weights.to(torch.bfloat16))
    # Parts that train, and the small frozen head, stay in full precision.
    assert model.backbone.vision_backbone.convs.weight.dtype == model.transformer.weight.dtype == torch.float32
    assert model.segmentation_head.pixel_decoder[1].num_batches_tracked.dtype == counter.dtype
    assert model.segmentation_head.semantic_seg_head.weight.dtype == torch.float32


def test_training_model_builder_freezes_the_backbones_and_the_unused_head(monkeypatch):
    import sys
    import types

    calls = {}

    def build_sam3_image_model(**kwargs):
        calls.update(kwargs)
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.vision_backbone = torch.nn.Module()
        model.backbone.vision_backbone.trunk = torch.nn.Linear(4, 4)
        model.backbone.vision_backbone.convs = torch.nn.Linear(4, 4)
        model.backbone.language_backbone = torch.nn.Linear(4, 4)
        model.transformer = torch.nn.Linear(4, 4)
        model.segmentation_head = torch.nn.Module()
        model.segmentation_head.semantic_seg_head = torch.nn.Linear(4, 1)
        model.segmentation_head.instance_seg_head = torch.nn.Linear(4, 4)
        return model

    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = build_sam3_image_model
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder)

    model = build_frozen_backbone_model(checkpoint_path="sam3.pt", device="cpu", eval_mode=False)
    assert calls == {"checkpoint_path": "sam3.pt", "device": "cpu", "eval_mode": False}
    trainable = {name.rsplit(".", 1)[0] for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"backbone.vision_backbone.convs", "transformer", "segmentation_head.instance_seg_head"}
    assert FROZEN_BACKBONES == ("backbone.vision_backbone.trunk", "backbone.language_backbone")
    assert FROZEN_PARTS == FROZEN_BACKBONES + ("segmentation_head.semantic_seg_head",)
    # The two backbones are held in half precision; the small frozen head is not.
    assert model.backbone.vision_backbone.trunk.weight.dtype == torch.bfloat16
    assert model.backbone.language_backbone.weight.dtype == torch.bfloat16
    assert model.segmentation_head.semantic_seg_head.weight.dtype == torch.float32
    model.train()
    assert not model.backbone.vision_backbone.trunk.training and not model.backbone.language_backbone.training
    assert model.transformer.training


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


def released_checkpoint(path, model):
    """The released file's layout: detector weights under a prefix, beside tracker weights."""
    state = {f"detector.{key}": value.clone() for key, value in model.state_dict().items()}
    state["tracker.memory.weight"] = torch.ones(2)
    torch.save({"model": state}, path)


def frozen_training_checkpoint(path, model, *, keep_backbones=False):
    """One optimizer step with the backbones frozen in half precision, saved as the recipe saves it."""
    freeze_modules(model, FROZEN_PARTS, half=FROZEN_BACKBONES)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):  # the trainer's mixed precision
        loss = model(torch.randn(8, 4)).float().pow(2).mean()
    loss.backward()
    optimizer.step()
    prefixes = tuple(f"{name}." for name in FROZEN_BACKBONES)
    state = {key: value for key, value in model.state_dict().items() if keep_backbones or not key.startswith(prefixes)}
    torch.save({"model": state, "optimizer": optimizer.state_dict(), "epoch": 1}, path)


def test_frozen_backbones_come_back_exactly_from_the_released_checkpoint(tmp_path):
    torch.manual_seed(0)
    original = TinyDetector()
    base = tmp_path / "sam3.pt"
    released_checkpoint(base, original)
    released = {key: value.clone() for key, value in original.state_dict().items()}

    trained = TinyDetector()
    trained.load_state_dict(released)
    source = tmp_path / "checkpoint.pt"
    frozen_training_checkpoint(source, trained)
    saved = torch.load(source, map_location="cpu", weights_only=True)["model"]
    assert not any(key.startswith("backbone.language_backbone.") for key in saved)

    with pytest.raises(ValueError, match="needs --base-checkpoint"):
        export_checkpoint(source, tmp_path / "no-base.pt")
    assert not (tmp_path / "no-base.pt").exists()

    output = tmp_path / "inference.pt"
    result = export_checkpoint(source, output, base)
    assert result["base_checkpoint_sha256"] == sha256_file(base)
    assert result["frozen_backbones_from_base"] == list(FROZEN_BACKBONES)
    assert result["tensor_count"] == len(released)
    exported = meta_inference_state(output)
    assert set(exported) == set(released)  # nothing of the tracker comes along
    changed = set()
    for key, value in exported.items():
        assert value.dtype == released[key].dtype, key
        if not torch.equal(value, released[key]):
            changed.add(key.rsplit(".", 1)[0])
    # Bit-identical backbones and frozen head; only parts that trained differ.
    assert changed and not any(name.startswith(FROZEN_PARTS) for name in changed)
    assert {"transformer", "backbone.vision_backbone.convs"} <= changed
    fresh = TinyDetector()
    fresh.load_state_dict(exported, strict=True)
    inference.check_checkpoint_keys(output, "text")


def test_half_precision_backbones_never_reach_an_export(tmp_path):
    torch.manual_seed(0)
    original = TinyDetector()
    base = tmp_path / "sam3.pt"
    released_checkpoint(base, original)
    released = {key: value.clone() for key, value in original.state_dict().items()}
    source = tmp_path / "checkpoint.pt"
    frozen_training_checkpoint(source, original, keep_backbones=True)

    with pytest.raises(ValueError, match="Half-precision weights"):
        export_checkpoint(source, tmp_path / "half.pt")
    assert not (tmp_path / "half.pt").exists()
    export_checkpoint(source, tmp_path / "inference.pt", base)
    exported = meta_inference_state(tmp_path / "inference.pt")
    for key in ("backbone.vision_backbone.trunk.weight", "backbone.language_backbone.weight"):
        assert exported[key].dtype == torch.float32 and torch.equal(exported[key], released[key])

    # A released file without those backbones cannot stand in for them.
    torch.save({"model": {"detector.transformer.weight": torch.ones(4, 4)}}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="holds no weights for backbone"):
        export_checkpoint(source, tmp_path / "wrong-base.pt", tmp_path / "wrong.pt")
    with pytest.raises(FileNotFoundError):
        export_checkpoint(source, tmp_path / "missing-base.pt", tmp_path / "absent.pt")


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
