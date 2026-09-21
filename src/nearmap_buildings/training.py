"""Prepare spatially separated instance labels and launch an external SAM 3 checkout."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SPLITS = ("train", "val", "test")
SAM3_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
CHECKPOINT_SCHEMA = "sam3-building-inference-checkpoint-v1"
NATIVE_COMPONENTS = frozenset({"backbone", "transformer", "geometry_encoder",
                               "segmentation_head", "dot_prod_scoring"})


def encode_rle(mask):
    """COCO uncompressed binary RLE: integer runs, column-major, background first."""
    import numpy as np

    values = np.asarray(mask)
    if values.ndim != 2 or not np.isin(values, [0, 1]).all():
        raise ValueError("RLE input must be a 2D binary instance mask")
    flat = values.astype(np.uint8).ravel(order="F")
    if not flat.size:
        raise ValueError("RLE input must not be empty")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.r_[0, changes, flat.size]).astype(int).tolist()
    if flat[0]:
        runs.insert(0, 0)
    return {"size": list(values.shape), "counts": runs}


def _read_polygons(path, layer=None):
    import geopandas as gpd

    frame = gpd.read_file(path, **({"layer": layer} if layer else {}))
    if frame.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    if (frame.empty or frame.geometry.isna().any() or frame.geometry.is_empty.any()
            or not frame.geometry.is_valid.all()
            or not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError(f"Expected valid, nonempty Polygon/MultiPolygon features: {path}")
    return frame.reset_index(drop=True)


def validate_splits(frame, field="split", min_distance_m=0):
    """Keep the user's AOIs; never make a random split or move a split boundary."""
    import math
    from pyproj import CRS
    from shapely.ops import unary_union

    if field not in frame or set(frame[field]) != set(SPLITS):
        raise ValueError(f"AOI field {field!r} must contain exactly train, val, test")
    if not math.isfinite(min_distance_m) or min_distance_m < 0:
        raise ValueError("min_distance_m must be finite and nonnegative")
    if frame.crs is None:
        raise ValueError("Split AOIs need a CRS")
    if (frame.geometry.isna().any() or frame.geometry.is_empty.any()
            or not frame.geometry.is_valid.all()
            or not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Split AOIs must contain valid, nonempty polygons")
    crs = CRS.from_user_input(frame.crs)
    if not crs.is_projected:
        raise ValueError("Split AOIs must use a projected CRS for spatial validation")
    factor = crs.axis_info[0].unit_conversion_factor
    areas = {name: unary_union(frame.loc[frame[field] == name, "geometry"]) for name in SPLITS}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if areas[left].intersection(areas[right]).area > 0:
                raise ValueError(f"Split AOIs overlap: {left} and {right}")
            if areas[left].distance(areas[right]) * factor < min_distance_m:
                raise ValueError(f"Split AOIs are closer than {min_distance_m} m: {left}, {right}")
    return areas


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_coco(manifest_path, ground_truth, split_aois, output, *, split_field="split",
                 ground_truth_layer=None, split_layer=None, bands=(1, 2, 3),
                 min_split_distance_m=0, skip_unassigned=False, labels_complete=False):
    """Write COCO RLE, uint32 instance rasters and RGB PNGs after explicit spatial checks.

    One polygon feature is one building instance; MultiPolygon components and holes
    remain one instance. All visible buildings must be labeled, including in tiles
    that will become negative examples.
    """
    import geopandas as gpd
    import numpy as np
    from pyproj import CRS
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import Polygon

    if not labels_complete:
        raise ValueError("Confirm exhaustive building labels with --labels-complete; missing labels become negatives")
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if len(bands) != 3 or len(set(bands)) != 3 or any(b < 1 for b in bands):
        raise ValueError("bands must be three distinct positive band numbers in RGB order")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiles = manifest.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("Manifest must contain a nonempty tiles array")
    truth = _read_polygons(ground_truth, ground_truth_layer)
    splits = _read_polygons(split_aois, split_layer)
    areas = validate_splits(splits, split_field, min_split_distance_m)
    truth_in_split_crs = truth.to_crs(splits.crs)
    # A building crossing AOIs could appear with different clipped masks in two splits.
    for index, geom in enumerate(truth_in_split_crs.geometry):
        hits = [name for name in SPLITS if geom.intersection(areas[name]).area > 0]
        if len(hits) > 1:
            raise ValueError(f"Ground-truth building {index + 1} crosses split AOIs: {hits}")

    jobs, skipped, seen_ids, seen_paths = [], [], set(), set()
    split_counts = dict.fromkeys(SPLITS, 0)
    required = {"id", "path", "row_off", "col_off", "width", "height", "crs", "bounds"}
    for tile in tiles:
        if not isinstance(tile, dict) or not required.issubset(tile):
            raise ValueError(f"Each tile requires {sorted(required)}")
        tile_id = str(tile["id"])
        if not tile_id or tile_id in seen_ids:
            raise ValueError(f"Empty or duplicate tile ID: {tile_id}")
        seen_ids.add(tile_id)
        relative = Path(tile["path"])
        if relative.is_absolute():
            raise ValueError("Tile paths must be relative to the manifest directory")
        path = (manifest_path.parent / relative).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError("Tile path escapes the manifest directory")
        if path in seen_paths:
            raise ValueError(f"Duplicate tile file: {relative}")
        seen_paths.add(path)
        for key in ("row_off", "col_off", "width", "height"):
            if isinstance(tile[key], bool) or not isinstance(tile[key], int) or tile[key] < (1 if key in ("width", "height") else 0):
                raise ValueError(f"Tile {tile_id}: invalid integer {key}")
        with rasterio.open(path) as source:
            if source.crs is None or CRS.from_user_input(tile["crs"]) != CRS.from_user_input(source.crs):
                raise ValueError(f"Tile {tile_id}: CRS missing or disagrees with raster")
            if (source.width, source.height) != (tile["width"], tile["height"]):
                raise ValueError(f"Tile {tile_id}: dimensions disagree with raster")
            recorded_bounds = np.asarray(tile["bounds"], dtype=float)
            if (recorded_bounds.shape != (4,) or not np.isfinite(recorded_bounds).all()
                    or not np.allclose(recorded_bounds, tuple(source.bounds), rtol=0, atol=1e-6)):
                raise ValueError(f"Tile {tile_id}: bounds disagree with raster")
            transform = source.transform
            if not np.isfinite(tuple(transform)).all() or transform.determinant == 0:
                raise ValueError(f"Tile {tile_id}: invalid affine transform")
            if max(bands) > source.count or any(source.dtypes[b - 1] != "uint8" for b in bands):
                raise ValueError(f"Tile {tile_id}: choose three existing uint8 RGB bands; no implicit stretch")
            if not (source.dataset_mask() > 0).all():
                raise ValueError(f"Tile {tile_id}: nodata pixels present; select fully valid imagery for exhaustive training labels")
            footprint = Polygon([transform * point for point in
                                 [(0, 0), (source.width, 0), (source.width, source.height), (0, source.height)]])
            split_footprint = gpd.GeoSeries([footprint], crs=source.crs).to_crs(splits.crs).iloc[0]
            owners = [name for name, area in areas.items() if area.covers(split_footprint)]
            if len(owners) != 1:
                if skip_unassigned:
                    skipped.append({"tile_id": tile_id, "reason": "not wholly within exactly one split AOI"})
                    continue
                raise ValueError(f"Tile {tile_id} is not wholly within exactly one split AOI; trim the manifest or explicitly --skip-unassigned")
            split_name = owners[0]
            jobs.append((tile, path, split_name, footprint, source.crs, transform))
            split_counts[split_name] += 1
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"Each split must contain at least one complete tile: {split_counts}")

    # Validate and prepare annotations before creating output, so invalid labels do not
    # leave a plausible-looking partial dataset. RGB imagery is read only when writing.
    datasets = {name: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "building"}]}
                for name in SPLITS}
    mask_jobs, small_instances, ann_id = [], [], 1
    projected_truth = {}
    for image_id, (tile, path, split_name, footprint, crs, transform) in enumerate(jobs, 1):
        crs_key = crs.to_wkt()
        if crs_key not in projected_truth:
            projected_truth[crs_key] = truth.to_crs(crs)
        layer = projected_truth[crs_key]
        candidates = sorted(layer.sindex.query(footprint, predicate="intersects").tolist())
        h, w = tile["height"], tile["width"]
        instances = np.zeros((h, w), dtype=np.uint32)
        local_id = 0
        for source_index in candidates:
            geom = layer.geometry.iloc[source_index].intersection(footprint)
            if geom.is_empty or geom.area == 0:
                continue
            mask = rasterize([(geom, 1)], out_shape=(h, w), transform=transform,
                             all_touched=False, dtype="uint8")
            rows, cols = np.nonzero(mask)
            if not len(rows):
                small_instances.append({"tile_id": str(tile["id"]), "source_id": source_index + 1})
                continue
            if np.any(instances[mask != 0]):
                raise ValueError(f"Tile {tile['id']}: ground-truth instances overlap in raster pixels")
            local_id += 1
            instances[mask != 0] = local_id
            datasets[split_name]["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "bbox": [int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1),
                         int(rows.max() - rows.min() + 1)],
                "area": int(len(rows)), "segmentation": encode_rle(mask),
                "source_instance_id": source_index + 1, "instance_value": local_id,
            })
            ann_id += 1
        name = f"tile_{image_id:06d}"
        datasets[split_name]["images"].append({
            "id": image_id, "file_name": f"{name}.png", "width": w, "height": h,
            "tile_id": str(tile["id"]), "source_tile": tile["path"],
        })
        mask_jobs.append((name, path, split_name, instances, crs, transform))
    output.mkdir(parents=True)
    for split_name in SPLITS:
        (output / split_name / "images").mkdir(parents=True)
        (output / split_name / "instances").mkdir()
        (output / split_name / "annotations.json").write_text(
            json.dumps(datasets[split_name], indent=2), encoding="utf-8")
    for name, path, split_name, instances, crs, transform in mask_jobs:
        with rasterio.open(path) as source:
            rgb = source.read(list(bands))
        h, w = instances.shape
        with rasterio.open(output / split_name / "images" / f"{name}.png", "w", driver="PNG",
                           width=w, height=h, count=3, dtype="uint8") as target:
            target.write(rgb)
        with rasterio.open(output / split_name / "instances" / f"{name}.tif", "w", driver="GTiff",
                           width=w, height=h, count=1, dtype="uint32", crs=crs,
                           transform=transform, nodata=0, compress="deflate") as target:
            target.write(instances, 1)
    report = {"schema": "building-coco-spatial-v1", "labels_complete_confirmed": True,
              "manifest_sha256": _sha256(manifest_path), "split_field": split_field,
              "ground_truth_geometry_sha256": hashlib.sha256(b"".join(truth.geometry.to_wkb())).hexdigest(),
              "split_geometry_sha256": hashlib.sha256(b"".join(splits.geometry.to_wkb())
                  + json.dumps(splits[split_field].tolist()).encode()).hexdigest(),
              "min_split_distance_m": min_split_distance_m, "rgb_bands": list(bands),
              "counts": {name: {"images": len(data["images"]), "instances": len(data["annotations"])}
                         for name, data in datasets.items()},
              "skipped_tiles": skipped, "subpixel_instances_omitted": small_instances,
              "segmentation": "COCO uncompressed RLE; one annotation per building; uint32 instance rasters",
              "sam3_loader": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON"}
    (output / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def launch_sam3(checkout, config, *, python=sys.executable, num_gpus=1, execute=False):
    """Use the real upstream entry point and its package-relative Hydra config names."""
    checkout, config = Path(checkout).resolve(), Path(config).resolve()
    entry = checkout / "sam3" / "train" / "train.py"
    config_root = checkout / "sam3" / "train"
    if not entry.is_file() or not config.is_file():
        raise FileNotFoundError("SAM 3 checkout entry point or YAML config does not exist")
    if config.suffix.lower() not in (".yaml", ".yml") or not config.is_relative_to(config_root):
        raise ValueError("Place the reviewed YAML config inside checkout/sam3/train/configs before launching")
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    command = [str(python), str(entry), "-c", config.relative_to(config_root).as_posix(),
               "--use-cluster", "0", "--num-gpus", str(num_gpus), "--num-nodes", "1"]
    plan = {"cwd": str(checkout), "command": command, "execute": execute,
            "config_sha256": _sha256(config)}
    if execute:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(checkout) + os.pathsep + environment.get("PYTHONPATH", "")
        subprocess.run(command, cwd=checkout, env=environment, check=True)
    return plan


def detector_state_from_trainer(checkpoint, *, is_tensor):
    """Validate the native PCS state and prefix keys for Meta's inference loader.

    is_tensor is injected so this structural conversion can be tested without Torch.
    Export always supplies torch.is_tensor. This does not validate tensor shapes
    against an instantiated SAM 3 architecture.
    """
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("Expected a native trainer checkpoint containing a 'model' state dictionary")
    state = checkpoint["model"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("The trainer 'model' state must be a nonempty mapping")
    roots = set()
    for key, tensor in state.items():
        if not isinstance(key, str) or "." not in key or any(not part for part in key.split(".")):
            raise ValueError("Model state keys must be nonempty dotted parameter names")
        root = key.split(".", 1)[0]
        if root in {"detector", "tracker", "module", "model", "_orig_mod", "inst_interactive_predictor"}:
            raise ValueError(f"Already-prefixed, wrapped, or interactive checkpoint is unsupported: {root}")
        if root not in NATIVE_COMPONENTS:
            raise ValueError(f"Unexpected native model component: {root}")
        if not is_tensor(tensor):
            raise ValueError(f"Model state value is not a tensor: {key}")
        roots.add(root)
    if roots != NATIVE_COMPONENTS:
        raise ValueError(f"Missing native PCS model components: {sorted(NATIVE_COMPONENTS - roots)}")
    for prefix in ("backbone.vision_backbone.", "backbone.language_backbone."):
        if not any(key.startswith(prefix) for key in state):
            raise ValueError(f"Missing native backbone component: {prefix}")
    return {f"detector.{key}": tensor for key, tensor in state.items()}


def export_checkpoint(source, output):
    """Export a native trainer checkpoint for text/exemplar inference, locally on CPU."""
    source, output = Path(source).resolve(), Path(output).resolve()
    metadata_path = Path(str(output) + ".metadata.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.suffix.lower() not in (".pt", ".pth"):
        raise ValueError("Export output must end in .pt or .pth")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Checkpoint or metadata output exists; choose a new output name")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Run export-checkpoint in your existing Torch training environment") from exc
    # Never fall back to pickle's unrestricted loader or allowlist arbitrary classes.
    checkpoint = torch.load(source, weights_only=True, map_location="cpu")
    state = detector_state_from_trainer(checkpoint, is_tensor=torch.is_tensor)
    for key, tensor in state.items():
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"Expected a dense, non-quantized CPU tensor: {key}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite model weights: {key}")
    metadata = {"schema": CHECKPOINT_SCHEMA, "format": "sam3-detector-prefixed",
                "model_id": "facebook/sam3", "upstream_commit": SAM3_COMMIT,
                "supported_methods": ["text", "exemplar"], "tensor_count": len(state),
                "source_checkpoint_sha256": _sha256(source),
                "exported_utc": datetime.now(timezone.utc).isoformat(),
                "validation": "native tensor components and finite values; model shapes not instantiated"}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing the training checkpoint.
    with output.open("xb") as target:
        torch.save({"model": state, "metadata": metadata}, target)
    metadata = dict(metadata, checkpoint_sha256=_sha256(output))
    with metadata_path.open("x", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2)
    return {"checkpoint": str(output), "metadata": str(metadata_path), **metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-coco", help="Prepare local imagery and exhaustive polygon labels")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--ground-truth", required=True, type=Path)
    prepare.add_argument("--split-aois", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--ground-truth-layer")
    prepare.add_argument("--split-layer")
    prepare.add_argument("--split-field", default="split")
    prepare.add_argument("--bands", type=int, nargs=3, default=[1, 2, 3])
    prepare.add_argument("--min-split-distance-m", type=float, default=0)
    prepare.add_argument("--skip-unassigned", action="store_true")
    prepare.add_argument("--labels-complete", action="store_true", help="Confirm all visible buildings in retained tiles are annotated")
    launch = commands.add_parser("launch", help="Print official SAM 3 training command; --execute runs it")
    launch.add_argument("--checkout", required=True, type=Path)
    launch.add_argument("--config", required=True, type=Path)
    launch.add_argument("--python", default=sys.executable)
    launch.add_argument("--num-gpus", type=int, default=1)
    launch.add_argument("--execute", action="store_true")
    export = commands.add_parser("export-checkpoint", help="Convert native training weights for text/exemplar inference")
    export.add_argument("--input", required=True, type=Path, help="Native trainer checkpoint.pt")
    export.add_argument("--output", required=True, type=Path, help="New inference .pt and adjacent metadata JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-coco":
            result = prepare_coco(args.manifest, args.ground_truth, args.split_aois, args.output,
                                  split_field=args.split_field, ground_truth_layer=args.ground_truth_layer,
                                  split_layer=args.split_layer, bands=args.bands,
                                  min_split_distance_m=args.min_split_distance_m,
                                  skip_unassigned=args.skip_unassigned, labels_complete=args.labels_complete)
        elif args.command == "launch":
            result = launch_sam3(args.checkout, args.config, python=args.python,
                                 num_gpus=args.num_gpus, execute=args.execute)
        else:
            result = export_checkpoint(args.input, args.output)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
