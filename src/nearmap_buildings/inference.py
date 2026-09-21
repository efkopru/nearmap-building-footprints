"""SAM 3 tile inference. Without --execute this validates the plan and downloads nothing."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .common import provenance, read_json, sha256_file, write_json

METHODS = ("text", "exemplar", "box", "point")

def checkpoint_metadata(path, method, checkpoint_hash):
    sidecar = Path(str(path) + ".metadata.json")
    if not sidecar.exists():
        return None
    metadata = read_json(sidecar)
    if metadata.get("schema") != "sam3-building-inference-checkpoint-v1":
        raise ValueError("Unrecognized checkpoint metadata schema.")
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint does not match its export metadata hash.")
    if method not in metadata.get("supported_methods", []):
        raise ValueError(f"This exported checkpoint does not support {method}; use text or exemplar.")
    return metadata

def check_checkpoint_keys(path, method):
    # Meta loads non-strictly after filtering prefixes. Reject incompatible native
    # trainer checkpoints before that can silently discard the fine-tuned weights.
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not any(str(k).startswith("detector.") for k in state):
        raise ValueError("Expected Meta detector-prefixed weights. Export a trainer checkpoint with 'nbf train export-checkpoint' first.")
    if method in ("box", "point") and not any(str(k).startswith("tracker.") for k in state):
        raise ValueError("Instance box/point modes require the official checkpoint with tracker/interactive weights.")
    if not all(torch.is_tensor(v) for k, v in state.items() if str(k).startswith(("detector.", "tracker."))):
        raise ValueError("Checkpoint weights must be tensors.")

def load_prompts(path, method):
    if method == "text":
        if path:
            raise ValueError("Text-only mode does not consume a prompt vector. Choose exemplar, box or point.")
        return None
    if not path:
        raise ValueError(f"--prompts is required for {method} mode.")
    frame = gpd.read_file(path)
    if frame.crs is None or frame.empty:
        raise ValueError("Prompts require a CRS and at least one feature.")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ValueError("Prompt geometries must be nonempty and valid.")
    allowed = ["Point"] if method == "point" else ["Polygon", "MultiPolygon"]
    if not frame.geom_type.isin(allowed).all():
        raise ValueError(f"{method} prompts require {allowed} geometry.")
    if "label" not in frame:
        frame["label"] = 1
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Prompt label must be 1 (include) or 0 (exclude).")
    if method == "box" and (frame.label != 1).any():
        raise ValueError("Instance boxes must be positive; negative concept boxes belong to exemplar mode.")
    if method == "point":
        if "object_id" not in frame or frame.object_id.isna().any():
            raise ValueError("Point prompts need object_id; points with the same ID describe one object.")
        for identifier, group in frame.groupby("object_id"):
            if not (group.label == 1).any():
                raise ValueError(f"Point group {identifier} has no positive point.")
    return frame

def prompts_for_tile(frame, src, method):
    if frame is None:
        return None
    local = frame.to_crs(src.crs)
    extent = box(*src.bounds)
    # Complete boxes/groups only. Clipping a prompt silently changes its meaning.
    if method == "point":
        selected = []
        for _, group in local.groupby("object_id", sort=False):
            if all(extent.covers(g) for g in group.geometry):
                selected.extend(group.index)
        return local.loc[selected]
    return local.loc[[extent.covers(g) for g in local.geometry]]

def pixel_box(geom, transform):
    x0, y0, x1, y1 = geom.bounds
    corners = [(~transform) * p for p in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]]
    xs, ys = zip(*corners)
    return [min(xs), min(ys), max(xs), max(ys)]

def numpy_value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def normalize_predictions(masks, scores, height, width):
    a, s = numpy_value(masks), numpy_value(scores).reshape(-1)
    if a.size == 0:
        return []
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3 or a.shape[1:] != (height, width) or len(s) != len(a):
        raise ValueError(f"Unexpected SAM mask/score shapes: {a.shape}, {s.shape}; expected Nx{height}x{width}.")
    if not np.isfinite(s).all():
        raise ValueError("Non-finite model scores.")
    # SamGeo's methods return thresholded binary masks. Do not silently accept logits.
    if a.dtype != bool and not np.isin(a, [0, 1]).all():
        raise ValueError("Expected thresholded binary masks, not logits/probabilities.")
    return [(mask.astype(bool), float(score)) for mask, score in zip(a, s)]

def predict_tile(model, image_path, method, prompts):
    with rasterio.open(image_path) as src:
        model.set_image(str(image_path))
        if method == "text":
            raise ValueError("Text mode is handled with its explicit text prompt.")
        if method == "exemplar":
            boxes = [pixel_box(g, src.transform) for g in prompts.geometry]
            model.generate_masks_by_boxes(boxes=boxes, box_labels=[bool(v) for v in prompts.label])
            return normalize_predictions(model.masks, model.scores, src.height, src.width)
        result = []
        if method == "box":
            groups = [(pixel_box(g, src.transform), None, None) for g in prompts.geometry]
        else:
            groups = []
            for _, group in prompts.groupby("object_id", sort=False):
                coords = [(~src.transform) * (g.x, g.y) for g in group.geometry]
                groups.append((None, coords, group.label.tolist()))
        for bounds, coords, labels in groups:
            masks, scores, _ = model.predict_inst(box=bounds, point_coords=coords, point_labels=labels, multimask_output=True, return_logits=False)
            choices = normalize_predictions(masks, scores, src.height, src.width)
            if choices:
                result.append(max(choices, key=lambda pair: pair[1]))
        return result

def vectorize(predictions, valid, transform, tile_id, method, min_pixels, threshold):
    rows = []
    for n, (mask, score) in enumerate(predictions, 1):
        if score < threshold:
            continue
        mask = mask & valid
        if int(mask.sum()) < min_pixels or not mask.any():
            continue
        polygons = [shape(g) for g, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform) if value == 1]
        geometry = unary_union(polygons)
        edge_touch = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        rows.append({"instance_id": f"{tile_id}_{n:06d}", "tile_id": tile_id, "score": score, "edge_touch": edge_touch, "method": f"sam3_{method}", "pixels": int(mask.sum()), "geometry": geometry})
    return rows

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--method", choices=METHODS, default="text")
    p.add_argument("--text", default="building")
    p.add_argument("--prompts", type=Path, help="Single-layer GeoJSON/GPKG; see docs/PROMPTS.md")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--bpe-path", type=Path)
    p.add_argument("--allow-model-download", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--min-pixels", type=int, default=16)
    p.add_argument("--limit", type=int, help="Process only the first N manifest tiles for a pilot.")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p

def run(args, model_factory=None):
    if not 0 <= args.confidence <= 1 or args.min_pixels < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Invalid confidence, min-pixels or limit.")
    if args.method != "text" and args.text != "building":
        raise ValueError("--text is only used in text mode; exemplar mode uses visual concepts alone.")
    manifest = read_json(args.manifest)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Use a completed schema_version=1 tile manifest.")
    tiles = manifest["tiles"][:args.limit] if args.limit else manifest["tiles"]
    if not tiles:
        raise ValueError("Manifest has no valid tiles.")
    prompts = load_prompts(args.prompts, args.method)
    root = args.manifest.resolve().parent
    for item in tiles:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]):
            raise ValueError("Tile IDs must contain only letters, digits, underscore or hyphen.")
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe tile path: {item['path']}")
    for file in [args.checkpoint, args.bpe_path]:
        if file and not file.is_file():
            raise ValueError(f"File does not exist: {file}")
    plan = {"method": args.method, "text": args.text if args.method == "text" else None, "tile_count": len(tiles), "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest), "prompts": str(args.prompts.resolve()) if args.prompts else None, "prompts_sha256": sha256_file(args.prompts) if args.prompts else None, "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else "facebook/sam3 (Hugging Face)", "confidence": args.confidence, "min_pixels": args.min_pixels, "device": args.device, "limit": args.limit}
    if not args.execute:
        print(json.dumps({"dry_run": True, **plan}, indent=2))
        return plan
    if not args.checkpoint and not args.allow_model_download and model_factory is None:
        raise ValueError("Supply --checkpoint or explicitly --allow-model-download after obtaining model access.")
    if args.resume and not args.checkpoint and model_factory is None:
        raise ValueError("Resume requires a local --checkpoint so its content hash can be verified.")
    plan["checkpoint_sha256"] = sha256_file(args.checkpoint) if args.checkpoint else None
    plan["checkpoint_metadata"] = checkpoint_metadata(args.checkpoint, args.method, plan["checkpoint_sha256"]) if args.checkpoint else None
    plan["bpe_sha256"] = sha256_file(args.bpe_path) if args.bpe_path else None
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    if args.resume:
        previous = read_json(output / "run.json")
        if previous["signature"] != signature:
            raise ValueError("Resume configuration/input signature changed. Use a new output folder.")
    else:
        output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    run_info = {"signature": signature, "plan": plan, "provenance": provenance(), "status": "running", "tiles": []}
    write_json(output / "run.json", run_info)
    start = time.perf_counter()
    try:
        if model_factory is None:
            if args.checkpoint:
                check_checkpoint_keys(args.checkpoint, args.method)
            from samgeo import SamGeo3
            model_factory = SamGeo3
        model = model_factory(backend="meta", model_id="facebook/sam3", device=args.device, checkpoint_path=str(args.checkpoint) if args.checkpoint else None, bpe_path=str(args.bpe_path) if args.bpe_path else None, load_from_HF=args.checkpoint is None, confidence_threshold=args.confidence, enable_inst_interactivity=args.method in ("box", "point"))
        for item in tiles:
            path = root / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"Tile modified since preparation: {path}")
            receipt_path = raw / f"{item['id']}.json"
            vector_path = raw / f"{item['id']}.gpkg"
            if args.resume and receipt_path.exists():
                receipt = read_json(receipt_path)
                if receipt.get("signature") != signature or not vector_path.exists() or sha256_file(vector_path) != receipt.get("vector_sha256"):
                    raise ValueError(f"Incomplete or changed tile output: {item['id']}")
                run_info["tiles"].append(receipt)
                continue
            t0 = time.perf_counter()
            with rasterio.open(path) as src:
                local_prompts = prompts_for_tile(prompts, src, args.method)
                skipped = local_prompts is not None and (local_prompts.empty or not (local_prompts.label == 1).any())
                if skipped:
                    predictions = []
                elif args.method == "text":
                    model.set_image(str(path))
                    model.generate_masks(prompt=args.text)
                    predictions = normalize_predictions(model.masks, model.scores, src.height, src.width)
                else:
                    predictions = predict_tile(model, path, args.method, local_prompts)
                rows = vectorize(predictions, src.dataset_mask() > 0, src.transform, item["id"], args.method, args.min_pixels, args.confidence)
                columns = ["instance_id", "tile_id", "score", "edge_touch", "method", "pixels", "geometry"]
                frame = gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs=src.crs)
            temporary = raw / f"{item['id']}.part.gpkg"
            frame.to_file(temporary, layer="predictions", driver="GPKG")
            temporary.replace(vector_path)
            receipt = {"tile_id": item["id"], "signature": signature, "status": "skipped_no_complete_positive_prompt" if skipped else "complete", "instances": len(rows), "seconds": time.perf_counter() - t0, "vector_sha256": sha256_file(vector_path)}
            write_json(receipt_path, receipt)
            run_info["tiles"].append(receipt)
            write_json(output / "run.json", run_info)
            print(f"{item['id']}: {receipt['status']}, {len(rows)} instances")
        run_info["status"] = "complete"
    except Exception as error:
        run_info["status"] = "failed"
        run_info["error_type"] = type(error).__name__
        raise
    finally:
        run_info["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "run.json", run_info)
    return run_info

def main(argv=None):
    return run(parser().parse_args(argv))

if __name__ == "__main__":
    main()
