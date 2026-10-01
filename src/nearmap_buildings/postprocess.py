"""Conservative CPU polygon cleanup. Original input files are never modified.

Deduplication keeps one of each set of overlapping instances: by default a complete
detection (not cut by a tile edge or nodata) before a truncated one, then the higher
score. It never dissolves neighbors. Optional seam merging joins the pieces of a
building longer than a tile, only where two cut pieces agree inside their tiles'
overlap. Orthogonalization is optional and only attempts already near-orthogonal
simple rings. Geometry cleanup cannot turn a roof outline into a ground footprint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.strtree import STRtree

from .common import read_json


DUPLICATE_PRIORITIES = ("complete", "score")
# Raw-mask flags written by inference; either one means the polygon may be cut off.
TRUNCATION_FLAGS = ("edge_touch", "nodata_touch")


def metric_crs(value: str | CRS) -> CRS:
    """Require a projected horizontal CRS whose coordinate units are metres."""
    crs = CRS.from_user_input(value)
    if not crs.is_projected or len(crs.axis_info) < 2:
        raise ValueError("metric CRS must be projected, with metre units")
    if any(not math.isclose(axis.unit_conversion_factor, 1.0, rel_tol=0, abs_tol=1e-12)
           or axis.unit_name.lower() not in {"metre", "meter", "metres", "meters"}
           for axis in crs.axis_info[:2]):
        raise ValueError("metric CRS must use metres, not degrees or feet")
    return crs


def polygon_parts(geometry):
    """Retain polygonal components of make_valid/intersection results."""
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    parts = []
    for part in getattr(geometry, "geoms", []):
        polygon = polygon_parts(part)
        if isinstance(polygon, Polygon):
            parts.append(polygon)
        elif isinstance(polygon, MultiPolygon):
            parts.extend(polygon.geoms)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _orthogonal_ring(coords: np.ndarray, angle: float, angle_tolerance: float):
    """Fit axis-aligned edge lines after rotation; preserve vertex/ring topology."""
    rotation = np.array([[math.cos(angle), math.sin(angle)],
                         [-math.sin(angle), math.cos(angle)]])
    origin = coords[0]
    points = (coords[:-1] - origin) @ rotation.T
    edges = np.roll(points, -1, axis=0) - points
    if len(points) < 4 or np.any(np.linalg.norm(edges, axis=1) < 1e-9):
        return None
    deviations = np.minimum(np.abs(np.arctan2(edges[:, 1], edges[:, 0])) % (math.pi / 2),
                            math.pi / 2 - np.abs(np.arctan2(edges[:, 1], edges[:, 0])) % (math.pi / 2))
    if np.any(deviations > math.radians(angle_tolerance)):
        return None
    horizontal = np.abs(edges[:, 0]) >= np.abs(edges[:, 1])
    # Adjacent collinear edges or acute reversals are left unchanged rather than
    # guessing which vertices to remove or which building shape was intended.
    if np.any(horizontal == np.roll(horizontal, 1)):
        return None
    midpoints = (points + np.roll(points, -1, axis=0)) / 2
    fitted = []
    for index in range(len(points)):
        previous = (index - 1) % len(points)
        x = midpoints[previous if horizontal[index] else index, 0]
        y = midpoints[index if horizontal[index] else previous, 1]
        fitted.append([x, y])
    return np.asarray(fitted) @ rotation + origin


def conservative_regularize(geometry, max_displacement: float = 0.3,
                           max_area_change: float = 0.05,
                           angle_tolerance: float = 10.0):
    """Return (geometry, flag); accept only a bounded, valid orthogonal fit.

    This intentionally leaves multipart geometries, holes, curves, diagonal edges,
    and complex/collinear rings unchanged. Bounds compare against the input to this
    function. Maximum corresponding-vertex displacement bounds every point of
    the corresponding straight edges, in the projected CRS's metres.
    """
    if not all(math.isfinite(value) for value in (max_displacement, max_area_change, angle_tolerance)) or max_displacement < 0 or not 0 <= max_area_change <= 1 or not 0 < angle_tolerance < 45:
        raise ValueError("invalid regularization bounds")
    if not isinstance(geometry, Polygon) or geometry.interiors or not geometry.is_valid or geometry.area <= 0:
        return geometry, "regularize_skipped_complex"
    coords = np.asarray(geometry.exterior.coords)[:, :2]
    edges = np.diff(coords, axis=0)
    longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    angle = math.atan2(longest[1], longest[0])
    fitted = _orthogonal_ring(coords, angle, angle_tolerance)
    if fitted is None:
        return geometry, "regularize_skipped_angles"
    candidate = Polygon(fitted)
    if candidate.is_empty or not candidate.is_valid or candidate.area <= 0:
        return geometry, "regularize_rejected_invalid"
    displacement = float(np.max(np.linalg.norm(coords[:-1] - fitted, axis=1)))
    relative_area_change = abs(candidate.area - geometry.area) / geometry.area
    if displacement > max_displacement or relative_area_change > max_area_change:
        return geometry, "regularize_rejected_bounds"
    return candidate, "regularized" if not candidate.equals_exact(geometry, 1e-9) else "regularize_unchanged"


def _within_change_bounds(original, candidate, *, min_area, max_displacement, max_area_change):
    """Whether candidate stays within the regularization limits of the repaired raw geometry."""
    if candidate.area < min_area or abs(candidate.area - original.area) / original.area > max_area_change:
        return False
    # Polygonal round buffers are conservative subsets of a true distance
    # buffer. Mutual coverage bounds all boundary points, not only samples.
    return original.equals(candidate) or (max_displacement > 0
        and original.boundary.buffer(max_displacement).covers(candidate.boundary)
        and candidate.boundary.buffer(max_displacement).covers(original.boundary))


def _flag(value, column) -> bool:
    """A raw-mask flag as a bool; missing counts as False."""
    if pd.isna(value):
        return False
    if value not in (True, False):
        raise ValueError(f"{column} must be boolean, not {value!r}")
    return bool(value)


def _truncated(data: gpd.GeoDataFrame) -> np.ndarray:
    """True where a raw mask reached a tile edge or nodata; missing flags count as complete."""
    truncated = np.zeros(len(data), dtype=bool)
    for column in TRUNCATION_FLAGS:
        if column not in data:
            continue
        for index, value in enumerate(data[column]):
            truncated[index] |= _flag(value, column)
    return truncated


def manifest_tiles(path: str | Path) -> gpd.GeoDataFrame:
    """Tile extents and pixel sizes from an nbf tile manifest, for seam merging."""
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete" or not manifest.get("tiles"):
        raise ValueError("seam merging needs a completed schema_version=1 tile manifest")
    tiles = manifest["tiles"]
    if len({tile["crs"] for tile in tiles}) != 1:
        raise ValueError("manifest tiles must share one CRS")
    return gpd.GeoDataFrame({"tile_id": [tile["id"] for tile in tiles],
                             "width": [tile["width"] for tile in tiles],
                             "height": [tile["height"] for tile in tiles]},
                            geometry=[box(*tile["bounds"]) for tile in tiles], crs=tiles[0]["crs"])


def _cut_inside(geometry, extent, other, tolerance):
    """Whether geometry reaches its own tile's edge where that edge lies inside the other tile."""
    seam = extent.boundary.intersection(other)
    return not seam.is_empty and geometry.distance(seam) <= tolerance


def _seam_groups(data, candidates, tiles, seam_merge_iou):
    """Groups of edge-cut pieces that continue each other across tile seams."""
    for column in ("tile_id", "edge_touch"):
        if column not in data:
            raise ValueError(f"seam merging needs the {column} field written by nbf infer")
    lookup = {tile_id: index for index, tile_id in enumerate(tiles.tile_id)}
    edge = [index for index in candidates if _flag(data.at[index, "edge_touch"], "edge_touch")]
    unknown = sorted({str(data.at[index, "tile_id"]) for index in edge} - set(lookup))
    if unknown:
        raise ValueError(f"predictions refer to tiles missing from the manifest, e.g. {unknown[0]}")
    extents = list(tiles.geometry)
    # Half a pixel, from each tile's perimeter in metres against its perimeter in pixels.
    tolerance = [0.5 * extent.length / (2 * (width + height))
                 for extent, width, height in zip(extents, tiles.width, tiles.height)]
    geometries = [data.geometry.iloc[index] for index in edge]
    tree = STRtree(geometries)
    pairs = []
    for position, index in enumerate(edge):
        tile = lookup[data.at[index, "tile_id"]]
        for other_position in tree.query(geometries[position], predicate="intersects"):
            other_position = int(other_position)
            if other_position <= position:
                continue
            other_tile = lookup[data.at[edge[other_position], "tile_id"]]
            if other_tile == tile:
                continue
            overlap = extents[tile].intersection(extents[other_tile])
            if overlap.area <= 0:
                continue
            first, second = geometries[position], geometries[other_position]
            # Each piece must be cut where the other tile can still see the building.
            if not (_cut_inside(first, extents[tile], extents[other_tile], tolerance[tile])
                    and _cut_inside(second, extents[other_tile], extents[tile], tolerance[other_tile])):
                continue
            # Inside the overlap both tiles saw the same pixels, so pieces of one building agree there.
            a, b = first.intersection(overlap), second.intersection(overlap)
            shared = a.intersection(b).area
            union = a.area + b.area - shared
            if union > 0 and shared / union >= seam_merge_iou:
                pairs.append((position, other_position))
    if not pairs:
        return [], extents, lookup, tolerance
    graph = coo_matrix((np.ones(len(pairs)), ([a for a, _ in pairs], [b for _, b in pairs])),
                       shape=(len(edge), len(edge))).tocsr()
    _, labels = connected_components(graph, directed=False)
    groups = {}
    for position, label in enumerate(labels):
        groups.setdefault(int(label), []).append(edge[position])
    return [sorted(group) for group in groups.values() if len(group) > 1], extents, lookup, tolerance


def _merge_seams(data, candidates, tiles, seam_merge_iou, scores):
    """Merge each seam group into its first member; return the candidates that remain."""
    groups, extents, lookup, tolerance = _seam_groups(data, candidates, tiles, seam_merge_iou)
    merged_away = set()
    for group in groups:
        merged = polygon_parts(shapely.union_all([data.geometry.iloc[index] for index in group]))
        if not isinstance(merged, Polygon) or not merged.is_valid:
            for index in group:
                data.at[index, "cleanup_flags"] = ";".join(filter(None, [data.at[index, "cleanup_flags"], "seam_merge_rejected"]))
            continue
        keep, others = group[0], group[1:]
        member_tiles = [lookup[data.at[index, "tile_id"]] for index in group]
        covered = shapely.union_all([extents[tile] for tile in member_tiles])
        data.at[keep, data.geometry.name] = merged
        data.at[keep, "area_m2"] = merged.area
        data.at[keep, "merged_cleanup_ids"] = ";".join(str(int(data.at[index, "cleanup_id"])) for index in group)
        # Still cut only if it reaches the outer edge of the tiles it was merged from.
        data.at[keep, "edge_touch"] = bool(merged.distance(covered.boundary) <= max(tolerance[tile] for tile in member_tiles))
        if "nodata_touch" in data:
            data.at[keep, "nodata_touch"] = any(_flag(data.at[index, "nodata_touch"], "nodata_touch") for index in group)
        if "score" in data:
            best = max((float(scores.iloc[index]) for index in group), default=-np.inf)
            data.at[keep, "score"] = best if np.isfinite(best) else data.at[keep, "score"]
        data.at[keep, "cleanup_flags"] = ";".join(filter(None, [data.at[keep, "cleanup_flags"], "seam_merged"]))
        for index in others:
            data.at[index, "removed_reason"] = "seam_merged"
            data.at[index, "kept_cleanup_id"] = keep
        merged_away.update(others)
    return [index for index in candidates if index not in merged_away]


@dataclass
class CleanupResult:
    cleaned: gpd.GeoDataFrame
    removed: gpd.GeoDataFrame


def clean_polygons(frame: gpd.GeoDataFrame, *, crs: str | CRS,
                   min_area: float = 4.0, simplify: float = 0.0,
                   iou_threshold: float = 0.7, containment_threshold: float = 0.98,
                   regularize: bool = False, max_displacement: float = 0.3,
                   max_area_change: float = 0.05,
                   angle_tolerance: float = 10.0,
                   duplicate_priority: str = "complete",
                   seam_tiles: gpd.GeoDataFrame | None = None,
                   seam_merge_iou: float = 0.5) -> CleanupResult:
    """Clean then prioritize duplicates, without merging adjacent instances.

    seam_tiles (tile_id, width, height and tile extents, as manifest_tiles returns)
    turns on seam merging before duplicate suppression: edge_touch pieces from two
    tiles merge when each is cut at its own tile's edge inside the other tile and,
    inside the two tiles' overlap, the pieces reach seam_merge_iou. The first piece
    keeps the union and the highest score; the others are audited as seam_merged.

    A duplicate has positive intersection area and either IoU >= threshold or
    intersection/min(area1, area2) >= containment_threshold. With the default
    "complete" priority, instances flagged edge_touch or nodata_touch rank after
    complete ones; "score" ranks by score alone. Missing/nonfinite scores rank
    last; ties follow input order. Raw files remain the authoritative unmodified
    geometry record. The removed layer records suppression decisions.
    """
    target = metric_crs(crs)
    if frame.crs is None:
        raise ValueError("input CRS is missing; assign the correct source CRS before cleanup")
    if duplicate_priority not in DUPLICATE_PRIORITIES:
        raise ValueError(f"duplicate_priority must be one of {DUPLICATE_PRIORITIES}")
    if not all(math.isfinite(value) for value in (min_area, simplify, iou_threshold, containment_threshold)) or min_area < 0 or simplify < 0 or not 0 < iou_threshold <= 1 or not 0 < containment_threshold <= 1:
        raise ValueError("areas/tolerances must be nonnegative and overlap thresholds in (0, 1]")
    if not all(math.isfinite(value) for value in (max_displacement, max_area_change, angle_tolerance)) or max_displacement < 0 or not 0 <= max_area_change <= 1 or not 0 < angle_tolerance < 45:
        raise ValueError("invalid regularization bounds")
    if not math.isfinite(seam_merge_iou) or not 0 < seam_merge_iou <= 1:
        raise ValueError("seam merge IoU must be in (0, 1]")
    data = frame.to_crs(target).copy().reset_index(drop=True)
    reserved = {"cleanup_id", "cleanup_flags", "raw_area_m2", "area_m2", "removed_reason",
                "kept_cleanup_id", "duplicate_iou", "duplicate_containment", "merged_cleanup_ids"}
    if reserved.intersection(data.columns):
        raise ValueError("input contains reserved cleanup columns; use raw predictions, not an earlier cleanup")
    data["cleanup_id"] = np.arange(len(data), dtype=np.int64)
    data["cleanup_flags"] = ""
    data["raw_area_m2"] = [float(g.area) if g is not None and not g.is_empty else np.nan for g in data.geometry]
    data["area_m2"] = np.nan
    data["removed_reason"] = ""
    data["kept_cleanup_id"] = -1
    data["duplicate_iou"] = np.nan
    data["duplicate_containment"] = np.nan
    candidates = []
    for index, geometry in enumerate(data.geometry):
        flags = []
        if geometry is None or geometry.is_empty:
            data.at[index, "removed_reason"] = "null_or_empty"
            continue
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
            flags.append("repaired_invalid")
        polygon = polygon_parts(geometry)
        if polygon is None or polygon.is_empty or not polygon.is_valid:
            data.at[index, "removed_reason"] = "nonpolygon_or_unrepairable"
            data.at[index, "cleanup_flags"] = ";".join(flags)
            continue
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            flags.append("nonpolygon_components_discarded")
        data.at[index, data.geometry.name] = polygon
        data.at[index, "area_m2"] = polygon.area
        data.at[index, "cleanup_flags"] = ";".join(flags)
        if polygon.area <= 0 or polygon.area < min_area:
            data.at[index, "removed_reason"] = "below_min_area"
        else:
            candidates.append(index)
    scores = pd.to_numeric(data["score"], errors="coerce") if "score" in data else pd.Series(np.nan, index=data.index)
    scores = scores.where(np.isfinite(scores), -np.inf)
    if seam_tiles is not None:
        if seam_tiles.crs is None:
            raise ValueError("seam merging tiles need a CRS")
        data["merged_cleanup_ids"] = ""
        candidates = _merge_seams(data, candidates, seam_tiles.to_crs(target).reset_index(drop=True),
                                  seam_merge_iou, scores)
        scores = pd.to_numeric(data["score"], errors="coerce") if "score" in data else scores
        scores = scores.where(np.isfinite(scores), -np.inf)
    geometries = [data.geometry.iloc[index] for index in candidates]
    tree = STRtree(geometries)
    # At a tile seam the copy cut by one tile's edge can outscore the whole
    # building seen by its neighbor, so completeness outranks score by default.
    truncated = _truncated(data) if duplicate_priority == "complete" else np.zeros(len(data), dtype=bool)
    priority = sorted(candidates, key=lambda index: (bool(truncated[index]), -float(scores.iloc[index]), index))
    kept = set()
    rank = {index: place for place, index in enumerate(priority)}
    for index in priority:
        geometry = data.geometry.iloc[index]
        possible = sorted((candidates[int(position)] for position in tree.query(geometry)
                           if candidates[int(position)] in kept), key=rank.get)
        for other in possible:
            previous = data.geometry.iloc[other]
            intersection = geometry.intersection(previous).area
            if intersection <= 0:
                continue
            iou = intersection / (geometry.area + previous.area - intersection)
            containment = intersection / min(geometry.area, previous.area)
            if iou >= iou_threshold or containment >= containment_threshold:
                data.at[index, "removed_reason"] = "duplicate_overlap"
                data.at[index, "kept_cleanup_id"] = other
                data.at[index, "duplicate_iou"] = iou
                data.at[index, "duplicate_containment"] = containment
                break
        else:
            kept.add(index)
    # Suppression uses repaired raw masks, so simplification does not manufacture
    # or erase overlap evidence. Never union, snap, or dissolve neighboring masks.
    limits = {"min_area": min_area, "max_displacement": max_displacement, "max_area_change": max_area_change}
    for index in sorted(kept):
        original = data.geometry.iloc[index]
        geometry = original
        flags = list(filter(None, data.at[index, "cleanup_flags"].split(";")))
        if simplify:
            candidate = geometry.simplify(simplify, preserve_topology=True)
            if isinstance(candidate, (Polygon, MultiPolygon)) and candidate.is_valid and candidate.area >= min_area:
                geometry = candidate
                flags.append("simplified")
            else:
                flags.append("simplify_rejected")
        if regularize:
            candidate, flag = conservative_regularize(geometry, max_displacement, max_area_change, angle_tolerance)
            # Bound the combined simplify+regularize change against repaired raw
            # geometry too. No simplification loophole in the regularization gate.
            if _within_change_bounds(original, candidate, **limits):
                geometry = candidate
                flags.append(flag)
            elif geometry is not original and _within_change_bounds(original, geometry, **limits):
                # Only regularizing pushed the change past the limits: keep the simplification.
                flags.append("combined_change_rejected")
            else:
                if "simplified" in flags:
                    flags[flags.index("simplified")] = "simplify_reverted"
                geometry = original
                flags.append("combined_change_rejected")
        data.at[index, data.geometry.name] = geometry
        data.at[index, "area_m2"] = geometry.area
        data.at[index, "cleanup_flags"] = ";".join(flags)
    return CleanupResult(data.loc[sorted(kept)].copy(), data.loc[data.removed_reason != ""].copy())


def read_polygon_inputs(path: str | Path, layer: str | None = None) -> tuple[gpd.GeoDataFrame, list[Path]]:
    source = Path(path)
    if source.suffix.lower() == ".gdb":
        # Esri's Confidence (0-100) and outline repair need the importer's handling first.
        raise ValueError("Import a file geodatabase layer first, e.g. nbf import-vectors --input "
                         f"{source.name} --layer <feature class> --preset esri --crs <EPSG> --output <new .gpkg>")
    # Inference receipts are JSON sidecars; interrupted writes use *.part.gpkg.
    # Neither is a completed prediction dataset. A standalone .json GeoJSON can
    # still be supplied explicitly, without guessing the meaning of sidecars.
    files = sorted(file for file in source.iterdir()
                   if file.is_file() and file.suffix.lower() in {".gpkg", ".geojson"}
                   and not file.name.lower().endswith(".part.gpkg")) if source.is_dir() else [source]
    if not files:
        raise ValueError("no GPKG or GeoJSON inputs found")
    frames = []
    first_crs = None
    for file in files:
        if file.suffix.lower() not in {".gpkg", ".geojson", ".json"}:
            raise ValueError(f"unsupported input format: {file.suffix}")
        frame = gpd.read_file(file, layer=layer) if layer else gpd.read_file(file)
        if frame.crs is None:
            raise ValueError(f"missing CRS in {file.name}")
        if {"input_file", "input_row"}.intersection(frame.columns):
            raise ValueError("input contains reserved provenance fields input_file/input_row")
        frame["input_file"] = str(file.resolve())
        frame["input_row"] = np.arange(len(frame), dtype=np.int64)
        first_crs = first_crs or frame.crs
        frames.append(frame.to_crs(first_crs))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry=frames[0].geometry.name, crs=first_crs), files


def write_layers(path: str | Path, layers: dict[str, gpd.GeoDataFrame], *, overwrite: bool = False):
    """Write every layer, including empty layers, with the pyogrio GPKG driver."""
    destination = Path(path)
    if destination.suffix.lower() != ".gpkg":
        raise ValueError("output must have a .gpkg extension")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in layers.items():
        frame.to_file(destination, layer=name, driver="GPKG", engine="pyogrio", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw GPKG/GeoJSON or flat directory of them")
    parser.add_argument("--layer", help="input layer name; otherwise the first layer is read")
    parser.add_argument("--output", required=True, help="GPKG with cleaned and removed_audit layers")
    parser.add_argument("--metric-crs", required=True, help="projected CRS in metres, e.g. EPSG:26914")
    parser.add_argument("--min-area-m2", type=float, default=4.0)
    parser.add_argument("--simplify-m", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--containment-threshold", type=float, default=0.98)
    parser.add_argument("--duplicate-priority", choices=DUPLICATE_PRIORITIES, default="complete",
                        help="complete: keep detections not cut by a tile edge or nodata first, then higher score; "
                             "score: higher score only")
    parser.add_argument("--seam-merge-manifest", type=Path,
                        help="tile manifest of the inference run; merges pieces of buildings cut at tile seams")
    parser.add_argument("--seam-merge-iou", type=float, default=0.5,
                        help="how closely two pieces must agree inside their tiles' overlap to merge")
    parser.add_argument("--regularize", action="store_true", help="attempt a bounded near-orthogonal fit")
    parser.add_argument("--max-displacement-m", type=float, default=0.3)
    parser.add_argument("--max-area-change", type=float, default=0.05, help="relative area change, e.g. 0.05 = 5%%")
    parser.add_argument("--angle-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    frame, files = read_polygon_inputs(args.input, args.layer)
    output = Path(args.output).resolve()
    if any(output == file.resolve() or (output.exists() and output.samefile(file)) for file in files):
        parser.error("output must not replace an input file; raw input must be preserved")
    result = clean_polygons(frame, crs=args.metric_crs, min_area=args.min_area_m2,
                            simplify=args.simplify_m, iou_threshold=args.iou_threshold,
                            containment_threshold=args.containment_threshold, regularize=args.regularize,
                            max_displacement=args.max_displacement_m, max_area_change=args.max_area_change,
                            angle_tolerance=args.angle_tolerance_deg, duplicate_priority=args.duplicate_priority,
                            seam_tiles=manifest_tiles(args.seam_merge_manifest) if args.seam_merge_manifest else None,
                            seam_merge_iou=args.seam_merge_iou)
    write_layers(output, {"cleaned": result.cleaned, "removed_audit": result.removed}, overwrite=args.overwrite)
    print(json.dumps({"input_count": len(frame), "cleaned_count": len(result.cleaned),
                      "removed_count": len(result.removed), "duplicate_priority": args.duplicate_priority,
                      "seam_merged": int((result.removed.removed_reason == "seam_merged").sum()),
                      "raw_inputs_preserved": True}))


if __name__ == "__main__":
    main()
