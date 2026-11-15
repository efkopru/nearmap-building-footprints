"""Bookkeeping for the Lewisville fine-tuning protocol: freeze the reviewed labels, choose
settings on the validation squares, and score the test squares once, with ranges.

    python scripts/finetune_protocol.py freeze --package data/lewisville/finetune_areas --output data/reference/finetune_v1
    python scripts/finetune_protocol.py choose --frozen data/reference/finetune_v1 --runs runs_val.json --output outputs/finetune/v1/choice.json
    python scripts/finetune_protocol.py score --frozen data/reference/finetune_v1 --choice outputs/finetune/v1/choice.json --runs runs_test.json --output outputs/finetune/v1/test_scores.json

The rules are in results/lewisville/finetuning_protocol.md. Tiling, training and inference are
ordinary nbf commands; a runs file names the folders of raw outlines they left:

    {"published": {"raw": "..."},
     "released": [{"tiles": 1024, "raw": "..."}, {"tiles": 1536, "raw": "..."}],
     "fine_tuned": [{"tiles": 1024, "epoch": 5, "raw": "..."}, ...]}

For the test, "released" and "fine_tuned" each name the one run with the chosen tile size and epoch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_name, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LISTS = ("test", "val", "train")
MINIMUM = {"test": 8, "val": 4, "train": 8}
CRS = "EPSG:26914"
PUBLISHED = {"min_score": 0.7, "small_min_score": None}
MAIN_CUTOFFS = (0.5, 0.6, 0.7, 0.8)
SMALL_CUTOFFS = (0.3, 0.4, 0.5, 0.6, 0.7)
SMALL_M2 = 20.0
RESAMPLES, SEED = 1000, 20261114
COUNTS = ("tp", "fp", "fn", "tp20", "fp20", "fn20", "small_labels", "small_found", "small_drawn", "small_matched")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_squares(areas, minimum=MINIMUM):
    """The squares marked reviewed, which must be the first k of each list with no gap."""
    done = areas[areas["reviewed"].fillna("").astype(str).str.strip() != ""]
    for name in LISTS:
        orders = sorted(int(value) for value in done.loc[done["split"] == name, "order_in_split"])
        if orders != list(range(1, len(orders) + 1)):
            raise ValueError(f"The reviewed {name} squares must be the first of their list with none skipped; got {orders}")
        if len(orders) < minimum[name]:
            raise ValueError(f"The protocol needs at least {minimum[name]} reviewed {name} squares; {len(orders)} are marked")
    return done.sort_values(["split", "order_in_split"]).reset_index(drop=True)


def freeze(package, output, minimum=MINIMUM):
    """Copy the reviewed outlines and squares out of the labeling package, with their fingerprints."""
    import geopandas as gpd
    from nearmap_buildings.evaluation import geometry_fingerprint

    package, output = Path(package), Path(output)
    if output.exists():
        raise FileExistsError(f"Frozen labels are never replaced; choose a new version folder: {output}")
    squares = reviewed_squares(gpd.read_file(package / "review.gdb", layer="areas"), minimum)
    region = squares.geometry.union_all()
    labels = gpd.read_file(package / "review.gdb", layer="buildings")
    labels = labels[labels.geometry.notna() & ~labels.geometry.is_empty]
    labels = labels[labels.geometry.intersects(region)].reset_index(drop=True)
    if not labels.geometry.is_valid.all():
        raise ValueError(f"{int((~labels.geometry.is_valid).sum())} reviewed outlines are not valid polygons; fix them in the review layer")
    original = gpd.read_file(package / "inventory_start.gpkg", layer="buildings")
    original = original[original.geometry.intersects(region)].reset_index(drop=True)
    output.mkdir(parents=True)
    labels.to_file(output / "labels.gpkg", layer="buildings", driver="GPKG")
    original.to_file(output / "original.gpkg", layer="buildings", driver="GPKG")
    keep = [column for column in ("area_id", "split", "order_in_split", "kind", "reviewed", "review_date", "notes", "geometry")
            if column in squares.columns]
    squares[keep].to_file(output / "squares.gpkg", layer="squares", driver="GPKG")
    for name in LISTS:
        squares.loc[squares["split"] == name, keep].to_file(output / "squares.gpkg", layer=name, driver="GPKG")
    record = {"schema": "finetune-frozen-labels-v1", "package": str(package),
              "squares": {name: squares.loc[squares["split"] == name, "area_id"].tolist() for name in LISTS},
              "outlines": len(labels), "outlines_started_from": len(original),
              "labels_fingerprint": geometry_fingerprint(labels), "squares_fingerprint": geometry_fingerprint(squares),
              "files": {name: sha256(output / name) for name in ("labels.gpkg", "original.gpkg", "squares.gpkg")}}
    (output / "freeze.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def read_raw(folder):
    """Every raw outline of one inference run."""
    import geopandas as gpd
    import pandas as pd

    files = sorted(Path(folder).glob("*.gpkg"))
    if not files:
        raise FileNotFoundError(f"No raw outlines in {folder}")
    frames = [gpd.read_file(path) for path in files]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def score_squares(raw, labels, squares, *, min_score, small_min_score, reference_status="independent_holdout"):
    """Clean one run with its cut-offs and count matches square by square.

    Each square is scored on its own, so an outline crossing a square's edge is left out
    even where two squares of a list touch, and whole squares can be resampled.
    """
    from nearmap_buildings.evaluation import evaluate
    from nearmap_buildings.postprocess import clean_polygons

    cleaned = clean_polygons(raw, crs=CRS, min_area=4.0, min_score=min_score, small_min_score=small_min_score,
                             small_area=SMALL_M2).cleaned.reset_index(drop=True)
    labels, squares = labels.to_crs(CRS), squares.to_crs(CRS)
    rows = []
    for _, square in squares.iterrows():
        aoi = squares[squares["area_id"] == square["area_id"]][["geometry"]]
        near = square.geometry.buffer(1.0)
        predictions = cleaned[cleaned.geometry.intersects(near)].reset_index(drop=True)
        reference = labels[labels.geometry.intersects(near)].reset_index(drop=True)
        common = dict(crs=CRS, edge_policy="exclude", reference_status=reference_status)
        every = evaluate(predictions, reference, aoi, size_bins=[SMALL_M2], **common).report
        main = evaluate(predictions, reference, aoi, min_area_m2=SMALL_M2, **common).report
        small = every["size_classes"][0]
        rows.append({"area_id": square["area_id"], "kind": square.get("kind", ""),
                     "tp": every["true_positives"], "fp": every["false_positives"], "fn": every["false_negatives"],
                     "tp20": main["true_positives"], "fp20": main["false_positives"], "fn20": main["false_negatives"],
                     "small_labels": small["reference"], "small_found": small["matched_reference"],
                     "small_drawn": small["predictions"], "small_matched": small["matched_predictions"]})
    return rows


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def pooled(rows):
    """Counts summed over squares, and the figures the protocol reports."""
    total = {key: sum(row[key] for row in rows) for key in COUNTS}
    figures = {
        "precision": ratio(total["tp"], total["tp"] + total["fp"]), "recall": ratio(total["tp"], total["tp"] + total["fn"]),
        "f1": ratio(2 * total["tp"], 2 * total["tp"] + total["fp"] + total["fn"]),
        "precision_20": ratio(total["tp20"], total["tp20"] + total["fp20"]),
        "recall_20": ratio(total["tp20"], total["tp20"] + total["fn20"]),
        "f1_20": ratio(2 * total["tp20"], 2 * total["tp20"] + total["fp20"] + total["fn20"]),
        "small_recall": ratio(total["small_found"], total["small_labels"]),
        "small_precision": ratio(total["small_matched"], total["small_drawn"]),
    }
    return {**total, **figures, "squares": len(rows)}


def ranges(rows, resamples=RESAMPLES, seed=SEED):
    """A 90% range for each figure, from resampling whole squares with replacement."""
    import numpy as np

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(resamples, len(rows)))
    samples = [pooled([rows[index] for index in draw]) for draw in draws]
    result = {}
    for key in ("precision", "recall", "f1", "precision_20", "recall_20", "f1_20", "small_recall", "small_precision"):
        values = [sample[key] for sample in samples if sample[key] is not None]
        result[key] = [float(np.percentile(values, 5)), float(np.percentile(values, 95))] if values else None
    return result


def summarize(rows):
    """All squares together and each kind of square, each with its ranges."""
    groups = {"all": rows, **{kind: [row for row in rows if row["kind"] == kind] for kind in sorted({row["kind"] for row in rows})}}
    return {name: {**pooled(group), "ranges_90": ranges(group)} for name, group in groups.items() if group}


def choose_setting(candidates, published_f1_20):
    """The protocol's rule for one run to tune, from its scored combinations.

    Highest F1 over all sizes among those within 0.01 of the published run's F1 for
    20 m2 and up; if none qualifies, the highest F1 for 20 m2 and up. Among combinations
    within 0.005 of the best: the higher small cut-off, then the higher main cut-off,
    then 1536 px tiles, then the earlier checkpoint.
    """
    def value(candidate, key):
        return candidate[key] if candidate[key] is not None else -1.0

    qualifying = [c for c in candidates if value(c, "f1_20") >= published_f1_20 - 0.01]
    metric, pool = ("f1", qualifying) if qualifying else ("f1_20", candidates)
    best = max(value(c, metric) for c in pool)
    close = [c for c in pool if value(c, metric) >= best - 0.005]
    chosen = max(close, key=lambda c: (c["small_min_score"], c["min_score"], c["tiles"], -c.get("epoch", 0)))
    return {**chosen, "chosen_by": metric, "qualifying": len(qualifying), "considered": len(candidates)}


def load_frozen(folder, name):
    import geopandas as gpd

    folder = Path(folder)
    return gpd.read_file(folder / "labels.gpkg", layer="buildings"), gpd.read_file(folder / "squares.gpkg", layer=name)


def choose(frozen, runs, output, reference_status="independent_holdout"):
    """Score every combination on the validation squares and write down the choice."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"The choice is written once, before any test score: {output}")
    labels, squares = load_frozen(frozen, "val")
    status = dict(reference_status=reference_status)
    published = pooled(score_squares(read_raw(runs["published"]["raw"]), labels, squares, **PUBLISHED, **status))
    record = {"schema": "finetune-choice-v1", "frozen": sha256(Path(frozen) / "freeze.json"),
              "validation_squares": squares["area_id"].tolist(), "published": {**PUBLISHED, "tiles": 1536, **published}}
    table = []
    for arm in ("released", "fine_tuned"):
        candidates = []
        for run in runs[arm]:
            raw = read_raw(run["raw"])
            for main in MAIN_CUTOFFS:
                for small in SMALL_CUTOFFS:
                    figures = pooled(score_squares(raw, labels, squares, min_score=main, small_min_score=small, **status))
                    candidates.append({"arm": arm, **{k: v for k, v in run.items() if k != "raw"},
                                       "min_score": main, "small_min_score": small, **figures})
        record[arm] = choose_setting(candidates, published["f1_20"])
        table.extend(candidates)
    record["combinations"] = table
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def score(frozen, choice_path, runs, output, reference_status="independent_holdout"):
    """Score the three runs on the test squares with their frozen settings. Runs once."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"The test squares are scored once; this result already exists: {output}")
    choice = json.loads(Path(choice_path).read_text(encoding="utf-8"))
    if choice["frozen"] != sha256(Path(frozen) / "freeze.json"):
        raise ValueError("The choice was made on a different frozen version of the labels")
    labels, squares = load_frozen(frozen, "test")
    record = {"schema": "finetune-test-scores-v1", "frozen": choice["frozen"], "choice": sha256(choice_path),
              "test_squares": squares["area_id"].tolist(), "resamples": RESAMPLES, "seed": SEED, "runs": {}}
    for arm in ("published", "released", "fine_tuned"):
        setting = PUBLISHED if arm == "published" else {key: choice[arm][key] for key in ("min_score", "small_min_score")}
        run = runs[arm]
        if arm != "published" and any(run.get(key) != choice[arm].get(key) for key in ("tiles", "epoch")):
            raise ValueError(f"The {arm} test run does not use the chosen tile size and checkpoint")
        rows = score_squares(read_raw(run["raw"]), labels, squares, **setting, reference_status=reference_status)
        record["runs"][arm] = {"setting": {**setting, **{k: v for k, v in run.items() if k != "raw"}},
                               "summary": summarize(rows), "squares": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("freeze", help="copy the reviewed outlines and squares out of the labeling package")
    step.add_argument("--package", required=True, type=Path)
    step.add_argument("--output", required=True, type=Path)
    for name in ("choose", "score"):
        step = commands.add_parser(name, help="choose settings on the validation squares" if name == "choose"
                                   else "score the test squares once, with ranges")
        step.add_argument("--frozen", required=True, type=Path)
        step.add_argument("--runs", required=True, type=Path, help="JSON naming the raw outline folders of each run")
        step.add_argument("--output", required=True, type=Path)
        if name == "score":
            step.add_argument("--choice", required=True, type=Path)
    for step in commands.choices.values():
        step.add_argument("--rehearsal", action="store_true",
                          help="development data: no minimum number of squares, and scores are recorded as agreement only")
    args = parser.parse_args(argv)
    status = "agreement_only" if args.rehearsal else "independent_holdout"
    try:
        if args.command == "freeze":
            result = freeze(args.package, args.output, dict.fromkeys(LISTS, 1) if args.rehearsal else MINIMUM)
            summary = {key: result[key] for key in ("squares", "outlines", "labels_fingerprint")}
        elif args.command == "choose":
            result = choose(args.frozen, json.loads(args.runs.read_text(encoding="utf-8")), args.output, status)
            keys = ("tiles", "epoch", "min_score", "small_min_score", "f1", "f1_20", "small_recall", "chosen_by", "qualifying")
            summary = {"published": {k: result["published"][k] for k in ("f1", "f1_20", "small_recall")},
                       **{arm: {k: result[arm].get(k) for k in keys} for arm in ("released", "fine_tuned")}}
        else:
            result = score(args.frozen, args.choice, json.loads(args.runs.read_text(encoding="utf-8")), args.output, status)
            keys = ("f1", "f1_20", "small_recall", "small_precision")
            summary = {arm: {k: run["summary"]["all"][k] for k in keys} | {"ranges_90": run["summary"]["all"]["ranges_90"]}
                       for arm, run in result["runs"].items()}
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
