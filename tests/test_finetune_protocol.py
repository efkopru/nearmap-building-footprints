"""The fine-tuning protocol's bookkeeping on small synthetic squares; no imagery or model runs."""
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

spec = importlib.util.spec_from_file_location("finetune_protocol", Path(__file__).parents[1] / "scripts/finetune_protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

CRS = "EPSG:26914"


def areas_frame(reviewed):
    """Two test, one validation and one training square, 100 m each; reviewed maps area ids to a mark."""
    rows = [("test-01", "test", 1, "S", 0), ("test-02", "test", 2, "G", 100), ("val-01", "val", 1, "S", 500), ("train-01", "train", 1, "S", 900)]
    frame = gpd.GeoDataFrame({"area_id": [r[0] for r in rows], "split": [r[1] for r in rows], "order_in_split": [r[2] for r in rows],
                              "kind": [r[3] for r in rows], "reviewed": [reviewed.get(r[0], "") for r in rows]},
                             geometry=[box(r[4], 0, r[4] + 100, 100) for r in rows], crs=CRS)
    return frame.astype({"order_in_split": "int32"})


def test_reviewed_squares_must_be_the_first_of_each_list():
    everything = {"test-01": "yes", "test-02": "yes", "val-01": "yes", "train-01": "yes"}
    one_each = dict.fromkeys(protocol.LISTS, 1)
    assert protocol.reviewed_squares(areas_frame(everything), one_each)["area_id"].tolist() == ["test-01", "test-02", "train-01", "val-01"]
    skipped = {key: value for key, value in everything.items() if key != "test-01"}
    with pytest.raises(ValueError, match="none skipped"):
        protocol.reviewed_squares(areas_frame(skipped), one_each)
    with pytest.raises(ValueError, match="at least 8 reviewed test squares"):
        protocol.reviewed_squares(areas_frame(everything))


def row(area_id, kind="S", **counts):
    return {"area_id": area_id, "kind": kind, **dict.fromkeys(protocol.COUNTS, 0), **counts}


def test_pooled_figures_sum_squares_before_dividing():
    rows = [row("a", tp=8, fp=2, fn=2, tp20=8, fp20=0, fn20=0, small_labels=4, small_found=1, small_drawn=5, small_matched=1),
            row("b", tp=1, fp=3, fn=9, tp20=1, fp20=1, fn20=3, small_labels=6, small_found=0, small_drawn=0, small_matched=0)]
    figures = protocol.pooled(rows)
    assert (figures["tp"], figures["fp"], figures["fn"], figures["squares"]) == (9, 5, 11, 2)
    assert figures["precision"] == pytest.approx(9 / 14) and figures["recall"] == pytest.approx(9 / 20)
    assert figures["f1"] == pytest.approx(18 / 34) and figures["f1_20"] == pytest.approx(18 / 22)
    assert figures["small_recall"] == pytest.approx(0.1) and figures["small_precision"] == pytest.approx(0.2)
    # Nothing drawn or labeled gives no ratio, not a zero.
    assert protocol.pooled([row("empty")])["f1"] is None


def test_ranges_resample_whole_squares_and_repeat_exactly():
    rows = [row("a", tp=9, fp=1, fn=1, tp20=9, fp20=1, fn20=1), row("b", tp=5, fp=5, fn=5, tp20=5, fp20=5, fn20=5),
            row("c", tp=1, fp=9, fn=9, tp20=1, fp20=9, fn20=9)]
    first, second = protocol.ranges(rows), protocol.ranges(rows)
    assert first == second
    low, high = first["f1"]
    assert 0.1 <= low < protocol.pooled(rows)["f1"] < high <= 0.9
    assert first["small_recall"] is None  # no small label in any square
    # One square cannot vary: its range is its own value.
    assert protocol.ranges(rows[:1])["f1"] == [pytest.approx(0.9), pytest.approx(0.9)]
    summary = protocol.summarize([rows[0], {**rows[1], "kind": "G"}])
    assert set(summary) == {"all", "S", "G"} and summary["G"]["squares"] == 1


def candidate(f1, f1_20, small, main=0.7, tiles=1024, epoch=None):
    extra = {} if epoch is None else {"epoch": epoch}
    return {"f1": f1, "f1_20": f1_20, "small_min_score": small, "min_score": main, "tiles": tiles, **extra}


def test_choice_takes_the_best_all_size_f1_that_keeps_main_buildings():
    candidates = [candidate(0.80, 0.90, 0.5), candidate(0.85, 0.86, 0.3), candidate(0.70, 0.91, 0.7)]
    chosen = protocol.choose_setting(candidates, published_f1_20=0.90)
    # 0.85 is higher but costs more than 0.01 on buildings of 20 m2 and up.
    assert (chosen["f1"], chosen["chosen_by"], chosen["qualifying"], chosen["considered"]) == (0.80, "f1", 2, 3)
    # Nothing keeps main buildings: fall back to the best F1 for 20 m2 and up.
    fallback = protocol.choose_setting(candidates, published_f1_20=0.99)
    assert (fallback["f1_20"], fallback["chosen_by"], fallback["qualifying"]) == (0.91, "f1_20", 0)


def test_choice_breaks_near_ties_towards_the_more_careful_setting():
    near = [candidate(0.800, 0.9, 0.3), candidate(0.797, 0.9, 0.6), candidate(0.796, 0.9, 0.6, main=0.8),
            candidate(0.796, 0.9, 0.6, main=0.8, tiles=1536), candidate(0.790, 0.9, 0.7)]
    chosen = protocol.choose_setting(near, published_f1_20=0.9)
    # Within 0.005 of the best: the higher small cut-off, then the higher main cut-off, then 1536 px.
    assert (chosen["small_min_score"], chosen["min_score"], chosen["tiles"]) == (0.6, 0.8, 1536)
    epochs = [candidate(0.8, 0.9, 0.5, epoch=20), candidate(0.8, 0.9, 0.5, epoch=5), candidate(0.8, 0.9, 0.5, epoch=10)]
    assert protocol.choose_setting(epochs, published_f1_20=0.9)["epoch"] == 5


def raw_frame(shapes):
    """Raw outlines as nbf infer leaves them: geometry, score and the tile fields cleanup reads."""
    return gpd.GeoDataFrame({"score": [score for _, score in shapes], "tile_id": ["t"] * len(shapes),
                             "edge_touch": [False] * len(shapes), "nodata_touch": [False] * len(shapes)},
                            geometry=[shape for shape, _ in shapes], crs=CRS)


def test_squares_are_scored_one_by_one_with_cutoffs_by_size():
    squares = areas_frame({})[lambda frame: frame["split"] == "test"].reset_index(drop=True)
    labels = gpd.GeoDataFrame(geometry=[box(10, 10, 30, 30), box(50, 50, 53, 53), box(95, 40, 105, 60), box(150, 10, 170, 30)], crs=CRS)
    raw = raw_frame([(box(10, 10, 30, 30), 0.9),      # house in test-01, found
                     (box(50, 50, 53, 53), 0.55),     # 9 m2 shed in test-01: kept only by the small cut-off
                     (box(95, 40, 105, 60), 0.9),     # crosses the edge between the two squares: left out of both
                     (box(150, 60, 170, 80), 0.9)])   # nothing there in test-02
    one = {r["area_id"]: r for r in protocol.score_squares(raw, labels, squares, min_score=0.7, small_min_score=None)}
    assert (one["test-01"]["tp"], one["test-01"]["fp"], one["test-01"]["fn"]) == (1, 0, 1)
    assert (one["test-01"]["small_labels"], one["test-01"]["small_found"], one["test-01"]["small_drawn"]) == (1, 0, 0)
    assert (one["test-02"]["tp"], one["test-02"]["fp"], one["test-02"]["fn"], one["test-02"]["kind"]) == (0, 1, 1, "G")
    two = {r["area_id"]: r for r in protocol.score_squares(raw, labels, squares, min_score=0.7, small_min_score=0.5)}
    assert (two["test-01"]["tp"], two["test-01"]["fn"], two["test-01"]["small_found"], two["test-01"]["small_matched"]) == (2, 0, 1, 1)
    # The 20 m2 cohort is filtered before matching, so the shed changes nothing there.
    assert (two["test-01"]["tp20"], two["test-01"]["fp20"], two["test-01"]["fn20"]) == (1, 0, 0)


@pytest.fixture
def package(tmp_path):
    """A labeling package with every square marked reviewed, and raw runs to score."""
    folder = tmp_path / "package"
    folder.mkdir()
    areas = areas_frame({"test-01": "yes", "test-02": "yes", "val-01": "yes", "train-01": "yes"})
    labels = gpd.GeoDataFrame({"id_2015": [1, 2, 3, 4, 5]}, geometry=[
        box(10, 10, 30, 30), box(50, 50, 53, 53), box(150, 10, 170, 30), box(510, 10, 530, 30), box(550, 50, 553, 53)],
        crs=CRS).astype({"id_2015": "int32"})
    labels.to_file(folder / "review.gdb", layer="buildings", driver="OpenFileGDB")
    areas.to_file(folder / "review.gdb", layer="areas", driver="OpenFileGDB")
    labels.to_file(folder / "inventory_start.gpkg", layer="buildings", driver="GPKG")
    runs = tmp_path / "runs"
    house, shed = box(510, 10, 530, 30), box(550, 50, 553, 53)
    for name, shapes in {"published_val": [(house, 0.9), (shed, 0.55)], "released_1024_val": [(house, 0.9), (shed, 0.55)],
                         "tuned_5_val": [(house, 0.9), (shed, 0.75), (box(560, 70, 563, 73), 0.45)],
                         "published_test": [(box(10, 10, 30, 30), 0.9)],
                         "released_test": [(box(10, 10, 30, 30), 0.9), (box(50, 50, 53, 53), 0.55)],
                         "tuned_test": [(box(10, 10, 30, 30), 0.9), (box(50, 50, 53, 53), 0.8), (box(150, 10, 170, 30), 0.9)]}.items():
        (runs / name).mkdir(parents=True)
        raw_frame(shapes).to_file(runs / name / "tile.gpkg", driver="GPKG")
    return folder, runs


def test_freeze_choose_and_score_run_once_each(package, tmp_path, capsys):
    folder, runs = package
    frozen = tmp_path / "frozen"
    protocol.main(["freeze", "--package", str(folder), "--output", str(frozen), "--rehearsal"])
    record = json.loads((frozen / "freeze.json").read_text())
    assert record["squares"] == {"test": ["test-01", "test-02"], "val": ["val-01"], "train": ["train-01"]}
    assert record["outlines"] == 5 and set(record["files"]) == {"labels.gpkg", "original.gpkg", "squares.gpkg"}
    with pytest.raises(SystemExit):  # a frozen version is never replaced
        protocol.main(["freeze", "--package", str(folder), "--output", str(frozen), "--rehearsal"])

    val_runs = tmp_path / "val_runs.json"
    val_runs.write_text(json.dumps({"published": {"raw": str(runs / "published_val")},
                                    "released": [{"tiles": 1024, "raw": str(runs / "released_1024_val")}],
                                    "fine_tuned": [{"tiles": 1024, "epoch": 5, "raw": str(runs / "tuned_5_val")}]}))
    choice_path = tmp_path / "choice.json"
    protocol.main(["choose", "--frozen", str(frozen), "--runs", str(val_runs), "--output", str(choice_path), "--rehearsal"])
    choice = json.loads(choice_path.read_text())
    assert choice["published"]["f1"] == pytest.approx(2 / 3) and choice["published"]["small_recall"] == 0
    # The released run finds the shed once small outlines may score 0.5; 0.5 is the highest cut-off that keeps it.
    assert (choice["released"]["f1"], choice["released"]["small_min_score"], choice["released"]["min_score"]) == (1.0, 0.5, 0.8)
    # The fine-tuned run scores it 0.75, so 0.7 keeps it and drops the false outline at 0.45.
    assert (choice["fine_tuned"]["f1"], choice["fine_tuned"]["small_min_score"], choice["fine_tuned"]["epoch"]) == (1.0, 0.7, 5)
    assert len(choice["combinations"]) == 2 * len(protocol.MAIN_CUTOFFS) * len(protocol.SMALL_CUTOFFS)
    with pytest.raises(SystemExit):  # the choice is written once
        protocol.main(["choose", "--frozen", str(frozen), "--runs", str(val_runs), "--output", str(choice_path), "--rehearsal"])

    test_runs = tmp_path / "test_runs.json"
    wrong = {"published": {"raw": str(runs / "published_test")}, "released": {"tiles": 1536, "raw": str(runs / "released_test")},
             "fine_tuned": {"tiles": 1024, "epoch": 5, "raw": str(runs / "tuned_test")}}
    test_runs.write_text(json.dumps(wrong))
    scores_path = tmp_path / "test_scores.json"
    with pytest.raises(SystemExit):  # not the tile size that was chosen
        protocol.main(["score", "--frozen", str(frozen), "--choice", str(choice_path), "--runs", str(test_runs),
                       "--output", str(scores_path), "--rehearsal"])
    assert not scores_path.exists()
    test_runs.write_text(json.dumps({**wrong, "released": {"tiles": 1024, "raw": str(runs / "released_test")}}))
    capsys.readouterr()
    protocol.main(["score", "--frozen", str(frozen), "--choice", str(choice_path), "--runs", str(test_runs),
                   "--output", str(scores_path), "--rehearsal"])
    printed = json.loads(capsys.readouterr().out)
    scores = json.loads(scores_path.read_text())
    summary = {arm: scores["runs"][arm]["summary"] for arm in ("published", "released", "fine_tuned")}
    # Three labels in the two test squares: a house and a shed in the first, a house in the second.
    assert (summary["published"]["all"]["tp"], summary["published"]["all"]["fn"]) == (1, 2)
    assert (summary["released"]["all"]["tp"], summary["released"]["all"]["small_found"]) == (2, 1)
    assert summary["fine_tuned"]["all"]["f1"] == 1.0 and summary["fine_tuned"]["G"]["tp"] == 1
    assert scores["runs"]["fine_tuned"]["setting"] == {"min_score": 0.8, "small_min_score": 0.7, "tiles": 1024, "epoch": 5}
    assert printed["fine_tuned"]["f1"] == 1.0 and len(scores["runs"]["released"]["squares"]) == 2
    with pytest.raises(SystemExit):  # the test squares are scored once
        protocol.main(["score", "--frozen", str(frozen), "--choice", str(choice_path), "--runs", str(test_runs),
                       "--output", str(scores_path), "--rehearsal"])
