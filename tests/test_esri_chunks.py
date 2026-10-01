import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from nearmap_buildings import cli
from nearmap_buildings.esri import plan_digest, run_chunk_batch
from nearmap_buildings.esri_chunks import _stream, main, merge_chunks, plan_chunks, run_chunks
from nearmap_buildings.preprocessing import tile

ROOT = Path(__file__).resolve().parents[1]


CRS = "EPSG:32614"


@pytest.fixture
def plan_dir(tmp_path):
    """A 300 x 200 px raster at 1 m, tiled, then planned in 128 px cores with 16 px overlap."""
    source = tmp_path / "city.tif"
    pixels = np.random.default_rng(0).integers(1, 255, (3, 200, 300), dtype=np.uint8)
    with rasterio.open(source, "w", driver="GTiff", width=300, height=200, count=3, dtype="uint8",
                       crs=CRS, transform=from_origin(1000, 5000, 1, 1)) as dst:
        dst.write(pixels)
    tile(source, tmp_path / "tiles", tile_size=100, overlap=20)
    plan_chunks(tmp_path / "tiles" / "manifest.json", tmp_path / "chunks", core=128, overlap=16)
    return tmp_path / "chunks"


def test_plan_writes_vrts_that_read_the_source_window_and_cores_that_tile_it(plan_dir):
    plan = json.loads((plan_dir / "chunks.json").read_text())
    assert [chunk["name"] for chunk in plan["chunks"]] == [
        "chunk_r000_c000", "chunk_r000_c001", "chunk_r000_c002", "chunk_r001_c000", "chunk_r001_c001", "chunk_r001_c002"]
    with rasterio.open(plan["source"]) as src:
        source = src.read()
    for chunk in plan["chunks"]:
        left, top, width, height = chunk["window"]
        with rasterio.open(plan_dir / f"{chunk['name']}.vrt") as vrt:
            assert np.array_equal(vrt.read(), source[:, top:top + height, left:left + width])
            assert vrt.transform.c == 1000 + left and vrt.transform.f == 5000 - top
    first = plan["chunks"][0]
    assert first["window"] == [0, 0, 144, 144] and first["core_bounds"] == [1000, 4872, 1128, 5000]
    cores = [box(*chunk["core_bounds"]) for chunk in plan["chunks"]]
    assert sum(core.area for core in cores) == 300 * 200


def write_chunk_outputs(gdb, plan_dir, detections):
    """detections: chunk name -> list of (box, Confidence). Writes Esri-style layers and markers."""
    done = plan_dir.parent / "done"
    done.mkdir(exist_ok=True)
    for name, rows in detections.items():
        gpd.GeoDataFrame({"Class": "Building", "Confidence": [score for _, score in rows],
                          "method": "esri_building_usa"},
                         geometry=[geometry for geometry, _ in rows], crs=CRS).to_file(
                             gdb, layer=name, driver="OpenFileGDB", geometry_type="Polygon")
        (done / f"{name}.json").write_text(json.dumps({"plan_sha256": plan_digest(plan_dir)}))
    return done


def test_merge_keeps_each_detection_only_in_the_core_that_holds_it(plan_dir, tmp_path):
    plan = json.loads((plan_dir / "chunks.json").read_text())
    names = [chunk["name"] for chunk in plan["chunks"]]
    # Core 0 spans x 1000-1128, core 1 x 1128-1256 (top row). A building at x 1120-1134 is centred
    # at x 1127 in core 0; both chunks saw it in their overlap.
    seam = box(1120, 4950, 1134, 4960)
    detections = {name: [] for name in names}
    detections[names[0]] = [(box(1010, 4980, 1020, 4990), 95.0), (seam, 90.0)]
    detections[names[1]] = [(seam, 92.0), (box(1200, 4980, 1210, 4990), 60.0)]
    done = write_chunk_outputs(tmp_path / "esri.gdb", plan_dir, detections)
    output = tmp_path / "merged.gpkg"
    result = merge_chunks(tmp_path / "esri.gdb", plan_dir, done, "EPSG:26914", output, min_score=0.7)
    assert (result["detected"], result["kept_by_core"], result["at_min_score"]) == (4, 3, 2)
    merged = gpd.read_file(output)
    assert merged.crs.to_epsg() == 26914
    assert sorted(merged.source_id) == [f"{names[0]}:1", f"{names[0]}:2"]
    assert sorted(merged.score) == [0.9, 0.95]


def test_merge_refuses_unfinished_chunks(plan_dir, tmp_path):
    names = [chunk["name"] for chunk in json.loads((plan_dir / "chunks.json").read_text())["chunks"]]
    done = write_chunk_outputs(tmp_path / "esri.gdb", plan_dir, {names[0]: [(box(1010, 4980, 1020, 4990), 95.0)]})
    with pytest.raises(ValueError, match="has not finished"):
        merge_chunks(tmp_path / "esri.gdb", plan_dir, done, CRS, tmp_path / "merged.gpkg")


def fake_process(lines, returncode=0, finish=(), done=None):
    """A stand-in ArcGIS process: prints lines, writes done-markers for finish, exits with returncode."""
    def run(command, env, on_line):
        for name in finish:
            (done / f"{name}.json").write_text("{}")
        for line in lines:
            on_line(line)
        return returncode
    return run


def driver(plan_dir, tmp_path, processes, **options):
    calls = []
    outcomes = iter(processes)

    def runner(command, env, on_line):
        calls.append((command, env))
        return next(outcomes)(command, env, on_line)

    result = run_chunks(plan_dir, tmp_path / "esri.gdb", tmp_path / "done", tmp_path / "usa.dlpk",
                        execute=True, runner=runner, **options)
    return result, calls


def test_run_restarts_arcgis_until_complete_and_restores_gis_settings(plan_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(cli.REMOVED_GIS_ENVIRONMENT, json.dumps({"PROJ_LIB": "arcgis/proj"}))
    dry = run_chunks(plan_dir, tmp_path / "esri.gdb", tmp_path / "done", tmp_path / "usa.dlpk", per_process=2)
    assert dry["dry_run"] and dry["chunks"] == 6 and dry["command"][-1] == "--execute"
    assert dry["command"][dry["command"].index("--budget") + 1] == "2"
    # Odd lines that start with "{" are shown, never parsed as a status.
    result, calls = driver(plan_dir, tmp_path, [
        fake_process(['{"chunk": "a"}', "{'batch_size': 4}", "{ partial", '{"status": "budget"}']),
        fake_process(["CUDA out of memory"], returncode=1),
        fake_process(['{"status": "complete"}'])])
    assert result["processes"] == 3
    assert all(env["PROJ_LIB"] == "arcgis/proj" and cli.REMOVED_GIS_ENVIRONMENT not in env for _, env in calls)
    assert Path(calls[0][0][1]).name == "esri.py"
    assert "{'batch_size': 4}" in capsys.readouterr().out


def test_run_gives_up_after_repeated_failures_without_progress(plan_dir, tmp_path):
    with pytest.raises(RuntimeError, match="2 failed processes"):
        driver(plan_dir, tmp_path, [fake_process(["license unavailable"], returncode=1)] * 2, max_failures=2)


def test_failing_processes_that_finish_chunks_keep_the_run_going(plan_dir, tmp_path):
    # Out of GPU memory part-way through every process: each still finishes a chunk.
    names = [chunk["name"] for chunk in json.loads((plan_dir / "chunks.json").read_text())["chunks"]]
    done = tmp_path / "done"
    done.mkdir()
    processes = [fake_process(["out of memory"], 1, finish=[name], done=done) for name in names[:4]]
    result, _ = driver(plan_dir, tmp_path, [*processes, fake_process(['{"status": "complete"}'])], max_failures=2)
    assert result["processes"] == 5


def test_stream_passes_lines_on_as_they_arrive():
    lines = []
    code = _stream([sys.executable, "-c", "import sys; print('one'); print('two', file=sys.stderr); sys.exit(3)"],
                   None, lines.append)
    assert (code, sorted(lines)) == (3, ["one", "two"])


def test_python_m_cli_gives_arcgis_its_gis_settings_back(plan_dir, tmp_path):
    # A stand-in ArcPy records the GDAL_DATA the ArcGIS process received, then fails.
    fake = tmp_path / "fake_arcpy"
    fake.mkdir()
    (fake / "arcpy.py").write_text("import os, pathlib\n"
                                   "pathlib.Path(os.environ['FAKE_ARCPY_RECORD']).write_text(os.environ.get('GDAL_DATA', 'unset'))\n"
                                   "raise RuntimeError('stand-in ArcPy')\n")
    record = tmp_path / "record.txt"
    env = {**os.environ, "GDAL_DATA": "arcgis/gdal", "FAKE_ARCPY_RECORD": str(record),
           "PYTHONPATH": os.pathsep.join([str(fake), str(ROOT / "src")])}
    env.pop(cli.REMOVED_GIS_ENVIRONMENT, None)
    result = subprocess.run([sys.executable, "-m", "nearmap_buildings.cli", "esri-chunks", "run",
                             "--chunks", str(plan_dir), "--gdb", str(tmp_path / "esri.gdb"), "--done", str(tmp_path / "done"),
                             "--model", str(tmp_path / "usa.dlpk"), "--arcgis-python", sys.executable,
                             "--max-failures", "1", "--execute"], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "stopped after 1 failed" in result.stderr
    assert record.read_text() == "arcgis/gdal"


def chunk_arcpy(calls, existing):
    """Enough ArcPy for run_chunk_batch and run_esri; records what ran and what was deleted."""
    def detect(**kwargs):
        calls.setdefault("detected", []).append(Path(kwargs["out_detected_objects"]).name)
        existing.add(kwargs["out_detected_objects"])
    return SimpleNamespace(
        Exists=lambda path: path in existing or path.endswith(".gdb"),
        Raster=lambda path: SimpleNamespace(bandCount=3, pixelType="U8", spatialReference=SimpleNamespace(name="UTM")),
        CheckExtension=lambda name: "Available", CheckOutExtension=lambda name: None, CheckInExtension=lambda name: None,
        EnvManager=lambda **kwargs: contextlib.nullcontext(),
        Describe=lambda path: SimpleNamespace(shapeType="Polygon"),
        ia=SimpleNamespace(DetectObjectsUsingDeepLearning=detect),
        management=SimpleNamespace(
            CreateFileGDB=lambda folder, name: None, AddField=lambda *a, **k: None, CalculateField=lambda *a, **k: None,
            Delete=lambda path: (existing.discard(path), calls.setdefault("deleted", []).append(Path(path).name)),
            GetCount=lambda path: ["3"]))


def test_batch_runs_its_budget_resumes_and_replaces_partial_outputs(plan_dir, tmp_path, monkeypatch):
    model = tmp_path / "usa.dlpk"
    model.touch()
    gdb, done = tmp_path / "esri.gdb", tmp_path / "done"
    calls, existing = {}, {str((gdb / "chunk_r000_c000").resolve())}  # left by an interrupted run
    monkeypatch.setitem(sys.modules, "arcpy", chunk_arcpy(calls, existing))
    assert run_chunk_batch(plan_dir, gdb, done, model, budget=4) == "budget"
    assert calls["deleted"] == ["chunk_r000_c000"]
    assert calls["detected"] == ["chunk_r000_c000", "chunk_r000_c001", "chunk_r000_c002", "chunk_r001_c000"]
    assert json.loads((done / "chunk_r000_c000.json").read_text())["features"] == 3
    assert run_chunk_batch(plan_dir, gdb, done, model, budget=4) == "complete"
    assert calls["detected"][4:] == ["chunk_r001_c001", "chunk_r001_c002"]


def test_plan_command_refuses_an_existing_folder(plan_dir, tmp_path):
    with pytest.raises(FileExistsError):
        main(["plan", "--manifest", str(tmp_path / "tiles" / "manifest.json"), "--output", str(plan_dir)])


def test_esri_script_chunk_mode_needs_its_own_arguments(plan_dir, tmp_path):
    from nearmap_buildings.esri import main as esri_main

    for argv in (["--chunks", str(plan_dir)],
                 ["--chunks", str(plan_dir), "--gdb", "a.gdb", "--done", "d", "--execute", "--raster", "x.tif"],
                 []):
        with pytest.raises(SystemExit):
            esri_main(["--model", str(tmp_path / "usa.dlpk"), *argv])


def test_done_markers_from_another_plan_are_refused(plan_dir, tmp_path, monkeypatch):
    model = tmp_path / "usa.dlpk"
    model.touch()
    done = tmp_path / "done"
    done.mkdir()
    (done / "chunk_r000_c000.json").write_text(json.dumps({"plan_sha256": "0" * 64}))
    monkeypatch.setitem(sys.modules, "arcpy", chunk_arcpy({}, set()))
    with pytest.raises(ValueError, match="another chunk plan"):
        run_chunk_batch(plan_dir, tmp_path / "esri.gdb", done, model, budget=1)
    names = [chunk["name"] for chunk in json.loads((plan_dir / "chunks.json").read_text())["chunks"]]
    done = write_chunk_outputs(tmp_path / "esri.gdb", plan_dir, {name: [] for name in names})
    (done / f"{names[0]}.json").write_text(json.dumps({"plan_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="another chunk plan"):
        merge_chunks(tmp_path / "esri.gdb", plan_dir, done, CRS, tmp_path / "merged.gpkg")


def raster(path, width, height, **profile):
    pixels = np.random.default_rng(1).integers(1, 240, (3, height, width), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8",
                       crs=CRS, transform=from_origin(0, height, 1, 1), **profile) as dst:
        dst.write(pixels)
    return path


def test_every_chunk_keeps_the_full_overlap_where_there_is_imagery(tmp_path):
    tile(raster(tmp_path / "city.tif", 600, 600), tmp_path / "tiles", tile_size=100, overlap=20)
    plan = plan_chunks(tmp_path / "tiles" / "manifest.json", tmp_path / "chunks", core=200, overlap=50)
    for chunk in plan["chunks"]:
        row, column = (int(part[1:]) * 200 for part in chunk["name"].split("_")[1:])
        expected = [max(column - 50, 0), max(row - 50, 0)]
        expected += [min(column + 250, 600) - expected[0], min(row + 250, 600) - expected[1]]
        assert chunk["window"] == expected, chunk["name"]


def test_source_nodata_reaches_the_model_as_zero(tmp_path):
    source = raster(tmp_path / "city.tif", 200, 200, nodata=249)
    with rasterio.open(source, "r+") as dst:
        pixels = dst.read()
        pixels[:, :10, :] = 249
        dst.write(pixels)
    tile(source, tmp_path / "tiles", tile_size=100, overlap=20)
    plan = plan_chunks(tmp_path / "tiles" / "manifest.json", tmp_path / "chunks", core=128, overlap=16)
    with rasterio.open(tmp_path / "chunks" / f"{plan['chunks'][0]['name']}.vrt") as vrt:
        chunk = vrt.read()
    assert (chunk[:, :10, :] == 0).all() and (chunk[:, 10:, :] == pixels[:, 10:chunk.shape[1], :chunk.shape[2]]).all()


def test_masked_sources_are_refused_and_leave_no_folder(tmp_path):
    source = raster(tmp_path / "city.tif", 200, 200)
    with rasterio.open(source, "r+") as dst:
        mask = np.full((200, 200), 255, dtype=np.uint8)
        mask[:10] = 0
        dst.write_mask(mask)
    tile(source, tmp_path / "tiles", tile_size=100, overlap=20)
    with pytest.raises(ValueError, match="internal mask"):
        plan_chunks(tmp_path / "tiles" / "manifest.json", tmp_path / "chunks", core=128, overlap=16)
    assert not (tmp_path / "chunks").exists()
