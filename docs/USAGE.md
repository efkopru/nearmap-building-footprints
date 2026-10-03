# Usage guide

Every command in this guide works on local files. Nothing downloads imagery, calls a paid API or fetches model weights for you. Commands that run a model (`nbf infer`, `nbf train launch` and the Esri runner) only print a plan until you add `--execute`. Every output path must be new: commands refuse to overwrite earlier results.

Start small. Run the demo first, then a few tiles of real imagery, before a whole city. The [methodology](METHODOLOGY.md) explains why each step works the way it does.

1. [Install](#1-install)
2. [Try the demo](#2-try-the-demo)
3. [Prepare your imagery](#3-prepare-your-imagery)
4. [Run SAM 3](#4-run-sam-3)
5. [Run Esri's model](#5-run-esris-model)
6. [Import other building layers](#6-import-other-building-layers)
7. [Clean up outlines](#7-clean-up-outlines)
8. [Score and compare](#8-score-and-compare)
9. [Fine-tune SAM 3](#9-fine-tune-sam-3)
10. [Run a whole experiment from a config file](#10-run-a-whole-experiment-from-a-config-file)
11. [Troubleshooting](#troubleshooting)

## 1. Install

Use Python 3.12, 3.13 or 3.14. On Windows, in PowerShell:

```powershell
Set-Location path\to\nearmap-building-footprints
.\scripts\setup_cpu.ps1
.\.venv\Scripts\Activate.ps1
nbf doctor
```

The setup script creates `.venv` with the first of Python 3.12, 3.13 and 3.14 that the `py` launcher finds, installs the locked dependencies and runs the tests. To choose the Python, pass it: `.\scripts\setup_cpu.ps1 -Python C:\Python314\python.exe`. It refuses an existing `.venv` made with another version. Activation is optional; `.\.venv\Scripts\nbf.exe` works too.

On Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock
python -m pip install --no-deps -e .
python -m pytest -q
nbf doctor
```

This is the CPU environment, used for everything except running SAM 3, which needs the GPU environment in [section 4](#4-run-sam-3).

## 2. Try the demo

The demo writes a small synthetic scene with reference outlines, prompts and an Esri-style geodatabase:

```bash
nbf demo --output data/demo
nbf inspect data/demo/imagery.tif
nbf tile data/demo/imagery.tif --output data/demo_tiles --tile-size 256 --overlap 64
nbf infer --manifest data/demo_tiles/manifest.json --method text --output outputs/demo_plan
nbf clean --input data/demo/perfect_predictions.gpkg --output outputs/demo_cleaned.gpkg --metric-crs EPSG:32614
nbf evaluate --predictions outputs/demo_cleaned.gpkg --predictions-layer cleaned --reference data/demo/reference.gpkg --aoi data/demo/aoi.geojson --metric-crs EPSG:32614 --output-json outputs/demo_metrics.json --output-gpkg outputs/demo_matches.gpkg --independent-holdout
```

`infer` prints a plan without loading a model. The demo's predictions copy its reference outlines, so the perfect scores only show that files and metrics are handled correctly; they are not model accuracy. Open `outputs/demo_matches.gpkg` in QGIS or ArcGIS Pro to see the match layers.

## 3. Prepare your imagery

Use a local, north-up, georeferenced, 8-bit RGB GeoTIFF or GDAL VRT. A VRT can point at tiles you already have, so you don't need to build one huge mosaic. Raw JPEG or PNG tiles need georeferencing first.

```bash
nbf inspect path/to/aoi_rgb.tif
nbf tile path/to/aoi_rgb.tif --output data/prepared/pilot01 --tile-size 1024 --overlap 128
```

`nbf tile` writes overlapping GeoTIFF tiles and a `manifest.json` with the CRS, each tile's transform, the valid-data mask and a hash of every tile. It never resamples, reprojects or stretches colours. Rotated or non-8-bit input is refused with a message saying what to fix. `--bands 1 2 3` picks the RGB bands, and `--hash-source` also hashes the source file (a VRT's hash does not cover the rasters it points to).

Tile size and overlap are starting points, not best values. Big roofs need bigger tiles or more overlap, and more overlap means more disk space. `--overlap` is a minimum: every tile is full size, so the last row and column sit flush with the raster's edge and overlap their neighbours by more.

## 4. Run SAM 3

### GPU environment

SAM 3 runs in a separate Linux or WSL2 environment with an NVIDIA GPU. Don't install it into ArcGIS Pro's Python. Request model access and download the checkpoint from Meta yourself. With Python 3.12 and an NVIDIA driver in place:

```bash
cd /path/to/nearmap-building-footprints
bash scripts/setup_sam3.sh
source .venv-sam3/bin/activate
nbf doctor
```

The script pins Meta's code to commit `2345a4ad109ac29c569da749c91d84f10dc08c40` and installs PyTorch 2.10.0 and torchvision 0.25.0 for CUDA 12.8, and `segment-geospatial[samgeo3]==1.4.2`. Add `--training` for the fine-tuning dependencies. The CPU lock file is not a tested GPU lock, so check memory and package compatibility in your own GPU environment. For big runs, copy the data into the Linux filesystem: reading across from Windows is slow. Tile paths in a manifest are relative, so a manifest made on Windows works in WSL.

### Text prompt

Save the checkpoint as `models/sam3.pt` (or pass another path) and print a plan for five tiles:

```bash
nbf infer --manifest data/prepared/pilot01/manifest.json --method text --text building --checkpoint models/sam3.pt --output outputs/text_pilot01 --limit 5
```

Add `--execute` to run it. For the whole area, drop `--limit` and use a new output folder. `--resume` continues an interrupted run, but only with the same checkpoint, inputs and settings. `--allow-model-download` loads weights from Hugging Face instead of a local file; such runs cannot resume.

Each run writes `run.json` and, per tile, `raw/<tile>.gpkg` with a JSON receipt. Every mask stays its own polygon, holes included, and touching buildings are never merged. A polygon flagged `edge_touch` reached a tile edge and `nodata_touch` reached missing imagery; either way it may be cut off.

### Prompt files for the guided modes

Three modes take a single-layer GeoJSON or GeoPackage drawn in QGIS or ArcGIS. Coordinates are map coordinates in the file's own CRS, not tile pixels; the tool reprojects them to each tile.

| Mode | Geometry | Attributes | What it does |
| --- | --- | --- | --- |
| `text` | none | none | One phrase, such as `building`, applied to every tile. |
| `exemplar` | boxes (polygons) | `label`: 1 for a good example, 0 for a bad one; default 1 | Finds other objects in the same tile that look like the positive examples. |
| `box` | boxes (polygons) | all positive | Outlines the one building in each box; finds nothing outside the boxes. |
| `point` | points | `object_id`, and `label` 1 (inside) or 0 (outside) | Points sharing an `object_id` outline one building. Each group needs a positive point. |

```bash
nbf infer --manifest data/prepared/pilot01/manifest.json --method exemplar --prompts data/prompts/exemplars.gpkg --checkpoint models/sam3.pt --output outputs/exemplar_pilot01 --execute
nbf infer --manifest data/prepared/pilot01/manifest.json --method box --prompts data/prompts/boxes.gpkg --checkpoint models/sam3.pt --output outputs/box_pilot01 --execute
nbf infer --manifest data/prepared/pilot01/manifest.json --method point --prompts data/prompts/points.gpkg --checkpoint models/sam3.pt --output outputs/point_pilot01 --execute
```

Only boxes and point groups that fit wholly inside a tile are used, and examples only apply to their own tile. A tile without a complete positive prompt is recorded as `skipped_no_complete_positive_prompt`, which is different from a tile where the model found nothing, so check `run.json` before treating a run as complete coverage. Text and exemplar prompts run separately; they are not combined. The box and point modes' score is predicted mask quality, not the text detector's confidence, so tune cut-offs for each mode on its own. `nbf demo` writes example prompt files to look at. Report how many prompts you placed: guided results are not comparable with fully automatic ones.

## 5. Run Esri's model

### One raster

Use a clone of ArcGIS Pro's Python environment with Image Analyst and Esri's deep learning libraries, and download the Building Footprint Extraction – USA package (`.dlpk`) yourself. It expects orthorectified 8-bit RGB at about 10 to 40 cm per pixel. ([Esri's instructions](https://doc.arcgis.com/en/pretrained-models/latest/imagery/using-building-footprint-extraction-usa.htm))

```powershell
& 'C:\path\to\arcgis-clone\python.exe' src/nearmap_buildings/esri.py --raster data/imagery/aoi.tif --model 'C:\models\BuildingFootprintExtractionUSA.dlpk' --output 'C:\analysis\results.gdb\esri_buildings'
```

This prints the call; add `--execute` to run it. Create the geodatabase first and use a new feature class name, because Esri's tool appends to an existing one. Options: `--threshold`, `--batch-size` (a perfect square), `--padding`, `--processor CPU|GPU` and `--gpu-id`. The runner calls `arcpy.ia.DetectObjectsUsingDeepLearning` with polygon output, `NO_NMS` and no regularization, and checks the band count, pixel type and CRS before it starts.

### A whole city in chunks

One call over a city runs for hours with no way to resume, and ArcGIS Pro's GPU memory grows with every call in one process: the Lewisville run ran out of memory after 42 chunks. `nbf esri-chunks` splits the run into overlapping chunks, restarts ArcGIS Pro's Python every few chunks and merges the results without counting overlaps twice. Tile the raster with `nbf tile` first.

```powershell
nbf esri-chunks plan --manifest data/prepared/city/manifest.json --output data/prepared/city_chunks --core 10240 --overlap 512
nbf esri-chunks run --chunks data/prepared/city_chunks --gdb 'C:\analysis\esri_chunks.gdb' --done outputs/esri_city/done --model 'C:\models\BuildingFootprintExtractionUSA.dlpk' --per-process 8
nbf esri-chunks merge --gdb 'C:\analysis\esri_chunks.gdb' --chunks data/prepared/city_chunks --done outputs/esri_city/done --crs EPSG:26914 --min-score 0.9 --output outputs/esri_city/raw.gpkg
nbf clean --input outputs/esri_city/raw.gpkg --output outputs/esri_city/cleaned.gpkg --metric-crs EPSG:26914
```

- **plan** writes a small VRT per chunk. Each chunk has a core that no other core overlaps, plus `--overlap` pixels on every side to provide context for buildings near a core's edge. Buildings extending beyond that overlap can still be cut. A source that marks missing imagery with a mask or alpha band is refused; give it a nodata value and tile it again.
- **run** prints the ArcGIS command; add `--execute` to start. It runs `--per-process` chunks in one ArcGIS Pro Python process (`--arcgis-python`), then starts a fresh one. Each finished chunk leaves a done-marker, so rerunning the same command resumes. Use a separate `--done` folder and `--gdb` for each plan. The run stops after `--max-failures` processes in a row fail without finishing a chunk.
- **merge** keeps each detection only in the chunk whose core holds it, so nothing is counted twice. `--min-score` applies a frozen confidence cut-off, and `source_id` (`<chunk>:<OBJECTID>`) finds any outline again in ArcGIS Pro.

Lewisville took 150 chunks of 10,240 px with 512 px overlap and 8 h 46 min of model time on an RTX 4050.

### Import Esri output

Back in the CPU environment:

```powershell
nbf import-vectors --input 'C:\analysis\results.gdb' --layer esri_buildings --preset esri --crs EPSG:26914 --output outputs/esri_run01/raw.gpkg
```

The `esri` preset converts Esri's 0–100 `Confidence` to a 0–1 `score`, keeps each feature's OBJECTID as `source_id`, and leaves self-intersecting outlines for `nbf clean` to repair. It refuses scores that are already 0–1; use `--score-scale 1` for such a model. Then clean, score and compare Esri's outlines exactly like SAM 3's (sections 7 and 8).

## 6. Import other building layers

Any local building polygon layer, such as a Nearmap AI export, can join the comparison:

```powershell
nbf import-vectors --input data/inputs/nearmap_buildings.gpkg --layer buildings --output outputs/nearmap_ai.gpkg --crs EPSG:26914 --id-field building_id --score-field confidence
```

The layer needs a known CRS. Each polygon becomes one building in a `buildings` layer with `source_id` and `method` fields; other fields are dropped. Invalid polygons are refused unless `--allow-invalid` passes them on for `nbf clean` to repair. A confidence field is copied only when you name it and its values are 0–1 (use `--score-scale 100` for percentages); a score is never invented. Record the product, survey date and licence yourself.

## 7. Clean up outlines

Use a projected CRS in metres for your area; EPSG:32614 is only an example.

```bash
nbf clean --input outputs/text_pilot01/raw --output outputs/text_pilot01/cleaned.gpkg --metric-crs EPSG:32614 --min-area-m2 4 --simplify-m 0.15
```

Cleanup repairs invalid outlines, drops slivers and removes duplicates from overlapping tiles. Where two copies overlap, it keeps one that touches neither a tile edge nor missing imagery before comparing scores, because the copy cut by a tile edge can score higher than the whole building seen by the next tile. `--duplicate-priority score` ranks by score alone. Neighbouring buildings are never merged. Every removed outline is in the `removed_audit` layer with the reason.

**Long buildings.** A building longer than a tile is cut in every tile that sees it. To join the pieces, pass the tile manifest:

```bash
nbf clean --input outputs/text_pilot01/raw --output outputs/text_pilot01/cleaned_seams.gpkg --metric-crs EPSG:32614 --seam-merge-manifest data/prepared/pilot01/manifest.json
```

Two pieces from different tiles merge only when each is cut at its own tile's edge and they agree, at IoU `--seam-merge-iou` (default 0.5), inside the strip both tiles saw. Neighbours that only touch stay apart, and two pieces from one tile are never joined (they get `seam_merge_rejected`). The merged outline lists its pieces in `merged_cleanup_ids`. This needs the `tile_id` and `edge_touch` fields that `nbf infer` writes, so it doesn't apply to imported layers.

**Straight edges.** Regularization is off by default. To try it, add `--regularize --max-displacement-m 0.3 --max-area-change 0.05`; any change beyond those limits is undone and flagged. Curved and unusual roofs should not be forced into rectangles.

These thresholds are starting values. Measure their effect on validation data and freeze them before scoring the test area.

## 8. Score and compare

Draw an area of interest (AOI) in which every building has been outlined in the reference. Use the same imagery date, outline convention (roof or ground) and cleanup settings for every method, and keep it apart from any area used for tuning.

```bash
nbf evaluate --predictions outputs/text_pilot01/cleaned.gpkg --predictions-layer cleaned --reference data/reference/test_buildings.gpkg --aoi data/reference/test_aoi.gpkg --metric-crs EPSG:32614 --iou-threshold 0.5 --output-json outputs/text_pilot01/evaluation.json --output-gpkg outputs/text_pilot01/evaluation.gpkg --independent-holdout
```

Each predicted outline matches at most one reference building, choosing the most matches above the IoU threshold and then the best overlap. The report gives precision, recall, F1, overlap, area error and boundary distance, and the GeoPackage has matched, unmatched and excluded layers. Buildings crossing the AOI's edge are left out (`--edge-policy clip` clips them instead). A metric with nothing to count is `null`, not 0.

**By size.** `--size-bins-m2 20 50 100` adds size classes (under 20, 20–50, 50–100 and 100 m² and up) plus the same metrics for everything above each edge, such as F1 for buildings of 20 m² and up.

**Outdated reference.** If the only reference is an older map, use `--agreement-only` instead of `--independent-holdout`. The scores then measure agreement with that map, not accuracy, since an unmatched outline may be a real new building.

**Compare methods.** Once every method is scored on the same reference and AOI:

```bash
nbf compare --reports outputs/text_pilot01/evaluation.json outputs/esri_run01/evaluation.json --labels sam3_text esri --output outputs/comparison.csv --per-building outputs/per_building.gpkg
```

`nbf compare` refuses reports made with different references, AOIs or settings, and never mixes agreement-only reports with independent ones. If every report was made with the same `--size-bins-m2`, `--by-size FILE` adds a table by size class. `--per-building` writes one row per reference building with an `outcome` such as `found by both`, `only esri` or `missed by both`; style it in GIS to see where the methods disagree. It first checks that the reference file still has the same rows in the same order (pass `--reference` if it has moved). The demo's `esri_format_predictions.gdb` lets you try this whole sequence without ArcGIS.

**Compare two models without a reference.** When the reference is old, two models agreeing is the most useful signal:

```bash
nbf agree --predictions outputs/text_pilot01/cleaned.gpkg outputs/esri_run01/cleaned.gpkg --layers cleaned cleaned --labels sam3_text esri --reports outputs/text_pilot01/evaluation.json outputs/esri_run01/evaluation.json --aoi data/reference/test_aoi.gpkg --metric-crs EPSG:32614 --output-json outputs/agreement.json --output-gpkg outputs/agreement.gpkg
```

It matches the two methods' outlines one to one and writes `both` and `only_<label>` layers. With evaluation reports (`--reports`, or `-` for a method without one) it also lists `candidate_new`: outlines both models drew that the reference lacks, most likely new buildings. Review them before using any as labels; two models can make the same mistake.

**Turn an old map into a real reference.** An outdated map becomes an independent reference only after a person checks every outline in the AOI against the imagery, adding, removing and correcting buildings. Do that in GIS on a copy, never by copying model output, then check what changed:

```bash
nbf review-check --original data/reference/lewisville_2015.gpkg --reviewed data/reference/test_area_reviewed.gpkg --aoi data/reference/test_area_aoi.gpkg --metric-crs EPSG:26914 --reviewer "Your Name" --imagery-date 2026-05 --label-convention "roof outline; every structure of 4 m2 and up" --output-json outputs/accuracy/review_check.json --output-gpkg outputs/accuracy/review_check.gpkg
```

It counts outlines as `unchanged`, `modified`, `added` or `removed`, flags outlines evaluation would refuse and ones that overlap, and records the reviewer and imagery date. `ready_for_evaluation` is true once there are no problems. [`configs/accuracy.example.json`](../configs/accuracy.example.json) runs the whole accuracy check for the Lewisville test area in one go (see section 10).

## 9. Fine-tune SAM 3

Fine-tuning is optional and has not been run on real data yet. It needs exhaustive, reviewed building outlines.

**Prepare training data.** Supply the tile manifest, your outlines, and a polygon layer with a `split` field of `train`, `val` and `test`:

```bash
nbf train prepare-coco --manifest data/tiles/manifest.json --ground-truth data/labels/buildings.gpkg --split-aois data/labels/splits.gpkg --output data/training/buildings_v1 --labels-complete --min-split-distance-m 100
```

`--labels-complete` confirms that every building in those areas is outlined, since a missing one would teach the model it isn't a building. Splits are always by area, never random tiles; overlapping splits, buildings crossing a split and tiles in more than one split are refused (`--skip-unassigned` drops boundary tiles instead). The 100 m gap is only an example. Training tiles must have no missing imagery. Each split gets `images/*.png`, a COCO `annotations.json` with one `building` category (masks as RLE, which keeps holes) and `instances/*.tif` instance rasters. Everything is held in memory, so use a subset that fits in RAM.

**Train.** `configs/sam3_training.example.yaml` targets Meta's SAM 3 commit `2345a4ad109ac29c569da749c91d84f10dc08c40` and follows Meta's Roboflow recipe, validating on `val` and leaving `test` untouched. Batch size, epochs and memory still need a pilot run. In the Linux training environment:

```bash
cp configs/sam3_training.example.yaml /path/to/sam3/sam3/train/configs/buildings.yaml
export SAM3_DATASET_ROOT=/path/to/prepared/buildings_v1
export SAM3_LOG_DIR=/path/to/outputs/sam3-buildings-v1
export SAM3_BPE_PATH=/path/to/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
export SAM3_CHECKPOINT=/path/to/models/sam3.pt
nbf train launch --checkout /path/to/sam3 --config /path/to/sam3/sam3/train/configs/buildings.yaml --python /path/to/training-env/bin/python --num-gpus 1
```

This prints the plan; add `--execute` to train. The config must sit inside Meta's checkout, because Meta's launcher looks for it there. ([Meta's training guide](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/README_TRAIN.md))

**Export the weights.** A raw training checkpoint must not go straight to inference: its weights are stored under different names, so the inference loader would silently skip them. Export it first:

```bash
nbf train export-checkpoint --input /path/to/outputs/sam3-buildings-v1/checkpoints/checkpoint.pt --output /path/to/models/buildings-inference.pt
```

The export loads with `torch.load(weights_only=True)`, checks every tensor and writes inference-ready weights plus `buildings-inference.pt.metadata.json` with both files' hashes. Keep the original checkpoint to resume training. The fine-tuned model supports the **text** and **exemplar** modes only; use the original checkpoint for box and point. Score it on your reserved test tiles like any other method:

```bash
nbf infer --manifest data/test-tiles/manifest.json --method text --text building --checkpoint /path/to/models/buildings-inference.pt --bpe-path /path/to/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --output outputs/sam3_finetuned_text --limit 4
```

## 10. Run a whole experiment from a config file

Copy `configs/text.example.json` or `configs/guided.example.json` to a `*.local.json` file, set the paths, checkpoint and CRS, and run:

```bash
python scripts/run_experiment.py configs/text.local.json
python scripts/run_experiment.py configs/text.local.json --execute
```

The first command prints the steps; `--execute` runs them in order and stops at the first failure. Don't put tokens in config files.

## Troubleshooting

- **Short error messages.** Input problems print one line, `nbf <command>: error: ...`, and exit with status 2. Set `NBF_DEBUG=1` to see the full traceback.
- **PROJ database conflicts on Windows.** A PostGIS install can set `PROJ_LIB` or `PROJ_DATA` and break rasterio's projections. `nbf` clears these, and `GDAL_DATA`, for its own process only, leaving ArcGIS untouched. In your own Python scripts, clear them in that shell only.
