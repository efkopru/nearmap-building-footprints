# Building footprints from aerial imagery

[![CPU tests](https://github.com/efkopru/nearmap-building-footprints/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/efkopru/nearmap-building-footprints/actions/workflows/cpu-tests.yml)
![Python 3.12 | 3.13](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/efkopru/nearmap-building-footprints/blob/main/LICENSE)

**A reproducible GIS toolkit that extracts building footprints from aerial imagery with Meta's SAM 3, then compares extraction methods fairly against the same independent reference labels.**

`nbf` is a Python command-line tool. It tiles GeoTIFF or VRT mosaics, runs four SAM 3 prompting modes alongside an Esri Mask R-CNN baseline and imported vendor vectors, suppresses duplicate polygons from overlapping tiles, and scores each method with one-to-one matching. Every stage records hashes and receipts, so an output polygon can be traced back to the pixels, prompts, and model weights that produced it.

> **Status:** The CPU pipeline is implemented and covered by 146 automated tests, run in CI on Windows and Ubuntu with Python 3.12 and 3.13. The Esri integration has run in ArcGIS Pro on real aerial imagery. The SAM 3 and fine-tuning integrations target pinned upstream versions but have not yet run on a GPU. No accuracy results are claimed yet, and the figures are synthetic illustrations. [Details](#project-status)

![Workflow from local georeferenced imagery through tiling, SAM 3 and baseline extraction, polygon cleanup, and held-out evaluation, with an optional fine-tuning route](https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/workflow.png)

**Docs:** [Setup and usage](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/USAGE.md) · [Methodology](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/METHODOLOGY.md) · [Prompt formats](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/PROMPTS.md) · [Validation record](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/VALIDATION.md)

## The problem

Building footprints underpin insurance, urban planning, solar, and disaster-response work. Promptable models such as SAM 3 can now outline buildings from a phrase like `building`, and imagery vendors ship their own extractions. Choosing between them is harder than running a model, because the usual shortcuts produce misleading numbers:

| Pitfall | How this toolkit handles it |
| --- | --- |
| Tiled inference cuts buildings at tile seams and detects them twice in overlaps. | Full-size overlapping tiles, `edge_touch` and `nodata_touch` flags on cut masks, and audited duplicate suppression that keeps the complete copy and never dissolves neighboring buildings. |
| Random tile splits put the same building in both training and test data. | Spatial train, validation, and test AOIs, with checks for overlap, minimum separation, and buildings that cross split boundaries. |
| Greedy matching undercounts true positives, and AOI edge handling quietly shifts scores. | Maximum-cardinality one-to-one matching and an explicit, recorded AOI edge policy. |
| Confidence scores and vendor benchmarks are not accuracy on your imagery. | Every method is scored against the same independent labels, and reports refuse to be compared if their holdouts differ. |
| A roof outline is not a ground footprint. | The label convention is recorded per experiment. No smoothing or regularization claims to convert one into the other. |

![Synthetic aerial-style scene beside its reference building polygons, including a courtyard hole](https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/synthetic-example.png)

*Synthetic illustration: an invented scene and the reference polygons used to draw it. It contains no Nearmap imagery and no model output. See the [figure provenance](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/images/README.md) to reproduce it.*

## How it works

| Stage | Command | Produces |
| --- | --- | --- |
| 1. Prepare | `nbf inspect`, `nbf tile` | Full-size, overlapping GeoTIFF tiles and a `manifest.json` with CRS, transforms, valid-data masks, and a SHA-256 hash per tile. No resampling, reprojection, or contrast stretch. |
| 2. Extract | `nbf infer`, `esri.py` in ArcGIS Pro, `nbf import-vectors` | Raw per-tile SAM 3 polygons with scores, `edge_touch` and `nodata_touch` flags, and JSON receipts, or baseline polygons from ArcGIS or an existing export. |
| 3. Clean | `nbf clean` | A `cleaned` layer, with duplicates resolved in favor of complete detections, plus a `removed_audit` layer recording why each polygon was dropped. Regularization is opt-in and bounded. |
| 4. Evaluate | `nbf evaluate` | Precision, recall, F1, matched IoU, area error, and boundary distance as JSON, with matched, unmatched, and excluded features as GeoPackage layers. |
| 5. Compare | `nbf compare` | One CSV across methods, written only when every report used the same reference labels, AOI, and settings, plus an optional map layer of which methods found each labeled building. |
| Optional | `nbf train prepare-coco`, `launch`, `export-checkpoint` | A leakage-checked COCO dataset, a run of Meta's official SAM 3 trainer, and inference-ready weights. |

### Extraction methods

| Method | Human input | Comparison group |
| --- | --- | --- |
| SAM 3 text | A phrase such as `building` | Automated |
| SAM 3 visual exemplars | Positive and negative example boxes in each tile | Guided concept detection |
| SAM 3 instance boxes | One box per building | Assisted delineation |
| SAM 3 grouped points | Foreground and background points per building | Assisted delineation |
| Fine-tuned SAM 3 | Spatially separated training labels | Separate trained experiment (text and exemplar inference) |
| Esri Building Footprint Extraction USA | A downloaded model and licensed ArcGIS Pro | Automated baseline (Mask R-CNN) |
| Imported vectors, such as Nearmap AI | An existing local export | Vendor product, with no local inference |

Assisted results are reported separately from automated extraction, together with the prompting effort they required.

### Comparing Esri with other models

Esri's model runs in ArcGIS Pro and writes a file geodatabase. `nbf import-vectors --preset esri` brings that output into the same pipeline as SAM 3: it converts Esri's 0-100 `Confidence` to a 0-1 score, keeps each feature's OBJECTID, and leaves self-intersecting outlines for cleanup to repair and record. Both methods are then cleaned with the same settings and evaluated against the same labels.

`nbf compare --per-building` writes a GeoPackage with one polygon per labeled building, showing which methods found it (`found by both`, `only esri`, `only sam3_text`, or `missed by both`) and each match's IoU. It first checks that every report was scored on the same reference rows in the same order. The [usage guide](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/USAGE.md#compare-esri-with-other-models) has the full command sequence, and the synthetic demo below runs it without ArcGIS.

## Engineering highlights

- **Optimal one-to-one matching.** Predictions and reference buildings are matched for maximum cardinality at the IoU threshold, then maximum total IoU, using SciPy's `linear_sum_assignment` on each connected overlap component instead of one city-sized matrix. [A test](https://github.com/efkopru/nearmap-building-footprints/blob/main/tests/test_evaluation.py) pins a case where greedy highest-IoU matching finds fewer true positives. ([`evaluation.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/evaluation.py))
- **Georeferencing preserved end to end.** Tiles keep the source CRS, affine transform, and valid-data mask. Masks are polygonized through each tile's own transform, keeping courtyard holes and keeping touching buildings as separate instances. ([`preprocessing.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/preprocessing.py), [`inference.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/inference.py))
- **Seam-aware duplicate suppression.** Overlapping tiles often see a building twice: once whole, and once cut by a tile edge, where the cut copy can score higher. Cleanup keeps detections that touch neither a tile edge nor missing imagery before comparing scores, never dissolves neighbors, and records every suppression in an audit layer. ([`postprocess.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/postprocess.py))
- **Traceable, resumable runs.** Tile, prompt, manifest, and checkpoint hashes form a run signature. `--resume` refuses to continue after any input or setting changes, and per-tile receipts with atomic writes make an interrupted run safe to pick up. ([`inference.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/inference.py))
- **Leakage-safe training data.** Training data comes from user-drawn spatial AOIs, never random tile splits. Overlapping or too-close splits and buildings that cross them are rejected. Output is COCO RLE, which keeps holes, plus `uint32` instance rasters that stay correct beyond 255 buildings per tile. ([`training.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/training.py))
- **Comparable by construction.** Evaluation reports carry order-independent SHA-256 fingerprints of the predictions, reference labels, and AOI, and `nbf compare` refuses to tabulate reports whose holdout or settings differ. The per-building view joins reports on reference row positions, so it also checks a row-order fingerprint: a re-sorted reference file would otherwise put each method's results on the wrong buildings. ([`evaluation.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/evaluation.py), [`compare.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/compare.py))
- **Safe defaults.** Inference, training, and ArcGIS calls are dry runs until `--execute`. Checkpoints load with `torch.load(weights_only=True)`, and commands refuse to overwrite their inputs. Fine-tuned weights go through an explicit export step, because the inference loader would otherwise accept a trainer checkpoint while silently ignoring its weights. ([`training.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/training.py))
- **Real-world GIS environments.** The CLI clears inherited `PROJ_LIB`, `PROJ_DATA`, and `GDAL_DATA` for its own process, so a system PostGIS install cannot override the projection database bundled with the wheels, while ArcPy keeps its own environment. ([`cli.py`](https://github.com/efkopru/nearmap-building-footprints/blob/main/src/nearmap_buildings/cli.py))

## Tech stack

| Area | Tools |
| --- | --- |
| Geospatial | Rasterio and GDAL, GeoPandas, Shapely 2, PyProj, pyogrio |
| Numerical | NumPy, SciPy (optimal assignment, sparse connected components), pandas |
| Segmentation | Meta SAM 3 through SamGeo 1.4.2, PyTorch 2.10 with CUDA 12.8 on Linux or WSL2, Hydra training configs |
| Baselines | ArcGIS Pro and ArcPy Image Analyst (Esri Mask R-CNN), vendor vector exports |
| Quality | pytest, coverage.py, and ruff; GitHub Actions on Windows and Ubuntu with Python 3.12 and 3.13 and SHA-pinned actions; locked CPU dependencies |

## Quick start

### 1. Install the CPU toolkit

Python 3.12 or 3.13 is required. On Windows PowerShell, the setup script uses 3.12, creates `.venv`, installs the locked dependencies, and runs the test suite:

```powershell
git clone https://github.com/efkopru/nearmap-building-footprints.git
Set-Location nearmap-building-footprints
.\scripts\setup_cpu.ps1
.\.venv\Scripts\Activate.ps1
```

On Linux:

```bash
git clone https://github.com/efkopru/nearmap-building-footprints.git
cd nearmap-building-footprints
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock
python -m pip install --no-deps -e .
```

### 2. Run the synthetic pipeline

No model, GPU, or real imagery is needed:

```bash
nbf doctor
nbf demo --output data/demo
nbf tile data/demo/imagery.tif --output data/demo_tiles --tile-size 256 --overlap 64
nbf infer --manifest data/demo_tiles/manifest.json --method text --output outputs/demo_plan
nbf clean --input data/demo/perfect_predictions.gpkg --output outputs/demo_cleaned.gpkg --metric-crs EPSG:32614
nbf evaluate --predictions outputs/demo_cleaned.gpkg --predictions-layer cleaned --reference data/demo/reference.gpkg --aoi data/demo/aoi.geojson --metric-crs EPSG:32614 --output-json outputs/demo_metrics.json --output-gpkg outputs/demo_matches.gpkg --independent-holdout
```

`infer` prints an inference plan without loading a model; adding `--execute` runs it in a GPU environment. The demo's predictions are copies of its reference polygons, so the perfect scores only confirm file handling and metric calculations. They are **not** model accuracy. Open `outputs/demo_matches.gpkg` in QGIS or ArcGIS to inspect the match layers. Output paths must be new, because commands refuse to overwrite earlier results by default.

### 3. Compare Esri-format output with another method

The demo also writes `esri_format_predictions.gdb`: synthetic predictions in the layout Esri's tool produces, with a duplicate detection, a self-intersecting outline, a missed building, and a false detection. Continuing from step 2:

```bash
nbf import-vectors --input data/demo/esri_format_predictions.gdb --layer esri_buildings --preset esri --crs EPSG:32614 --output outputs/demo_esri/raw.gpkg
nbf clean --input outputs/demo_esri/raw.gpkg --output outputs/demo_esri/cleaned.gpkg --metric-crs EPSG:32614
nbf evaluate --predictions outputs/demo_esri/cleaned.gpkg --predictions-layer cleaned --reference data/demo/reference.gpkg --aoi data/demo/aoi.geojson --metric-crs EPSG:32614 --output-json outputs/demo_esri/evaluation.json --independent-holdout
nbf compare --reports outputs/demo_metrics.json outputs/demo_esri/evaluation.json --labels identity esri --output outputs/demo_comparison.csv --per-building outputs/demo_per_building.gpkg
```

Open `outputs/demo_per_building.gpkg` and style it by `outcome` to find the building only one method detected. The fixtures are synthetic, so this shows how the comparison works, not how Esri's model performs.

### 4. Move to real imagery

Supply a local, north-up, georeferenced **8-bit RGB GeoTIFF or VRT**, and start with a small area that has independent reference labels:

```bash
nbf inspect path/to/pilot_rgb.tif
nbf tile path/to/pilot_rgb.tif --output data/prepared/pilot01 --tile-size 1024 --overlap 128
```

SAM 3 runs in a separate Linux or WSL2 environment with an NVIDIA GPU. `scripts/setup_sam3.sh` builds it from a pinned SAM 3 commit; model access and the checkpoint are obtained from Meta separately. A five-tile pilot:

```bash
nbf infer --manifest data/prepared/pilot01/manifest.json --method text --text building --checkpoint models/sam3.pt --output outputs/text_pilot01 --limit 5 --execute
```

The [usage guide](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/USAGE.md) covers guided prompts, resume behavior, cleanup, evaluation, and fine-tuning.

## Project status

**Verified**

- Every push runs lint, the test suite with coverage, and a synthetic end-to-end run of every CPU pipeline stage (demo data, tiling, inference plan, cleanup, evaluation, vector and Esri-format import, per-building comparison, and COCO preparation), on Windows and Ubuntu with Python 3.12 and 3.13.
- All four SAM 3 adapter modes run against a fake model, exercising prompt handling, georeferencing, and output plumbing. Checkpoint export is tested against a mocked Torch interface.
- The adapter was checked against the SamGeo 1.4.2 source, and the training recipe against Meta's pinned SAM 3 commit.
- `esri.py` ran Esri's model in ArcGIS Pro on a GPU over real aerial imagery, and its output passed through import and cleanup. That first real run exposed a validation bug, now fixed and tested.

**Not yet run**

- SAM 3 on a GPU, fine-tuning, and a scored comparison on real imagery.
- As a result, there are **no accuracy results** yet. Synthetic fixtures verify data handling and metric calculations, and their perfect scores are expected by construction.

**Next steps**

1. Label a pilot area and a spatially separate test area, then run SAM 3 text inference on a GPU.
2. Tune prompts, thresholds, and cleanup on validation data only, then freeze them.
3. Evaluate every method on the held-out test area and publish aggregate results in the [comparison report format](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/METHODOLOGY.md#comparison-report-template).

## Development

With the CPU environment active, run the same checks as CI:

```bash
python -m pytest -q
python -m coverage run -m pytest -q
python -m coverage combine
python -m coverage report
python -m pip install ruff==0.16.9
ruff check .
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/nearmap_buildings/` | CLI, tiling, SAM 3 adapter, cleanup, evaluation, comparison, training, and baseline adapters |
| `tests/` | CPU tests for geometry, raster I/O, adapters, matching, spatial splits, and checkpoint formats |
| `scripts/` | CPU and GPU environment setup, a config-driven experiment runner, and the figure generator |
| `configs/` | Example experiment runs and the SAM 3 training recipe |
| `docs/` | Usage, methodology, prompt formats, optional methods, sources, and the validation record |

## Documentation

- [Setup and usage](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/USAGE.md): installation, every command, and GPU setup
- [Methodology](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/METHODOLOGY.md): label conventions, spatial splits, metrics, and reporting templates
- [Prompt formats](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/PROMPTS.md): the GIS data contract for exemplar, box, and point prompts
- [Optional methods](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/OPTIONAL_METHODS.md): the Esri baseline, vector imports, SAM 3 fine-tuning, and checkpoint export
- [Sources](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/SOURCES.md): primary references and the pinned upstream revisions
- [Validation record](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/VALIDATION.md): what has been executed, and what has not

## License and third-party terms

The code in this repository is released under the [MIT License](https://github.com/efkopru/nearmap-building-footprints/blob/main/LICENSE). No imagery, labels, checkpoints, or model outputs are included, and `data/`, `outputs/`, and model files are ignored by Git. The toolkit makes no imagery downloads or paid API calls; it works only on data you are already authorized to use. This is an independent project, not affiliated with or endorsed by Nearmap, Meta, Esri, or the SamGeo project. Their software, models, and data remain under their respective terms.
