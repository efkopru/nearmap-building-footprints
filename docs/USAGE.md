# Setup and usage

A local project for preparing georeferenced imagery, running SAM 3, cleaning building polygons, and comparing independently labeled results. Source imagery is read, never changed. No Nearmap imagery, API key, checkpoint, or real inference result is included.

**Start with a small labeled area.** Complete the CPU smoke test first, then run a limited GPU pilot before processing a full mosaic. This project prepares the methods; it does not claim that SAM 3 has been tested on your imagery.

## Included methods

| Method | Command | What it does |
| --- | --- | --- |
| SAM 3 text | `nbf infer --method text` | Finds building instances from a text concept. |
| SAM 3 visual exemplars | `nbf infer --method exemplar` | Uses positive/negative example boxes in the current tile to discover matching objects. |
| SAM 3 individual boxes | `nbf infer --method box` | Delineates individual buildings identified by supplied bounding boxes. |
| SAM 3 grouped points | `nbf infer --method point` | Delineates one building per object_id using foreground/background points. |
| SAM 3 fine-tuning | `nbf train prepare-coco` / `launch` / `export-checkpoint` | Prepares spatially separated COCO instance masks, launches Meta's training recipe, and converts trained weights for concept inference. |
| Esri USA baseline | ArcGIS Python + `src/nearmap_buildings/esri.py`, then `nbf import-vectors --preset esri` | Runs an already downloaded Esri building model in its separate licensed environment, then brings its output into the comparison. |
| Nearmap AI / other vector baseline | `nbf import-vectors` | Imports an existing local building export for the same cleanup/evaluation workflow. |

All `infer`, training `launch`, and Esri calls default to **dry run**. Add `--execute` to run the model. No paid imagery/API requests are implemented. The importer consumes an export you already have.

Read [METHODOLOGY.md](METHODOLOGY.md) for the experimental design, [PROMPTS.md](PROMPTS.md) for vector prompt formats, and [OPTIONAL_METHODS.md](OPTIONAL_METHODS.md) for Esri, imported vectors, and fine-tuning. [SOURCES.md](SOURCES.md) records primary references and the audited source version.

## 1. CPU environment on Windows

Use Python **3.12** for this project; CI also tests 3.13. In PowerShell:

```powershell
Set-Location path\to\nearmap-building-footprints
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\Activate.ps1
nbf doctor
```

`scripts/setup_cpu.ps1` performs the same creation/install/test steps. Activation is optional: use `.\.venv\Scripts\nbf.exe` wherever the examples use `nbf`.

For the CPU workflow on Linux, start in the cloned repository and run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock
python -m pip install --no-deps -e .
python -m pytest -q
nbf doctor
```

These commands install only the CPU dependencies. Use the separate GPU environment below for SAM 3.

Use the `nbf` entry point for GIS commands. It clears inherited `PROJ_LIB`, `PROJ_DATA`, and `GDAL_DATA` **within that child Python process** so a system PostGIS installation cannot override the wheel's projection database. It leaves ArcPy commands untouched. No machine environment settings are changed.

Input problems, such as an invalid setting, an unreadable file, or an unknown CRS, print a one-line `nbf <command>: error: ...` message and exit with status 2. Set `NBF_DEBUG=1` to see the full traceback instead. Unexpected failures always show their traceback.

## 2. CPU smoke test, no model or real imagery

```powershell
nbf demo --output data/demo
nbf inspect data/demo/imagery.tif
nbf tile data/demo/imagery.tif --output data/demo_tiles --tile-size 256 --overlap 64
nbf infer --manifest data/demo_tiles/manifest.json --output outputs/demo_plan --method text
nbf clean --input data/demo/perfect_predictions.gpkg --output outputs/demo_cleaned.gpkg --metric-crs EPSG:32614
nbf evaluate --predictions outputs/demo_cleaned.gpkg --predictions-layer cleaned --reference data/demo/reference.gpkg --aoi data/demo/aoi.geojson --metric-crs EPSG:32614 --output-json outputs/demo_metrics.json --output-gpkg outputs/demo_matches.gpkg --independent-holdout
nbf train prepare-coco --manifest data/demo_tiles/manifest.json --ground-truth data/demo/reference.gpkg --split-aois data/demo/splits.geojson --output data/demo_training --labels-complete --skip-unassigned
```

For this smoke test only, `--independent-holdout` exercises the evaluator's explicit confirmation gate; the data are **synthetic identity fixtures, not independent evidence or model accuracy**. Predictions intentionally copy reference geometry, so perfect identity metrics are expected and establish only file/metric plumbing. Keep this output separate from real benchmark reports. Every output path must be new; choose a new run suffix instead of deleting earlier work.

## 3. Prepare your Nearmap raster

Point to a local georeferenced, north-up, unsigned 8-bit RGB GeoTIFF or GDAL VRT. A VRT can reference your already downloaded tiles without creating another massive mosaic. Preserve the acquisition date and actual ground resolution. Raw JPEG/PNG tiles need correct georeferencing before this stage.

```powershell
nbf inspect 'D:\Nearmap\aoi_rgb.tif'
nbf tile 'D:\Nearmap\aoi_rgb.tif' --output data/prepared/pilot01 --tile-size 1024 --overlap 128 --bands 1 2 3
```

The tiler writes overlapping GeoTIFFs and `manifest.json`, preserves valid-data masks, records CRS/transforms, and hashes each prepared tile. It does not stretch radiometry, reproject, or resample implicitly. Non-8-bit or rotated inputs fail with a specific preparation requirement. `--hash-source` optionally hashes the source file; a VRT hash alone does not hash its referenced rasters. Prepared tile hashes capture the actual pixels read.

Tile size and overlap are starting parameters, not universal optimums. Large roofs may require larger windows or overlap. Disk use increases with overlap. `--overlap` is a minimum: every tile is full size unless the raster itself is smaller, so the last row and column of tiles sit flush with the raster edge and can overlap their neighbors by more.

Inference flags each raw polygon whose mask reached a processing-tile boundary (`edge_touch`) or missing imagery (`nodata_touch`). Either flag means the polygon may be cut off. A polygon without them is not thereby proven complete.

## 4. SAM 3 GPU environment

Use a **separate Linux/WSL2 environment with a compatible NVIDIA CUDA setup**. Do not install the SAM stack into ArcGIS Pro's environment. SAM 3 model access and a checkpoint must be obtained through Meta/Hugging Face; this project does not accept those terms or download weights for you.

In Linux/WSL2, with Python 3.12 and an appropriate NVIDIA driver already available:

```bash
cd /path/to/nearmap-building-footprints
bash scripts/setup_sam3.sh
source .venv-sam3/bin/activate
nbf doctor
```

Add `--training` to the setup script when preparing fine-tuning dependencies. Setup follows the audited Meta recipe, pins Meta code to `2345a4ad109ac29c569da749c91d84f10dc08c40`, installs PyTorch 2.10.0 with torchvision 0.25.0 (CUDA 12.8 wheels), and installs `segment-geospatial[samgeo3]==1.4.2`. The CPU dependency lock is **not** a tested CUDA lock. GPU package compatibility, memory requirements, and actual checkpoint execution must be verified in your GPU environment. Keep Windows `.venv` and Linux `.venv-sam3` separate.

For substantial GPU work, copying the project/data to the Linux filesystem avoids cross-filesystem I/O overhead. A Windows-generated manifest contains a Windows source path for provenance, but tile paths are relative to the manifest and remain usable in WSL/Linux. Pass all CLI paths using the current operating system's syntax.

## 5. Run text and guided methods

Save your checkpoint at `models/sam3.pt`, or supply another local path. First print a plan:

```bash
nbf infer --manifest data/prepared/pilot01/manifest.json --method text --text building --checkpoint models/sam3.pt --output outputs/text_pilot01 --limit 5
```

Add `--execute` to run the pilot. Remove `--limit` and use a **new output directory** for the full area. The checkpoint is hashed for provenance; `--resume` requires that same local checkpoint and an unchanged configuration/input signature. `--allow-model-download` is available only when you explicitly choose Hugging Face loading; remote-weight runs cannot resume without a local checkpoint.

Use GIS-created prompts according to [PROMPTS.md](PROMPTS.md):

```bash
nbf infer --manifest data/prepared/pilot01/manifest.json --method exemplar --prompts data/prompts/exemplars.gpkg --checkpoint models/sam3.pt --output outputs/exemplar_pilot01 --execute
nbf infer --manifest data/prepared/pilot01/manifest.json --method box --prompts data/prompts/boxes.gpkg --checkpoint models/sam3.pt --output outputs/box_pilot01 --execute
nbf infer --manifest data/prepared/pilot01/manifest.json --method point --prompts data/prompts/points.gpkg --checkpoint models/sam3.pt --output outputs/point_pilot01 --execute
```

Concept examples are local to each tile. Guided modes can skip tiles without complete positive prompts. The receipt distinguishes `skipped_no_complete_positive_prompt` from a completed inference with zero detections. Review this coverage before evaluating an entire AOI. The point/box confidence score is predicted mask quality and is not calibrated to the text detector's confidence; tune thresholds on validation data per method.

Each run creates `run.json` and `raw/<tile>.gpkg` plus matching JSON receipts. Individual mask geometries and holes are retained; no binary union joins touching buildings. The runner does not save a full-area raster mask. Keep these raw vectors for diagnosing seams and merges.

Fine-tuned checkpoints require the explicit `nbf train export-checkpoint` step described in the optional-methods guide. Native trainer checkpoints use different weight keys from Meta's public inference checkpoint. The runner checks this format and the exported file's metadata/hash; the supplied fine-tuning recipe supports **text and exemplar inference only**, not fine-tuning the separate interactive box/point branch.

## 6. Reconcile overlap and clean polygons

Choose the correct local **projected CRS with metre units**. EPSG:32614 below is illustrative, not a universal choice.

```bash
nbf clean --input outputs/text_pilot01/raw --output outputs/text_pilot01/cleaned.gpkg --metric-crs EPSG:32614 --min-area-m2 4 --simplify-m 0.15
```

Cleanup removes duplicates using IoU/containment thresholds. Where copies overlap, it keeps a complete detection before one flagged `edge_touch` or `nodata_touch`, then the higher score: along a tile seam, the copy cut by one tile's edge can outscore the whole building seen by the neighboring tile. `--duplicate-priority score` ranks by score alone. Cleanup does not union adjacent roofs or promise to reconstruct every clipped building; a building cut in every tile that sees it stays in pieces. Review tile-edge features and conflicting overlaps in GIS. `removed_audit` records exclusions; inputs are retained.

Regularization is **off by default**. To test it, add `--regularize --max-displacement-m 0.3 --max-area-change 0.05` and review rejected/accepted changes. Curves and unusual roof shapes should not be forced into rectangles. With regularization on, the limits apply to the total change from the repaired raw polygon. If regularizing a simplified polygon would exceed them, the simplification alone is kept when it fits (`combined_change_rejected`); otherwise both are undone (`simplify_reverted`). Each polygon's `cleanup_flags` lists what was applied.

The thresholds above are pilot values. Measure their effects on validation data and freeze them before final test evaluation.

## 7. Evaluate the same held-out area

Create a polygon AOI inside which every building has been labeled independently. Match imagery dates and the target definition (roof outline or ground footprint). Exclude training/prompt-tuning areas. Use the same test coverage and cleanup policy across methods.

```bash
nbf evaluate --predictions outputs/text_pilot01/cleaned.gpkg --predictions-layer cleaned --reference data/reference/test_buildings.gpkg --aoi data/reference/test_aoi.gpkg --metric-crs EPSG:32614 --iou-threshold 0.5 --output-json outputs/text_pilot01/evaluation.json --output-gpkg outputs/text_pilot01/evaluation.gpkg --independent-holdout
```

Evaluation uses one-to-one matching, maximum match count above the IoU threshold, then IoU as the tie-break. It reports precision/recall/F1, matched overlap, area errors, and explicitly defined boundary-distance measures. Default AOI policy excludes polygons crossing the AOI boundary; `--edge-policy clip` is an explicit alternative. Inspect matched, unmatched, and excluded layers. Missing metrics for empty denominators are JSON null, not misleading zeros. The report also fingerprints the predictions and records each input's path and layer, so a result can be traced back to the file that produced it.

Aggregate method reports only after evaluating the same reference and AOI:

```bash
nbf compare --reports outputs/text_pilot01/evaluation.json outputs/esri_pilot01/evaluation.json --output outputs/comparison.csv
```

The comparator checks holdout geometry fingerprints and evaluation settings before writing the table. It cannot establish label independence or matching capture dates; record those decisions in the experiment metadata.

### Compare Esri with other models

Run Esri's model in ArcGIS Pro as described in [OPTIONAL_METHODS.md](OPTIONAL_METHODS.md#esri-building-footprint-extraction-usa-baseline), then give its output the same treatment as SAM 3's:

```bash
nbf import-vectors --input /path/to/results.gdb --layer esri_buildings --preset esri --crs EPSG:32614 --output outputs/esri_pilot01/raw.gpkg
nbf clean --input outputs/esri_pilot01/raw.gpkg --output outputs/esri_pilot01/cleaned.gpkg --metric-crs EPSG:32614 --min-area-m2 4 --simplify-m 0.15
nbf evaluate --predictions outputs/esri_pilot01/cleaned.gpkg --predictions-layer cleaned --reference data/reference/test_buildings.gpkg --aoi data/reference/test_aoi.gpkg --metric-crs EPSG:32614 --iou-threshold 0.5 --output-json outputs/esri_pilot01/evaluation.json --output-gpkg outputs/esri_pilot01/evaluation.gpkg --independent-holdout
nbf compare --reports outputs/text_pilot01/evaluation.json outputs/esri_pilot01/evaluation.json --labels sam3_text esri --output outputs/comparison.csv --per-building outputs/per_building.gpkg
```

Use the same cleanup and evaluation settings as for the other methods. `--per-building` writes a `buildings` layer with one row per evaluated reference building. Each row has the building's original attributes and outline, `<label>_match_iou` and `<label>_match_id` for each method (NaN and -1 when missed), `found_by`, and an `outcome` such as `found by both`, `only esri`, or `missed by both`. Style `outcome` in QGIS or ArcGIS Pro to see where the methods disagree on the same labeled buildings. Each method's false detections are in the `unmatched_predictions` layer of its own evaluation GeoPackage.

Reports refer to reference buildings by row position, so the comparison first checks that the reference file still holds the same rows in the same order. It reads the reference file recorded in the reports; pass `--reference` if the file has moved. The demo includes `esri_format_predictions.gdb`, synthetic predictions in Esri's output layout, so the whole sequence can be tried without ArcGIS.

**A valid polygon is not proof of an accurate ground footprint.** Roof overhang and displacement require a separate target/geometry decision. Confidence scores are not measured accuracy.

## Configuration files and reruns

Copy `configs/text.example.json` or `configs/guided.example.json` to a `*.local.json` file and edit input paths, output run names, checkpoint, and CRS. The experiment runner prints its steps by default:

```powershell
python scripts/run_experiment.py configs/text.local.json
python scripts/run_experiment.py configs/text.local.json --execute
```

It runs commands without a shell and stops on the first failed stage. Do not include tokens in configuration. A complete experiment can include a separately reviewed evaluation step; the examples stop before claiming any independent reference is available.

## Validation status

See [VALIDATION.md](VALIDATION.md) for executed checks and limits. CPU geometry/data tests and synthetic adapters are distinct from actual model inference. The project does not publish local data.
