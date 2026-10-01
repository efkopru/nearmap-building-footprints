# Optional baselines and SAM 3 fine-tuning

These commands operate on local files. They do not download Nearmap imagery or vectors, call a paid API, install ArcGIS libraries, download model weights, or start training by default. Use imagery and labels you are permitted to process. Keep imagery, exports, prepared datasets, checkpoints, and predictions in the ignored data/output directories.

## Esri Building Footprint Extraction USA baseline

Use your ArcGIS Pro cloned Python environment with Image Analyst and the matching Esri deep learning libraries already installed. Download the USA `.dlpk` yourself. The supported input is orthorectified RGB, unsigned 8-bit, with approximately 10 to 40 cm pixels; off-nadir imagery is not equivalent input. The runner verifies band count, pixel type, and known CRS on execution. Review resolution and orthorectification yourself. [Esri model instructions](https://doc.arcgis.com/en/pretrained-models/latest/imagery/using-building-footprint-extraction-usa.htm)

From this project's root in PowerShell, replace the example paths:

```powershell
& 'C:\path\to\arcgis-clone\python.exe' src/nearmap_buildings/esri.py --raster data/imagery/aoi.tif --model 'C:\models\BuildingFootprintExtractionUSA.dlpk' --output 'C:\analysis\results.gdb\esri_buildings'
```

This prints the call without importing ArcPy. Add `--execute` to run it. Create the output geodatabase beforehand and use a new feature-class name. Optional arguments are `--threshold`, `--batch-size` (a perfect square), `--padding`, `--processor CPU|GPU`, and `--gpu-id`. Padding must fit the selected package's tile size.

The runner uses `arcpy.ia.DetectObjectsUsingDeepLearning` with the local package, polygon output (`return_bboxes False`), `NO_NMS`, and mosaicked-image processing. It adds `method=esri_building_usa`. Output is the raw polygon baseline, without regularization. It refuses existing outputs because this Esri tool can append detections. Postprocess and evaluate with the same settings used for the other methods. [Current ArcPy interface](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/detect-objects-using-deep-learning.htm)

### A whole city in chunks

One call over a city runs for hours with no way to resume, and ArcGIS Pro's GPU memory grows with each call in one process: in the Lewisville run the model failed to load with a CUDA out-of-memory error after 42 chunks. `nbf esri-chunks` splits the run into overlapping chunks, restarts ArcGIS Pro's Python every few chunks, and merges the results without counting overlaps twice. Tile the raster with `nbf tile` first; the plan covers only tiles with imagery.

```powershell
nbf esri-chunks plan --manifest data/prepared/city/manifest.json --output data/prepared/city_chunks --core 10240 --overlap 512
nbf esri-chunks run --chunks data/prepared/city_chunks --gdb 'C:\analysis\esri_chunks.gdb' --done outputs/esri_city/done --model 'C:\models\BuildingFootprintExtractionUSA.dlpk' --per-process 8
nbf esri-chunks merge --gdb 'C:\analysis\esri_chunks.gdb' --chunks data/prepared/city_chunks --done outputs/esri_city/done --crs EPSG:26914 --min-score 0.9 --output outputs/esri_city/raw.gpkg
nbf clean --input outputs/esri_city/raw.gpkg --output outputs/esri_city/cleaned.gpkg --metric-crs EPSG:26914
```

- **plan** writes one small VRT per chunk, reading a window of the source raster, and `chunks.json`. Each chunk has a core, in source pixels, that no other chunk's core overlaps, plus `--overlap` pixels on every side so buildings on a core's edge are seen whole.
- **run** prints the ArcGIS command; add `--execute` to start it. It runs `esri.py --chunks` in ArcGIS Pro's Python (`--arcgis-python`, by default the standard install path) for `--per-process` chunks at a time, then starts a fresh process. Each finished chunk writes a done-marker, so rerunning the same command resumes; a chunk interrupted midway is run again from the start. The run stops after `--max-failures` failed processes in a row. The detection options are those of `esri.py`. `nbf` clears inherited PROJ and GDAL settings for its own libraries but passes them on to ArcGIS Pro.
- **merge** imports each chunk with the `esri` preset and keeps a detection only in the chunk whose core holds its representative point, decided in the raster's own CRS with half-open core bounds, so every detection belongs to exactly one chunk. Empty outlines are dropped and counted. `--min-score` applies a frozen confidence cut-off. `source_id` is `<chunk>:<OBJECTID>`, so every outline can be found again in ArcGIS Pro.

Lewisville: 150 chunks of 10,240 px cores with 512 px overlap, 8 h 46 min of model time on an RTX 4050, 98,307 raw detections, 83,530 kept by core ownership, 41,751 at ≥ 0.9 and 39,840 after cleanup. That run used the notebook version of these steps ([notebook 3](../notebooks/03_run_esri.ipynb)), which the commands now package.

### Bring Esri output into the comparison

Back in the CPU environment, import the feature class with the `esri` preset:

```powershell
nbf import-vectors --input 'C:\analysis\results.gdb' --layer esri_buildings --preset esri --crs EPSG:26914 --output outputs/esri_run01/raw.gpkg
```

The preset labels the method `esri_building_usa` and converts Esri's `Confidence` field to a 0-1 `score`. The Mask R-CNN inference code in `arcgis.learn` writes `Confidence` as a percentage: `score * 100` in `arcgis/learn/models/_inferencing/_maskrcnn_inferencing.py`, checked in arcgis 2.4.3. Esri's generic detector template writes 0-1 instead, so the importer refuses to divide values that are already at most 1; use `--score-scale 1` for such a model. The preset also keeps self-intersecting outlines, which raster-to-polygon conversion can produce, for `nbf clean` to repair and record, exactly as for SAM 3 masks. `source_id` holds each feature's OBJECTID, so any polygon can be found again in ArcGIS Pro.

Then clean with the same settings as the other methods, evaluate against the same reference and AOI, and compare. Because `NO_NMS` keeps the model's overlapping-chip duplicates, cleanup's score-ranked duplicate suppression applies to Esri output too. [Compare Esri with other models](USAGE.md#compare-esri-with-other-models) walks through the whole sequence.

## Import an existing Nearmap AI building export

Use a locally exported building polygon layer, not a mixed AI-feature layer. The importer requires a known CRS, nonempty polygons, and an explicit output CRS. Invalid polygons are refused unless `--allow-invalid` keeps them for `nbf clean` to repair and record. A source with several layers needs `--layer`. The importer preserves each Polygon or MultiPolygon as one instance, writes a new GeoPackage layer named `buildings`, and standardizes `source_id` and `method`. `source_id` comes from `--id-field`, or else from the source feature ID. Other vendor fields are omitted. Confidence is copied only when its field is supplied and holds scores in `[0,1]`, after an explicit `--score-scale` such as 100 for a percentage; it is never synthesized.

```powershell
nbf import-vectors --input data/inputs/nearmap_buildings.gpkg --layer buildings --output outputs/nearmap_ai.gpkg --crs EPSG:26914 --id-field building_id --score-field confidence
```

Use the imagery's projected CRS instead of the illustrative EPSG code. Omit optional fields when absent. The method label is `--method`, or the preset's, or the input's own `method` field, and these must agree; `nearmap_ai` is the fallback. Import Esri output with `--preset esri`, described above. An export's product, survey date, and licensing context must be supplied separately by its owner; the importer does not infer them.

## Prepare spatial train, validation, and test data

Supply:

1. The tile manifest with `tiles` entries containing `id`, `path`, `row_off`, `col_off`, `width`, `height`, `crs`, and `bounds`. Tile paths are relative to the manifest folder and remain within it.
2. Exhaustive, manually reviewed building polygons with a known CRS. One feature is one building instance. MultiPolygon parts and courtyard holes are supported. Unlabeled buildings become false negative supervision, so the CLI requires `--labels-complete` as an explicit assertion about your labels.
3. A polygon AOI file in a projected CRS with a `split` field containing exactly `train`, `val`, and `test`. Each split may contain multiple AOI features. Choose spatial regions before examining model test results.

```powershell
nbf train prepare-coco --manifest data/tiles/manifest.json --ground-truth data/labels/buildings.gpkg --split-aois data/labels/splits.gpkg --output data/training/buildings_v1 --labels-complete --min-split-distance-m 100
```

The 100 m gap is an example, not a validated independence threshold. Default distance is zero. Choose a separation appropriate to your imagery and use case. The preparer never assigns random tile splits. It rejects overlapping AOIs, buildings crossing split AOIs, tiles not fully contained within exactly one split, missing splits, and CRS/dimension/bounds disagreements with the actual raster. `--skip-unassigned` explicitly drops boundary/outside tiles and records them, while still requiring at least one retained tile in each split. A spatial split alone does not eliminate every form of geographic or acquisition-related correlation.

Optional `--ground-truth-layer`, `--split-layer`, and `--split-field` select dataset fields/layers. `--bands 1 2 3` chooses existing unsigned 8-bit bands in RGB order. Training tiles must have no nodata pixels; use fully valid interior imagery rather than labeling black nodata regions. No implicit contrast stretch, reprojection of imagery, or random partition is performed. A new output directory is required.

The output for each of `train`, `val`, and `test` contains:

- `images/*.png`: three-band RGB imagery, unchanged pixel values.
- `annotations.json`: COCO images, one `building` category, and per-instance uncompressed column-major RLE. Pixel areas and bounding boxes are derived from the actual rasterized mask.
- `instances/*.tif`: georeferenced `uint32` instance IDs with zero background. Adjacent buildings stay separate, and IDs above 255 remain intact. COCO annotations carry `instance_value` and the one-based source feature row.

`preparation.json` records split counts, manifest and geometry hashes, selected bands, requested distance, skipped tiles, and subpixel instances omitted after rasterization. Empty tiles remain negative examples. Overlapping labeled instances are rejected rather than silently overwriting one another. RLE preserves holes; a list of polygon exteriors would not. The current preparer validates and holds integer masks in memory before writing, so use a bounded labeled subset that fits RAM.

## Official SAM 3 training adapter

The supplied `configs/sam3_training.example.yaml` targets Meta's official SAM 3 commit **2345a4ad109ac29c569da749c91d84f10dc08c40**, verified on 2026-09-20. Use that checkout with its training dependencies and your already obtained checkpoint. This is separate from the ArcGIS environment. The inherited CUDA/NCCL configuration is intended for a compatible Linux or WSL2 GPU environment, not the Windows CPU preparation environment. [Official training instructions](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/README_TRAIN.md)

This version's `COCO_FROM_JSON` builds concept queries from category names and supports polygon or uncompressed RLE masks. It performs the COCO-to-query adaptation during loading; no guessed PCS file format or extra converter is needed. The prepared category text is `building`. This compatibility depends on the explicit loader in our template. [Official loader](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/train/data/coco_json_loaders.py)

The template inherits the official Roboflow recipe and enables instance segmentation, mask collation, RLE decoding, mask loss, and checkpoint saving. It overrides the original validation paths to use **val**, leaving **test** unused by the training configuration. Its inherited validation monitor reports box metrics; use the project's footprint evaluation on held-out test predictions for the final comparison. Batch size, epochs, tile density, and device capacity still need a pilot run. The inherited training filters can discard empty queries or images with too many objects, so inspect training logs before a full run. [Base recipe](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml)

In the Linux/WSL2 training shell, with this package and Meta's training environment available:

```bash
cp configs/sam3_training.example.yaml /path/to/sam3/sam3/train/configs/buildings.yaml
export SAM3_DATASET_ROOT=/path/to/prepared/buildings_v1
export SAM3_LOG_DIR=/path/to/outputs/sam3-buildings-v1
export SAM3_BPE_PATH=/path/to/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
export SAM3_CHECKPOINT=/path/to/models/sam3.pt

nbf train launch --checkout /path/to/sam3 --config /path/to/sam3/sam3/train/configs/buildings.yaml --python /path/to/training-env/bin/python --num-gpus 1
```

This prints a dry-run plan and config hash. Add `--execute` only when ready to train. It invokes the actual entry point, equivalent to:

```bash
python sam3/train/train.py -c configs/buildings.yaml --use-cluster 0 --num-gpus 1 --num-nodes 1
```

The official launcher resolves configs relative to `sam3.train`, so the wrapper requires the config inside that checkout. It does not copy files into the checkout, create a virtual environment, fetch weights, or validate GPU memory. Environment paths are intentionally user supplied. [Official launcher](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/train/train.py)

## Export trained weights before inference

Do not pass the raw trainer checkpoint directly to SamGeo. At the pinned revision, the trainer stores native `Sam3Image` parameter names inside `checkpoint['model']`. The inference builder selects keys containing `detector.` and strips that prefix. A native training checkpoint can therefore be accepted without applying the intended weights. [Trainer save format](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/train/trainer.py), [inference checkpoint loader](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model_builder.py)

After training, run the local export in your existing Torch environment:

```bash
nbf train export-checkpoint --input /path/to/outputs/sam3-buildings-v1/checkpoints/checkpoint.pt --output /path/to/models/buildings-inference.pt
```

The command reads with `torch.load(weights_only=True, map_location="cpu")`, validates tensor values and expected native model components, and writes a new `model` dictionary whose keys start with `detector.`. It rejects already-prefixed, wrapped, interactive, partial-component, non-tensor, non-finite, and unknown-component states. It never retries with unrestricted pickle loading. Keep the original trainer checkpoint for resuming training because the export omits optimizer and training state. Architecture-level parameter shapes still require verification by loading the model in the compatible inference environment. A CPU test with real PyTorch (`tests/test_checkpoint_torch.py`, its own CI job) saves a trainer-style checkpoint from a small model with SAM 3's native component names, exports it, passes the inference preflight checks, applies the inference builder's key filter, and loads the result strictly into a fresh model: every weight and buffer arrives unchanged and the model's output matches. It also shows that an unexported trainer checkpoint would leave nothing for that filter to keep.

The adjacent file is **`buildings-inference.pt.metadata.json`**, formed by appending `.metadata.json` to the full checkpoint filename. Its contract is:

```json
{
  "schema": "sam3-building-inference-checkpoint-v1",
  "format": "sam3-detector-prefixed",
  "model_id": "facebook/sam3",
  "upstream_commit": "2345a4ad109ac29c569da749c91d84f10dc08c40",
  "supported_methods": ["text", "exemplar"],
  "checkpoint_sha256": "SHA256 of the exported checkpoint",
  "source_checkpoint_sha256": "SHA256 of the original trainer checkpoint",
  "tensor_count": 0
}
```

The shown hash strings and zero count illustrate fields; the exporter writes actual hashes/counts plus an export timestamp. Keep this metadata with the exported weights. This PCS training recipe does not train the interactive tracker required by **box** or **point** mode. Use the exported model only with **text** or **exemplar**. Use an appropriate original pretrained checkpoint for the other modes.

For a text inference pilot, use a manifest containing only your reserved test tiles:

```bash
nbf infer --manifest data/test-tiles/manifest.json --method text --text building --checkpoint /path/to/models/buildings-inference.pt --bpe-path /path/to/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz --output outputs/sam3_finetuned_text --limit 4
```

This prints a plan. Add `--execute` to run it, then use a separate new output directory for the complete test set without `--limit`. For exemplar inference, select `--method exemplar --prompts data/prompts/building-exemplars.gpkg` using the format in `PROMPTS.md`. Check missing/unexpected model-key diagnostics before interpreting results. Exporting or loading weights does not establish accuracy; compare held-out footprint results with the same evaluation settings as the baselines.

## Verified scope

Synthetic CPU tests exercise mask holes, column-major RLE, touching instances, IDs above 255, negative imagery, spatial exclusion, manifest/raster consistency, vector reprojection, dry-run behavior, and checkpoint key conversion. A mocked serialization test verifies safe load arguments and metadata hashes without installing Torch, and a real-PyTorch CPU test exports and reloads a small stand-in model's checkpoint. No SAM 3 checkpoint from real training has been exported or loaded, and no Hydra model instantiation or GPU fine-tuning was performed. The configuration is a concrete source-verified starting recipe, not a demonstrated accuracy result.

On Windows, a system `PROJ_LIB` or `PROJ_DATA` from PostGIS can override rasterio's bundled database. If the CPU environment reports a PROJ database-version conflict, clear those variables **only in the preparation/test shell** so the wheel uses its own database. Do not remove ArcGIS environment settings from the ArcGIS clone.
