# Methodology

How to compare ways of extracting building outlines fairly, how the code enforces it, and what has been tested. The commands are in the [usage guide](USAGE.md).

A model's confidence score, a vendor's benchmark or a passing synthetic test is not accuracy on your imagery. Only outlines scored against independent reference outlines are.

## Common pitfalls

| Pitfall | How this project handles it |
| --- | --- |
| Tiles cut buildings at their edges, and overlapping tiles see them twice. | Full-size overlapping tiles, flags on cut outlines, and cleanup that keeps the complete copy and never merges neighbours. |
| Random tile splits put the same building in training and test data. | Training, validation and test areas are separate regions, checked for overlap, distance and buildings crossing them. |
| Greedy matching undercounts correct outlines, and edge handling quietly shifts scores. | Optimal one-to-one matching, and a recorded rule for buildings on the area's edge. |
| Confidence scores and vendor benchmarks are taken as accuracy. | Every method is scored against the same reference outlines, and reports with different references refuse to be compared. |
| A roof outline is treated as a ground footprint. | The outline convention is recorded per experiment; smoothing never claims to turn one into the other. |

## 1. Decide what counts as an outline

Pick one convention per experiment and write it down:

- **Roof outline:** the roof edge visible in the image, with an agreed treatment of overhangs, carports, sheds and attached structures.
- **Ground footprint:** where the building meets the ground. A roof mask alone cannot show this.

Off-nadir lean, overhangs, shadows, trees and changes since the reference was made all cause honest disagreement between the two. Also write down how to handle courtyards, partly hidden buildings, the smallest building to include and uncertain cases. Never use the model being tested to draw reference outlines. Nearmap's documentation describes its Building Footprints generations 1 to 5 as roof outlines, and its fidelity score as agreement with its own prediction, not with ground truth.

## 2. Fix the areas before tuning

An area of interest (AOI) is the boundary within which results are scored. Use several separate AOIs that cover dense and sparse development, large and small roofs, different materials, trees and shadows, plus some areas with no buildings, so false detections show up.

Outline **every** building inside an evaluation AOI; scattered examples are not a reference. Decide in advance how to treat buildings crossing the AOI's edge, and record any area left out and why.

Split the AOIs by region into training, validation and test sets, and record the split before tuning. Training changes model weights; validation picks prompts, cut-offs, tile sizes and cleanup settings; the test set is scored once, with everything frozen. For a fair comparison every method uses the same imagery, bands, resolution, AOIs and outline convention. A vendor layer from another survey date is a different comparison and should be reported as one.

## 3. Record the human input for each method

| Method | Input beyond the image | Comparison group |
| --- | --- | --- |
| SAM 3 text | A fixed phrase such as `building` | Automatic, after picking the phrase on validation data |
| SAM 3 example boxes | Positive and negative example boxes | Guided; report how many examples |
| SAM 3 boxes or points | A box or points for each chosen building | Assisted; report selection and correction effort |
| Fine-tuned SAM 3 | A locally trained checkpoint | A separate trained experiment |
| Esri Mask R-CNN | A fixed model package and settings | Automatic |
| Imported layer, such as Nearmap AI | An existing export | Imported product; no local model run |

- **Text prompts.** Choose the phrase and cut-off on validation areas, then freeze them. Each tile starts fresh. Running several phrases or mixing text with boxes would be a different experiment.
- **Example boxes** show the model what to look for in their own tile and can affect predictions anywhere in it; they don't restrict output to the boxed building. Tiles without a complete positive example are logged as skipped, and skipped areas must not count as processed.
- **Boxes and points** outline a chosen building. If they come from the reference outlines, call the result *oracle-prompted*: it measures outline quality given the location, not detection. For a full assisted workflow, also count missed buildings and correction time.
- **Fine-tuning** is a separate experiment. Run the pretrained model first, keep it as a control, choose the checkpoint on validation data and score once on the untouched test areas. Fine-tuning the concept detector does not adapt the box and point path. Native SAM 3 training needs Linux or WSL2 with an NVIDIA GPU.
- **Esri's model.** Record the package version and hash and every setting (padding, threshold, tile size, batch size, duplicate suppression). Its published benchmark is not a result on your areas.
- **Imported layers.** Keep the original file and its IDs, survey date, AI generation and scores, kept apart from scores of locally run models. Report it as an import, not a model run.

## 4. Tile without losing the georeferencing

Keep the original raster and record its CRS, transform, size, valid-data mask, bands, capture date, resolution and hash. Tile size and overlap are in pixels, so their ground size depends on the resolution: pick them on validation areas so that large roofs fit whole with some context. Too little overlap splits roofs; too much creates more duplicates and work. There is no universally right value.

Each tile keeps its own transform, and masks are turned into polygons through it, never through coordinates guessed from file names. Check an overlay on the imagery before scaling up.

## 5. Clean conservatively

Raw per-tile outlines, duplicate removal and geometric cleanup are separate, recorded stages. In overlaps, prefer the complete copy of a building over one cut by a tile edge; confidence alone can favour the cut copy. Never dissolve every touching polygon: neighbouring buildings touch. Repair invalid geometry deliberately and log what was removed or split. Measure areas and tolerances in metres in a suitable projected CRS. Keep courtyard holes and separate building identities. Squaring off outlines is an opt-in treatment to test, not a default: curved and angled buildings are real.

## 6. Score detection, shape and effort separately

Freeze cut-offs, matching rules and cleanup settings before looking at test results. Match outlines to reference buildings one to one at a fixed IoU (intersection over union) threshold such as 0.5. Unmatched outlines are false positives (FP) and unmatched reference buildings false negatives (FN):

- precision = TP / (TP + FP), the share of outlines that are right
- recall = TP / (TP + FN), the share of buildings found
- F1 = 2TP / (2TP + FP + FN), which balances the two

Report the IoU of matched pairs for shape, alongside recall so that missed buildings stay visible, and mark undefined ratios rather than treating them as perfect. Report merges, splits and area bias, and break errors down by area and difficulty. Keep assisted and oracle-prompted results in their own groups, with their prompting effort, and record the time each stage took and on what hardware.

**Record for every run:** run ID, method and repository commit; input hashes, capture date, CRS, resolution and AOI; reference version, outline convention and reviewer; model file, hash and upstream commit; environment, GPU and drivers; prompts, cut-offs, tile size and overlap; cleanup settings; and scoring settings and timings. Report a method that was not run as "Not run", never as a zero, and add rows for new variants instead of replacing poor results.

## How the code enforces this

- **Optimal one-to-one matching.** Outlines and reference buildings are matched for the most matches above the IoU threshold, then the highest total IoU, with SciPy's `linear_sum_assignment` on each group of overlapping shapes rather than one city-sized matrix. A test pins a case where greedy matching finds fewer. ([`evaluation.py`](../src/nearmap_buildings/evaluation.py))
- **Georeferencing end to end.** Tiles keep the source CRS, transform and valid-data mask. Masks become polygons through each tile's own transform, keeping holes and keeping touching buildings apart. ([`preprocessing.py`](../src/nearmap_buildings/preprocessing.py), [`inference.py`](../src/nearmap_buildings/inference.py))
- **Seam-aware duplicate removal.** Cleanup keeps outlines that touch neither a tile edge nor missing imagery before comparing scores, never merges neighbours, and records every removal. A building longer than a tile can be rejoined, but only where two cut pieces agree inside the strip both tiles saw. ([`postprocess.py`](../src/nearmap_buildings/postprocess.py))
- **Traceable, resumable runs.** Tile, prompt, manifest and checkpoint hashes form a run signature. `--resume` refuses to continue after any change, and per-tile receipts with atomic writes make an interrupted run safe to pick up. ([`inference.py`](../src/nearmap_buildings/inference.py))
- **Training data without leakage.** Splits come from drawn regions, never random tiles; overlapping or too-close splits and buildings crossing them are refused. Masks are COCO RLE, which keeps holes, plus `uint32` instance rasters that stay correct past 255 buildings per tile. ([`training.py`](../src/nearmap_buildings/training.py))
- **Comparable by construction.** Reports carry fingerprints of the predictions, reference and AOI, and `nbf compare` refuses reports that differ. The per-building view also checks the reference's row order, since a re-sorted file would put each method's results on the wrong buildings. ([`compare.py`](../src/nearmap_buildings/compare.py))
- **Safe defaults.** Model runs are dry runs until `--execute`, checkpoints load with `torch.load(weights_only=True)`, and commands never overwrite their inputs. Fine-tuned weights need an explicit export, because the inference loader would otherwise accept a training checkpoint while silently ignoring its weights.
- **Real GIS machines.** `nbf` clears inherited `PROJ_LIB`, `PROJ_DATA` and `GDAL_DATA` for its own process, so a PostGIS install cannot override the projection database bundled with the wheels, while ArcPy keeps its own settings. ([`cli.py`](../src/nearmap_buildings/cli.py))

## What has been tested

Last updated October 2026. This records what has been run, not accuracy.

**Run and passing**

- The test suite, lint and a synthetic run of every CPU command, on Windows and Linux with Python 3.12, 3.13 and 3.14, on every push ([CI history](https://github.com/efkopru/nearmap-building-footprints/actions/workflows/cpu-tests.yml)). Package branch coverage was 81% when last measured. Deprecation warnings from this package fail the tests.
- Geometry and raster handling: transforms, CRS, nodata, courtyard holes, touching buildings, IDs above 255, duplicate removal, regularization limits, one-to-one matching, AOI edge rules, empty inputs and split leakage.
- All four SAM 3 modes against a fake model, which checks prompts, georeferencing and outputs but not the network itself.
- Checkpoint export and loading with real PyTorch 2.10 on CPU, using a small stand-in model in SAM 3's layout: its weights survive export and load strictly into a fresh model.
- Real imagery, held locally: SAM 3 with the official checkpoint (its published SHA-256 verified) on an RTX 4050 under WSL2, at a median 0.82 s per 1024 px tile and 5.8 GB peak memory, then across the whole of Lewisville; and Esri's model in ArcGIS Pro 3.7, on one area and then the whole city. The first real Esri run exposed a validation bug (multiband rasters were always rejected), now fixed and tested. The results are in [results/lewisville](../results/lewisville/README.md).
- The [walkthrough notebooks](../notebooks/README.md), on the synthetic demo, with SAM 3 on the GPU and Esri's model in ArcGIS Pro.
- Esri's 0–100 confidence scale, confirmed in the arcgis 2.4.3 source, and the training recipe and checkpoint conversion, reviewed against Meta's pinned code.

**Not run yet**

- Scoring against reference outlines reviewed for the 2026 imagery, so there are **no accuracy results** yet: the Lewisville scores are agreement with a 2015 map.
- Fine-tuning on real data, and loading a really fine-tuned checkpoint.
- Seam merging, `nbf agree` and the chunked Esri commands on the Lewisville imagery; they are covered by synthetic tests.
- Any licensed Nearmap AI layer.

The demo's predictions copy its reference outlines, so its perfect scores are expected by construction and are not SAM 3, Esri or Nearmap accuracy.

## Sources

Checked 2026-09-20. They describe upstream tools and requirements, not this project's results. SAM 3 is pinned to Meta's commit [`2345a4ad109ac29c569da749c91d84f10dc08c40`](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40), and SamGeo separately to 1.4.2; record the versions you actually install.

| Source | Used for |
| --- | --- |
| [Meta SAM 3 repository](https://github.com/facebookresearch/sam3) | Concept segmentation, model access, Python, PyTorch and CUDA requirements |
| [Meta image prompting example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb) | Text prompts and positive and negative example boxes |
| [Meta image processor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py) | Text and box prompt formats and prompt state |
| [Meta image model](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image.py) and [instance predictor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam1_task_predictor.py) | The separate path for boxes and points per building, and predicted mask quality |
| [SamGeo 3 reference](https://samgeo.gishub.org/samgeo3/) and [SamGeo 1.4.2](https://pypi.org/project/segment-geospatial/1.4.2/) | The geospatial wrapper the adapter uses |
| [Meta training guide](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md) and [evaluation config](https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/eval_base.yaml) | Training setup, Hydra configs, CUDA and NCCL |
| [Triton compatibility](https://github.com/triton-lang/triton#compatibility) and [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) | Why training needs Linux or WSL2, and the WSL2 driver setup |
| [Esri Building Footprint Extraction – USA](https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-building-footprint-extraction-usa.htm) | Model architecture and scope |
| [Esri Detect Objects Using Deep Learning](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/image-analyst/detect-objects-using-deep-learning.html) | Tool settings, duplicate suppression, append behaviour and licensing |
| [Nearmap AI building footprints](https://help.nearmap.com/kb/articles/787-ai-pack-building-footprints) | Roof-outline meaning and the fidelity score |
| [Rasterio transforms](https://rasterio.readthedocs.io/en/stable/topics/transforms.html) and [Shapely `make_valid`](https://shapely.readthedocs.io/en/stable/reference/shapely.make_valid.html) | Pixel-to-map transforms and geometry repair |
