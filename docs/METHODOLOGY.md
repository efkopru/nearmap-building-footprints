# Building extraction methodology

This protocol compares ways to extract building polygons from **local, authorized Nearmap imagery**. It defines what to measure and what evidence to retain. The [usage guide](USAGE.md) provides the actual commands and supported adapters. A documented method, a successful synthetic test, or a model confidence score is not evidence of accuracy on Nearmap imagery.

Keep imagery, labels, checkpoints, prompts containing real locations, and generated vectors local. This workflow does not require uploading data or calling a paid Nearmap API. Import already available, authorized imagery and AI vector exports. Model acquisition and software installation are separate setup steps, subject to the model's access requirements and license.

## 1. Define the target before extracting it

Use one explicit label convention per experiment:

- **Visible roof outline:** the roof perimeter visible in the image, including the agreed treatment of overhangs, attached structures, carports, and sheds.
- **Ground footprint:** the building's intersection with the ground, supported by appropriate reference data. A roof mask alone cannot establish that boundary.

Off-nadir displacement, roof overhangs, shadows, vegetation, and capture-date differences can produce legitimate disagreement between roofs and ground footprints. Polygon smoothing or right-angle regularization does not convert one convention into the other. Preserve separate layers if both are needed.

Nearmap's Building Footprints documentation specifically identifies generations 1 through 5 with roof outlines. Check the actual imported product and generation rather than generalizing this to every export. Its fidelity score describes agreement with the provider's prediction raster, not agreement with independent ground truth. [Nearmap Building Footprints](https://help.nearmap.com/kb/articles/787-ai-pack-building-footprints)

Write a short annotation specification covering attached buildings, courtyards, partial visibility, minimum included size, temporary structures, and uncertain objects. Give each labeled building a stable instance ID. Do not use the model being evaluated to create unreviewed reference labels.

## 2. Freeze imagery, AOIs, and evaluation splits

An area of interest (AOI) is the geographic boundary within which results are assessed. Use several spatially separate AOIs that cover the intended operating conditions: dense and sparse development, large and small roofs, varied materials, tree cover, shadows, and difficult adjoining buildings. Include genuine building-free areas so false detections can be measured.

Fully label every target building inside each evaluation AOI. Sparse example polygons do not constitute exhaustive ground truth. Declare ignored areas for unusable imagery or unresolved annotation before scoring. Record the reason and excluded area. Predetermine how objects crossing AOI boundaries are handled, for example by scoring only buildings wholly inside a buffered evaluation core.

Divide AOIs geographically into training, validation, and a held-out test set. Keep overlapping tiles, the same building, adjacent context, and repeat captures of that building within one split. Use separation buffers appropriate to tile context. Record the split assignment before tuning. Training changes weights; validation selects prompts, thresholds, tile sizes, and cleanup parameters; the held-out test measures the final frozen configuration.

For directly comparable runs, use the same imagery capture, band preparation, ground sample distance (GSD), labeled AOIs, and target convention. If an imported vendor layer represents another survey or an unknown resolution, report it as an unmatched comparison and inspect change-related disagreements separately. Do not present it as a controlled model ranking.

## 3. Choose the method and record the human input

SAM 3 supports concept segmentation using text or visual exemplars. These inputs can ask for multiple matching objects in an image. They are different from prompts that identify one particular instance. [Meta SAM 3](https://github.com/facebookresearch/sam3)

| Method | Input supplied beyond the image | Meaning of the result | Comparison category |
| --- | --- | --- | --- |
| SAM 3 text | A fixed phrase such as `building roof` | Candidate instances matching the phrase | Automated extraction after validation-time prompt selection |
| SAM 3 visual concept | Positive and optional negative exemplar boxes | Candidate instances matching the demonstrated concept | Guided concept extraction; report exemplar effort |
| SAM 3 instance prompts | A box or grouped foreground/background points for a selected building | Mask for the prompted object | Assisted segmentation; report object-selection and correction effort |
| Fine-tuned SAM 3 | A locally trained checkpoint plus its inference prompts | Predictions from the adapted model | Separate trained experiment |
| Esri Mask R-CNN | A fixed building extraction model package and inference settings | Building instances from the baseline | Automated extraction |
| Imported Nearmap AI | A preexisting local vector export | Provider-generated building polygons | Imported product comparison; no local inference claimed |

### SAM 3 text prompts

Choose candidate phrases on validation AOIs, then freeze the exact phrase and confidence threshold. Start a fresh image state for each tile and preserve the raw instance masks and scores. The repository's current adapter runs text or visual concept prompts separately. A combined-prompt or multiple-phrase workflow would be a different experiment requiring explicit implementation and duplicate-resolution rules.

### SAM 3 visual concept exemplars

A positive box demonstrates a desired object; a negative box demonstrates an unwanted example. This guides the concept detector and can affect predictions elsewhere in the same image. It does not constrain the output to exactly the boxed building. Meta's image example demonstrates positive and negative visual boxes. [Official image example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb)

The native processor accepts these boxes as normalized center-x, center-y, width, height values in `[0, 1]`, with a Boolean positive/negative label. Its state can also contain text, but this repository's SamGeo adapter uses separate text and exemplar modes because the wrapper resets prompts. [Processor implementation](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py), [SamGeo wrapper reference](https://samgeo.gishub.org/samgeo3/)

Store the source image ID, box coordinates, coordinate convention, labels, and prompt order. Boxes belong to their source image. The current adapter scopes exemplars to that same tile and does not transfer them across images. Tiles without a complete positive prompt are logged as skipped. Record this coverage: skipped areas cannot disappear from a full-AOI evaluation or be reported as successfully processed. Fix the prompting budget before test evaluation.

### SAM 3 per-building boxes and points

Use the instance-interactive path for a selected building. The native image model exposes `predict_inst` separately from concept grounding. Its predictor uses XYXY boxes, XY pixel points, and point labels `1` for foreground and `0` for background under its normal image-coordinate interface. Alternative masks have predicted quality scores. [Image model](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image.py), [instance predictor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam1_task_predictor.py)

The current adapter accepts an instance box or a group of points, rather than a combined box-plus-points call. Record one prompt group per intended instance, including corrections and elapsed annotation time. A negative point excludes pixels from that instance; a negative concept box teaches which example is unwanted. They are not interchangeable. Select alternative masks with a predeclared rule, not whichever best matches the hidden test polygon.

If boxes or points are derived from reference labels, label the result **oracle-prompted segmentation**. It measures delineation given object location and does not establish automatic building detection recall. To measure a complete assisted workflow, include missed buildings, object selection, and correction time.

### Optional SAM 3 fine-tuning

First run the frozen pretrained baseline. Fine-tune only if validation errors and sufficient local labels justify a separate experiment. Keep the pretrained checkpoint as a control. Training is not presumed to improve results.

Meta provides Hydra-configured training with local single-GPU and distributed options. Adapt the pinned upstream dataset/configuration contract, including segmentation targets and exhaustive-label indicators; an arbitrary GeoJSON file is not automatically a valid training dataset. Preserve resolved configuration, checkpoints, training logs, and validation results. [Meta training guide](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)

Convert only training labels into training examples, with stable image/instance IDs and traceable pixel-to-map transforms. Verify polygons, holes, class labels, empty images, and image/mask alignment visually. Confirm one batch loads and produces a finite loss before a longer run. Choose checkpoint, stopping rule, augmentations, and hyperparameters using validation data. Evaluate the selected checkpoint on the untouched test AOIs using the same procedure as the baseline. Record which parts of the model were trained; do not assume fine-tuning the concept detector also adapts the instance-interactive path.

**Environment target:** use Linux, or a Linux environment under Windows WSL2 with a compatible NVIDIA GPU, for native Meta SAM 3 training. This is the project's supported route, not a claim that Meta certifies every WSL2 configuration. Meta currently lists Python 3.12+, PyTorch 2.7+, and CUDA 12.6+; its current installation example uses a newer PyTorch/CUDA build. Pin a coherent, tested dependency set instead of mixing minimum versions with current examples. [Meta prerequisites](https://github.com/facebookresearch/sam3#installation)

The upstream training configuration uses CUDA and NCCL, and Triton lists Linux as its supported platform. Native Windows training is therefore outside this protocol's supported setup. NVIDIA documents CUDA on WSL2 using the Windows NVIDIA driver; do not install a Linux display driver inside WSL2. Check GPU architecture, available VRAM, driver/runtime compatibility, and actual framework GPU visibility before attempting training. These are setup requirements; **CUDA installation and a successful training run are not established by this document**. [Meta configuration](https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/eval_base.yaml), [Triton compatibility](https://github.com/triton-lang/triton#compatibility), [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Esri Mask R-CNN baseline

Use a fixed, locally available Building Footprint Extraction USA model package and record its version/hash. Esri identifies this model as Mask R-CNN. Its published benchmark does not measure this project's AOIs. Confirm the selected package's intended geography, input bands, cell size, and model definition before inference. [Esri model description](https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-building-footprint-extraction-usa.htm)

Run in the compatible ArcGIS Pro deep learning environment with the required Image Analyst license. Preserve the `.emd`/`.dlpk` provenance and all inference arguments, including padding, threshold, tile size, batch size, and NMS. Use a fresh output per run: the detection tool can append to an existing feature class. Avoid double suppression when both the tool and a later merge stage remove duplicates. [Esri detection tool](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/image-analyst/detect-objects-using-deep-learning.html)

### Imported Nearmap AI vectors

Import a local export without requesting fresh imagery or AI features. Preserve the original file and its provider identifiers, survey/capture date, AI generation, feature type, coordinate system, and available confidence/fidelity fields. Map these fields into the run manifest without assuming every export has the same schema. Keep provider scores distinct from scores emitted by locally run models. Evaluate a copy through the common AOI and scoring rules. Report the method as an import, not as a SAM 3 or Esri inference run.

## 4. Tile with enough context and preserve georeferencing

Retain the original raster and a manifest of CRS, affine transform, dimensions, nodata/valid-data mask, bands, capture date, GSD, and checksum. Apply a declared, repeatable band/color transformation rather than independently stretching each test tile.

Tile size and overlap are in **pixels**. Their ground extent depends on GSD. Choose candidate values on validation AOIs so representative large roofs can appear whole with surrounding context. Small overlap can leave split roofs; excessive overlap increases duplicate predictions and processing. There is no universal tile size or overlap that guarantees good extraction. Record any model resizing and effective inference resolution.

For each tile retain its raster window, padding, image ID, and transform. Transform mask boundaries back through any resize/padding operation, then through the full pixel-to-map transform. Do not reconstruct coordinates from filename guesses or GSD alone. Rasterio distinguishes pixel coordinates from spatial coordinates and supports the appropriate transform operations. [Rasterio transforms](https://rasterio.readthedocs.io/en/stable/topics/transforms.html)

Check alignment with a local overlay before scaling up. Clip padded predictions to valid imagery and use the predetermined AOI rule. Preserve flags for masks touching tile edges, nodata, and evaluation boundaries.

## 5. Stitch instances and clean polygons conservatively

The extraction adapter produces raw per-tile polygons with edge flags; cleanup and duplicate suppression are separate stages. These outputs do not guarantee seamless stitching. Retain the contributing tile/instance IDs in a common projected CRS. Identify duplicate hypotheses in overlap areas using spatial agreement and a declared selection rule, such as overlap plus preference for a complete interior prediction. Store the chosen and suppressed IDs. Confidence alone may favor a truncated prediction.

Do not dissolve every intersecting polygon: adjacent buildings can touch, while repeated predictions of one building may disagree. Do not assume ordinary duplicate suppression reconnects edge fragments. Inspect residual cuts, missing pieces, duplicate roofs, and falsely joined buildings; flag unresolved cases for review. Evaluate the stitched layer before geometric cleanup.

Repair invalid geometry, retain polygonal components deliberately, and log anything removed or split. A validity repair can return a geometry collection or lower-dimensional remnants, so a successful function call alone is insufficient. [Shapely validity repair](https://shapely.readthedocs.io/en/stable/reference/shapely.make_valid.html)

Choose a suitable local projected CRS for metric operations. Express minimum area in square metres and simplification/boundary tolerances in metres, with explicit unit conversion. Do not treat geographic degrees or arbitrary projected units as metres. For square, north-up pixels, area is `pixel_count × GSD²`; use the affine determinant for general raster cell area and account for CRS units.

Preserve meaningful holes and separate building identities. Keep the unmodified polygon alongside the cleaned version or retain a reproducible change log. Track area change, vertex reduction, splits, merges, and removals. Do not apply blanket polygon orthogonalization: curved roofs, diagonal wings, and irregular buildings are legitimate. Any rectilinear regularization is a separately evaluated, opt-in treatment for appropriate structures.

## 6. Evaluate detection, shape, and effort separately

Freeze confidence thresholds, matching rules, and cleanup settings before opening test results. Confidence and predicted mask quality are model outputs, **not measured accuracy**. Show each method's raw, stitched, and cleaned stage when comparing the effect of postprocessing.

For instance detection, compute polygon intersection-over-union (IoU), `intersection area / union area`, within the valid evaluation region. Match predictions and labels one-to-one at a declared threshold, such as 0.50, with a deterministic matching rule. Report the exact assignment algorithm. Count unmatched predictions as false positives and unmatched reference buildings as false negatives. Report `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, and `F1 = 2TP/(2TP+FP+FN)`. Retain TP/FP/FN counts; mark undefined ratios rather than silently treating empty denominators as perfect scores.

Report matched polygon IoU for shape quality, while retaining recall so missed buildings cannot disappear from the assessment. Raster foreground IoU/Dice can supplement this, but unioned masks do not expose instance merges and splits. If boundary quality matters, use a declared tolerance in metres and record its definition. Count merge/split errors, count bias, area bias, and review-required objects. Evaluate errors by AOI and relevant difficulty strata. Any uncertainty interval should respect spatial grouping, not treat neighboring pixels as independent observations.

Assisted and oracle-prompted results belong in separate comparison groups. Report prompts, selection effort, and corrections alongside quality. Record preprocessing, inference, stitching, cleanup, and human-review time separately, along with hardware and warm/cold execution conditions. A vendor import has import time, not a locally measured model inference time.

### Run manifest template

| Field group | Required record |
| --- | --- |
| Identity | Run ID; timestamp; method; automated/assisted/imported; execution status; repository commit |
| Input | Local input manifest hash; capture date; CRS; GSD; bands; AOI and split IDs; valid/ignored area |
| Reference | Label version/hash; roof/ground convention; complete-label policy; boundary policy; reviewer |
| Model | Provider; exact checkpoint/package; file hash; upstream commit; license/access provenance |
| Environment | OS or WSL distribution; Python/packages; GPU/VRAM; driver; framework CUDA runtime; verified GPU visibility |
| Inference | Text; exemplar/instance prompt file hash and coordinate convention; thresholds; tile pixels; overlap pixels; resize; precision; seed where applicable |
| Processing | CRS and units; duplicate rule; repair policy; area/simplification tolerances; original and output hashes |
| Evaluation | Split; IoU threshold; matching rule; metrics definitions; stage; script/config version; excluded objects |
| Cost and review | Stage runtimes; prompt/correction counts; human minutes; failed tiles; unresolved cases |

### Comparison report template

Populate only from retained outputs and reference labels. `Not run` is a status, not a zero-valued metric. Add rows for each stage or frozen variant rather than replacing a poor result.

| Method and stage | Status | Comparable imagery/labels? | Test AOIs / reference count | TP / FP / FN | Precision / recall / F1 | Matched IoU | Human effort | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 3 text | Not run | Pending verification | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| SAM 3 visual concept | Not run | Pending verification | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| SAM 3 instance-assisted | Not run | Separate assisted group | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Fine-tuned SAM 3 | Not run | Pending verification | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Esri Mask R-CNN | Not run | Pending verification | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Nearmap AI import | Not run | Verify survey and convention | Not measured | Not measured | Not measured | Not measured | Not measured | Import only |

Retain representative success and failure overlays locally, selected by a declared rule rather than only appearance. Public reporting can describe methods and aggregate results after checking their release scope. This workflow does not publish source imagery, real-location prompts, ground truth, or generated building vectors.
