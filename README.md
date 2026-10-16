# Building footprints from aerial imagery

[![CPU tests](https://github.com/efkopru/nearmap-building-footprints/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/efkopru/nearmap-building-footprints/actions/workflows/cpu-tests.yml)
![Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/efkopru/nearmap-building-footprints/blob/main/LICENSE)

**Turn an aerial image into building outlines with deep learning, then check fairly how well each method did.**

`nbf` is a Python command-line tool for GIS work. It runs two deep learning models, Meta's SAM 3 and Esri's Mask R-CNN building model, on the same imagery, cleans up their outlines with the same rules, and scores them against the same reference outlines.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/how-it-works-dark.svg">
  <img alt="How it works: an aerial image is cut into overlapping tiles, SAM 3 and Esri's Mask R-CNN model extract building outlines, and the outlines are cleaned and scored the same way for each method." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/how-it-works.svg">
</picture>

## Results: Lewisville, Texas

Both models ran on Nearmap's May 2026 imagery of Lewisville, Texas, at 10 cm per pixel. **Reviewed accuracy is now measured in one held-out 700 m test square.** Esad completed the reference review on October 15, 2026 and confirmed that the test review used only the imagery and historical outlines, without exposure to SAM 3 or Esri predictions before or during it.

For buildings of at least 20 m², both predictions and reference outlines are filtered before matching at IoU ≥ 0.5. The reviewed reference contains 468 buildings in this cohort:

| Method | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Esri, city run | 0.965 | 0.951 | **0.958** |
| SAM 3 | 0.828 | 0.923 | **0.873** |
| Esri, local raw run | 0.936 | 0.904 | **0.920** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/reviewed-test-accuracy-dark.svg">
  <img alt="Reviewed test-area accuracy for three frozen methods, showing precision, recall and F1 separately for all sizes and for buildings of at least 20 square metres. Primary F1: city Esri 0.958, SAM 3 0.873 and local Esri 0.920." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/reviewed-test-accuracy.svg">
</picture>

The separate all-size evaluation includes 600 reviewed buildings; 40 boundary-crossing outlines are excluded. These are results for one area and a reference corrected from an inherited inventory. Independence is recorded as the reviewer's attestation. Settings were selected against the historical Old Town inventory and were not retuned on these test results. See the [reviewed scores and provenance](results/lewisville/reviewed_2026/README.md).

**Historical city-wide comparison.** The preserved city-wide scores measure agreement with 38,882 evaluated outlines from the city's 38,924-feature inventory. Its dates are primarily 2015 or earlier, with 33 source features undated; see the [historical reference provenance](results/lewisville/reference_provenance.json). They do not establish city-wide accuracy against the 2026 imagery.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/found-by-size-dark.svg">
  <img alt="Bar chart of recall by building area, the share of the 2015 outlines each model found. 100 m² and up: Esri Mask R-CNN 94%, SAM 3 90%. 50 to 100 m²: 52% and 39%. 20 to 50 m²: 8% and 25%. Under 20 m²: 0% and 3%." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/found-by-size.svg">
</picture>

- **Historical inventory, buildings of 100 m² and up:** both models find about 9 in 10 (recall 0.94 for Esri's model, 0.90 for SAM 3).
- **Small structures:** recall remains low in the reviewed test area: SAM 3 and the local Esri run each match 4 of 132 outlines under 20 m²; the city Esri run matches none.
- **Candidates for review:** about 4,900 SAM 3 outlines without an inventory match also match Esri's model. These may reflect inventory omissions, different outline conventions, changed structures or shared model errors. Agreement alone does not establish construction or demolition dates.

The [full results](https://github.com/efkopru/nearmap-building-footprints/blob/main/results/lewisville/README.md) have every number, and the [PDF report](https://github.com/efkopru/nearmap-building-footprints/blob/main/results/lewisville/Lewisville_building_footprint_comparison.pdf) shows both models on the imagery. The [project summary](portfolio/nearmap-building-footprints.md) and [two-page PDF](portfolio/nearmap-building-footprints.pdf) provide a shorter account of the reviewed results and their limits.

## Project status

As of October 15, 2026, the reference review ([#2](https://github.com/efkopru/nearmap-building-footprints/issues/2)) and frozen accuracy evaluation ([#3](https://github.com/efkopru/nearmap-building-footprints/issues/3)) are complete. [PR #29](https://github.com/efkopru/nearmap-building-footprints/pull/29) is merged. The reviewed references, existing predictions and published scores remain frozen.

The remaining tracked work is [SAM 3 fine-tuning for small buildings (#4)](https://github.com/efkopru/nearmap-building-footprints/issues/4), deferred until a later session. No real-data training or new training-label packages have been completed. Resume by defining separate training, validation and fresh test areas, then reviewing **every building** in those areas, including small sheds and garages. Keep the completed test area out of training and setting selection. The [fine-tuning guide](docs/USAGE.md#9-fine-tune-sam-3) lists the preparation, hardware pilot and evaluation steps.

## Try it in five minutes

The demo makes a small synthetic scene, so no GPU, model or real imagery is needed.

On Windows, in PowerShell:

```powershell
git clone https://github.com/efkopru/nearmap-building-footprints.git
cd nearmap-building-footprints
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

Then make the demo data, clean it and score it:

```bash
nbf demo --output data/demo
nbf clean --input data/demo/perfect_predictions.gpkg --output outputs/demo/cleaned.gpkg --metric-crs EPSG:32614
nbf evaluate --predictions outputs/demo/cleaned.gpkg --predictions-layer cleaned --reference data/demo/reference.gpkg --aoi data/demo/aoi.geojson --metric-crs EPSG:32614 --output-json outputs/demo/scores.json --independent-holdout
```

The demo's "predictions" are copies of its reference outlines, so the perfect score only shows that the steps work. To run on real imagery, follow the [usage guide](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/USAGE.md).

## What's inside

- **Seven methods:** SAM 3 with a text prompt, image exemplars, or a box or points per building; a fine-tuned SAM 3; Esri's Mask R-CNN model in ArcGIS Pro; or an imported layer such as Nearmap AI.
- **Reduce tile-edge errors:** tiles overlap, and cleanup prefers complete outlines over cut copies. Buildings longer than a tile can still be split.
- **Fair scoring:** one-to-one matching, so each reference building matches at most one outline, and reports made against different references refuse to be compared.
- **Traceable results:** hashes link every outline to the pixels, prompts and model weights that made it, and interrupted runs resume safely.

Tested on Windows and Linux with Python 3.12 to 3.14. Reviewed accuracy is available for one test area; fine-tuning and validation across several independent areas remain future work. The [methodology](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/METHODOLOGY.md) explains the design and what has been tested.

## Project layout

| Folder | What's in it |
| --- | --- |
| `src/nearmap_buildings/` | The `nbf` tool |
| `tests/` | Automated tests |
| `docs/` | The [usage guide](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/USAGE.md), the [methodology](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/METHODOLOGY.md) and the charts |
| `notebooks/` | [Step-by-step walkthroughs](https://github.com/efkopru/nearmap-building-footprints/blob/main/notebooks/README.md) on the demo data |
| `results/lewisville/` | The [Lewisville results](https://github.com/efkopru/nearmap-building-footprints/blob/main/results/lewisville/README.md), their CSV files and the PDF report |
| `configs/`, `scripts/` | Example run settings, setup scripts, the chart maker and the PDF report maker |

Your imagery and results go in `data/` and `outputs/`, which git ignores.

## License

The code is MIT licensed. Source rasters, label datasets, model weights and vector outputs stay local; the PDF contains credited Nearmap imagery screenshots. The tool never downloads imagery or calls paid APIs. This is an independent project, not affiliated with or endorsed by Nearmap, Meta, Esri or the SamGeo project.
