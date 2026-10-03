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

Both models ran on Nearmap's May 2026 imagery of the whole city of Lewisville, Texas, at 10 cm per pixel. Their outlines were compared with the city's own 38,882 building outlines. Those date from 2015, so the scores measure agreement with them, not accuracy.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/found-by-size-dark.svg">
  <img alt="Bar chart of recall by building area, the share of the 2015 outlines each model found. 100 m² and up: Esri Mask R-CNN 94%, SAM 3 90%. 50 to 100 m²: 52% and 39%. 20 to 50 m²: 8% and 25%. Under 20 m²: 0% and 3%." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/found-by-size.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/test-area-dark.svg">
  <img alt="Bar chart for the held-out test area. Recall: Esri Mask R-CNN 94% and SAM 3 91% of the 471 buildings. Precision: 97% and 82%." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/test-area.svg">
</picture>

- **Buildings of 100 m² and up:** both models find about 9 in 10 (recall 0.94 for Esri's model, 0.90 for SAM 3).
- **Small structures:** both miss almost all of those under 20 m². Between 20 and 50 m², SAM 3 finds more than Esri's model.
- **Likely new buildings:** SAM 3 draws more outlines that have no 2015 match. About 4,900 of them are also drawn by Esri's model, so many are probably buildings built since 2015.

The [full results](https://github.com/efkopru/nearmap-building-footprints/blob/main/results/lewisville/README.md) have every number, and the [PDF report](https://github.com/efkopru/nearmap-building-footprints/blob/main/results/lewisville/Lewisville_building_footprint_comparison.pdf) shows both models on the imagery.

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

Tested on Windows and Linux with Python 3.12 to 3.14. Not done yet: fine-tuning, and scoring against reference outlines reviewed against the 2026 imagery, which is the true accuracy test. The [methodology](https://github.com/efkopru/nearmap-building-footprints/blob/main/docs/METHODOLOGY.md) explains the design and what has been tested.

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

The code is MIT licensed. No imagery, labels, model weights or model outputs are included, and the tool never downloads imagery or calls paid APIs. This is an independent project, not affiliated with or endorsed by Nearmap, Meta, Esri or the SamGeo project.
