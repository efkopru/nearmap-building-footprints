# Notebooks

Four walkthroughs of how SAM 3 and Esri's model are run and scored, from raw imagery to a frozen, once-scored result. Each notebook runs on the synthetic demo data from `nbf demo`, so none of them needs licensed imagery, and none contains any. Where the real Lewisville run used different values, the notebook gives them.

| Notebook | What it covers | Environment it was executed in |
|---|---|---|
| [01_prepare_imagery](01_prepare_imagery.ipynb) | Inspecting a raster, fixing a mislabelled CRS with a VRT (no resampling), tiling | The CPU environment |
| [02_run_sam3](02_run_sam3.ipynb) | Checkpoint check, inference plan, a resumable GPU run, cut-offs from stored scores, cleanup, scoring | WSL2 Ubuntu with an RTX 4050 and the SAM 3 environment |
| [03_run_esri](03_run_esri.ipynb) | Esri's model through ArcGIS Pro, city-scale overlapping chunks with process restarts, core-ownership merge, importing Esri output | The CPU environment, calling ArcGIS Pro 3.7's Python for the model |
| [04_score_and_compare](04_score_and_compare.ipynb) | Cleanup, evaluation, per-building comparison, tuning, the freezing rule, a once-only test-area score | The CPU environment |

## Running them

```bash
pip install -e .[test] jupyterlab
jupyter lab notebooks/
```

Run each notebook from the repository root or from `notebooks/`. Generated files go to `notebook_runs/`, which git ignores, and each notebook clears its own folder when it starts.

Without a CUDA GPU and `models/sam3.pt`, notebook 2 prints the plan and skips the model steps. Without ArcGIS Pro (with Image Analyst) and `usa_building_footprints.dlpk`, notebook 3 does the same and still runs the import and cleanup. For the GPU setup, see [Setup and usage](../docs/USAGE.md).

## What the demo can and cannot show

The demo image is nine flat roofs on a flat background, with exact reference outlines. It checks that every step runs and that the numbers add up. It says nothing about accuracy on real imagery. Two results make that plain:

- **SAM 3 finds nothing for the prompt "building":** the flat rectangles don't look like buildings. For "rectangle" it finds some of them, and the rest of notebook 2 cleans and scores that run.
- **Esri's model finds all nine roofs, but at confidence 0.5 to 0.7:** that is below the 0.9 cut-off frozen for Lewisville, so notebook 3 keeps the run's own 0.5 threshold.

The real results, including screenshots of both models on Lewisville's imagery, are summarized in [results/lewisville](../results/lewisville/README.md). The screenshots themselves stay in a private report because the imagery is licensed.
