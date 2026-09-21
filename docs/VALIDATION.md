# Validation record

Prepared September 20, 2026. This records implementation verification, not building-extraction accuracy.

## Executed

- **97 automated tests passed** on Windows with Python 3.12.1 and the versions in `requirements-cpu.lock`.
- The 97-test suite passed again during GitHub release preparation. Editable package metadata built successfully, and the GitHub Actions workflow parsed successfully. The [CPU tests workflow](../.github/workflows/cpu-tests.yml) runs the tests and a synthetic CLI smoke test on Windows and Linux; its [live run history](https://github.com/efkopru/nearmap-building-footprints/actions/workflows/cpu-tests.yml) is the source for CI status.
- CPU end-to-end CLI sequence: synthetic imagery creation, raster inspection, overlapping tile preparation, SAM inference dry run, polygon cleanup, evaluation with vector match layers, spatial COCO preparation, local vector import, and dependency reporting.
- Compared cleaned and imported synthetic identity fixtures through two distinct evaluation reports and wrote a CSV. Incompatible/duplicate report rejection is covered by tests.
- Verified raster transforms, pixel coverage, CRS preservation, nodata handling, courtyard holes, touching instances, IDs above 255, duplicate suppression, conservative regularization limits, one-to-one maximum-cardinality matching, AOI policies, empty detections/reference, and spatial split leakage rejection.
- Exercised all four SAM adapter modes using a synthetic fake model. This checks adapters and geospatial output plumbing, not the neural network.
- Checked safe resume receipts, changed-input rejection, exported-checkpoint metadata/hash restrictions, native trainer key conversion, and rejection of incompatible interactive modes. Checkpoint serialization/load tests use a mocked Torch interface.
- `compileall` passed for Python source/scripts. Experiment JSON parsed. PowerShell parser and Bash `-n` checks passed for setup scripts; the setup scripts' GPU installation path was not executed.
- Source-reviewed SAM 3 configuration, loader, loss, transforms, and training-to-inference checkpoint conversion against Meta commit `2345a4ad109ac29c569da749c91d84f10dc08c40`.
- Verified the published SamGeo 1.4.2 wheel source against the official repository's interface used by the adapter.

The test run reported dependency deprecation warnings from Affine/Rasterio and expected non-georeferenced PNG fixture warnings. No failing test was suppressed. Georeferenced imagery/instance masks are retained in GeoTIFFs; training PNG coordinates are image pixels.

Synthetic identity reports return 9 matched objects and perfect overlap because predictions deliberately copy the reference geometry. These values are **not SAM, Esri, Nearmap, or real-world accuracy measurements** and must not be presented as such.

## Not executed

- No real Nearmap imagery was processed or copied into this project.
- No SAM 3 weights were downloaded, loaded, or used for inference.
- No actual trained checkpoint was exported or loaded.
- No Hydra model instantiation, CUDA execution, or fine-tuning was performed.
- No ArcPy/Esri model or licensed Nearmap AI service was run.
- No model accuracy, GPU memory requirement, or full-mosaic throughput was measured.

The CUDA/training configuration is a concrete source-verified starting recipe. Run the documented small pilot in the target GPU environment before expanding to a full area. Obtain independent reference labels before interpreting extraction accuracy.

## Files and environment

The delivered folder contains source, tests, documentation, and example configuration only. It does not bundle the test virtual environment, downloaded upstream source, generated synthetic datasets, checkpoints, or working outputs. Setup creates new local environments when you run it. The `nbf` entry point isolates inherited PROJ/GDAL settings within its Python process while leaving ArcPy's environment intact.
