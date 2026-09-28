# Validation record

Prepared September 20, 2026, and updated September 27, 2026. This records implementation verification, not building-extraction accuracy.

## Executed

- **146 automated tests passed** on Windows with Python 3.12.1 and again with Python 3.13.14, using the versions in `requirements-cpu.lock` (September 27, 2026). The original 97-test suite passed on September 20, 2026, including during GitHub release preparation.
- Branch coverage of the package, including CLI subprocesses, was 81%. Ruff found no errors under the configured correctness rules. The editable package metadata, including the MIT license expression, built successfully.
- The [CPU tests workflow](../.github/workflows/cpu-tests.yml) is configured to lint, run the tests with coverage, and run a synthetic CLI smoke test of every CPU pipeline stage on Windows and Linux with Python 3.12 and 3.13. The smoke sequence passed locally on both Python versions. Its [live run history](https://github.com/efkopru/nearmap-building-footprints/actions/workflows/cpu-tests.yml) is the source for CI status.
- Added on September 27, 2026: duplicate suppression that keeps a complete detection before one cut by a tile edge or nodata, full-size edge tiles with at least the requested overlap, `nodata_touch` flags, typed fields for tiles without detections, accurate cleanup flags when regularization is rejected, prediction fingerprints in evaluation reports, one-line CLI input errors, and a check that the pinned SAM 3 commit matches across scripts and documentation.
- Esri comparison, added September 27, 2026: `nbf import-vectors --preset esri` imported a synthetic file geodatabase in Esri's output layout (0-100 `Confidence`, an overlapping-chip duplicate, a self-intersecting outline). Cleanup removed the duplicate and repaired the outline, and `nbf compare --per-building` joined its report with two others building by building. A re-sorted reference file is refused. The 0-100 scale was confirmed in the arcgis 2.4.3 source, where `_maskrcnn_inferencing.py` multiplies each score by 100.
- First real-imagery run, September 27, 2026, held locally; no imagery, labels or outputs from it are in this repository. `esri.py` ran Esri's Building Footprint Extraction – USA model in ArcGIS Pro 3.7 on a GPU over real aerial imagery. The run exposed a validation bug, now fixed and tested with a stand-in ArcPy: dataset-level `Describe` has no pixel type for multiband rasters, so every run had been rejected. The real output then passed through `--preset esri` import, with `Confidence` observed between 50 and 99, and cleanup removed its overlapping-chip duplicates. The same imagery showed that a nodata value of 0 also masks isolated pure-black shadow pixels, so `nodata_touch` now ignores masked patches under 256 pixels. `nbf evaluate --agreement-only` was added for references known to be outdated.
- SAM 3 ran with the official checkpoint, verified against its published SHA-256, on a 6 GB RTX 4050 laptop GPU under WSL2: 90 tiles of 1024 px at a median 0.82 s each, with peak GPU memory of 5.8 GB. The comparison of SAM 3 and Esri, from city-wide runs to the first tuning runs, is in [results/lewisville](../results/lewisville/README.md); it is agreement with an older inventory, not accuracy.
- CPU end-to-end CLI sequence: synthetic imagery creation, raster inspection, overlapping tile preparation, SAM inference dry run, polygon cleanup, evaluation with vector match layers, spatial COCO preparation, local vector import, and dependency reporting.
- Compared cleaned and imported synthetic identity fixtures through two distinct evaluation reports and wrote a CSV. Incompatible/duplicate report rejection is covered by tests.
- Verified raster transforms, pixel coverage, CRS preservation, nodata handling, courtyard holes, touching instances, IDs above 255, duplicate suppression, conservative regularization limits, one-to-one maximum-cardinality matching, AOI policies, empty detections/reference, and spatial split leakage rejection.
- Exercised all four SAM adapter modes using a synthetic fake model. This checks adapters and geospatial output plumbing, not the neural network.
- Checked safe resume receipts, changed-input rejection, exported-checkpoint metadata/hash restrictions, native trainer key conversion, and rejection of incompatible interactive modes. Checkpoint serialization/load tests use a mocked Torch interface.
- `compileall` passed for Python source/scripts. Experiment JSON parsed. PowerShell parser and Bash `-n` checks passed for setup scripts; the setup scripts' GPU installation path was not executed.
- Source-reviewed SAM 3 configuration, loader, loss, transforms, and training-to-inference checkpoint conversion against Meta commit `2345a4ad109ac29c569da749c91d84f10dc08c40`.
- Verified the published SamGeo 1.4.2 wheel source against the official repository's interface used by the adapter.

The project's own Affine calls now use the `@` operator, and deprecation warnings raised by this package fail the test suite. rasterio 1.5 still triggers Affine's deprecation warning internally; the pytest configuration filters only that third-party warning. Training PNGs are deliberately written without georeferencing, so their expected non-georeferenced warning is suppressed where they are written and read. No failing test was suppressed. Georeferenced imagery/instance masks are retained in GeoTIFFs; training PNG coordinates are image pixels.

Synthetic identity reports return 9 matched objects and perfect overlap because predictions deliberately copy the reference geometry. These values are **not SAM, Esri, Nearmap, or real-world accuracy measurements** and must not be presented as such.

## Not executed

- No real imagery, labels or model output is stored in this repository.
- No actual trained checkpoint was exported or loaded.
- No Hydra model instantiation or fine-tuning was performed.
- No licensed Nearmap AI service was used.
- No model accuracy has been measured against reference outlines reviewed for the imagery date.

The CUDA/training configuration is a concrete source-verified starting recipe. Run the documented small pilot in the target GPU environment before expanding to a full area. Obtain independent reference labels before interpreting extraction accuracy.

## Files and environment

The delivered folder contains source, tests, documentation, and example configuration only. It does not bundle the test virtual environment, downloaded upstream source, generated synthetic datasets, checkpoints, or working outputs. Setup creates new local environments when you run it. The `nbf` entry point isolates inherited PROJ/GDAL settings within its Python process while leaving ArcPy's environment intact.
