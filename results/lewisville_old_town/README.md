# Lewisville, Texas: untuned baseline on one sample area

This is the first comparison of SAM 3 and Esri's building model on real imagery: a 700 m square (0.49 km²) around Old Town Lewisville. Every method used its default settings, before any tuning.

**These scores measure agreement with the city's building outlines collected in 2015 or earlier, not accuracy.** Old Town has had substantial construction since 2015, so some "extra" detections are real new buildings. Accuracy will come from outlines reviewed against the 2026 imagery.

## Setup

- **Imagery:** Nearmap, May 2026, 10.16 cm, 8-bit RGB. Not included in this repository.
- **Area:** a 700 m square centred on Old Town (33.0462° N, 96.9942° W). Buildings crossing its edge are excluded.
- **Reference:** the city's public building outlines, collected in 2015 or earlier; 319 buildings in the square.
- **Methods:**
  - **SAM 3:** text prompt "building", confidence 0.5, 1024 px tiles with 128 px overlap (90 tiles), SAM 3 at the pinned commit.
  - **Esri, city run:** Building Footprint Extraction – USA through `ExtractFeaturesUsingAIModels` with default settings, including Esri's post-processing. The city published it for the same May 2026 capture.
  - **Esri, local raw run:** the same model through `esri.py` (`DetectObjectsUsingDeepLearning`), threshold 0.5, padding 128, batch 4, no NMS.
- **Same cleanup for all:** EPSG:26914, minimum area 4 m², duplicate suppression at IoU 0.7 or containment 0.98, no simplification or regularization.
- **Evaluation:** one-to-one matching at IoU ≥ 0.5, run with `nbf evaluate --agreement-only`.

## Results

| Method | Matched | Extra | Missed | Precision | Recall | F1 | Median IoU of matches |
|---|---|---|---|---|---|---|---|
| SAM 3 | 142 | 287 | 177 | 0.33 | 0.45 | 0.38 | 0.74 |
| Esri, city run | 135 | 63 | 184 | 0.68 | 0.42 | 0.52 | 0.75 |
| Esri, local raw run | 158 | 381 | 161 | 0.29 | 0.50 | 0.37 | 0.74 |

By building size, a split chosen before any results were seen:

| Method | Recall, ≥ 50 m² (208) | Recall, < 50 m² (111) | Precision, ≥ 50 m² |
|---|---|---|---|
| SAM 3 | 0.63 | 0.10 | 0.39 |
| Esri, city run | 0.64 | 0.01 | 0.69 |
| Esri, local raw run | 0.71 | 0.10 | 0.36 |

Per building: 109 of the 319 outlines were found by all three methods, 38 by two, 32 by only one, and 140 by none. 96 of those 140 are under 50 m² (median 22 m²).

## Reading the numbers

- **Sheds:** none of the methods finds buildings under 50 m² reliably.
- **Larger buildings:** the three find 63–71% of the 2015 outlines. The city's Esri run is the most precise.
- **Extra detections:** SAM 3 and the local Esri run produce many more. Some are buildings built after 2015; others are fragments, duplicates or non-buildings, which is what tuning targets.
- **Outline quality** on matched buildings is similar across methods (median IoU 0.74–0.75).

## Runtime on a laptop RTX 4050 (6 GB)

- **SAM 3:** 0.82 s per 1024 px tile after loading, with peak GPU memory of 5.8 GB. The 90 tiles took 105 s.
- **Esri, local run:** 4.9 minutes for the 0.64 km² crop, which includes a 50 m margin.

## Next

1. Review the sample's outlines against the 2026 imagery, so the scores measure accuracy.
2. Tune prompts, thresholds, tile size and cleanup on validation data.
3. Score the frozen settings once on an untouched test area, and only then run the whole city.

Files: [`metrics.csv`](metrics.csv) (the `nbf compare` output without local paths) and [`size_split.csv`](size_split.csv).
