# Lewisville, Texas: results so far

This page compares SAM 3 and Esri's building model on Nearmap's May 2026 imagery of Lewisville (10.16 cm). **Every score here measures agreement with the city's building outlines collected in 2015 or earlier, not accuracy.** Accuracy comes once two sample areas are reviewed against the 2026 imagery. Maps of the results stay in a private report because the imagery is licensed; this page has aggregate numbers only.

## Key findings

- **Houses are found.** City-wide, the city's Esri run matches 94% of the 2015 outlines of 100 m² and up. In residential streets, SAM 3 and Esri outline nearly the same houses.
- **Sheds are missed.** No method reliably finds structures under 20 m². Esri's city run matched 2 of 7,867 such outlines.
- **Large buildings break up.** Buildings longer than a processing tile come out in pieces: SAM 3 finds only parts, and Esri's outlines are cut at its chip edges.
- **SAM 3's settings matter.** On main buildings in Old Town, 1536 px tiles, the prompt "building" and a 0.7 cut-off raised its F1 to 0.53, from 0.46 with the best 1024 px setting. That ties the local Esri run and trails the city's Esri run at 0.61.

## City-wide: the city's Esri run against the 2015 outlines

The city ran Esri's Building Footprint Extraction – USA model on the same May 2026 capture. Scored against the 38,882 older outlines inside the city limits: 26,502 matched, 6,106 Esri buildings with no 2015 outline, 12,380 2015 outlines unmatched. Precision 0.81, recall 0.68, F1 0.74, median IoU of matches 0.77.

| Building size | 2015 outlines | Found by Esri | Share |
|---|---|---|---|
| under 20 m² | 7,867 | 2 | 0.0% |
| 20–50 m² | 2,346 | 194 | 8.3% |
| 50–100 m² | 1,710 | 896 | 52.4% |
| 100 m² and up | 26,959 | 25,410 | 94.3% |

Most unmatched 2015 outlines are sheds: 10,017 of the 12,380 are under 50 m². The Esri buildings with no 2015 outline likely include buildings built since 2015.

## Old Town sample: the untuned baseline

A 700 m square around Old Town with 319 of the 2015 outlines, every method at its default settings. Details are in [lewisville_old_town](../lewisville_old_town/README.md).

| Method | Matched | Extra | Missed | Precision | Recall | F1 | Median IoU |
|---|---|---|---|---|---|---|---|
| SAM 3 | 142 | 287 | 177 | 0.33 | 0.45 | 0.38 | 0.74 |
| Esri, city run | 135 | 63 | 184 | 0.68 | 0.42 | 0.52 | 0.75 |
| Esri, local raw | 158 | 381 | 161 | 0.29 | 0.50 | 0.37 | 0.74 |

| Method | Recall ≥ 50 m² | Precision ≥ 50 m² | Recall < 50 m² |
|---|---|---|---|
| SAM 3 | 0.63 | 0.39 | 0.10 |
| Esri, city run | 0.64 | 0.69 | 0.01 |
| Esri, local raw | 0.71 | 0.36 | 0.10 |

## First tuning runs (preliminary)

The objective, fixed in advance in the [tuning protocol](tuning_protocol.md), is F1 at IoU ≥ 0.5 on main buildings of 20 m² and up in the Old Town square. These scores use the 2015 outlines. Settings are chosen once Old Town's outlines have been reviewed, then scored once on a separate test area. Every combination is in [tuning_preliminary.csv](tuning_preliminary.csv).

Best F1 for each setting. These scores count only buildings of 20 m² and up, so they run higher than the baseline table above. Shed recall is the share of 2015 outlines under 20 m² that were found.

| Method | Setting | Precision | Recall | F1 | Shed recall |
|---|---|---|---|---|---|
| Esri, city run | as published | 0.69 | 0.55 | 0.61 | 0.00 |
| Esri, local raw | ≥ 0.9 | 0.56 | 0.50 | 0.53 | 0.03 |
| SAM 3 | "building", 1536 px, ≥ 0.7 | 0.53 | 0.52 | 0.53 | 0.03 |
| SAM 3 | "building", 2048 px, ≥ 0.7 | 0.55 | 0.50 | 0.53 | 0.01 |
| SAM 3 | "house", 1536 px, ≥ 0.7 | 0.60 | 0.42 | 0.50 | 0.03 |
| SAM 3 | "house", 2048 px, ≥ 0.5 | 0.49 | 0.48 | 0.48 | 0.04 |
| SAM 3 | "roof", 2048 px, ≥ 0.7 | 0.48 | 0.47 | 0.47 | 0.01 |
| SAM 3 | "building", 1024 px, ≥ 0.7 | 0.44 | 0.49 | 0.46 | 0.04 |
| SAM 3 | "roof", 1536 px, ≥ 0.7 | 0.41 | 0.50 | 0.45 | 0.03 |
| SAM 3 | "house", 1024 px, ≥ 0.4 | 0.38 | 0.51 | 0.44 | 0.14 |
| SAM 3 | "roof", 1024 px, ≥ 0.7 | 0.30 | 0.47 | 0.36 | 0.08 |

SAM 3 F1 by prompt, tile size and confidence cut-off:

| Prompt | Tile size | ≥ 0.3 | ≥ 0.4 | ≥ 0.5 | ≥ 0.6 | ≥ 0.7 |
|---|---|---|---|---|---|---|
| "building" | 1024 px | 0.38 | 0.41 | 0.43 | 0.44 | 0.46 |
| "building" | 1536 px | 0.43 | 0.46 | 0.46 | 0.49 | 0.53 |
| "building" | 2048 px | 0.43 | 0.46 | 0.47 | 0.51 | 0.53 |
| "house" | 1024 px | 0.43 | 0.44 | 0.43 | 0.43 | 0.42 |
| "house" | 1536 px | 0.44 | 0.47 | 0.46 | 0.47 | 0.50 |
| "house" | 2048 px | 0.44 | 0.47 | 0.48 | 0.48 | 0.45 |
| "roof" | 1024 px | 0.29 | 0.31 | 0.34 | 0.35 | 0.36 |
| "roof" | 1536 px | 0.34 | 0.37 | 0.39 | 0.41 | 0.45 |
| "roof" | 2048 px | 0.36 | 0.40 | 0.42 | 0.44 | 0.47 |

Bigger tiles and higher cut-offs help. The best cut-off, 0.7, is the top of the range fixed in advance, so testing higher cut-offs is proposed as a protocol change before the reviewed outlines are used. The local Esri run's cut-off sweep: ≥ 0.5: 0.43 · ≥ 0.6: 0.46 · ≥ 0.7: 0.49 · ≥ 0.8: 0.51 · ≥ 0.9: 0.53.

## Runtime on a laptop RTX 4050 (6 GB)

| Run | Area or tiles | Time | City-wide estimate |
|---|---|---|---|
| SAM 3, 1024 px tiles | 90 tiles | 105 s (0.82 s per tile) | about 3.3 h for 14,491 tiles |
| SAM 3, 1536 px tiles | 36 tiles | 93 s, including model loading | about 3.5 h |
| SAM 3, 2048 px tiles | 25 tiles | 126 s, including model loading | about 4 h |
| Esri, local raw run | 0.64 km² | 4.9 min | about 15 h |

SAM 3 used up to 96% of the GPU's memory.

## How this was run

- **Imagery:** a Nearmap mosaic from May 2026, 10.16 cm, clipped to the city limits. Its GeoTIFF declared a local CRS in metres; it is EPSG:2276 in US survey feet, corrected through a VRT and checked against the city limits.
- **Reference:** the City of Lewisville's public building outlines, collected in 2015 or earlier. They align with the imagery to within about 0.4 m.
- **Methods:**
  - **SAM 3:** a text prompt through SamGeo 1.4.2, at a pinned commit.
  - **Esri, city run:** Building Footprint Extraction – USA through `ExtractFeaturesUsingAIModels` with default settings, published by the city.
  - **Esri, local raw run:** the same model through `DetectObjectsUsingDeepLearning` without NMS.
- **Scoring,** identical for every method: EPSG:26914, 4 m² minimum, duplicate suppression, one-to-one matching at IoU ≥ 0.5, and buildings crossing an area's edge excluded.

## Next

1. Review Old Town's outlines against the 2026 imagery, then choose each model's settings by the fixed rule.
2. Review the test area, a 700 m square chosen by a seeded random rule, and score the frozen settings there once. That is the accuracy result.
3. Then decide on city-wide runs, and on fine-tuning if sheds matter.
