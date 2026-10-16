# Lewisville, Texas: results

SAM 3 and Esri's Mask R-CNN building model on Nearmap's May 2026 imagery of Lewisville, Texas (10.16 cm per pixel). **Reviewed accuracy is reported for one held-out test square.** The earlier city-wide, test-area and Old Town inventory comparisons are retained below as historical agreement results. The [PDF report](Lewisville_building_footprint_comparison.pdf) includes the reviewed results and eight existing screenshots of both models on the imagery (imagery: Nearmap, May 2026).

The historical tables and charts use "2015 inventory" as shorthand for the city's 38,924-feature source, primarily dated 2015 or earlier with 33 undated features. Its 38,882 evaluated city-wide outlines are a separate count; see [reference provenance](reference_provenance.json). Collection dates do not establish individual construction or demolition dates.

Three runs are compared:

- **Esri, city run:** Esri's Building Footprint Extraction – USA model (Mask R-CNN) with its standard post-processing, run by the city on the same imagery and published.
- **SAM 3:** the text prompt "building", 1536 px tiles and a confidence threshold of 0.7.
- **Esri, local raw run:** the same Esri model run here without that post-processing, with a confidence threshold of 0.9.

## Reviewed test-area accuracy

Esad completed the Old Town and test-area reference reviews on October 15, 2026 (America/Chicago). The reviewer confirmed that the test review used only aerial imagery and historical outlines, without exposure to SAM 3 or Esri predictions before or during review, and that the documented rules were followed with no unresolved cases. The frozen test reference has **640 outlines: 600 fully inside the 700 m square and 40 crossing its edge**. Coverage and independence are human attestations recorded in the [review provenance](reviewed_2026/provenance.json).

Settings were selected against the historical Old Town inventory under the [frozen protocol](tuning_protocol.md) and were not retuned on the reviewed test results. The existing predictions were reused. Differences from the historical scores reflect the changed reference and explicit filtering policy, not improved models. The primary evaluation applies the AOI boundary rule, filters both datasets to **at least 20 m² before matching**, then matches one to one at IoU ≥ 0.5 in EPSG:26914. It includes 468 reviewed reference buildings.

| Method | Matched (TP) | Extra (FP) | Missed (FN) | Precision | Recall | F1 | Median IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Esri, city run | 445 | 16 | 23 | 0.965 | 0.951 | **0.958** | 0.761 |
| SAM 3 | 432 | 90 | 36 | 0.828 | 0.923 | **0.873** | 0.758 |
| Esri, local raw run | 423 | 29 | 45 | 0.936 | 0.904 | **0.920** | 0.793 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/reviewed-test-accuracy-dark.svg">
  <img alt="Reviewed test-area accuracy for three frozen methods, with separate all-size and at-least-20-square-metre panels. Primary F1: city Esri 0.958, SAM 3 0.873 and local Esri 0.920." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/reviewed-test-accuracy.svg">
</picture>

**Separate all-size evaluation.** The full 600-building reference is matched without a new minimum-area filter. The review convention was approximately 4 m²; its three retained outlines just below 4 m² are included as reviewed. These results are not inferred from the primary run:

| Method | Matched (TP) | Extra (FP) | Missed (FN) | Precision | Recall | F1 | Median IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Esri, city run | 445 | 16 | 155 | 0.965 | 0.742 | **0.839** | 0.761 |
| SAM 3 | 436 | 93 | 164 | 0.824 | 0.727 | **0.772** | 0.757 |
| Esri, local raw run | 428 | 31 | 172 | 0.932 | 0.713 | **0.808** | 0.792 |

Of the 132 reviewed outlines under 20 m², SAM 3 and the local Esri run each match 4 (3.0% recall); the city Esri run matches none. Size classes summarize each object's own area in the all-size matching, so a pair can cross a class boundary. Their threshold summaries must not be substituted for the primary before-matching filter. Full-precision [scores](reviewed_2026/test_area_scores.csv), [size diagnostics](reviewed_2026/test_area_by_size.csv) and [scope and provenance](reviewed_2026/README.md) are stored separately.

**Limits.** This is one reviewed test square, with labels corrected from the historical inventory rather than independently redrawn. Its unchanged geometry and building identities remain inherited, and no second reviewer is recorded. The result does not establish city-wide or out-of-area accuracy. Old Town was also reviewed but remains the area used for tuning, not an additional held-out test. Labels, imagery, full receipts and individual matches remain local.

## Historical inventory agreement

All numbers in the following city-wide, historical test-area and Old Town sections use the older inventory. Their original CSVs are unchanged.

- **On the historical test-area comparison,** F1 on inventory outlines of 20 m² and up was 0.95 for the city's Esri run, 0.91 for the local Esri run and 0.87 for SAM 3.
- **Esri leads city-wide,** with F1 0.83 against SAM 3's 0.74 on buildings of 20 m² and up. Both find most of them (recall 0.85 and 0.82), but SAM 3 draws more outlines the 2015 map lacks (precision 0.68 against 0.82).
- **Small structures are missed.** Of the 7,867 outlines under 20 m², SAM 3 found 267 and the city's Esri run 2.
- **Model agreement identifies review candidates.** 4,923 SAM 3 outlines without an inventory match also match an Esri outline. These are candidates to inspect, not dated evidence of new or changed buildings.
- **Large buildings break up.** SAM 3 splits buildings longer than a tile where tiles meet, and Esri's outlines are cut at its chip edges.
- **Settings matter.** Without Esri's post-processing, the same model scores F1 0.77 instead of 0.83. For SAM 3, 1536 px tiles and a confidence threshold of 0.7 raised Old Town F1 from 0.48 to 0.53.

## Whole city: historical inventory agreement

All three runs went through the same cleanup and were scored against 38,882 inventory outlines within the city-wide evaluation boundary. The 38,924 source features and 42-feature difference are distinct counts; see [provenance](reference_provenance.json). SAM 3 ran on 6,538 tiles; the local Esri run in 150 chunks.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/found-by-size-dark.svg">
  <img alt="Bar chart of recall by building area, the share of the 2015 outlines each model found. 100 m² and up: Esri Mask R-CNN 94%, SAM 3 90%. 50 to 100 m²: 52% and 39%. 20 to 50 m²: 8% and 25%. Under 20 m²: 0% and 3%." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/found-by-size.svg">
</picture>

| Method | Outlines | Matched (TP) | Extra (FP) | Missed (FN) | Precision | Recall | F1 | F1, 20 m² and up | Median IoU |
|---|---|---|---|---|---|---|---|---|---|
| Esri, city run | 32,608 | 26,502 | 6,106 | 12,380 | 0.81 | 0.68 | 0.74 | 0.83 | 0.77 |
| SAM 3 | 37,906 | 25,758 | 12,148 | 13,124 | 0.68 | 0.66 | 0.67 | 0.74 | 0.80 |
| Esri, local raw run | 38,860 | 26,536 | 12,324 | 12,346 | 0.68 | 0.68 | 0.68 | 0.77 | 0.80 |

Extra outlines are false positives against the inventory. They may reflect omitted structures, changed structures, roof-versus-ground conventions, registration differences or model errors. Model agreement cannot distinguish these explanations by itself. SAM 3's matched outlines fit slightly more tightly (median IoU 0.80 against 0.77).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/outcomes-by-size-dark.svg">
  <img alt="Stacked bars of what happened to each 2015 outline, by building area. 100 m² and up: 89% found by both. 50 to 100 m²: 31% both, 21% only Esri Mask R-CNN, 8% only SAM 3, 39% missed. 20 to 50 m²: 20% only SAM 3, 72% missed. Under 20 m²: 97% missed." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/outcomes-by-size.svg">
</picture>

| Building area | 2015 outlines | Found by Esri | Found by SAM 3 | Both | Only Esri | Only SAM 3 | Neither |
|---|---|---|---|---|---|---|---|
| 100 m² and up | 26,959 | 25,410 (94%) | 24,245 (90%) | 23,947 | 1,463 | 298 | 1,251 |
| 50–100 m² | 1,710 | 896 (52%) | 669 (39%) | 530 | 366 | 139 | 675 |
| 20–50 m² | 2,346 | 194 (8%) | 577 (25%) | 107 | 87 | 470 | 1,682 |
| under 20 m² | 7,867 | 2 (0%) | 267 (3%) | 1 | 1 | 266 | 7,599 |

Most of the 11,207 outlines missed by both (7,599) are under 20 m². A missing detection does not establish demolition: occlusion, inventory geometry and model errors also need review. Adding the local Esri run, 10,463 are missed by all three.

**SAM 3 against Esri, without the inventory.** Matched to each other, the two models form 29,567 pairs with a median IoU of 0.81: 91% of Esri's outlines have a SAM 3 partner and 78% of SAM 3's have an Esri one. Agreement on an outline without an inventory match is a review candidate. Both models can reproduce the same error; agreement supplies neither a construction date nor independent reference labels.

**What the screenshots show.** In a residential area missing from the inventory, both models outline the visible houses. At an apartment complex, Esri outlines long carport rows that SAM 3 mostly misses. In another residential area, SAM 3 outlines more back-yard garages and sheds. Long industrial roofs are split where SAM 3's tiles meet, and some inventory outlines overlap apparently open ground. These are observations in the May 2026 imagery, not verified construction, change or demolition dates.

## Test area: historical inventory agreement

Each model's settings were frozen first (see [tuning](#old-town-baseline-and-tuning) below), and they are exactly the settings of the city-wide runs. This earlier comparison used the historical inventory in the same seeded-random 700 m test square, before the October 2026 human review. It is preserved separately from the reviewed evaluation above.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/test-area-dark.svg">
  <img alt="Historical test-area inventory agreement, not reviewed accuracy. Recall: Esri Mask R-CNN 94% and SAM 3 91% of the 471 inventory buildings. Precision: 97% and 82%." src="https://raw.githubusercontent.com/efkopru/nearmap-building-footprints/main/docs/images/test-area.svg">
</picture>

| Method | Matched (TP) | Extra (FP) | Missed (FN) | Precision | Recall | F1 | Median IoU | F1, all sizes |
|---|---|---|---|---|---|---|---|---|
| Esri, city run | 445 | 16 | 26 | 0.97 | 0.94 | 0.95 | 0.76 | 0.83 |
| SAM 3 | 430 | 92 | 41 | 0.82 | 0.91 | 0.87 | 0.75 | 0.76 |
| Esri, local raw run | 422 | 30 | 49 | 0.93 | 0.90 | 0.91 | 0.79 | 0.80 |

Buildings of 20 m² and up, except the last column. Historical inventory agreement was much higher here than in Old Town, with the same ordering of methods. The cause of that difference has not been established. In this historical test comparison, at most 2 of the 136 inventory outlines under 20 m² were found.

The protocol requires filtering both reference and prediction polygons to at least 20 m² before matching. `nbf evaluate --min-area-m2 20` implements that policy; an all-size run summarized with `--size-bins-m2 20 50 100` retains its global matching and can give different threshold metrics. The historical CSV values are preserved, not replaced by a new run. Their execution policy must be established from evidence, not inferred from the protocol; see the [scoring audit](scoring_audit.md).

The audit reproduces all six published city/test ≥20 m² rows by keeping global matches only when both objects are at least 20 m². A fresh before-matching filter in the small test AOI produces the same counts, so those values do not prove the historical order. The whole city and 176 tuning variants were not rerun with that filter. Generic size-bin summaries use each side's matches separately and need not equal either procedure.

## Old Town: baseline and tuning

**Baseline.** The first comparison used a 700 m square around Old Town with 319 of the 2015 outlines, with every method at its default settings: SAM 3 at 1024 px tiles and a confidence threshold of 0.5, and the local Esri run at 0.5.

| Method | Matched (TP) | Extra (FP) | Missed (FN) | Precision | Recall | F1 | Median IoU |
|---|---|---|---|---|---|---|---|
| SAM 3 | 142 | 287 | 177 | 0.33 | 0.45 | 0.38 | 0.74 |
| Esri, city run | 135 | 63 | 184 | 0.68 | 0.42 | 0.52 | 0.75 |
| Esri, local raw run | 158 | 381 | 161 | 0.29 | 0.50 | 0.37 | 0.74 |

All three found 63–71% of the outlines of 50 m² and up, but at most 10% of the smaller ones.

**Tuning.** The goal, fixed in advance in the [tuning protocol](tuning_protocol.md), was the best F1 on buildings of 20 m² and up in the Old Town square. With no reviewed outlines available, tuning used the 2015 outlines ([Amendment 1](tuning_protocol.md#amendment-1-2026-09-29)). The rule froze the highest F1, or, within 0.01 of it, the variant closest to the defaults. Best F1 for some of the 176 variants (all are in [tuning_final.csv](tuning_final.csv)):

| Method | Setting | Precision | Recall | F1 |
|---|---|---|---|---|
| Esri, city run | as published (frozen) | 0.69 | 0.55 | 0.61 |
| SAM 3 | "building", 2048 px, ≥ 0.75 | 0.60 | 0.48 | 0.54 |
| SAM 3 | "building", 1536 px, ≥ 0.7 (frozen) | 0.53 | 0.52 | 0.53 |
| Esri, local raw run | ≥ 0.9 (frozen) | 0.56 | 0.50 | 0.53 |
| SAM 3 | "house", 1536 px, ≥ 0.7 | 0.60 | 0.42 | 0.50 |
| SAM 3 | "roof", 2048 px, ≥ 0.75 | 0.54 | 0.46 | 0.49 |
| SAM 3 | "building", 1024 px, ≥ 0.75 | 0.48 | 0.47 | 0.48 |

For SAM 3, bigger tiles and higher confidence thresholds help up to about 0.75; above that, recall falls faster than precision rises. Its best variant (2048 px, 0.54) was within 0.01 of 1536 px at 0.7, so the rule kept the latter. Regularization never changed a score and stays off.

## Runtime on a laptop RTX 4050 (6 GB)

| Run | Area | Time |
|---|---|---|
| SAM 3, 1536 px tiles | whole city, 6,538 tiles | 2 h 41 min (median 1.08 s per tile) |
| SAM 3, 1024 px tiles | Old Town, 90 tiles | 105 s (0.82 s per tile) |
| Esri, local raw run | whole city, 150 chunks | 8 h 46 min of model time |
| Esri, local raw run | Old Town, 0.64 km² | 4.9 min |

SAM 3 used up to 96% of the GPU's memory, close to the limit but stable through the city-wide run.

## Historical runs and shared scoring settings

- **Imagery:** a Nearmap mosaic from May 2026 at 10.16 cm, clipped to the city limits. Its GeoTIFF declared the wrong CRS (a local one in metres); it is EPSG:2276 in US survey feet, fixed with a VRT without resampling and checked against the city limits.
- **Historical reference:** the City of Lewisville's public inventory, primarily dated 2015 or earlier, with 33 undated source features. It has 38,924 source features and 38,882 evaluated city-wide outlines; [provenance](reference_provenance.json) records the distinction. The report observed alignment to within about 0.4 m; that does not establish label completeness or temporal accuracy. The reviewed test uses the separate October 2026 snapshot described above.
- **Esri, city run:** `ExtractFeaturesUsingAIModels` with default settings. **Local raw run:** `DetectObjectsUsingDeepLearning` without non-maximum suppression (NMS); city-wide in 150 overlapping chunks of about 1.1 km, each keeping only the outlines centred in its own core.
- **SAM 3:** text prompt through SamGeo 1.4.2 at a pinned commit; 1536 px tiles overlapping by 192 px.
- **Cleanup and scoring:** prediction cleanup uses a 4 m² minimum and duplicate removal. Every method is scored in EPSG:26914 with one-to-one matching at IoU ≥ 0.5 and buildings crossing an area's edge left out. The reviewed primary run additionally filters both sets at 20 m² before matching; its all-size run retains the reviewed reference without another area filter. Historical threshold rows retain the policy limitations documented in the [scoring audit](scoring_audit.md).
- **Screenshot areas** were picked by a fixed rule and kept at least 500 m from the Old Town and test squares.

## Next steps

The October 2026 review and frozen test evaluation are complete. Further experiments need separate references and versions so these test results remain preserved.

1. **For broader accuracy evidence:** review additional geographically separate areas and measure agreement between independent annotators. Keep their labels blind to model outputs and freeze any new settings before scoring them.
2. **For small structures:** fine-tune a model on independently labelled structures under 20 m²; current recall is low.
3. **For large buildings:** merge outlines across tile seams, or use larger tiles in commercial areas.

**Files:** [reviewed test scores and provenance](reviewed_2026/README.md), [historical reference provenance](reference_provenance.json), [citywide_scores.csv](citywide_scores.csv), [citywide_by_size.csv](citywide_by_size.csv), [test_area_scores.csv](test_area_scores.csv), [old_town_scores.csv](old_town_scores.csv), [old_town_by_size.csv](old_town_by_size.csv), [tuning_final.csv](tuning_final.csv) and the [tuning protocol](tuning_protocol.md). Rebuild the PDF with `python scripts/build_lewisville_report.py` from the repository root; it requires pypdf, Pillow and Edge or Chrome. The [builder](../../scripts/build_lewisville_report.py) uses public aggregate files and the existing PDF's eight screenshots, without the original private report page. The October 2026 revision adds reviewed test results while retaining historical values and screenshots; no new model inference was run.
