# Lewisville: reviewed 2026 test-area accuracy

These results score the three frozen methods against a reference reviewed for the May 2026 Nearmap imagery. They cover one seeded-random 700 m test square. The [historical results](../README.md#historical-inventory-agreement) remain agreement with the older inventory and have not been overwritten.

**Completed October 15, 2026:** reference review [#2](https://github.com/efkopru/nearmap-building-footprints/issues/2) and evaluation [#3](https://github.com/efkopru/nearmap-building-footprints/issues/3) are closed following merged [PR #29](https://github.com/efkopru/nearmap-building-footprints/pull/29). This directory is the frozen baseline. The remaining small-building fine-tuning work in [#4](https://github.com/efkopru/nearmap-building-footprints/issues/4) is deferred and must use separate training, validation and fresh test data.

## Review and reference

Esad confirmed completion of the Old Town and test-area reviews on October 15, 2026, in America/Chicago. The reviewer confirmed following the documented rules, completing both areas with no unresolved cases, and using only aerial imagery and historical outlines for the test review, without seeing SAM 3 or Esri predictions before or during it. Coverage and independence are reviewer attestations. Geometry checks cannot establish them.

The convention is the visible roof edge of permanent, roofed buildings, including detached garages and sheds down to about 2 by 2 m. Open carports, canopies, pergolas, tents, trailers and vehicles are excluded. The review retains the inventory's splitting of attached buildings, uses one polygon per separately roofed building when adding missing outlines, and leaves slight offsets under about 1 m unchanged. Buildings crossing the scored square are included in the review and excluded from scoring.

| Area | Reviewed outlines | Fully inside | Crossing the boundary | Role |
| --- | ---: | ---: | ---: | --- |
| Test area | 640 | 600 | 40 | Held-out accuracy evaluation |
| Old Town | 393 | 374 | 19 | Previously used for tuning; not another test area |

The labels are reviewed corrections of the historical inventory. Unchanged geometry, building identities and its splitting convention remain inherited from that source. No second reviewer or independent redraw is recorded. The single test square does not establish city-wide accuracy or performance in other places.

## Scoring

The predictions and settings were frozen after selection against the historical Old Town inventory. This evaluation does not rerun inference or retune on the reviewed test labels. Differences from historical scores reflect the changed reference and explicit scoring policy, not model improvement. SAM 3 uses the prompt `building`, 1536 px tiles, 192 px overlap and confidence 0.7; the local Esri run uses confidence 0.9; the city's Esri run is used as published. Prediction cleanup uses a 4 m² minimum and no regularization.

Both scoring runs use EPSG:26914, one-to-one matching at IoU ≥ 0.5 and the `exclude` boundary policy:

- **Primary, at least 20 m²:** filter both predictions and reference outlines after AOI selection and before matching. The reference cohort has 468 buildings.
- **All sizes:** match the full 600-building reference separately, then summarize that matching by each object's area. The reference includes 132 outlines under 20 m². The review's approximate 4 m² convention left three outlines just below 4 m², with a minimum of about 3.5 m²; these are retained as reviewed, without an extra reference-area filter. A matched pair can cross a size-class boundary, so these diagnostics do not replace the primary filtered evaluation.

| Cohort | Method | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| At least 20 m² | Esri, city run | 445 | 16 | 23 | 0.965 | 0.951 | 0.958 |
| At least 20 m² | SAM 3 | 432 | 90 | 36 | 0.828 | 0.923 | 0.873 |
| At least 20 m² | Esri, local raw run | 423 | 29 | 45 | 0.936 | 0.904 | 0.920 |
| All sizes | Esri, city run | 445 | 16 | 155 | 0.965 | 0.742 | 0.839 |
| All sizes | SAM 3 | 436 | 93 | 164 | 0.824 | 0.727 | 0.772 |
| All sizes | Esri, local raw run | 428 | 31 | 172 | 0.932 | 0.713 | 0.808 |

Precision is the proportion of evaluated predictions matched to the reviewed reference; recall is the proportion of reviewed buildings matched; F1 combines them. IoU summaries describe matched pairs only and exclude missed buildings. Values called added or removed in the review audit describe inventory edits, not construction or demolition dates.

## Files and reproduction boundary

- [test_area_scores.csv](test_area_scores.csv): six aggregate rows, one per method for each scoring cohort, including counts, precision, recall, F1 and matched IoU.
- [test_area_by_size.csv](test_area_by_size.csv): size-class diagnostics from the separate all-size run.
- [provenance.json](provenance.json): sanitized review provenance, settings, counts and hashes linking the published scores to the frozen local inputs and receipts.

Imagery, reviewed labels, full receipts, prediction geometry and per-building matches remain local and outside Git. Public aggregates and fingerprints permit provenance checks but do not supply the source data needed to reproduce the geometric evaluation. The [usage guide](../../../docs/USAGE.md#8-score-and-compare) documents review receipts and both scoring modes. Historical CSVs, the original reference snapshot and the [historical scoring audit](../scoring_audit.md) are preserved separately.

With the frozen local evidence available, run `python scripts/publish_reviewed_accuracy.py --check` from the repository root to verify these three exports. The [exporter](../../../scripts/publish_reviewed_accuracy.py) checks source bindings and completed-run records without rerunning inference or evaluation. It refuses to overwrite existing exports.

The [accuracy recipe](../../../configs/accuracy.example.json) records the six evaluations and two comparisons used for these results. Its output paths belong to the completed run; do not rerun it over those artifacts. Future experiments need separate output directories, new input and review records, and a fresh test cohort. Preserve this test reference and its scores rather than using them to select fine-tuning settings.
