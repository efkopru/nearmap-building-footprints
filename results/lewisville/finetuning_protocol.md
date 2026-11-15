# Lewisville fine-tuning protocol

Fixed on 2026-11-14, before any outline in the new areas was reviewed and before any score exists for them. It governs the experiment in [#4](https://github.com/efkopru/nearmap-building-footprints/issues/4). The [tuning protocol](tuning_protocol.md), Old Town and the published test square take no part in it.

## Question

Does fine-tuning SAM 3 find more small roofs than the released model can with settings chosen for small roofs, without doing worse on main buildings?

## What is compared

Three runs, scored against the same test labels:

- **Published:** the released SAM 3 with the frozen published setting: prompt "building", 1536 px tiles, confidence 0.7.
- **Released, small-roof settings:** the released SAM 3 with settings chosen on the validation squares.
- **Fine-tuned:** SAM 3 trained on the training squares, with its checkpoint and settings chosen on the validation squares.

The second and third run choose from the same settings:

- tiles of 1024 or 1536 px, overlapping by one eighth;
- a main cut-off of 0.5, 0.6, 0.7 or 0.8;
- a cut-off for outlines under 20 m² of 0.3, 0.4, 0.5, 0.6 or 0.7.

The prompt stays "building". Scores are stored down to 0.3 and the cut-offs are applied with `nbf clean --min-score --small-min-score`. Cleanup is otherwise the default: a 4 m² minimum and no regularization.

**Training.** One run of [`configs/sam3_training.example.yaml`](../../configs/sam3_training.example.yaml) as it stands: image and text backbones frozen, 20 epochs, one image per step. Each training square gives nine 1024 px tiles at a 512 px step. Checkpoints are kept after 5, 10, 15 and 20 epochs, and the fine-tuned run also chooses among them. No other training setting is tried.

## Choosing on the validation squares

For each of the two tuned runs, take the combination with the highest F1 over all sizes, among those whose F1 for buildings of 20 m² and up is no more than 0.01 below the published run's on the same squares. If none qualifies, take the highest F1 for 20 m² and up. Among combinations within 0.005 of the best, prefer the higher small cut-off, then the higher main cut-off, then 1536 px tiles.

The choice is written down with its validation scores before any test square is scored.

## Scoring the test squares

Each run is scored once, with its frozen settings: IoU ≥ 0.5, one-to-one matching, EPSG:26914, outlines crossing a square's edge left out. For each run:

- all sizes: precision, recall and F1;
- 20 m² and up, filtering both sets before matching: precision, recall and F1;
- under 20 m²: how many labeled roofs are found, and how many small outlines are drawn and matched.

Results are given for all test squares together and for the two kinds of square separately. Each figure comes with a 90% range from resampling whole squares 1,000 times. Nothing is changed after the first test score.

## Areas

Chosen by a seeded rule from the city's inventory, the city limits and the imagery grid. No model output was read, so the choice cannot favour a model.

- **Candidates** are the cells of a grid of 2048 × 2048 imagery pixels (208 m), aligned to the mosaic's pixel grid, so each is exactly four 1024 px tiles. A candidate lies wholly inside the city limits, at least 500 m from Old Town and from the published test square, at least 300 m from the two areas whose model outlines appear in the report, and has complete imagery.
- **Two kinds**, from the inventory outlines touching a cell. *Shed-rich:* at least 20 outlines of 20 m² and up and at least 15 under 20 m² (129 cells). *General:* at least 10 outlines of 20 m² and up, or at least 2 of 1,000 m² and up, and not shed-rich (869 cells).
- **Order.** Each kind is put in one random order (NumPy `default_rng`, seed 20261114, over the cells sorted by grid row and column). Three lists are then filled in the order test (16), validation (6), training (16). Every fourth square of a list is general and the others are shed-rich. Each is the next cell of its kind that is unused and at least 300 m from every square of another list; 28 cells were passed over for that reason.
- **Stopping early.** Squares are reviewed in list order, so the first *k* of a list are a random sample too. The experiment uses the first *k* reviewed squares of each list, with at least 8 test, 4 validation and 8 training squares, and *k* is fixed before any scoring.

| Square | Kind | Centre (lat, lon) | Inventory outlines, 20 m² and up | Under 20 m² |
| --- | --- | --- | ---: | ---: |
| test-01 | shed-rich | 33.02881, -96.99879 | 54 | 31 |
| test-02 | shed-rich | 33.01583, -97.01238 | 48 | 18 |
| test-03 | shed-rich | 33.03076, -97.00544 | 54 | 19 |
| test-04 | general | 33.04786, -97.02298 | 27 | 6 |
| test-05 | shed-rich | 33.01219, -97.02135 | 76 | 21 |
| test-06 | shed-rich | 33.01203, -97.00799 | 48 | 16 |
| test-07 | shed-rich | 33.02167, -97.03010 | 124 | 26 |
| test-08 | general | 33.03416, -96.97642 | 13 | 2 |
| test-09 | shed-rich | 33.05930, -97.03839 | 54 | 22 |
| test-10 | shed-rich | 33.03084, -97.01212 | 64 | 27 |
| test-11 | shed-rich | 33.03256, -96.99873 | 64 | 30 |
| test-12 | general | 33.03319, -96.89846 | 23 | 0 |
| test-13 | shed-rich | 33.04020, -97.00974 | 36 | 22 |
| test-14 | shed-rich | 33.03425, -96.98310 | 46 | 16 |
| test-15 | shed-rich | 33.06120, -97.04059 | 50 | 20 |
| test-16 | general | 33.05550, -97.03399 | 18 | 0 |
| val-01 | shed-rich | 33.04434, -97.04309 | 50 | 15 |
| val-02 | shed-rich | 33.05328, -97.00506 | 63 | 28 |
| val-03 | shed-rich | 33.07399, -97.01140 | 36 | 15 |
| val-04 | general | 33.00597, -96.97245 | 4 | 0 |
| val-05 | shed-rich | 32.99149, -97.01724 | 85 | 51 |
| val-06 | shed-rich | 33.06099, -97.02276 | 49 | 15 |
| train-01 | shed-rich | 33.02513, -97.00554 | 48 | 17 |
| train-02 | shed-rich | 33.01192, -96.99908 | 69 | 22 |
| train-03 | shed-rich | 33.08174, -97.03133 | 48 | 20 |
| train-04 | general | 33.05507, -96.99835 | 9 | 0 |
| train-05 | shed-rich | 33.07428, -97.03591 | 58 | 18 |
| train-06 | shed-rich | 33.02326, -97.00557 | 49 | 17 |
| train-07 | shed-rich | 33.03557, -96.93852 | 89 | 26 |
| train-08 | general | 33.03882, -96.89836 | 29 | 0 |
| train-09 | shed-rich | 33.06081, -97.00716 | 30 | 19 |
| train-10 | shed-rich | 33.06860, -97.03155 | 93 | 17 |
| train-11 | shed-rich | 33.04114, -96.93397 | 68 | 27 |
| train-12 | general | 32.99457, -96.96151 | 3 | 0 |
| train-13 | shed-rich | 33.03652, -97.01648 | 75 | 26 |
| train-14 | shed-rich | 33.06268, -97.00713 | 55 | 34 |
| train-15 | shed-rich | 33.04984, -97.03186 | 39 | 15 |
| train-16 | general | 33.05671, -96.97826 | 11 | 2 |

The imagery clips and the outlines to edit are in a local labeling package, outside Git.

## Labels

The same rules for all three lists, so that validation and test are labeled alike:

- **What counts:** every permanent roofed building whose roof is visible in the May 2026 imagery, including detached garages and sheds down to about 2 × 2 m. Open carports, canopies, pergolas, gazebos, tents, trailers and vehicles are left out.
- **Visible roofs only.** The review starts from the inventory outlines touching each square. An outline stays only if its roof can be seen. One on grass, pavement, bare ground or unbroken tree canopy is deleted, whether the building is gone or hidden. Every visible roof without an outline gets one.
- **Outline:** the visible roof edge, one polygon per separately roofed building, keeping the inventory's way of splitting attached buildings. Offsets under about 1 m are left alone.
- **Coverage:** every building touching a square, including those crossing its edge.
- **Imagery only.** No model output of any kind is displayed while reviewing any square, and no candidate list is made from model output for these squares.

The reviewer records the date, the squares completed and any unresolved cases. The labels and squares are then frozen as version 1 with their fingerprints and `nbf review-check` receipts. Training and validation labels may be corrected later in a new version; test labels are not edited after the first test score.

## Development evidence so far

None of this is a test result. It comes from Old Town, the tuning area, with outlines read by Claude on image crops ([#43](https://github.com/efkopru/nearmap-building-footprints/issues/43)).

- Training runs on the 6 GB laptop GPU with both backbones frozen.
- Of 13 small roofs that are clearly visible and outlined, the published setting finds 2 (1 in the city-wide run, whose tile grid falls differently). The released model on 1024 px tiles with a cut-off of 0.5 for small outlines finds 9. Lower cut-offs and smaller tiles find no more.
- At that setting the model draws about 40 small outlines in the square. Of the 38 looked at by eye, 21 are real small roofs (14 of them without an outline in the reviewed layer), 8 are parts of larger roofs, 4 are not buildings and 5 are unclear. The 38 outlines scoring between 0.3 and 0.5 add 3 real roofs.

This is why the second run exists: fine-tuning has to beat the released model at its best settings, not at a setting chosen for main buildings.

## Amendments

Any change is added below with its date and reason, before the step it affects.
