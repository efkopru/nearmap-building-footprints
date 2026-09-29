# Lewisville tuning protocol

Fixed on 2026-09-27, before any tuned result was seen.

## Areas

- **Tuning area:** the Old Town 700 m square, scored against its outlines once they have been reviewed against the 2026 imagery.
- **Test area:** a 700 m square chosen by a seeded random rule (seed 20260927, accepted on attempt 1) centred at 33.03169° N, 97.02861° W. The rule required the square to lie fully inside the city limits, touch at least 150 of the 2015 outlines, sit at least 2 km from Old Town, and not overlap any area where model outlines had been shown. It is scored once, with frozen settings, against its reviewed outlines. Its outlines are reviewed without viewing any model output.

## Objective

- **Primary:** F1 at IoU ≥ 0.5 on main buildings. After cleanup, both reference outlines and predictions under 20 m² are dropped before matching.
- **Also reported, not optimized:** recall for structures under 20 m² (sheds), the 50 m² split used in the untuned baseline, median IoU of matches, and runtime.

## Variants

- **SAM 3:** text prompt "building", "house" or "roof" × tile size 1024, 1536 or 2048 px (overlap one eighth), all run at confidence 0.3. Confidence cut-offs 0.3, 0.4, 0.5, 0.6 and 0.7 are then applied afterwards, from the stored scores.
- **Esri, local raw run:** confidence cut-offs 0.5, 0.6, 0.7, 0.8 and 0.9, applied to the existing run's scores.
- **Esri, city run:** fixed as published; only cleanup varies.
- **Cleanup, for every method:** regularization off, or on at 0.5 m maximum displacement and 10% maximum area change. The minimum area stays 4 m², since the objective filters at 20 m².

## Selection

For each method, keep the variant with the highest primary F1 on the tuning area. When F1 values are within 0.01, prefer the variant closer to the defaults. Freeze those settings, then score the test area once.

## Amendment 1, 2026-09-29

Recorded before the extended variants below were scored.

- **Why:** the Old Town and test-area reviews have not been done, and the project is being finalized now. The review packages stay ready; reviewed scores can be added later with the frozen settings, without tuning again.
- **Tuning reference:** the 2015 outlines in the Old Town square, scored exactly as in the preliminary runs. The test area is scored once, with frozen settings, against its 2015 outlines. Every result is therefore agreement with the 2015 inventory, not accuracy.
- **Extended cut-offs:** in the preliminary runs the best cut-offs sat at the top of the fixed ranges (0.7 for SAM 3, 0.9 for the local Esri run), so the ranges are extended: SAM 3 adds 0.75, 0.8, 0.85 and 0.9; the local Esri run adds 0.95. This was decided after seeing the preliminary runs.
- **"Closer to the defaults", made explicit:** the defaults are SAM 3 "building", 1024 px, cut-off 0.5; the local Esri run at 0.5; regularization off. Among variants within 0.01 of the best F1, prefer the fewest settings changed from the defaults, then the smaller cut-off change, then the smaller tile-size change.
- **Test-area outputs:** each method is scored in the test area from its city-wide run when that run used the frozen settings; otherwise the frozen settings are run on the test area alone.
