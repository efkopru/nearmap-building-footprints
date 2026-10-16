# Reference provenance and historical scoring audit

Audited locally on 2026-10-15. This records evidence from existing files; it does not change historical scores, establish extraction accuracy, or reconstruct an unrecorded acquisition history.

## Reference snapshot

The saved source export records [3D_Buildings_082020, FeatureServer layer 0](https://services2.arcgis.com/kXGqZY4GIOcEYxoF/arcgis/rest/services/3D_Buildings_082020/FeatureServer/0). Its coordinate metadata is EPSG:2276. The name `082020` is not the collection date of every building.

The [provenance manifest](reference_provenance.json) records SHA-256 hashes for the original GeoJSON, source GeoPackage and normalized GeoPackage, plus geometry fingerprints, ordered ID hashes and links to all six existing result CSVs. It publishes no geometries, individual feature records, private absolute paths or credentials. A source URL identifies a service; the saved hashes identify the specific local snapshot used here. A fresh service download is not presumed identical.

| Scope | `Pre_2015` | `2015` | Unknown | Total |
| --- | ---: | ---: | ---: | ---: |
| Source and normalized inventory | 34,993 | 3,898 | 33 | 38,924 |
| Evaluated city reference | 34,967 | 3,882 | 33 | 38,882 |
| Evaluated Old Town reference | 263 | 33 | 23 | 319 |
| Evaluated test-area reference | 568 | 39 | 0 | 607 |

The city evaluation excludes 29 outlines crossing the boundary and 13 outside it. All 33 undated records remain in the city evaluation. The review packages contain 341 Old Town and 647 test-area outlines touching their AOIs; boundary exclusions reduce these to 319 and 607 scored outlines. Of those, 246 and 471 respectively have area at least 20 m².

Source and normalized GeoPackages retain the same 38,924 OBJECTIDs in the same order. The five retained attributes (`OBJECTID`, `Year_Colle`, `FEATURE_TY`, `Class`, `BLDGHEIGHT`) are unchanged. Exactly one geometry changes: the source has one invalid polygon, and applying `make_valid` reproduces the normalized geometries exactly. This establishes the observed transformation. The original normalization command was not retained in the inspected records.

The retrieval time, API query/filter/pagination parameters, service revision at retrieval, historical licence terms and authoritative roof-versus-ground convention are **unknown** from the inspected local evidence. Missing collection dates have not been filled in or inferred from edit timestamps. The current service cannot retroactively establish those facts.

## What the historical scores establish

The three stored city evaluation reports reference the normalized snapshot's exact geometry and row-order fingerprints, the same city AOI, EPSG:26914, IoU 0.5 and exclusion of boundary-crossing outlines. The audit recomputes the evaluated reference IDs and every stored matched pair's IoU against the local prediction/reference rows. These checks pass.

All six published city/test rows labelled **20 m² and up** can be numerically reconstructed as follows:

1. Use one global matching over the evaluated outlines of all sizes.
2. Retain a matched pair only if **both** its predicted outline and reference outline have area at least 20 m².
3. Use the retained pair count as TP. Subtract it from the number of predictions and references at least 20 m² to obtain FP and FN.

| Method | City TP / FP / FN | Test TP / FP / FN |
| --- | --- | --- |
| Esri, city run | 26,496 / 6,011 / 4,519 | 445 / 16 / 26 |
| SAM 3 | 25,482 / 11,990 / 5,533 | 430 / 92 / 41 |
| Esri, local raw run | 26,107 / 11,025 / 4,908 | 422 / 30 / 49 |

Precision, recall and F1 also reproduce the published four-decimal values. This is a verified numerical reconstruction, **not proof of the historical operation order**: the real-data script or complete scoring invocation was not retained in the inspected files.

Filtering both input layers before matching, as specified in the [tuning protocol](tuning_protocol.md), is a distinct procedure. A fresh geometric rerun of that procedure in the small test AOI gives the same TP/FP/FN values for all three methods. Consequently, those test values cannot distinguish the two procedures. This audit does not rerun filtering-before-matching over the whole city or all 176 tuning variants, and does not claim that every historical row complied with that protocol.

The distinction matters because global matches can cross the 20 m² threshold:

| City method | Prediction ≥20, reference <20 | Prediction <20, reference ≥20 |
| --- | ---: | ---: |
| Esri, city run | 0 | 4 |
| SAM 3 | 129 | 9 |
| Esri, local raw run | 12 | 77 |

The generic size-bin metrics are a third interpretation: they retain global matching and compute recall from reference-side matches and precision from prediction-side matches independently. They need not equal the historical pair-filtered counts or a new match after filtering both layers. Existing CSVs remain unchanged; future scoring must state the filter/matching order explicitly.

## Tuning and notebook evidence

The public 176-row tuning CSV matches the local full-precision tuning CSV after column renaming and rounding. The frozen-settings receipt retains the selected thresholds and counts. Nine SAM 3 inference receipts record the prompt, a 0.3 inference confidence floor, tile count, checkpoint hash and manifest hash. They establish inference inputs and completed runs, not the subsequent scoring filter order.

At baseline commit [`09adb7362162888187d7f750412cf057a4cd8512`](https://github.com/efkopru/nearmap-building-footprints/blob/09adb7362162888187d7f750412cf057a4cd8512/notebooks/04_score_and_compare.ipynb), the synthetic notebook's `score` function called the evaluator without applying an area filter, although the preceding prose described the Lewisville objective of F1 for buildings at least 20 m². The current correction applies the explicit area filter. It does not establish how the historical real-data thresholded scores were computed. The manifest records the baseline notebook's commit and file hash. A public CSV's snapshot association is evidence-bounded in the manifest; the tuning receipt lacks per-variant reference fingerprints.

## Reproduce the audit

Use the project's CPU environment with the original local `data/lewisville` inputs and `outputs/lewisville` receipts/predictions available. The audit reads a bounded set of known files and nine top-level inference receipts; it does not scan raw tile folders or fetch the service.

```bash
python scripts/audit_lewisville_reference.py --check results/lewisville/reference_provenance.json
```

To inspect a changed snapshot without replacing the published manifest, write a new local report:

```bash
python scripts/audit_lewisville_reference.py --output outputs/reference_audit_new.json
```

The script refuses an existing output path. `--check` fails if the recorded file hashes, counts, fingerprints or reconstructed scores differ. A matching hash proves file identity, not label completeness, independence, licensing or accuracy.
