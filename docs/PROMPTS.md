# Prompt data contract

Use a single-layer GeoJSON or GeoPackage exported from QGIS or ArcGIS. Declare its real CRS. Coordinates are GIS coordinates, not tile pixel coordinates; the adapter reprojects to each tile and applies its affine transform. GeoJSON coordinates in a projected CRS are accepted locally by GDAL, but GeoPackage is preferable for projected data exchange.

| Mode | Geometry | Required attributes | Meaning |
| --- | --- | --- | --- |
| `text` | None | None | One text concept, such as `building`, applied to every tile. |
| `exemplar` | Polygon bounding boxes | `label`: 1 or 0; defaults to 1 | Positive boxes show the desired visual concept; negative boxes show unwanted examples. The model can find other matching objects in the same tile. |
| `box` | Polygon bounding boxes | All labels positive | Each box targets a particular building. One best-scoring mask is retained per box. This does not discover buildings outside supplied prompts. |
| `point` | Point | `object_id` and `label` (1 foreground, 0 background) | Points sharing an object_id guide one building mask. Separate buildings require separate object IDs. Each group needs a positive point. |

Exemplars and instance boxes are different modes and use different model interfaces. The current runner intentionally supports either text concepts or visual exemplars; it does not imply that passing text and then boxes combines them. SamGeo's high-level concept methods reset previous prompts.

Only complete boxes and complete point groups contained in a tile are used. Boxes are not silently clipped. Tiles with no complete positive prompt are recorded as `skipped_no_complete_positive_prompt` in their receipt. Review `run.json` before treating a run as complete area coverage. A run can finish computationally while still containing skipped tiles. Enlarge the overlap/tile size or correct the prompt coverage for large buildings.

Visual examples are scoped to their input image. The runner does not transfer a marked roof from one tile into unrelated tiles as an exemplar. Provide local examples in each tile where that mode is evaluated. Negative boxes alone cannot define a positive concept.

Run `nbf demo --output data/demo` to create synthetic prompt examples and inspect them in GIS. These fixtures are not Nearmap data or model predictions.

Keep prompt creation separate from final test labels. Report the manual effort used to place boxes/points; prompt-assisted results are not directly equivalent to fully automatic extraction.
