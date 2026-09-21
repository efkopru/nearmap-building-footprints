# Local generated output

Each inference run gets a new directory containing run.json and raw per-tile vectors/receipts. Original masks are polygonized without cross-instance union. Cleanup writes a new GeoPackage with a retained/removal audit. Preserve raw outputs for reviewing duplicates and tile-edge fragments. No model results are supplied with the project.
