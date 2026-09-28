"""Create synthetic geospatial fixtures. These are not Nearmap imagery or accuracy evidence."""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from .common import write_json

def create_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    crs = "EPSG:32614"
    transform = from_origin(500000, 3600000, 0.25, 0.25)
    width, height = 1536, 512
    buildings = []
    for offset in (0, 512, 1024):
        for col, row in [(80, 90), (190, 330), (360, 140)]:
            xmin, ymax = transform @ (offset + col, row)
            buildings.append(box(xmin, ymax - 9, xmin + 12, ymax))
    label = rasterize([(g, i + 1) for i, g in enumerate(buildings)], out_shape=(height, width), transform=transform, dtype="uint8")
    rgb = np.zeros((3, height, width), dtype="uint8")
    rgb[:] = np.array([67, 100, 65], dtype="uint8")[:, None, None]
    rgb[:, label > 0] = np.array([180, 173, 158], dtype="uint8")[:, None]
    with rasterio.open(output / "imagery.tif", "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(rgb)
    reference = gpd.GeoDataFrame({"building_id": [f"synthetic_{i}" for i in range(len(buildings))], "target": "roof_outline"}, geometry=buildings, crs=crs)
    reference.to_file(output / "reference.gpkg", layer="buildings", driver="GPKG")
    reference.to_file(output / "boxes.geojson", driver="GeoJSON")
    points = gpd.GeoDataFrame({"object_id": reference.building_id, "label": 1}, geometry=reference.geometry.centroid, crs=crs)
    points.to_file(output / "points.geojson", driver="GeoJSON")
    exemplar = reference.iloc[[0, 3, 6]].copy()
    exemplar["label"] = 1
    exemplar.to_file(output / "exemplars.geojson", driver="GeoJSON")
    extent = box(500000, 3600000 - height * 0.25, 500000 + width * 0.25, 3600000)
    gpd.GeoDataFrame({"coverage": ["synthetic_complete"]}, geometry=[extent], crs=crs).to_file(output / "aoi.geojson", driver="GeoJSON")
    regions = [box(500000 + offset * .25, extent.bounds[1], 500000 + (offset + 512) * .25, extent.bounds[3]) for offset in (0, 512, 1024)]
    gpd.GeoDataFrame({"split": ["train", "val", "test"]}, geometry=regions, crs=crs).to_file(output / "splits.geojson", driver="GeoJSON")
    predictions = reference.copy()
    predictions["score"] = 1.0
    predictions["method"] = "synthetic_identity_fixture"
    predictions.to_file(output / "perfect_predictions.gpkg", layer="predictions", driver="GPKG")
    # Laid out like Esri's Detect Objects output (a file geodatabase feature class with
    # Class and a 0-100 Confidence), with the defects of a raw run: a duplicate from
    # overlapping chips (the tool runs with NO_NMS), a self-intersecting outline, one
    # missed building (5) and one false detection. No Esri model produced these.
    esri = []
    for i, geometry in enumerate(buildings):
        if i == 5:
            continue
        shifted = translate(geometry, 0.3, -0.2)
        if i == 7:
            x0, y0, x1, y1 = shifted.bounds
            shifted = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0 + 0.2, y0 - 0.2)])
        esri.append((shifted, 97.5 - i))
        if i == 2:
            esri.append((translate(geometry, 0.6, 0.1), 91.0))
    fx, fy = transform @ (1304, 420)
    esri.append((box(fx, fy - 6, fx + 8, fy), 62.0))
    gpd.GeoDataFrame({"Class": "Building", "Confidence": [score for _, score in esri], "method": "esri_building_usa"},
                     geometry=[geometry for geometry, _ in esri], crs=crs).to_file(
        output / "esri_format_predictions.gdb", layer="esri_buildings", driver="OpenFileGDB")
    write_json(output / "NOTICE.json", {"synthetic_only": True, "purpose": "CPU pipeline and evaluator checks", "model_inference": False, "accuracy_claim": "None. Perfect predictions are copies of synthetic reference geometries.", "esri_format_predictions": "Shifted copies of the synthetic reference in Esri's Detect Objects output layout, with a duplicate, a self-intersecting outline, a missed building and a false detection. Not Esri model output."})
    return output

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    print(create_demo(a.output))

if __name__ == "__main__":
    main()
