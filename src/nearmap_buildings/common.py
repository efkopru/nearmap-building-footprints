"""Small file/provenance helpers shared by CPU preparation and inference."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def provenance():
    packages = {}
    for name in ["nearmap-building-footprints", "segment-geospatial", "sam3", "torch", "numpy", "rasterio", "geopandas", "shapely", "pyproj", "scipy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform(), "packages": packages}
