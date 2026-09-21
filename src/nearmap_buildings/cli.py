"""Thin command dispatcher; optional ArcPy/CUDA are never needed for --help."""
import argparse
import importlib
import os
import sys

COMMANDS = {
    "inspect": ("preprocessing", "inspect_main"),
    "tile": ("preprocessing", "tile_main"),
    "infer": ("inference", "main"),
    "clean": ("postprocess", "main"),
    "evaluate": ("evaluation", "main"),
    "compare": ("compare", "main"),
    "train": ("training", "main"),
    "esri": ("esri", "main"),
    "import-vectors": ("import_vectors", "main"),
    "doctor": ("doctor", "main"),
    "demo": ("demo", "main"),
}

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Local building extraction and evaluation tools")
    parser.add_argument("command", choices=COMMANDS)
    if not argv or argv[0] not in COMMANDS:
        parser.parse_args(argv)
        return 0
    command, rest = argv[0], argv[1:]
    # Wheel-based GIS commands must not inherit a foreign PostGIS/ArcGIS database.
    # This is process-local. ArcPy deliberately keeps its own environment intact.
    if command != "esri":
        for variable in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
            os.environ.pop(variable, None)
    module, function = COMMANDS[command]
    getattr(importlib.import_module(f"nearmap_buildings.{module}"), function)(rest)
    return 0

if __name__ == "__main__":
    main()
