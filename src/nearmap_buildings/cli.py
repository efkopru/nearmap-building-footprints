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
# Bad inputs or settings get a one-line message; anything else keeps its traceback.
# The GIS libraries raise their own types for unreadable files and unknown CRSs.
USER_ERRORS = (ValueError, OSError)
GIS_ERROR_PACKAGES = ("pyogrio", "pyproj", "rasterio")

def is_user_error(error):
    return isinstance(error, USER_ERRORS) or type(error).__module__.split(".")[0] in GIS_ERROR_PACKAGES

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="nbf", description="Local building extraction and evaluation tools")
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
    try:
        getattr(importlib.import_module(f"nearmap_buildings.{module}"), function)(rest)
    except Exception as error:
        # NBF_DEBUG=1 restores the full traceback for diagnosing a failure.
        if not is_user_error(error) or os.environ.get("NBF_DEBUG"):
            raise
        print(f"nbf {command}: error: {error}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    # Propagate the exit code; the experiment runner stops on a nonzero status.
    raise SystemExit(main())
