"""Thin command dispatcher; optional ArcPy/CUDA are never needed for --help."""
import argparse
import importlib
import json
import os
import sys

# Module, function and the one-line summary `nbf --help` shows, in the order of a typical run.
COMMANDS = {
    "demo": ("demo", "main", "Write a small synthetic scene to try the other commands on"),
    "doctor": ("doctor", "main", "Report installed packages and whether a GPU is available"),
    "inspect": ("preprocessing", "inspect_main", "Check that a raster is usable: CRS, 8-bit RGB and north-up"),
    "tile": ("preprocessing", "tile_main", "Cut a raster into overlapping tiles and write their manifest"),
    "infer": ("inference", "main", "Run SAM 3 on the tiles; prints a plan until --execute"),
    "esri": ("esri", "main", "Run Esri's model on one raster, in ArcGIS Pro's Python"),
    "esri-chunks": ("esri_chunks", "main", "Run Esri's model over a whole city in resumable chunks"),
    "import-vectors": ("import_vectors", "main", "Bring an existing building layer into the comparison"),
    "clean": ("postprocess", "main", "Repair outlines and remove duplicates from overlapping tiles"),
    "evaluate": ("evaluation", "main", "Score outlines against reference outlines"),
    "compare": ("compare", "main", "Tabulate evaluation reports that share a reference"),
    "agree": ("agreement", "main", "Match two methods' outlines to each other, without a reference"),
    "review-check": ("review", "main", "Summarize what a manual review changed in a reference layer"),
    "train": ("training", "main", "Prepare training data, launch SAM 3 training or export weights"),
}
# Bad inputs or settings get a one-line message; anything else keeps its traceback.
# The GIS libraries raise their own types for unreadable files and unknown CRSs.
USER_ERRORS = (ValueError, OSError)
GIS_ERROR_PACKAGES = ("pyogrio", "pyproj", "rasterio")
GIS_VARIABLES = ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA")
# What main removed, as JSON, so a command that starts ArcGIS Pro's Python can give it back.
# It lives in the environment, not in this module: `python -m nearmap_buildings.cli` runs this
# file as __main__, so another module importing .cli would get a second, empty copy.
REMOVED_GIS_ENVIRONMENT = "NBF_REMOVED_GIS_ENVIRONMENT"

def is_user_error(error):
    return isinstance(error, USER_ERRORS) or type(error).__module__.split(".")[0] in GIS_ERROR_PACKAGES

def command_list():
    width = max(len(name) for name in COMMANDS)
    return "\n".join(["commands:", *(f"  {name:<{width}}  {summary}" for name, (_, _, summary) in COMMANDS.items()),
                      "", "Run 'nbf <command> --help' for a command's options."])

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="nbf", usage="nbf <command> [options]",
                                     description="Local building extraction and evaluation tools",
                                     epilog=command_list(), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS, metavar="command", help=argparse.SUPPRESS)
    if not argv:
        # A missing command is still an error, but the list of commands is the useful message.
        parser.print_help(sys.stderr)
        return 2
    if argv[0] not in COMMANDS:
        parser.parse_args(argv)
        return 0
    command, rest = argv[0], argv[1:]
    # Wheel-based GIS commands must not inherit a foreign PostGIS/ArcGIS database.
    # This is process-local. ArcPy deliberately keeps its own environment intact.
    if command != "esri":
        removed = {variable: os.environ.pop(variable) for variable in GIS_VARIABLES if variable in os.environ}
        if removed:
            os.environ[REMOVED_GIS_ENVIRONMENT] = json.dumps(
                {**json.loads(os.environ.get(REMOVED_GIS_ENVIRONMENT, "{}")), **removed})
    module, function, _ = COMMANDS[command]
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
