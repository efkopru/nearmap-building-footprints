"""Report installed dependencies and GPU availability; never download model weights."""
import argparse
import importlib.util
import json
from .common import provenance

def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = provenance()
    result["modules"] = {name: importlib.util.find_spec(name) is not None for name in ["samgeo", "sam3", "torch", "arcpy"]}
    if result["modules"]["torch"]:
        import torch
        result["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    result["doctor_performs_model_inference"] = False
    print(json.dumps(result, indent=2))
