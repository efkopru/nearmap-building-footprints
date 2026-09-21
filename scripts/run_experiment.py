"""Execute an explicit JSON list of nbf commands. Default: print plan only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    config_path = a.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    # Paths in these experiment files are relative to the project, not shell cwd.
    root = (config_path.parent / config.get("working_directory", "..")).resolve()
    for step in config["steps"]:
        if not isinstance(step, list) or not all(isinstance(value, str) for value in step):
            raise ValueError("Each step must be a JSON array of CLI argument strings.")
        command = [sys.executable, "-m", "nearmap_buildings.cli", *step]
        print(json.dumps({"cwd": str(root), "command": command}), flush=True)
        if a.execute:
            subprocess.run(command, cwd=root, check=True)

if __name__ == "__main__":
    main()
