#!/usr/bin/env bash
# Run inside Linux/WSL2 after installing a compatible NVIDIA driver and Python 3.12.
# Model weights/access are separate; this script does not accept terms or fetch weights.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
sam3_ref=2345a4ad109ac29c569da749c91d84f10dc08c40
training="${1:-}"
if [[ -n "$training" && "$training" != "--training" ]]; then
  printf '%s\n' 'Usage: bash scripts/setup_sam3.sh [--training]' >&2
  exit 2
fi
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
mkdir -p third_party
if [[ ! -d third_party/sam3 ]]; then
  git clone https://github.com/facebookresearch/sam3.git third_party/sam3
fi
if [[ -n "$(git -C third_party/sam3 status --porcelain)" ]]; then
  printf '%s\n' 'SAM 3 checkout has local changes. Preserve them and use a separate checkout.' >&2
  exit 1
fi
git -C third_party/sam3 checkout "$sam3_ref"
if [[ "$training" == "--training" ]]; then
  python -m pip install -e './third_party/sam3[train]'
else
  python -m pip install -e ./third_party/sam3
fi
python -m pip install -e '.[sam]'
python -m nearmap_buildings.cli doctor
python -m pip freeze > outputs/sam3-environment.txt
printf '%s\n' 'Environment prepared. Obtain SAM 3 checkpoint access separately; validate with a small pilot.'
