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
# Stock Ubuntu ships python3.12 without ensurepip, which venv needs to install pip.
if ! python3.12 -c 'import ensurepip' 2>/dev/null; then
  printf '%s\n' 'Python 3.12 venv support is missing. Install it first: sudo apt install python3.12-venv' >&2
  exit 1
fi
# Meta's training loss compiles a small GPU helper the first time it runs, which needs a
# C compiler and Python's headers. Checked here because it otherwise fails at the first step.
if [[ "$training" == "--training" ]]; then
  python_include="$(python3.12 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
  if ! printf '#include <Python.h>\n' | cc -E -x c -I"$python_include" - >/dev/null 2>&1; then
    printf '%s\n' 'Training needs a C compiler and the Python headers. Install them first: sudo apt install build-essential python3.12-dev' >&2
    exit 1
  fi
fi
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate
python -m pip install pip==26.2.1
# torchvision 0.25.0 is the release built against torch 2.10.0.
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
mkdir -p third_party
if [[ ! -d third_party/sam3 ]]; then
  git clone https://github.com/facebookresearch/sam3.git third_party/sam3
fi
# Added files, such as a training config copied into the checkout, are fine; edits to Meta's code are not.
if [[ -n "$(git -C third_party/sam3 status --porcelain --untracked-files=no)" ]]; then
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
# outputs/ is ignored by git, so a fresh clone does not have it yet.
mkdir -p outputs
python -m pip freeze > outputs/sam3-environment.txt
printf '%s\n' 'Environment prepared. Obtain SAM 3 checkpoint access separately; validate with a small pilot.'
