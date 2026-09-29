"""The walkthrough notebooks stay valid, parseable and free of machine-specific paths."""
import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
# Absolute user paths from Windows, WSL or Linux must never reach a published notebook.
MACHINE_PATH = re.compile(r"[A-Za-z]:\\\\+Users|[A-Za-z]:/Users|/home/[^/<\s]+/|/mnt/[a-z]/Users|/Users/[^/<\s]+/")


def test_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == ["01_prepare_imagery.ipynb", "02_run_sam3.ipynb",
                                           "03_run_esri.ipynb", "04_score_and_compare.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid_and_clean(path):
    text = path.read_text(encoding="utf-8")
    notebook = json.loads(text)
    assert notebook["nbformat"] == 4
    assert not MACHINE_PATH.search(text), "notebook outputs contain a machine-specific path"
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            ast.parse(source)  # plain Python only: no shell or IPython magics
            assert not [o for o in cell.get("outputs", []) if o["output_type"] == "error"]
