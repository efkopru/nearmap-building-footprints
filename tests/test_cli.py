import os
import subprocess
import sys

from pyproj.exceptions import CRSError

from nearmap_buildings.cli import is_user_error

def run_nbf(*args, **env):
    # Never inherit a developer's NBF_DEBUG; tests set it explicitly.
    base = {key: value for key, value in os.environ.items() if key != "NBF_DEBUG"}
    return subprocess.run([sys.executable,"-m","nearmap_buildings.cli",*args],capture_output=True,text=True,env={**base,**env})

def test_user_errors_print_one_line_instead_of_a_traceback(tmp_path):
    result = run_nbf("infer","--manifest",str(tmp_path/"manifest.json"),"--output",str(tmp_path/"out"),"--confidence","2")
    assert result.returncode == 2
    assert "nbf infer: error: Invalid confidence" in result.stderr
    assert "Traceback" not in result.stderr

def test_unreadable_raster_is_a_user_error(tmp_path):
    result = run_nbf("tile",str(tmp_path/"missing.tif"),"--output",str(tmp_path/"tiles"))
    assert result.returncode == 2
    assert result.stderr.startswith("nbf tile: error:")
    assert "Traceback" not in result.stderr
    assert not (tmp_path/"tiles").exists()

def test_debug_environment_restores_the_traceback(tmp_path):
    result = run_nbf("infer","--manifest",str(tmp_path/"manifest.json"),"--output",str(tmp_path/"out"),"--confidence","2",NBF_DEBUG="1")
    assert result.returncode == 1
    assert "Traceback" in result.stderr and "ValueError" in result.stderr

def test_only_input_and_gis_errors_count_as_user_errors():
    assert is_user_error(ValueError()) and is_user_error(FileNotFoundError()) and is_user_error(CRSError("unknown CRS"))
    assert not is_user_error(KeyError("tiles")) and not is_user_error(TypeError())

def test_subcommand_help_describes_actual_options():
    result = subprocess.run([sys.executable,"-m","nearmap_buildings.cli","infer","--help"],capture_output=True,text=True)
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout and "--manifest" in result.stdout

def test_doctor_works_despite_foreign_projection_environment():
    env = {**os.environ,"PROJ_LIB":"nonexistent/foreign/proj","PROJ_DATA":"nonexistent/foreign/proj"}
    result = subprocess.run([sys.executable,"-m","nearmap_buildings.cli","doctor"],env=env,capture_output=True,text=True)
    assert result.returncode == 0
    assert '"doctor_performs_model_inference": false' in result.stdout


def test_setup_script_passes_no_double_quotes_to_python():
    # Windows PowerShell 5.1 strips embedded double quotes from native command
    # arguments, which broke the venv version check in setup_cpu.ps1.
    import re
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "scripts" / "setup_cpu.ps1").read_text(encoding="utf-8")
    snippets = re.findall(r"-c '((?:[^']|'')*)'", script)
    assert snippets and not any('"' in snippet for snippet in snippets)
