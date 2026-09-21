import os
import subprocess
import sys

def test_subcommand_help_describes_actual_options():
    result = subprocess.run([sys.executable,"-m","nearmap_buildings.cli","infer","--help"],capture_output=True,text=True)
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout and "--manifest" in result.stdout

def test_doctor_works_despite_foreign_projection_environment():
    env = {**os.environ,"PROJ_LIB":"nonexistent/foreign/proj","PROJ_DATA":"nonexistent/foreign/proj"}
    result = subprocess.run([sys.executable,"-m","nearmap_buildings.cli","doctor"],env=env,capture_output=True,text=True)
    assert result.returncode == 0
    assert '"doctor_performs_model_inference": false' in result.stdout
