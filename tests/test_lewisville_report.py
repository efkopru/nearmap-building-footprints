import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("lewisville_report", ROOT / "scripts/build_lewisville_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_report_uses_reviewed_scores_and_keeps_historical_results_distinct():
    html = report.build_html()
    with (ROOT / "results/lewisville/reviewed_2026/test_area_scores.csv").open(encoding="utf-8", newline="") as stream:
        scores = {(r["cohort"], r["method_key"]): r for r in csv.DictReader(stream)}
    for cohort in ["all_sizes", "at_least_20m2"]:
        expected = " / ".join(f'{float(scores[cohort, method]["f1"]):.2f}'
                              for method in ["esri_city", "esri_local", "sam3"])
        assert f"<b>{expected}</b>" in html
    assert "Historical city-wide agreement" in html
    assert "Historical scores reported for at least 20 m²" in html
    assert "Review pending" not in html
    assert "REVIEWED_F1_" not in html
    assert "C:\\Users" not in html
    assert "file:///" not in html
    assert html.count('<img src="screenshot-') == 8


@pytest.mark.parametrize("count, size", [(7, (1000, 1000)), (9, (1000, 1000)), (8, (500, 1000))])
def test_source_pdf_must_contain_the_eight_full_size_screenshots(count, size):
    reader = SimpleNamespace(pages=[SimpleNamespace(images=[SimpleNamespace(image=SimpleNamespace(size=size))] * count)])
    with pytest.raises(ValueError, match="exactly eight 1000 x 1000"):
        report.screenshot_images(reader)


def test_existing_pdf_rebuild_keeps_screenshot_bytes_and_colour_profiles(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    source = ROOT / "results/lewisville/Lewisville_building_footprint_comparison.pdf"
    checked = tmp_path / "preserved.pdf"
    pages = report.preserve_screenshots(source, source, checked)
    assert pages == len(pypdf.PdfReader(source).pages)
    assert report.screenshot_hashes(pypdf.PdfReader(source)) == report.screenshot_hashes(pypdf.PdfReader(checked))


def test_report_credits_the_imagery_and_has_no_private_page_wording():
    html = report.build_html()
    assert html.count(report.CREDIT) == 2
    for wording in ["keep this page private", "Select a screenshot", "IMAGERY_CREDIT"]:
        assert wording not in html


def test_a_browser_path_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="--browser"):
        report.find_browser(str(tmp_path / "no-such-browser.exe"))


PDF = b"%PDF-1.7 new report"


@pytest.fixture
def source(tmp_path, monkeypatch):
    """An existing report, with small stand-ins for reading PDFs, the page and the browser."""
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: object()))
    monkeypatch.setattr(report, "screenshot_images", lambda reader: [(0, SimpleNamespace(data=b"image"))] * 8)
    monkeypatch.setattr(report, "build_html", lambda: "<html>prepared report</html>")
    monkeypatch.setattr(report, "find_browser", lambda explicit: "browser")
    path = tmp_path / "report.pdf"
    path.write_bytes(b"original public report")
    return path


def build(source, output, tmp_path):
    report.main(["--source-pdf", str(source), "--output", str(output), "--work-dir", str(tmp_path / "work")])


def print_destination(command):
    return Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--print-to-pdf=")))


@pytest.mark.parametrize("written", [None, b"", b"not a PDF", "browser error"])
def test_failed_print_preserves_existing_report(source, tmp_path, monkeypatch, capsys, written):
    def browser(command, **kwargs):
        assert print_destination(command) != source
        if written == "browser error":
            raise subprocess.CalledProcessError(1, command)
        if written is not None:
            print_destination(command).write_bytes(written)

    monkeypatch.setattr(report.subprocess, "run", browser)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        build(source, source, tmp_path)
    assert source.read_bytes() == b"original public report"
    assert "Wrote" not in capsys.readouterr().out


def test_failed_screenshot_validation_preserves_existing_report(source, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "print_pdf", lambda page, destination, browser: destination.write_bytes(PDF))

    def reject(*args):
        raise ValueError("Screenshot bytes changed during report generation")

    monkeypatch.setattr(report, "preserve_screenshots", reject)
    with pytest.raises(ValueError, match="Screenshot bytes changed"):
        build(source, source, tmp_path)
    assert source.read_bytes() == b"original public report"
    assert len(list((tmp_path / "work").glob("source-*.pdf"))) == 1


@pytest.mark.parametrize("existing", [False, True])
def test_checked_report_creates_or_replaces_the_output(source, tmp_path, monkeypatch, capsys, existing):
    output = tmp_path / "new folder" / "report.pdf"
    if existing:
        output.parent.mkdir()
        output.write_bytes(b"previous report")

    def browser(command, **kwargs):
        assert print_destination(command) != output
        assert command[-1] == (tmp_path / "work" / "report.html").resolve().as_uri()
        print_destination(command).write_bytes(PDF)

    def keep(baseline, rendered, checked):
        shutil.copyfile(rendered, checked)
        return 11

    monkeypatch.setattr(report.subprocess, "run", browser)
    monkeypatch.setattr(report, "preserve_screenshots", keep)
    build(source, output, tmp_path)
    assert output.read_bytes() == PDF
    assert source.read_bytes() == b"original public report"
    assert f"Wrote {output}" in capsys.readouterr().out
    record = json.loads((tmp_path / "work" / "artifact_validation.json").read_text(encoding="utf-8"))
    assert record["pages"] == 11 and record["output_sha256"] == hashlib.sha256(PDF).hexdigest()
