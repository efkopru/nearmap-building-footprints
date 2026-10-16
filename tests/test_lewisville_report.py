import csv
import importlib.util
from pathlib import Path
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


def test_failed_screenshot_validation_preserves_existing_report(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"original public report")
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: object()))
    monkeypatch.setattr(report, "screenshot_images", lambda reader: [(0, SimpleNamespace(data=b"image"))] * 8)
    monkeypatch.setattr(report, "build_html", lambda: "<html>prepared report</html>")

    def render(argv):
        Path(argv[1]).write_bytes(b"rendered PDF")

    monkeypatch.setitem(sys.modules, "make_report_pdf", SimpleNamespace(main=render))

    def reject(*args):
        raise ValueError("Screenshot bytes changed during report generation")

    monkeypatch.setattr(report, "preserve_screenshots", reject)
    with pytest.raises(ValueError, match="Screenshot bytes changed"):
        report.main(["--source-pdf", str(source), "--output", str(source),
                     "--work-dir", str(tmp_path / "work")])
    assert source.read_bytes() == b"original public report"
    assert len(list((tmp_path / "work").glob("source-*.pdf"))) == 1
