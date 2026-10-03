import importlib.util
from html.parser import HTMLParser
from pathlib import Path
import subprocess
from urllib.parse import urljoin, urlparse
from urllib.request import url2pathname

import pytest


spec = importlib.util.spec_from_file_location("make_report_pdf", Path(__file__).parents[1] / "scripts" / "make_report_pdf.py")
report_pdf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_pdf)

HTML = '''<!doctype html><html><head><meta charset="utf-8"></head><body>
<p class="meta">29 September 2026</p>
<p>Select a screenshot to see it larger.</p>
<p>The imagery is licensed from Nearmap: keep this page private.</p>
<img src="report_files/screenshot.png"></body></html>'''
PDF = b"%PDF-1.7\nnew report\n%%EOF\n"


@pytest.fixture
def report(tmp_path, monkeypatch):
    path = tmp_path / "saved report.html"
    path.write_text(HTML, encoding="utf-8")
    monkeypatch.setattr(report_pdf, "find_browser", lambda explicit: "browser")
    return path


def destination(command):
    return Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--print-to-pdf=")))


@pytest.mark.parametrize("result", [None, b"", b"not a PDF", "browser_error"])
def test_failed_render_preserves_previous_pdf(report, tmp_path, monkeypatch, capsys, result):
    output = tmp_path / "existing.pdf"
    output.write_bytes(b"previous report")

    def browser(command, **kwargs):
        assert destination(command) != output
        if result == "browser_error":
            raise subprocess.CalledProcessError(1, command)
        if result is not None:
            destination(command).write_bytes(result)

    monkeypatch.setattr(report_pdf.subprocess, "run", browser)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        report_pdf.main([str(report), str(output)])
    assert output.read_bytes() == b"previous report"
    assert "Wrote" not in capsys.readouterr().out


@pytest.mark.parametrize("existing", [False, True])
def test_successful_render_creates_or_replaces_pdf(report, tmp_path, monkeypatch, capsys, existing):
    output = tmp_path / "new folder" / "report.pdf"
    if existing:
        output.parent.mkdir()
        output.write_bytes(b"previous report")

    def browser(command, **kwargs):
        assert destination(command) != output
        destination(command).write_bytes(PDF)

    monkeypatch.setattr(report_pdf.subprocess, "run", browser)
    report_pdf.main([str(report), str(output)])
    assert output.read_bytes() == PDF
    assert f"Wrote {output}" in capsys.readouterr().out


@pytest.mark.parametrize("base", [None, "report_files/", "https://example.com/reports/"])
def test_saved_assets_keep_the_original_base(report, tmp_path, monkeypatch, base):
    html = HTML if base is None else HTML.replace("<head>", f'<head><base href="{base}" target="_blank">')
    report.write_text(html, encoding="utf-8")

    class Bases(HTMLParser):
        def __init__(self):
            super().__init__()
            self.attrs = []

        def handle_starttag(self, tag, attrs):
            if tag == "base":
                self.attrs.append(dict(attrs))

    def browser(command, **kwargs):
        page = Path(url2pathname(urlparse(command[-1]).path))
        transformed = page.read_text(encoding="utf-8")
        parsed = Bases()
        parsed.feed(transformed)
        assert len(parsed.attrs) == 1
        expected = urljoin(report.as_uri(), base or "")
        assert urljoin(parsed.attrs[0]["href"], "report_files/screenshot.png") == urljoin(expected, "report_files/screenshot.png")
        if base is not None:
            assert parsed.attrs[0]["target"] == "_blank"
        assert "keep this page private" not in transformed
        assert 'd.open = true' in transformed
        destination(command).write_bytes(PDF)

    monkeypatch.setattr(report_pdf.subprocess, "run", browser)
    report_pdf.main([str(report), str(tmp_path / "report.pdf")])
