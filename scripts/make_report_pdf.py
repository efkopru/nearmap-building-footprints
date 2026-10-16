"""Print the public PDF edition of the private Lewisville report page.

Save the private report page as HTML, then, from the repository root:

    python scripts/make_report_pdf.py path/to/report.html results/lewisville/Lewisville_building_footprint_comparison.pdf

The public edition replaces the "keep this page private" notice and the screenshot
zoom hint with an imagery credit, shows every data table the page folds away, and
prints US Letter pages with page numbers. It needs Microsoft Edge or Google Chrome;
pass --browser if neither is found. Only the PDF is written: the page itself, with
its licensed imagery, stays out of the repository.
"""

import argparse
from html import escape
from html.parser import HTMLParser
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin

CREDIT = "Imagery: Nearmap, May 2026"

# Each private wording and its public replacement; every one must match exactly once.
PUBLIC_EDITION = [
    ('<p class="meta">29 September 2026', f'<p class="meta">{CREDIT} · 29 September 2026'),
    ("Select a screenshot to see it larger.", "Zoom in on a screenshot to see the full 1000 px image."),
    ("The imagery is licensed from Nearmap: keep this page private.", f"{CREDIT}."),
]

# Known wording from the earlier private report. Keep historical numbers and
# screenshot observations, but do not turn unmatched geometry into dated change.
# Replacements are targeted and idempotent; unrelated uses of "new" are untouched.
REPORTING_CORRECTIONS = [
    ("Likely built since 2015", "Model-agreement candidates"),
    ("Built since 2015, 300 m square", "Inventory-unmatched area, 300 m square"),
    ("Two independent models agreeing where the old layer has nothing points to new buildings, such as whole new subdivisions.",
     "Model agreement without an inventory match identifies review candidates. Inventory omissions, geometry conventions and shared model errors remain possible."),
    ("many are buildings built since 2015.", "unmatched outlines need independent review."),
    ("some others have been demolished since 2015.", "missing detections need review and do not establish demolition."),
    ("mark buildings that have since gone.", "overlap apparently open ground; no demolition date is established."),
    ("those are counted here as likely built since 2015.", "these are model-agreement candidates, not dated evidence of change."),
    ("where much has been built or rebuilt since 2015,", "where agreement with the inventory is lower,"),
    ("building outlines collected in 2015 or earlier", "building inventory, primarily dated 2015 or earlier with 33 undated source features"),
    ("collected in 2015 or earlier (38,924 outlines, public)",
     "primarily dated 2015 or earlier (38,924 source outlines, 33 undated; 38,882 evaluated city-wide)"),
    ("The areas were picked by rule from the city-wide results and avoid the two review areas, so the reviews stay independent of the models.",
     "The areas were picked by rule outside the two review areas. Held-out annotation must still be completed blind to predictions and these candidate views."),
    ("For accuracy: review the Old Town and test-area outlines against the 2026 imagery (the review packages are ready), then score the frozen settings against them.",
     "For accuracy: review both complete AOIs against the 2026 imagery while predictions and agreement candidates are hidden, record a completed review receipt, then score the frozen settings."),
]


def normalize_reporting_language(html):
    """Correct known unsupported temporal claims in saved legacy report pages."""
    for old, new in REPORTING_CORRECTIONS:
        # Saved HTML may wrap a text sentence across source lines.
        pattern = r"\s+".join(re.escape(part) for part in old.split())
        html = re.sub(pattern, lambda match: new, html)
    return html

PRINT_LAYOUT = """
<style media="print">
@page { size: Letter; margin: 0.5in 0.5in 0.6in;
  @bottom-center { content: "Lewisville building footprints · SAM 3 and Esri · page " counter(page) " of " counter(pages);
    font: 9pt "Public Sans", "Segoe UI", sans-serif; color: #6f7a7c; } }
html { zoom: 0.78; }
body { background: #ffffff; }
.page { padding: 0; max-width: none; }
.gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); max-width: 50rem; margin: 0 auto; }
section, .card, .finding, figure, tr, .srow, .hrow { break-inside: avoid; }
h2, h3 { break-after: avoid; }
details > summary, .tip, dialog { display: none !important; }
th, td { white-space: normal; }
</style>
<script>document.querySelectorAll("details").forEach((d) => { d.open = true; });</script>
"""

BROWSERS = [
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "msedge", "microsoft-edge", "google-chrome", "chromium", "chrome",
]


def public_edition(html):
    html = normalize_reporting_language(html)
    for private, public in PUBLIC_EDITION:
        if html.count(private) != 1:
            raise ValueError(f"Expected the report page to contain {private!r} once; has the page changed?")
        html = html.replace(private, public)
    if html.count("</body>") == 1:
        return html.replace("</body>", PRINT_LAYOUT + "</body>")
    return html + PRINT_LAYOUT


def preserve_asset_base(html, source):
    """Keep saved-page assets relative to the source, including an explicit base."""
    class Page(HTMLParser):
        head_end = None
        base = None

        def handle_starttag(self, tag, attrs):
            if tag not in {"head", "base"}:
                return
            line, column = self.getpos()
            start = sum(len(part) for part in html.splitlines(keepends=True)[:line - 1]) + column
            end = start + len(self.get_starttag_text())
            if tag == "head" and self.head_end is None:
                self.head_end = end
            if tag == "base" and self.base is None and any(key == "href" for key, _ in attrs):
                self.base = (start, end, attrs)

    page = Page()
    page.feed(html)
    source_uri = source.resolve().as_uri()
    if page.base is not None:
        start, end, attrs = page.base
        attrs = [(key, urljoin(source_uri, value or "") if key == "href" else value) for key, value in attrs]
        tag = "<base" + "".join(f" {key}" if value is None else f' {key}="{escape(value, quote=True)}"' for key, value in attrs) + ">"
        return html[:start] + tag + html[end:]
    position = page.head_end if page.head_end is not None else 0
    return html[:position] + f'<base href="{escape(source_uri, quote=True)}">' + html[position:]


def find_browser(explicit=None):
    for candidate in [explicit] if explicit else BROWSERS:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    raise FileNotFoundError("Edge or Chrome not found; pass its path with --browser.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", type=Path, help="The private report page, saved as HTML")
    p.add_argument("output", type=Path, help="The PDF to write")
    p.add_argument("--browser", help="Path to Edge or Chrome")
    args = p.parse_args(argv)
    browser = find_browser(args.browser)
    output = args.output.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "public.html"
        html = public_edition(args.report.read_text(encoding="utf-8"))
        page.write_text(preserve_asset_base(html, args.report), encoding="utf-8")
        rendered = Path(tmp) / "report.pdf"
        # The time budget lets the page's script draw its charts and tables before printing.
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-first-run", f"--user-data-dir={Path(tmp) / 'profile'}",
                        "--no-pdf-header-footer", "--virtual-time-budget=15000", f"--print-to-pdf={rendered}", page.as_uri()],
                       check=True, capture_output=True)
        if not rendered.is_file():
            raise RuntimeError(f"{browser} did not write a PDF")
        with rendered.open("rb") as pdf:
            header = pdf.read(8)
        if len(header) < 8 or not header.startswith(b"%PDF-"):
            raise RuntimeError(f"{browser} did not write a PDF")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Stage beside the destination so replacement also works across drives.
        with tempfile.TemporaryDirectory(dir=output.parent) as staging:
            ready = Path(staging) / "report.pdf"
            shutil.copyfile(rendered, ready)
            ready.replace(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
