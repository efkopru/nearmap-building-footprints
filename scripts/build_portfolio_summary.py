"""Build the two-page public portfolio PDF from committed aggregate CSVs.

Run from any directory: python scripts/build_portfolio_summary.py
Optional reporting dependencies: reportlab and pypdf. No private data is read.
"""

import argparse
import csv
from html import escape
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/lewisville"
DEFAULT_PDF = ROOT / "portfolio/nearmap-building-footprints.pdf"
REPO = "https://github.com/efkopru/nearmap-building-footprints"
METHODS = ("esri_city", "esri_local", "sam3")
NAMES = {"esri_city": "Esri, city run", "esri_local": "Esri, local raw run", "sam3": "SAM 3"}


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_pdf(destination):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows = load_rows(RESULTS / "reviewed_2026/test_area_scores.csv")
    score = {(row["cohort"], row["method_key"]): row for row in rows}
    expected = {(cohort, method) for cohort in ("all_sizes", "at_least_20m2") for method in METHODS}
    if set(score) != expected or len(rows) != len(expected):
        raise ValueError("Expected exactly six unique reviewed aggregate rows")
    primary = [score[("at_least_20m2", method)] for method in METHODS]
    all_sizes = [score[("all_sizes", method)] for method in METHODS]
    if {int(row["reference"]) for row in primary} != {468} or {int(row["reference"]) for row in all_sizes} != {600}:
        raise ValueError("Reviewed cohort changed; update and review the summary narrative")
    historical = {row["method"]: row for row in load_rows(RESULTS / "citywide_scores.csv")
                  if row["buildings"] == "20 m² and up"}

    ink = colors.HexColor("#17324D")
    accent = colors.HexColor("#176B75")
    muted = colors.HexColor("#516271")
    light = colors.HexColor("#EAF2F4")
    styles = {
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=13.4, textColor=ink),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.1, leading=11.2, textColor=muted),
        "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12.2, leading=15, textColor=ink),
    }
    canvas = Canvas(str(destination), pagesize=(612, 792), invariant=1)
    canvas.setTitle("Building footprints from aerial imagery: SAM 3 and Esri")
    canvas.setAuthor("efkopru")
    canvas.setSubject("Reviewed 2026 test-area accuracy and historical inventory agreement")

    def paragraph(text, y, style="body", x=42, width=528):
        p = Paragraph(text, styles[style])
        _, height = p.wrap(width, 700)
        p.drawOn(canvas, x, y - height)
        return y - height

    def heading(text, y):
        return paragraph(text, y, "heading") - 8

    def footer(page):
        canvas.setStrokeColor(colors.HexColor("#CED9DF"))
        canvas.line(42, 42, 570, 42)
        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(42, 28, "Public summary | updated October 15, 2026")
        canvas.drawRightString(570, 28, f"{page} / 2")

    def table(headers, values, widths, y):
        t = Table([headers] + values, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), ink),
            ("BACKGROUND", (0, 0), (-1, 0), light),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FA")]),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#B8CDD2")),
        ]))
        _, height = t.wrap(528, 700)
        t.drawOn(canvas, 42, y - height)
        return y - height

    canvas.setFillColor(accent)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(42, 752, "GEOSPATIAL MACHINE LEARNING")
    canvas.setFillColor(ink)
    canvas.setFont("Helvetica-Bold", 25)
    canvas.drawString(42, 716, "Building footprints")
    canvas.drawString(42, 686, "from aerial imagery")
    y = paragraph("SAM 3 and Esri | Lewisville, Texas | May 2026 imagery", 668) - 20
    y = paragraph("A shared Python workflow prepares imagery, runs or imports predictions, cleans building outlines, and evaluates every method under the same rules. The city-scale workflow processed 21.1 gigapixels at approximately 10 cm resolution.", y) - 20
    y = heading("Reviewed test-area accuracy", y)
    y = paragraph("The primary cohort contains <b>468 reviewed buildings of at least 20 m²</b> in one 700 m square. Both inputs are area-filtered before matching. Settings and predictions were frozen; this evaluation did not rerun inference or retune on the reviewed test labels.", y) - 12
    y = table(["Method", "Precision", "Recall", "F1"],
              [[NAMES[row["method_key"]]] + [f"{float(row[k]):.3f}" for k in ("precision", "recall", "f1")] for row in primary],
              [246, 94, 94, 94], y) - 15
    y = paragraph("<b>All sizes:</b> a separate evaluation uses 600 reviewed buildings, including 132 outlines under 20 m². Forty boundary-crossing outlines are excluded. Matching is one-to-one at IoU &gt;= 0.5 in EPSG:26914.", y) - 10
    y = table(["Method", "All-size F1", "Missed reference"],
              [[NAMES[row["method_key"]], f"{float(row['f1']):.3f}", row["fn"]] for row in all_sizes],
              [270, 120, 138], y) - 13
    y = paragraph("In the all-size matching, SAM 3 and local Esri each match 4 of the 132 reference outlines under 20 m²; city Esri matches none. These area classes do not identify building types.", y, "small") - 10
    y = paragraph("<b>Scope:</b> these results describe one reviewed test square. They do not establish city-wide accuracy. Changes from historical scores reflect the changed reference and scoring policy, not an improvement in the models.", y, "small")
    if y < 58:
        raise ValueError("Page 1 content exceeds the safe layout")
    footer(1)
    canvas.showPage()

    canvas.setFillColor(accent)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(42, 752, "METHOD, EVIDENCE AND REMAINING WORK")
    y = heading("Historical agreement stays separate", 725)
    y = paragraph("Earlier city-wide scores use 38,882 historical outlines from a 38,924-record source inventory. Dates are primarily 2015 or earlier; 33 source records are undated and remain in the evaluated set. These scores measure historical agreement, not current accuracy.", y) - 10
    y = table(["Method", "City-wide F1, at least 20 m²"],
              [[NAMES[method], f"{float(historical[NAMES[method]]['f1']):.3f}"] for method in METHODS],
              [280, 248], y) - 14
    y = paragraph("An outline absent from the inventory is a candidate for review. Absence does not establish a construction date or prove that a prediction is a building. The city and local Esri outputs have different processing histories.", y, "small") - 18
    y = heading("Review and engineering", y)
    y = paragraph("<b>Human review:</b> Esad completed both review areas and confirmed the labeling rules and no unresolved cases. The test review used only aerials and historical outlines, with no prior or concurrent prediction exposure. These are reviewer attestations. The reference retains inherited geometry and splitting conventions; no second independent reviewer is recorded.", y) - 10
    y = paragraph("<b>Workflow:</b> hash-linked tile manifests, resumable inference, chunk ownership for ArcGIS execution, shared cleanup, explicit boundary handling, and one-to-one scoring. Automated tests cover Windows and Ubuntu. Tile overlap reduces edge losses; buildings longer than a tile can still be split.", y) - 10
    y = paragraph("<b>Publication:</b> committed aggregates, charts and provenance support report regeneration. Raw imagery, reviewed labels, prediction geometry and full receipts remain local. The full report retains its previously published screenshots. Public files alone cannot reproduce the geometric evaluation.", y) - 18
    y = heading("Completed baseline; fine-tuning deferred", y)
    y = paragraph("Reference review (#2) and frozen evaluation (#3) are complete following merged PR #29. Remaining issue #4 requires separate training, validation and fresh test areas, with every eligible building labeled and no prediction exposure during fresh test review. Preserve the completed test reference and scores. Labeling packages for #4 have not been prepared. Training feasibility on the 6 GB GPU remains unverified.", y) - 16
    y = heading("Public evidence and guides", y)
    links = [
        ("Repository and setup", REPO),
        ("Reviewed scores, protocol and provenance", REPO + "/tree/main/results/lewisville/reviewed_2026"),
        ("Full comparison report, 11 pages", REPO + "/blob/main/results/lewisville/Lewisville_building_footprint_comparison.pdf"),
        ("Usage and reproduction guide", REPO + "/blob/main/docs/USAGE.md"),
    ]
    for label, url in links:
        y = paragraph(f'<link href="{escape(url)}" color="#176B75">{escape(label)}</link>', y, "small") - 3
    if y < 58:
        raise ValueError("Page 2 content exceeds the safe layout")
    footer(2)
    canvas.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()
    from pypdf import PdfReader

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pdf", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        build_pdf(temporary)
        reader = PdfReader(temporary)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if len(reader.pages) != 2 or "Reviewed test-area accuracy" not in text:
            raise ValueError("Generated PDF did not pass the page and content checks")
        if any(page.images for page in reader.pages):
            raise ValueError("The public summary must not include imagery")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
