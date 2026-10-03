"""Draw the diagram and charts used in the README and the Lewisville results page.

Needs only Python 3.12 or later. From the repository root:

    python scripts/make_charts.py

The numbers come from the CSV files in results/lewisville. Each figure is written to
docs/images twice: NAME.svg for light pages and NAME-dark.svg for dark ones.
"""

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lewisville"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

# Esri is red, SAM 3 blue and "found by both" violet in every chart. The pairs pass the
# dataviz palette checks (colour-blind separation and contrast) on both surfaces.
THEMES = {
    "light": dict(surface="#fcfcfb", border="#e1e0d9", card="#f3f2ee", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", esri="#e34948", sam3="#2a78d6", both="#4a3aa7",
                  missed="#d6d5ce", on_color="#ffffff"),
    "dark": dict(surface="#1a1a19", border="#2c2c2a", card="#242422", ink="#ffffff", ink2="#c3c2b7",
                 muted="#898781", grid="#383835", esri="#e66767", sam3="#3987e5", both="#9085e9",
                 missed="#46463f", on_color="#0b0b0b"),
}

# The 2015 map's size classes, largest first, with a plain-language name for each.
SIZES = [("100 m² and up", "Houses and larger"), ("50–100 m²", "Small buildings"),
         ("20–50 m²", "Garages and big sheds"), ("under 20 m²", "Garden sheds")]


class Svg:
    def __init__(self, width, height, theme):
        self.width, self.height, self.t, self.parts = width, height, theme, []

    def rect(self, x, y, w, h, fill, rx=0, stroke=None):
        line = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" fill="{fill}"{line}/>')

    def bar(self, x, y, w, h, fill, r=4):
        """A horizontal bar with a rounded end, anchored square at the baseline."""
        if w <= r:
            self.rect(x, y, max(w, 0), h, fill)
            return
        self.parts.append(f'<path d="M{x:g} {y:g}h{w - r:g}a{r} {r} 0 0 1 {r} {r}v{h - 2 * r:g}'
                          f'a{r} {r} 0 0 1 -{r} {r}h-{w - r:g}z" fill="{fill}"/>')

    def line(self, x1, y1, x2, y2, stroke, width=1):
        self.parts.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{stroke}" stroke-width="{width}"/>')

    def path(self, d, stroke="none", width=2, fill="none"):
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
                          'stroke-linejoin="round" stroke-linecap="round"/>')

    def text(self, x, y, value, size=14, fill=None, weight=400, anchor="start"):
        self.parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" '
                          f'text-anchor="{anchor}" fill="{fill or self.t["ink"]}">{escape(value)}</text>')

    def header(self, title, subtitle):
        self.text(32, 44, title, 21, weight=700)
        self.text(32, 70, subtitle, 14, self.t["ink2"])

    def legend(self, y, items):
        x = 32
        for name, color in items:
            self.rect(x, y - 11, 14, 14, color, rx=3)
            self.text(x + 22, y + 1, name, 14, self.t["ink2"])
            x += 22 + 8.2 * len(name) + 28

    def save(self, path):
        body = "\n".join(self.parts)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" font-family="{FONT}">\n'
            f'<rect x="0.5" y="0.5" width="{self.width - 1}" height="{self.height - 1}" rx="12" '
            f'fill="{self.t["surface"]}" stroke="{self.t["border"]}"/>\n{body}\n</svg>\n',
            encoding="utf-8", newline="\n")


def read_csv(name, key):
    """Rows of a results CSV, by the value in their `key` column."""
    lines = (RESULTS / name).read_text(encoding="utf-8").splitlines()
    return {row[key]: row for row in csv.DictReader(lines)}


def grouped_bars(t, title, subtitle, rows, note):
    """rows: (label, sublabel, esri share, SAM 3 share), shares from 0 to 1."""
    x0, full, top, step = 300, 440, 128, 76
    bottom = top + step * len(rows) - 14
    height = bottom + 44 + 17 * len(note)
    svg = Svg(840, height, t)
    svg.header(title, subtitle)
    svg.legend(102, [("Esri", t["esri"]), ("SAM 3", t["sam3"])])
    for share in (0, 0.5, 1):
        x = x0 + full * share
        svg.line(x, top - 8, x, bottom, t["grid"] if share else t["muted"])
        svg.text(x, bottom + 18, f"{share:.0%}", 12, t["muted"], anchor="middle")
    for i, (label, sublabel, esri, sam3) in enumerate(rows):
        y = top + step * i
        svg.text(32, y + 15, label, 15, weight=600)
        svg.text(32, y + 35, sublabel, 13, t["muted"])
        for j, (share, color) in enumerate([(esri, t["esri"]), (sam3, t["sam3"])]):
            bar_y = y + j * 22
            svg.bar(x0, bar_y, full * share, 18, color)
            svg.text(x0 + full * share + 8, bar_y + 14, f"{share:.0%}", 13, weight=600)
    for k, part in enumerate(note):
        svg.text(32, bottom + 50 + 17 * k, part, 12, t["muted"])
    return svg


def found_by_size(t):
    rows = read_csv("citywide_by_size.csv", "size")
    total = sum(int(r["outlines_2015"]) for r in rows.values())
    data = []
    for size, name in SIZES:
        r = rows[size]
        n = int(r["outlines_2015"])
        data.append((name, f"{size} · {n:,} buildings",
                     int(r["found_by_esri_city"]) / n, int(r["found_by_sam3"]) / n))
    return grouped_bars(
        t, "Both models find 9 in 10 houses, but almost no sheds",
        f"Share of the {total:,} buildings on Lewisville's 2015 map that each model also outlined",
        data, ["Whole city, May 2026 imagery. The map is older than the imagery, so this measures",
               "agreement with it, not accuracy. Esri: the city's own run of Esri's model."])


def test_area(t):
    rows = read_csv("test_area_scores.csv", "method")
    esri, sam3 = rows["Esri, city run"], rows["SAM 3"]
    n = int(esri["reference_main"])

    def matched(r):
        return int(r["tp"]) / (int(r["tp"]) + int(r["fp"]))
    data = [("Buildings found", f"of the {n} on the 2015 map", int(esri["tp"]) / n, int(sam3["tp"]) / n),
            ("Outlines that match the map", "the rest may be new buildings or mistakes", matched(esri), matched(sam3))]
    return grouped_bars(
        t, "On a hidden test area, both find more than 9 in 10 buildings",
        "A 700 m square kept aside until the settings were frozen, then scored once",
        data, ["Buildings of 20 m² and up. SAM 3 draws more outlines that the 2015 map lacks."])


def outcomes(t):
    rows = read_csv("citywide_by_size.csv", "size")
    x0, full, top, step = 300, 500, 128, 52
    height = top + step * len(SIZES) + 40
    svg = Svg(840, height, t)
    svg.header("Most houses are found by both models; most sheds by neither",
               "What happened to each building on Lewisville's 2015 map, by size")
    keys = [("Found by both", "esri_city_and_sam3", t["both"]), ("Only Esri", "only_esri_city", t["esri"]),
            ("Only SAM 3", "only_sam3", t["sam3"]), ("Missed by both", "missed_by_esri_city_and_sam3", t["missed"])]
    svg.legend(102, [(name, color) for name, _, color in keys])
    for i, (size, name) in enumerate(SIZES):
        r, y = rows[size], top + step * i
        n = int(r["outlines_2015"])
        svg.text(32, y + 14, name, 15, weight=600)
        svg.text(32, y + 32, size, 13, t["muted"])
        x = x0
        for _, key, color in keys:
            w = full * int(r[key]) / n
            if w >= 1:
                svg.rect(x, y, max(w - 2, 1), 30, color)
            if w >= 44:
                ink = t["ink"] if color == t["missed"] else t["on_color"]
                svg.text(x + 8, y + 20, f"{int(r[key]) / n:.0%}", 13, ink, weight=600)
            x += w
    svg.text(32, height - 18, "Whole city, May 2026 imagery. Some buildings missed by both were demolished after 2015.",
             12, t["muted"])
    return svg


def icon_image(svg, x, y, t):
    svg.rect(x, y, 48, 40, "none", rx=6, stroke=t["ink2"])
    for dx, dy, w, h in [(7, 7, 14, 11), (27, 6, 14, 9), (8, 24, 10, 9), (24, 21, 17, 13)]:
        svg.rect(x + dx, y + dy, w, h, t["muted"], rx=1)


def icon_tiles(svg, x, y, t):
    for row in range(2):
        for col in range(2):
            svg.parts.append(f'<rect x="{x + col * 20}" y="{y + row * 16}" width="28" height="24" rx="3" '
                             f'fill="{t["muted"]}" fill-opacity="0.35" stroke="{t["ink2"]}" stroke-width="1.5"/>')


def icon_outlines(svg, x, y, t):
    svg.path(f"M{x + 3} {y + 18}l14 -14l14 14v20h-28z", t["esri"], 2.5)
    svg.path(f"M{x + 20} {y + 22}h26v18h-26z", t["sam3"], 2.5)


def icon_score(svg, x, y, t):
    svg.line(x, y + 40, x + 48, y + 40, t["ink2"], 2)
    svg.rect(x + 6, y + 6, 14, 34, t["esri"])
    svg.rect(x + 26, y + 14, 14, 26, t["sam3"])


def how_it_works(t):
    steps = [(icon_image, "Aerial image", ["A GeoTIFF or VRT,", "never changed"]),
             (icon_tiles, "Cut into tiles", ["Overlapping, so no", "building is lost"]),
             (icon_outlines, "AI draws outlines", ["SAM 3 and Esri's", "building model"]),
             (icon_score, "Clean and score", ["Same rules and same", "map for each model"])]
    pad, gap, card_h = 20, 28, 160
    width = 880
    card_w = (width - 2 * pad - 3 * gap) / 4
    svg = Svg(width, card_h + 2 * pad, t)
    for i, (icon, title, lines) in enumerate(steps):
        x = pad + i * (card_w + gap)
        svg.rect(x, pad, card_w, card_h, t["card"], rx=10)
        icon(svg, x + 18, pad + 20, t)
        svg.text(x + 18, pad + 96, title, 16, weight=700)
        for k, line in enumerate(lines):
            svg.text(x + 18, pad + 120 + 19 * k, line, 13.5, t["ink2"])
        if i < 3:
            ax = x + card_w + 7
            ay = pad + card_h / 2
            svg.path(f"M{ax} {ay}h13m-5 -5l5 5l-5 5", t["muted"], 2)
    return svg


FIGURES = {"how-it-works": how_it_works, "found-by-size": found_by_size, "test-area": test_area,
           "outcomes-by-size": outcomes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "images")
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        for mode, theme in THEMES.items():
            draw(theme).save(args.output_dir / (f"{name}.svg" if mode == "light" else f"{name}-dark.svg"))
    print(f"Wrote {2 * len(FIGURES)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
