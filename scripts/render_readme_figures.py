"""Render original, deterministic README figures without real imagery or a model.

Optional documentation dependencies: NumPy and Pillow. No GIS or model runtime is
used. Geometry, colors, and the random seed are declared here. Use explicit font
paths to reproduce the same typography on a different operating system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


SEED = 20260920
SCALE = 2
WIDTH = 1800
INK = "#142c40"
MUTED = "#4f6471"
TEAL = "#087f83"
CYAN = "#44e1e3"
BLUE = "#2d669b"
AMBER = "#986425"
PAGE = "#f6f8fa"
LINE = "#d3dfe5"


def choose_fonts(regular: str | None, bold: str | None) -> tuple[Path, Path]:
    if regular or bold:
        if not regular or not bold:
            raise ValueError("Supply both --font-regular and --font-bold.")
        candidates = [(Path(regular), Path(bold))]
    else:
        candidates = [
            (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
            (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
             Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
            (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
        ]
    for pair in candidates:
        if all(path.is_file() for path in pair):
            return pair
    raise FileNotFoundError("Supply existing TTF fonts with --font-regular and --font-bold.")


class Canvas:
    def __init__(self, height: int, fonts: tuple[Path, Path]):
        self.image = Image.new("RGB", (WIDTH * SCALE, height * SCALE), PAGE)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = fonts

    def text(self, xy, value, size=28, color=INK, bold=False, anchor="lt"):
        font = ImageFont.truetype(str(self.fonts[int(bold)]), size * SCALE)
        self.draw.text(tuple(int(v * SCALE) for v in xy), value, font=font,
                       fill=color, anchor=anchor, spacing=9 * SCALE)

    def rect(self, bounds, fill, radius=0, outline=None, width=1):
        bounds = tuple(int(v * SCALE) for v in bounds)
        if radius:
            self.draw.rounded_rectangle(bounds, radius=radius * SCALE, fill=fill,
                                        outline=outline, width=width * SCALE)
        else:
            self.draw.rectangle(bounds, fill=fill, outline=outline, width=width * SCALE)

    def line(self, points, fill=LINE, width=3):
        self.draw.line([(int(x * SCALE), int(y * SCALE)) for x, y in points],
                       fill=fill, width=width * SCALE, joint="curve")

    def arrow(self, start, end, fill=MUTED, width=3):
        self.line([start, end], fill, width)
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / length, dy / length
        back = (end[0] - 11 * ux, end[1] - 11 * uy)
        points = [end, (back[0] + 6 * uy, back[1] - 6 * ux),
                  (back[0] - 6 * uy, back[1] + 6 * ux)]
        self.draw.polygon([(int(x * SCALE), int(y * SCALE)) for x, y in points], fill=fill)

    def paste(self, image: Image.Image, xy, side):
        rendered = image.resize((side * SCALE, side * SCALE), Image.Resampling.LANCZOS)
        self.image.paste(rendered, (xy[0] * SCALE, xy[1] * SCALE))

    def finish(self, path: Path, title: str, description: str):
        final = self.image.resize((WIDTH, self.image.height // SCALE), Image.Resampling.LANCZOS)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Title", title)
        metadata.add_text("Description", description)
        metadata.add_text("Source", "Original deterministic schematic; no real imagery or model inference.")
        final.save(path, pnginfo=metadata, optimize=True)


def render_workflow(path: Path, fonts):
    c = Canvas(1170, fonts)
    c.text((80, 49), "LOCAL BUILDING-FOOTPRINT EXPERIMENTS", 23, TEAL, True)
    c.text((80, 94), "From imagery to an auditable comparison", 52, bold=True)
    c.text((80, 160), "Keep source imagery local. Preserve raw outputs. Measure against independent labels.", 28, MUTED)

    c.rect((80, 217, 1720, 337), "#e9f3f3", 18, "#b8d6d6")
    c.text((111, 244), "1  Prepare imagery", 33, TEAL, True)
    c.text((495, 244), "Local RGB GeoTIFF / VRT  →  overlapping tiles", 31, INK, True)
    c.text((495, 291), "Retain CRS, valid-data masks and tile hashes", 27, MUTED)
    c.arrow((900, 338), (900, 378))
    c.text((80, 379), "2  Choose an extraction method", 31, bold=True)

    cards = [
        (80, 600, "#ecf6f4", TEAL, "SAM 3 concepts",
         "Text or visual exemplars", "Find matching building instances"),
        (640, 1160, "#edf2f8", BLUE, "SAM 3 instances",
         "Boxes or grouped points", "Delineate selected buildings"),
        (1200, 1720, "#f6f0e6", AMBER, "Other baselines",
         "Esri Mask R-CNN", "Imported Nearmap AI vectors"),
    ]
    for left, right, fill, accent, title, line1, line2 in cards:
        c.rect((left, 433, right, 612), fill, 18, LINE)
        c.rect((left + 25, 458, left + 31, 485), accent, 3)
        c.text((left + 48, 455), title, 33, accent, True)
        c.text((left + 27, 512), line1, 28, INK)
        c.text((left + 27, 554), line2, 26, MUTED)
        center = (left + right) // 2
        c.line([(center, 613), (center, 643)], MUTED)
    c.line([(340, 643), (1460, 643)], MUTED)
    c.arrow((900, 643), (900, 677))

    c.rect((80, 679, 1720, 775), "#ffffff", 18, LINE)
    c.text((111, 708), "3  Retain raw vectors", 32, bold=True)
    c.text((495, 713), "Track provenance, processing coverage and tile-edge defects", 29, MUTED)
    c.line([(900, 777), (900, 809), (482, 809)], MUTED)
    c.arrow((482, 809), (482, 827))

    c.rect((80, 831, 855, 987), "#ffffff", 18, LINE)
    c.text((109, 857), "4  Reconcile overlap + clean", 32, bold=True)
    c.text((109, 906), "Remove duplicates; preserve building instances", 27, MUTED)
    c.text((109, 946), "Review edge cuts and geometric changes", 27, MUTED)
    c.arrow((868, 908), (924, 908))
    c.rect((945, 831, 1720, 987), "#ffffff", 18, LINE)
    c.text((974, 857), "5  Evaluate held-out AOIs", 32, bold=True)
    c.text((974, 906), "Match area, capture date and label convention", 27, MUTED)
    c.text((974, 946), "Measure detection, shape and review effort", 27, MUTED)

    c.rect((80, 1016, 1720, 1114), "#eaf0f4", 14)
    c.text((110, 1037), "OPTIONAL FINE-TUNING", 21, BLUE, True)
    c.text((110, 1071), "Spatial splits  →  COCO instance masks  →  native SAM 3 training  →  checkpoint export", 28, INK)
    c.text((80, 1138), "Workflow schematic. Guided methods require human prompts; this figure shows no model results.", 23, MUTED)
    c.finish(path, "Local building-footprint experiment workflow",
             "Workflow schematic covering local imagery preparation, SAM 3 concepts and instances, "
             "Esri and imported vector baselines, cleanup, held-out evaluation, and optional training.")


def geometry():
    def box(x0, y0, x1, y1):
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [
        (box(48, 58, 145, 118), [], (96, 87)),
        ([(178, 47), (231, 47), (231, 139), (199, 139), (199, 100), (178, 100)], [], (210, 79)),
        (box(335, 50, 474, 172), [box(369, 83, 440, 140)], (405, 66)),
        (box(58, 173, 145, 213), [], (100, 193)),
        (box(179, 173, 227, 219), [], (203, 197)),
        ([(48, 323), (154, 323), (154, 367), (118, 367), (118, 440), (48, 440)], [], (84, 365)),
        (box(176, 322, 231, 383), [], (202, 351)),
        (box(177, 404, 232, 459), [], (204, 431)),
        (box(326, 322, 408, 395), [], (366, 359)),
        (box(412, 322, 474, 395), [], (443, 359)),
        ([(330, 433), (387, 414), (401, 457), (344, 476)], [], (365, 445)),
        (box(429, 428, 481, 466), [], (455, 447)),
    ]


def make_scene() -> Image.Image:
    rng = np.random.default_rng(SEED)
    base = np.empty((512, 512, 3), dtype=np.float64)
    base[:] = [133, 151, 117]
    base += rng.normal(0, 2.8, (512, 512, 1))
    image = Image.fromarray(np.clip(base, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(image)
    # Illustrative lots, sidewalks, and crossing roads. No real geography.
    for bounds in [(23, 24, 250, 232), (299, 25, 490, 229),
                   (22, 299, 249, 490), (299, 300, 491, 492)]:
        d.rectangle(bounds, outline=(171, 183, 154), width=2)
    d.rectangle((0, 237, 511, 291), fill=(190, 193, 177))
    d.rectangle((245, 0, 298, 511), fill=(190, 193, 177))
    d.rectangle((0, 244, 511, 284), fill=(97, 111, 113))
    d.rectangle((252, 0, 291, 511), fill=(97, 111, 113))
    for pos in range(10, 510, 30):
        if not 226 < pos < 300:
            d.line([(pos, 264), (pos + 13, 264)], fill=(214, 209, 169), width=2)
            d.line([(272, pos), (272, pos + 13)], fill=(214, 209, 169), width=2)
    for bounds in [(80, 119, 112, 161), (200, 140, 219, 164), (190, 220, 219, 237),
                   (112, 298, 142, 321), (299, 338, 324, 368), (462, 396, 478, 427)]:
        d.rectangle(bounds, fill=(183, 180, 160))
    tree_centers = [(29, 54), (29, 183), (157, 155), (39, 223), (319, 35),
                    (488, 211), (318, 207), (444, 212), (41, 463), (156, 477),
                    (24, 367), (242, 474), (311, 487), (490, 311), (305, 411)]
    for x, y in tree_centers:
        r = int(rng.integers(9, 16))
        d.ellipse((x-r+5, y-r+6, x+r+5, y+r+6), fill=(79, 104, 80))
        d.ellipse((x-r, y-r, x+r, y+r), fill=(78, 126, 82))
        d.ellipse((x-r+3, y-r+2, x+3, y+4), fill=(107, 147, 91))
    roofs = [(189, 168, 137), (147, 153, 156), (174, 182, 178), (177, 139, 112),
             (163, 173, 179), (190, 168, 139), (174, 150, 121), (170, 177, 180),
             (181, 187, 182), (181, 166, 146), (164, 157, 149), (177, 141, 119)]
    for (shell, holes, _), roof in zip(geometry(), roofs):
        d.polygon([(x+6, y+7) for x, y in shell], fill=(81, 101, 90))
        d.polygon(shell, fill=roof, outline=(116, 125, 119), width=1)
        for hole in holes:
            d.polygon(hole, fill=(135, 153, 118), outline=(108, 125, 102), width=2)
    # Ridge lines give the invented roofs texture without pretending to be photos.
    for coords in [[(53, 88), (140, 88)], [(61, 193), (142, 193)],
                   [(53, 346), (145, 346)], [(83, 366), (83, 434)],
                   [(180, 353), (227, 353)], [(181, 432), (228, 432)],
                   [(330, 358), (403, 358)], [(416, 358), (470, 358)],
                   [(434, 447), (477, 447)], [(340, 448), (386, 433)]]:
        d.line(coords, fill=(222, 219, 198), width=2)
    return image


def render_synthetic(path: Path, fonts):
    c = Canvas(1280, fonts)
    c.text((80, 47), "SYNTHETIC REFERENCE ILLUSTRATION", 23, TEAL, True)
    c.text((80, 92), "The pixel-to-polygon task", 54, bold=True)
    c.text((80, 160), "An invented scene and its hand-defined building instances. No model was run.", 29, MUTED)
    c.text((80, 225), "A  Synthetic RGB raster", 34, bold=True)
    c.text((940, 225), "B  Reference footprints", 34, bold=True)
    source = make_scene()
    c.paste(source, (80, 283), 780)
    c.paste(source, (940, 283), 780)
    c.rect((79, 282, 861, 1064), None, outline=LINE, width=1)
    c.rect((939, 282, 1721, 1064), None, outline=LINE, width=1)
    factor = 780 / 512
    for idx, (shell, holes, center) in enumerate(geometry(), start=1):
        for ring in [shell, *holes]:
            points = [(940 + x * factor, 283 + y * factor) for x, y in [*ring, ring[0]]]
            c.line(points, "#174c55", 6)
            c.line(points, CYAN, 3)
        x, y = 940 + center[0] * factor, 283 + center[1] * factor
        c.rect((x-21, y-17, x+21, y+18), INK, 6)
        c.text((x, y+1), f"{idx:02}", 23, "#ffffff", True, "mm")
    # North and scale are illustrative and follow the declared synthetic GSD.
    for x in [80, 940]:
        c.rect((x+20, 298, x+89, 362), "#f6f8fa", 10)
        c.text((x+54, 304), "N", 23, INK, True, "mt")
        c.arrow((x+54, 353), (x+54, 333), INK, 3)
    c.line([(106, 1026), (106 + 125 * factor, 1026)], "#ffffff", 6)
    c.line([(106, 1018), (106, 1034)], "#ffffff", 3)
    c.line([(106 + 125 * factor, 1018), (106 + 125 * factor, 1034)], "#ffffff", 3)
    c.rect((106, 978, 303, 1009), "#142c40", 5)
    c.text((115, 982), "25 m (synthetic)", 23, "#ffffff")

    c.text((80, 1092), "512 × 512 pixels  •  0.20 m / pixel", 28, MUTED)
    c.line([(943, 1107), (999, 1107)], "#174c55", 7)
    c.line([(943, 1107), (999, 1107)], CYAN, 4)
    c.text((1016, 1092), "12 separate instances, including a courtyard hole", 26, MUTED)
    c.rect((80, 1151, 1720, 1246), "#eaf0f4", 14)
    c.text((108, 1173), "REFERENCE GEOMETRY, NOT MODEL OUTPUT", 24, BLUE, True)
    c.text((108, 1211), "Both panels use the same predefined shapes. This illustrates alignment, not extraction accuracy.", 26, INK)
    c.finish(path, "Synthetic imagery and reference building footprints",
             "An original synthetic 512 by 512 pixel scene at an illustrative 0.20 metres per pixel, "
             "with 12 predefined building instances including a courtyard hole. Reference geometry, "
             "not model output or independent validation. No real Nearmap imagery is used.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs" / "images")
    parser.add_argument("--font-regular")
    parser.add_argument("--font-bold")
    args = parser.parse_args()
    fonts = choose_fonts(args.font_regular, args.font_bold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow = args.output_dir / "workflow.png"
    synthetic = args.output_dir / "synthetic-example.png"
    render_workflow(workflow, fonts)
    render_synthetic(synthetic, fonts)
    manifest = {
        "description": "Original deterministic documentation figures; no model inference or real imagery.",
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "font_files": [{"name": p.name, "sha256": sha256(p)} for p in fonts],
        "generator_sha256": sha256(__file__),
        "outputs": [{"name": p.name, "sha256": sha256(p),
                     "width": Image.open(p).width, "height": Image.open(p).height}
                    for p in [workflow, synthetic]],
        "synthetic_scene": {"raster_pixels": [512, 512], "illustrative_gsd_m": 0.2,
                            "reference_instances": len(geometry()), "model_executed": False},
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Rendered workflow.png and synthetic-example.png; wrote provenance.json.")


if __name__ == "__main__":
    main()
