"""The committed charts are exactly what scripts/make_charts.py draws from the result CSVs."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("make_charts", ROOT / "scripts/make_charts.py")
make_charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_charts)


def test_committed_charts_match_the_generator(tmp_path):
    make_charts.main(["--output-dir", str(tmp_path)])
    drawn = sorted(path.name for path in tmp_path.glob("*.svg"))
    assert drawn == sorted(path.name for path in (ROOT / "docs/images").glob("*.svg"))
    for name in drawn:
        assert (tmp_path / name).read_text(encoding="utf-8") == (ROOT / "docs/images" / name).read_text(encoding="utf-8"), \
            f"{name} is out of date; run python scripts/make_charts.py"


def test_every_series_keeps_its_own_colour():
    # Violet means "found by both", so no method may borrow it, and no two series share a colour.
    for theme in make_charts.THEMES.values():
        series = [theme[key] for key in ("esri", "esri_local", "sam3", "both", "missed")]
        assert len(set(series)) == len(series)
