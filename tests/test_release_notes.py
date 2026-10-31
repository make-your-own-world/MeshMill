import pytest

from tools.release_notes import release_notes


def test_release_notes_include_only_requested_entry():
    text = "# Changelog\n## 0.3.1 - Unreleased\n\nNew\n[Details](benchmarks/report.md)\n## 0.3.0\nOld\n"
    notes = release_notes(text, "0.3.1")
    assert "New" in notes and "Old" not in notes
    assert "https://github.com/make-your-own-world/MeshMill/blob/main/benchmarks/report.md" in notes
    with pytest.raises(ValueError):
        release_notes(text, "0.4.0")


def test_current_release_notes_include_major_features():
    from pathlib import Path

    from meshmill_version import APP_VERSION as VERSION
    text = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8-sig")
    notes = release_notes(text, VERSION)
    for feature in ("Performance", "Solidify", "Cast", "Open and export", "Headless automation"):
        assert feature in notes
    assert "## 0.3.3" not in notes
