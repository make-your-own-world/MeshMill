from tools.release_notes import release_notes
import pytest


def test_release_notes_include_only_requested_entry():
    text = "# Changelog\n## 0.3.1 - Unreleased\n\nNew\n[Details](benchmarks/report.md)\n## 0.3.0\nOld\n"
    notes = release_notes(text, "0.3.1")
    assert "New" in notes and "Old" not in notes
    assert "https://github.com/make-your-own-world/MeshMill/blob/main/benchmarks/report.md" in notes
    with pytest.raises(ValueError):
        release_notes(text, "0.4.0")
