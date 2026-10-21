from tools.release_notes import release_notes
from tools.normalize_package_timestamps import normalize
import pytest


def test_release_notes_include_only_requested_entry():
    text = "# Changelog\n## 0.3.1 - Unreleased\n\nNew\n[Details](benchmarks/report.md)\n## 0.3.0\nOld\n"
    notes = release_notes(text, "0.3.1")
    assert "New" in notes and "Old" not in notes
    assert "https://github.com/make-your-own-world/MeshMill/blob/main/benchmarks/report.md" in notes
    with pytest.raises(ValueError):
        release_notes(text, "0.4.0")


def test_package_timestamps_share_one_build_time(tmp_path):
    directory = tmp_path/"package"
    directory.mkdir()
    file = directory/"file.txt"
    file.write_text("test")
    normalize(directory, 1700000000.)
    assert file.stat().st_mtime == directory.stat().st_mtime == 1700000000.


@pytest.mark.parametrize("suffix", ["a"*36, "APPID_"+"a"*170+"."+"b"*170+"."+"c"*170])
def test_release_screening_detects_short_and_long_installation_tokens(suffix):
    from tools.release_privacy_check import PATTERNS
    synthetic = "ghs"+"_"+suffix
    assert PATTERNS["GitHub credential"].search(synthetic)
    assert not PATTERNS["GitHub credential"].search("GH_TOKEN: ${{ github.token }}")
