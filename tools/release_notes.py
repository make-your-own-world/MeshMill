"""Extract one reviewed changelog entry for release publication."""
from pathlib import Path
import argparse
import re


def release_notes(changelog: str, version: str) -> str:
    headings = list(re.finditer(r"^## ([^\n]+)$", changelog, re.MULTILINE))
    for index, heading in enumerate(headings):
        if heading.group(1).split()[0] == version:
            end = headings[index+1].start() if index+1 < len(headings) else len(changelog)
            body = changelog[heading.end():end].strip()
            if not body:
                raise ValueError("Release entry is empty")
            return re.sub(r"\]\((?!https?://|#)([^)]+)\)",
                          r"](https://github.com/make-your-own-world/MeshMill/blob/main/\1)", body)+"\n"
    raise ValueError(f"No changelog entry for {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.write_text(release_notes((root/"CHANGELOG.md").read_text(encoding="utf-8"),
                                        args.version), encoding="utf-8")
