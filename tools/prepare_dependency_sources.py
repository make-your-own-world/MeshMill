"""Prepare verified source archives for redistributed surface-completion dependencies."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from create_reproducible_zip import create_archive

DEPENDENCIES = ("pymeshfix", "tetgen", "PyMaxflow", "python-sat")


def prepare(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for package in DEPENDENCIES:
        version = importlib.metadata.version(package)
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=60) as response:
            metadata = json.load(response)
        sources = [entry for entry in metadata["urls"] if entry["packagetype"] == "sdist"]
        if len(sources) != 1:
            raise ValueError(f"Expected one source distribution for {package} {version}")
        source = sources[0]
        url = source["url"]
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
            raise ValueError("Source archive is not hosted by the expected PyPI file service")
        name = source["filename"]
        if Path(name).name != name:
            raise ValueError("Source archive filename contains a directory")
        output = destination / name
        expected = source["digests"]["sha256"]
        if not output.exists() or hashlib.sha256(output.read_bytes()).hexdigest() != expected:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source checksum mismatch for {package}")
            output.write_bytes(data)
        manifest.append({"package": package, "version": version, "filename": name,
                         "url": url, "sha256": expected})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.directory)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    create_archive(args.directory, args.archive)


if __name__ == "__main__":
    main()
