"""Prepare an isolated source tree for the x86-64-v3 Python distribution."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PORTABLE = 'name = "calc-flow-python"'
_V3 = 'name = "calc-flow-python-v3"'


def prepare_v3_source(destination: Path, root: Path = ROOT) -> None:
    """Copy a source tree and change only its distribution name."""
    destination = destination.resolve()
    if destination == root.resolve() or destination.exists():
        raise ValueError("v3 source destination must be a new non-root directory")
    ignored = {".git", ".venv", "dist", "target", "__pycache__", destination.name}

    def ignore(path: str, names: list[str]) -> set[str]:
        del path
        return ignored.intersection(names)

    shutil.copytree(root, destination, ignore=ignore)
    pyproject = destination / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    if text.count(_PORTABLE) != 1:
        raise ValueError("expected one portable distribution declaration")
    pyproject.write_text(text.replace(_PORTABLE, _V3), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_v3_source(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
