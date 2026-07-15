from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

_FORBIDDEN_PARTS = {"fixtures", "tests"}
_FORBIDDEN_NAMES = {"agents.md", "claude.md"}
_PROJECT_DATA_SUFFIXES = {".json", ".yaml", ".yml"}


def inspect_wheel(wheel: Path) -> int:
    with ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]

    if "calc_flow/__init__.py" not in names:
        raise ValueError(f"{wheel}: missing calc_flow/__init__.py")
    if not {"calc_flow/_native.abi3.so", "calc_flow/_native.pyd"}.intersection(names):
        raise ValueError(f"{wheel}: missing abi3 native module")

    for name in names:
        path = PurePosixPath(name)
        in_package = path.parts[0] == "calc_flow"
        in_metadata = path.parts[0].startswith("calc_flow-") and path.parts[0].endswith(
            ".dist-info"
        )
        if not (in_package or in_metadata):
            raise ValueError(f"{wheel}: unexpected wheel entry: {name}")
        if in_package and (
            _FORBIDDEN_PARTS.intersection(part.lower() for part in path.parts)
            or path.name.lower() in _FORBIDDEN_NAMES
            or path.suffix.lower() in _PROJECT_DATA_SUFFIXES
        ):
            raise ValueError(f"{wheel}: forbidden wheel entry: {name}")

    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Calc Flow core wheel contents")
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        count = inspect_wheel(wheel)
        print(f"{wheel}: verified {count} files")


if __name__ == "__main__":
    main()
