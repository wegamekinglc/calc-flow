from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from tarfile import open as open_tar
from zipfile import ZipFile

_FORBIDDEN_PARTS = {"fixtures", "tests"}
_FORBIDDEN_NAMES = {"agents.md", "claude.md"}
_PROJECT_DATA_SUFFIXES = {".json", ".yaml", ".yml"}
_LICENSE_MARKERS = (b"Apache License", b"Version 2.0, January 2004")


def _is_repository_guidance(path: PurePosixPath) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    return (
        any(part in {".claude", ".codex"} for part in parts)
        or path.name.lower() in _FORBIDDEN_NAMES
        or any(
            parts[index : index + 2] == ("docs", "superpowers")
            for index in range(len(parts) - 1)
        )
        or "design" in parts
    )


def _validate_license(content: bytes, artifact: Path) -> None:
    if any(marker not in content for marker in _LICENSE_MARKERS):
        raise ValueError(f"{artifact}: invalid Apache-2.0 license")


def _require_wheel_license(names: list[str], metadata_prefix: str, wheel: Path) -> None:
    license_entries = {
        name
        for name in names
        if PurePosixPath(name).parts[0].startswith(metadata_prefix)
        and PurePosixPath(name).parts[0].endswith(".dist-info")
        and PurePosixPath(name).parts[1:] == ("licenses", "LICENSE")
    }
    if not license_entries:
        raise ValueError(f"{wheel}: missing Apache-2.0 license")
    with ZipFile(wheel) as archive:
        _validate_license(archive.read(sorted(license_entries)[0]), wheel)


def inspect_wheel(wheel: Path) -> int:
    with ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]

    if "calc_flow/__init__.py" not in names:
        raise ValueError(f"{wheel}: missing calc_flow/__init__.py")
    if not {"calc_flow/_native.abi3.so", "calc_flow/_native.pyd"}.intersection(names):
        raise ValueError(f"{wheel}: missing abi3 native module")
    _require_wheel_license(names, "calc_flow-", wheel)

    for name in names:
        path = PurePosixPath(name)
        in_package = path.parts[0] == "calc_flow"
        in_metadata = path.parts[0].startswith("calc_flow-") and path.parts[0].endswith(
            ".dist-info"
        )
        if not (in_package or in_metadata):
            raise ValueError(f"{wheel}: unexpected wheel entry: {name}")
        if _is_repository_guidance(path):
            raise ValueError(f"{wheel}: forbidden wheel entry: {name}")
        if in_package and (
            _FORBIDDEN_PARTS.intersection(part.lower() for part in path.parts)
            or path.suffix.lower() in _PROJECT_DATA_SUFFIXES
        ):
            raise ValueError(f"{wheel}: forbidden wheel entry: {name}")

    return len(names)


def inspect_studio_wheel(wheel: Path) -> int:
    with ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]

    required = {
        "calc_flow_studio/__init__.py",
        "calc_flow_studio/static/index.html",
    }
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{wheel}: missing Studio entries: {sorted(missing)}")
    _require_wheel_license(names, "calc_flow_studio-", wheel)

    for name in names:
        path = PurePosixPath(name)
        in_package = path.parts[0] == "calc_flow_studio"
        in_metadata = path.parts[0].startswith("calc_flow_studio-") and path.parts[
            0
        ].endswith(".dist-info")
        if not (in_package or in_metadata):
            raise ValueError(f"{wheel}: unexpected Studio wheel entry: {name}")
        if _is_repository_guidance(path):
            raise ValueError(f"{wheel}: forbidden Studio wheel entry: {name}")
        if in_package and _FORBIDDEN_PARTS.intersection(
            part.lower() for part in path.parts
        ):
            raise ValueError(f"{wheel}: forbidden Studio wheel entry: {name}")

    return len(names)


def _archive_names(archive_path: Path) -> list[str]:
    with open_tar(archive_path, "r:gz") as archive:
        return [member.name for member in archive.getmembers() if member.isfile()]


def _relative_archive_names(
    archive_path: Path,
) -> tuple[list[str], list[PurePosixPath]]:
    names = _archive_names(archive_path)
    paths = [PurePosixPath(name) for name in names]
    if not paths or any(len(path.parts) < 2 for path in paths):
        raise ValueError(f"{archive_path}: entries must share an archive root")
    roots = {path.parts[0] for path in paths}
    if len(roots) != 1:
        raise ValueError(f"{archive_path}: entries do not share one archive root")
    relative = [PurePosixPath(*path.parts[1:]) for path in paths]
    return names, relative


def _require_archive_license(archive_path: Path, relative: list[PurePosixPath]) -> None:
    if PurePosixPath("LICENSE") not in relative:
        raise ValueError(f"{archive_path}: missing Apache-2.0 license")
    with open_tar(archive_path, "r:gz") as archive:
        member = next(
            member
            for member in archive.getmembers()
            if member.isfile() and PurePosixPath(member.name).parts[1:] == ("LICENSE",)
        )
        license_file = archive.extractfile(member)
        if license_file is None:
            raise ValueError(f"{archive_path}: unreadable Apache-2.0 license")
        _validate_license(license_file.read(), archive_path)


def inspect_sdist(sdist: Path) -> int:
    names, relative = _relative_archive_names(sdist)
    _require_archive_license(sdist, relative)
    required = {
        PurePosixPath("Cargo.lock"),
        PurePosixPath("pyproject.toml"),
        PurePosixPath("crates/calc-flow/Cargo.toml"),
        PurePosixPath("crates/calc-flow/src/lib.rs"),
        PurePosixPath("crates/calc-flow-python/Cargo.toml"),
        PurePosixPath("python/calc_flow/__init__.py"),
    }
    missing = required.difference(relative)
    if missing:
        raise ValueError(f"{sdist}: missing sdist entries: {sorted(map(str, missing))}")

    for path in relative:
        forbidden = (
            path.parts[:2] == ("src", "calc_flow")
            or path.parts[:1] == ("web-ui",)
            or path.parts[:2] == ("tests", "fixtures")
            or _is_repository_guidance(path)
        )
        if forbidden:
            raise ValueError(f"{sdist}: forbidden sdist entry: {path}")
    return len(names)


def inspect_crate(crate: Path) -> int:
    names, relative = _relative_archive_names(crate)
    _require_archive_license(crate, relative)
    required = {PurePosixPath("Cargo.toml"), PurePosixPath("src/lib.rs")}
    missing = required.difference(relative)
    if missing:
        raise ValueError(f"{crate}: missing crate entries: {sorted(map(str, missing))}")
    for path in relative:
        if "tests" in (part.lower() for part in path.parts) or _is_repository_guidance(
            path
        ):
            raise ValueError(f"{crate}: forbidden crate entry: {path}")
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Calc Flow release artifacts")
    parser.add_argument(
        "kind", choices=("core-wheel", "studio-wheel", "sdist", "crate")
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    inspectors = {
        "core-wheel": inspect_wheel,
        "studio-wheel": inspect_studio_wheel,
        "sdist": inspect_sdist,
        "crate": inspect_crate,
    }
    for artifact in args.artifacts:
        count = inspectors[args.kind](artifact)
        print(f"{artifact}: verified {count} files")


if __name__ == "__main__":
    main()
