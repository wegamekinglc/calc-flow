#!/usr/bin/env python3
"""Validate the complete Calc Flow Python release before a PyPI upload."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from email.parser import BytesParser
from hashlib import sha256
from http.client import HTTPSConnection
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from zipfile import ZipFile

if __package__:
    from scripts.inspect_wheel import (
        inspect_sdist,
        inspect_studio_wheel,
        inspect_wheel,
    )
else:
    from inspect_wheel import inspect_sdist, inspect_studio_wheel, inspect_wheel


ROOT = Path(__file__).resolve().parents[1]
CORE_TARGETS = frozenset(
    {
        "linux-aarch64",
        "linux-x86_64",
        "macos-arm64",
        "macos-x86_64",
        "windows-amd64",
    }
)
_FINAL_VERSION_RE = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
)
_MANYLINUX_RE = re.compile(r"manylinux_([0-9]+)_([0-9]+)_(x86_64|aarch64)")
_MACOS_RE = re.compile(r"macosx_([0-9]+)_([0-9]+)_(x86_64|arm64)")
_VERSION_RE = re.compile(
    r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ReleaseConfig:
    version: str
    requires_python: str
    studio_requirement: str


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a table")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def project_configuration(root: Path = ROOT) -> ReleaseConfig:
    package = _mapping(_read_toml(root / "pyproject.toml")["project"], "project")
    studio = _mapping(
        _read_toml(root / "web-ui/backend/pyproject.toml")["project"],
        "studio.project",
    )
    workspace = _mapping(_read_toml(root / "Cargo.toml")["workspace"], "workspace")
    workspace_package = _mapping(workspace["package"], "workspace.package")
    binding = _read_toml(root / "crates/calc-flow-python/Cargo.toml")
    binding_dependencies = _mapping(binding["dependencies"], "binding.dependencies")
    core_dependency = _mapping(
        binding_dependencies["calc-flow"],
        "binding.dependencies.calc-flow",
    )
    connector_dependency = _mapping(
        binding_dependencies["calc-flow-connectors"],
        "binding.dependencies.calc-flow-connectors",
    )

    if package.get("name") != "calc-flow-python":
        raise ValueError("pyproject.toml project.name must equal 'calc-flow-python'")
    if studio.get("name") != "calc-flow-studio":
        raise ValueError(
            "web-ui/backend/pyproject.toml project.name must equal 'calc-flow-studio'"
        )

    version = _string(package.get("version"), "project.version")
    if _FINAL_VERSION_RE.fullmatch(version) is None:
        raise ValueError(f"project.version={version!r} must be a final X.Y.Z version")
    aligned_versions = {
        "workspace.package.version": workspace_package.get("version"),
        "studio.project.version": studio.get("version"),
        "binding.calc-flow.version": core_dependency.get("version"),
        "binding.calc-flow-connectors.version": connector_dependency.get("version"),
    }
    expected_dependency_version = f"={version}"
    for name, actual in aligned_versions.items():
        expected = (
            expected_dependency_version if name.startswith("binding.") else version
        )
        if actual != expected:
            raise ValueError(f"{name}={actual!r} does not match {expected!r}")

    source = (root / "python/calc_flow/__init__.py").read_text(encoding="utf-8")
    match = _VERSION_RE.search(source)
    if match is None:
        raise ValueError("python/calc_flow/__init__.py does not declare __version__")
    if match.group(1) != version:
        raise ValueError(
            f"calc_flow.__version__={match.group(1)!r} does not match {version!r}"
        )

    requires_python = _string(
        package.get("requires-python"),
        "project.requires-python",
    )
    studio_requires_python = _string(
        studio.get("requires-python"),
        "studio.project.requires-python",
    )
    if studio_requires_python != requires_python:
        raise ValueError(
            "core and Studio requires-python values must match: "
            f"{requires_python!r} != {studio_requires_python!r}"
        )

    dependencies = studio.get("dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(dependency, str) for dependency in dependencies
    ):
        raise ValueError("studio.project.dependencies must be a string list")
    studio_requirements = [
        dependency
        for dependency in dependencies
        if dependency.lower().startswith("calc-flow")
    ]
    if len(studio_requirements) != 1:
        raise ValueError("Studio must declare exactly one calc-flow-python requirement")
    studio_requirement = studio_requirements[0]
    if _normalized_requirement(studio_requirement) != (
        "calc-flow-python",
        tuple(sorted((f">={version}", "<5"))),
    ):
        raise ValueError(
            f"Studio requirement {studio_requirement!r} does not cover "
            f"{version} within v4"
        )
    return ReleaseConfig(version, requires_python, studio_requirement)


def ensure_version_is_new_on_pypi(project: str, version: str) -> None:
    path = f"/pypi/{quote(project, safe='')}/{quote(version, safe='')}/json"
    connection = HTTPSConnection("pypi.org", timeout=20)
    try:
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        response.read()
    finally:
        connection.close()
    if response.status == 404:
        return
    if response.status != 200:
        raise OSError(
            f"PyPI version check for {project} failed with HTTP {response.status}"
        )
    raise ValueError(
        f"{project} {version} already exists on PyPI and cannot be overwritten"
    )


def validate_versions(
    root: Path = ROOT,
    tag: str | None = None,
    check_pypi: bool = False,
) -> ReleaseConfig:
    config = project_configuration(root)
    expected_tag = f"v{config.version}"
    if tag is not None and tag != expected_tag:
        raise ValueError(f"release tag {tag!r} must equal {expected_tag!r}")
    if check_pypi:
        ensure_version_is_new_on_pypi("calc-flow-python", config.version)
    return config


def _wheel_parts(path: Path) -> tuple[str, str, str, str, str]:
    if path.suffix != ".whl":
        raise ValueError(f"not a wheel: {path}")
    parts = path.name[:-4].rsplit("-", 4)
    if len(parts) != 5:
        raise ValueError(f"invalid wheel filename: {path.name}")
    return tuple(parts)  # type: ignore[return-value]


def _single_header(path: Path, metadata, key: str) -> str:
    values = metadata.get_all(key, [])
    if len(values) != 1:
        raise ValueError(
            f"{path.name}: metadata has {len(values)} {key} headers; expected exactly 1"
        )
    return values[0]


def _metadata_from_wheel(
    path: Path,
    archive: ZipFile,
    dist_info: str,
    filename: str,
):
    expected = f"{dist_info}/{filename}"
    pattern = re.compile(rf"[^/]+\.dist-info/{re.escape(filename)}")
    matches = [name for name in archive.namelist() if pattern.fullmatch(name)]
    if matches != [expected]:
        raise ValueError(
            f"{path.name}: wheel must contain exactly one {expected}, found {matches}"
        )
    return BytesParser().parsebytes(archive.read(expected))


def _validate_project_metadata(
    path: Path,
    metadata,
    *,
    name: str,
    config: ReleaseConfig,
) -> None:
    expected = {
        "Name": name,
        "Version": config.version,
        "License-Expression": "Apache-2.0",
        "Requires-Python": config.requires_python,
    }
    for key, value in expected.items():
        actual = _single_header(path, metadata, key)
        if actual != value:
            raise ValueError(f"{path.name}: {key}={actual!r}, expected {value!r}")
    if not str(metadata.get_payload()).strip():
        raise ValueError(f"{path.name}: package description is empty")


def _validate_wheel_metadata(
    path: Path,
    metadata,
    *,
    pure: bool,
    tags: tuple[str, ...],
) -> None:
    _single_header(path, metadata, "Wheel-Version")
    expected_pure = str(pure).lower()
    actual_pure = _single_header(path, metadata, "Root-Is-Purelib")
    if actual_pure != expected_pure:
        raise ValueError(
            f"{path.name}: Root-Is-Purelib={actual_pure!r}, expected {expected_pure!r}"
        )
    actual_tags = metadata.get_all("Tag", [])
    if Counter(actual_tags) != Counter(tags):
        raise ValueError(
            f"{path.name}: WHEEL tags {actual_tags!r} do not match {list(tags)!r}"
        )


def _core_target(platform_tag: str) -> str:
    components = platform_tag.split(".")
    if not components or any(not component for component in components):
        raise ValueError(f"platform field {platform_tag!r} contains an empty component")
    if len(components) != len(set(components)):
        raise ValueError(
            f"platform field {platform_tag!r} contains duplicate components"
        )
    if components == ["win_amd64"]:
        return "windows-amd64"

    manylinux_matches = [_MANYLINUX_RE.fullmatch(component) for component in components]
    if all(match is not None for match in manylinux_matches):
        matches = [match for match in manylinux_matches if match is not None]
        architectures = {match.group(3) for match in matches}
        if len(architectures) != 1:
            raise ValueError(f"mixed manylinux architectures in {platform_tag!r}")
        architecture = architectures.pop()
        baseline = f"manylinux_2_28_{architecture}"
        if baseline not in components:
            raise ValueError(f"platform field {platform_tag!r} omits {baseline!r}")
        for match in matches:
            version = (int(match.group(1)), int(match.group(2)))
            if version > (2, 28):
                raise ValueError(
                    f"platform component {match.group(0)!r} exceeds manylinux 2.28"
                )
        return f"linux-{architecture}"

    macos_matches = [_MACOS_RE.fullmatch(component) for component in components]
    if all(match is not None for match in macos_matches):
        architectures = {match.group(3) for match in macos_matches if match is not None}
        if len(architectures) != 1:
            raise ValueError(f"mixed macOS architectures in {platform_tag!r}")
        return f"macos-{architectures.pop()}"
    raise ValueError(f"unsupported core wheel platform tag: {platform_tag}")


def _validate_native_extension(path: Path, names: list[str], target: str) -> None:
    suffix = ".pyd" if target == "windows-amd64" else ".so"
    extensions = [
        name
        for name in names
        if name.startswith("calc_flow/_native") and name.endswith(suffix)
    ]
    if len(extensions) != 1:
        raise ValueError(
            f"{path.name}: expected one {suffix} native extension, found {extensions}"
        )


def validate_core_wheel(path: Path, config: ReleaseConfig) -> tuple[str, str]:
    inspect_wheel(path)
    distribution, version, python_tag, abi_tag, platform_tag = _wheel_parts(path)
    if distribution != "calc_flow_python":
        raise ValueError(f"{path.name}: unexpected distribution {distribution!r}")
    if version != config.version:
        raise ValueError(f"{path.name}: version {version!r} != {config.version!r}")
    if python_tag != "cp313":
        raise ValueError(f"{path.name}: Python tag {python_tag!r} must equal 'cp313'")
    if abi_tag != "abi3":
        raise ValueError(f"{path.name}: ABI {abi_tag!r} must equal 'abi3'")
    target = _core_target(platform_tag)
    if target not in CORE_TARGETS:
        raise ValueError(f"{path.name}: unsupported release target {target!r}")

    dist_info = f"{distribution}-{version}.dist-info"
    with ZipFile(path) as archive:
        names = archive.namelist()
        _validate_native_extension(path, names, target)
        metadata = _metadata_from_wheel(path, archive, dist_info, "METADATA")
        wheel_metadata = _metadata_from_wheel(path, archive, dist_info, "WHEEL")
    _validate_project_metadata(path, metadata, name="calc-flow-python", config=config)
    expected_tags = tuple(
        f"{python_tag}-{abi_tag}-{platform}" for platform in platform_tag.split(".")
    )
    _validate_wheel_metadata(
        path,
        wheel_metadata,
        pure=False,
        tags=expected_tags,
    )
    return target, sha256(path.read_bytes()).hexdigest()


def _normalized_requirement(requirement: str) -> tuple[str, tuple[str, ...]]:
    compact = "".join(requirement.lower().split())
    match = re.fullmatch(r"([a-z0-9_.-]+)(.*)", compact)
    if match is None:
        raise ValueError(f"invalid requirement {requirement!r}")
    name = re.sub(r"[-_.]+", "-", match.group(1))
    specifiers = tuple(sorted(filter(None, match.group(2).split(","))))
    return name, specifiers


def validate_studio_wheel(path: Path, config: ReleaseConfig) -> str:
    inspect_studio_wheel(path)
    distribution, version, python_tag, abi_tag, platform_tag = _wheel_parts(path)
    expected_parts = ("calc_flow_studio", config.version, "py3", "none", "any")
    actual_parts = (distribution, version, python_tag, abi_tag, platform_tag)
    if actual_parts != expected_parts:
        raise ValueError(
            f"{path.name}: Studio wheel fields {actual_parts!r} != {expected_parts!r}"
        )

    dist_info = f"{distribution}-{version}.dist-info"
    with ZipFile(path) as archive:
        metadata = _metadata_from_wheel(path, archive, dist_info, "METADATA")
        wheel_metadata = _metadata_from_wheel(path, archive, dist_info, "WHEEL")
    _validate_project_metadata(
        path,
        metadata,
        name="calc-flow-studio",
        config=config,
    )
    requirements = metadata.get_all("Requires-Dist", [])
    expected_requirement = _normalized_requirement(config.studio_requirement)
    matching_requirements = [
        requirement
        for requirement in requirements
        if _normalized_requirement(requirement) == expected_requirement
    ]
    if len(matching_requirements) != 1:
        raise ValueError(
            f"{path.name}: expected one Requires-Dist {config.studio_requirement!r}, "
            f"found {requirements!r}"
        )
    _validate_wheel_metadata(
        path,
        wheel_metadata,
        pure=True,
        tags=("py3-none-any",),
    )
    return sha256(path.read_bytes()).hexdigest()


def validate_sdist(path: Path, config: ReleaseConfig) -> str:
    expected_name = f"calc_flow_python-{config.version}.tar.gz"
    if path.name != expected_name:
        raise ValueError(f"source distribution {path.name!r} != {expected_name!r}")
    inspect_sdist(path)
    return sha256(path.read_bytes()).hexdigest()


def _relative_name(path: Path, dist_dir: Path) -> str:
    return str(PurePosixPath(path.relative_to(dist_dir)))


def validate_release(
    dist_dir: Path,
    root: Path = ROOT,
    tag: str | None = None,
    check_pypi: bool = False,
) -> list[str]:
    config = validate_versions(root, tag, check_pypi)
    dist_dir = dist_dir.resolve()
    if not dist_dir.is_dir():
        raise ValueError(f"release directory does not exist: {dist_dir}")

    wheels = sorted(dist_dir.rglob("*.whl"))
    sdists = sorted(dist_dir.rglob("*.tar.gz"))
    core_wheels = [path for path in wheels if path.name.startswith("calc_flow_python-")]
    studio_wheels = [
        path for path in wheels if path.name.startswith("calc_flow_studio-")
    ]
    unknown_wheels = [
        path for path in wheels if path not in core_wheels and path not in studio_wheels
    ]
    if unknown_wheels:
        raise ValueError(
            f"unexpected wheel artifacts: {[path.name for path in unknown_wheels]}"
        )
    if len(studio_wheels) != 1:
        raise ValueError(
            f"expected one Studio wheel, found {[path.name for path in studio_wheels]}"
        )
    if len(sdists) != 1:
        raise ValueError(
            f"expected one source distribution, found {[path.name for path in sdists]}"
        )

    manifest_by_name: dict[str, str] = {}
    actual_targets: set[str] = set()
    for wheel in core_wheels:
        target, digest = validate_core_wheel(wheel, config)
        if target in actual_targets:
            raise ValueError(f"duplicate core wheel target {target!r}")
        actual_targets.add(target)
        manifest_by_name[_relative_name(wheel, dist_dir)] = digest
    if actual_targets != CORE_TARGETS:
        missing = sorted(CORE_TARGETS - actual_targets)
        unexpected = sorted(actual_targets - CORE_TARGETS)
        raise ValueError(
            f"core wheel matrix mismatch: missing={missing}, unexpected={unexpected}"
        )

    studio = studio_wheels[0]
    manifest_by_name[_relative_name(studio, dist_dir)] = validate_studio_wheel(
        studio, config
    )
    sdist = sdists[0]
    manifest_by_name[_relative_name(sdist, dist_dir)] = validate_sdist(sdist, config)
    return [f"{manifest_by_name[name]}  {name}" for name in sorted(manifest_by_name)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Calc Flow Python release artifacts"
    )
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--check-pypi", action="store_true")
    parser.add_argument("--version-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.version_only:
            config = validate_versions(tag=args.tag, check_pypi=args.check_pypi)
            print(config.version)
            return 0
        if args.dist_dir is None:
            parser.error("--dist-dir is required unless --version-only is used")
        manifest = validate_release(
            args.dist_dir,
            tag=args.tag,
            check_pypi=args.check_pypi,
        )
    except (OSError, ValueError) as error:
        print(f"Python release verification failed: {error}", file=sys.stderr)
        return 1
    print("\n".join(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
