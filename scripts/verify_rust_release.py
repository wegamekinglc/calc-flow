"""Inspect a Cargo distribution and run a consumer without workspace patches."""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath

if __package__:
    from .inspect_wheel import inspect_crate as inspect_crate_contents
    from .verify_python_release import project_configuration
else:
    from inspect_wheel import inspect_crate as inspect_crate_contents
    from verify_python_release import project_configuration

ROOT = Path(__file__).resolve().parents[1]
_SOURCE_KEYS = frozenset({"path", "git", "registry", "registry-index", "workspace"})


def validate_tag(tag: str, version: str) -> None:
    """Require the core tag namespace and the shared calendar version."""
    if tag != f"calc-flow-v{version}":
        raise ValueError(f"expected calc-flow-v{version}, got {tag!r}")


def _distribution_files(
    archive: tarfile.TarFile, prefix: str
) -> dict[str, tarfile.TarInfo]:
    files = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if (
            not path.parts
            or path.parts[0] != prefix
            or ".." in path.parts
            or path.is_absolute()
            or not (member.isfile() or member.isdir())
        ):
            raise ValueError(f"unexpected archive entry: {member.name}")
        if member.isfile():
            files[path.relative_to(prefix).as_posix()] = member
    return files


def _validate_manifest(source: str, version: str) -> None:
    manifest = tomllib.loads(source)
    package = manifest["package"]
    if (package["name"], package["version"]) != ("calc-flow", version):
        raise ValueError("crate identity differs from the shared version")
    if package.get("license") != "Apache-2.0" or package.get("readme") != "README.md":
        raise ValueError("crate must declare its license and README")
    if any(key in manifest for key in ("patch", "replace", "workspace")):
        raise ValueError("distribution manifest retains workspace overrides")
    for section in (manifest, *manifest.get("target", {}).values()):
        for group in ("dependencies", "build-dependencies", "dev-dependencies"):
            for name, dependency in section.get(group, {}).items():
                if isinstance(dependency, dict) and _SOURCE_KEYS.intersection(
                    dependency
                ):
                    raise ValueError(f"{group}.{name} is not a crates.io dependency")


def inspect_crate(archive_path: Path, version: str) -> None:
    """Require a self-contained crates.io manifest and distribution files."""
    inspect_crate_contents(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        files = _distribution_files(archive, f"calc-flow-{version}")
        required = {"Cargo.toml", "Cargo.lock", "LICENSE", "README.md", "src/lib.rs"}
        if missing := required.difference(files):
            raise ValueError(f"missing distribution files: {sorted(missing)}")
        if not all(
            any(name.startswith(f"{folder}/") for name in files)
            for folder in ("examples", "benches")
        ):
            raise ValueError("crate must retain examples and benches")
        if any(
            name.startswith("tests/")
            or PurePosixPath(name).name in {"AGENTS.md", "CLAUDE.md"}
            for name in files
        ):
            raise ValueError("crate includes repository-only files")

        def read(name: str) -> str:
            stream = archive.extractfile(files[name])
            if stream is None:
                raise ValueError(f"missing file content: {name}")
            return stream.read().decode("utf-8")

        _validate_manifest(read("Cargo.toml"), version)
        if not read("README.md").strip():
            raise ValueError("crate README is empty")


def run_consumer(archive_path: Path, version: str) -> None:
    """Execute the packaged README quickstart in a fresh external workspace."""
    directory = Path(
        tempfile.mkdtemp(prefix="rust-release-consumer-", dir=ROOT / "target")
    )
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(directory, filter="data")
    source = directory / "src"
    source.mkdir(exist_ok=True)
    readme = (directory / f"calc-flow-{version}/README.md").read_text(encoding="utf-8")
    quickstart = readme.split("```rust\n")
    if len(quickstart) != 2 or "\n```" not in quickstart[1]:
        raise ValueError("expected one runnable Rust quickstart in the README")
    (source / "main.rs").write_text(
        quickstart[1].split("\n```", 1)[0] + "\n", encoding="utf-8"
    )
    datafusion = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))[
        "workspace"
    ]["dependencies"]["datafusion"]["version"]
    (directory / "Cargo.toml").write_text(
        '[package]\nname = "calc-flow-release-consumer"\nversion = "0.0.0"\n'
        'edition = "2024"\npublish = false\n\n[workspace]\n\n[dependencies]\n'
        f'calc-flow = {{ path = "calc-flow-{version}", version = "={version}" }}\n'
        f'datafusion = {{ version = "{datafusion}", default-features = false }}\n'
        'tokio = { version = "1", features = ["macros", "rt-multi-thread"] }\n',
        encoding="utf-8",
    )
    subprocess.run(  # nosec B603 -- fixed cargo command on inspected local sources
        ["cargo", "run", "--manifest-path", str(directory / "Cargo.toml")],
        cwd=directory,
        env={**os.environ, "CARGO_TARGET_DIR": str(ROOT / "target")},
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--version-only", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--consumer", action="store_true")
    options = parser.parse_args()
    version = project_configuration().version
    if options.tag:
        validate_tag(options.tag, version)
    if options.version_only:
        print(version)
        return
    archive_path = options.archive or ROOT / f"target/package/calc-flow-{version}.crate"
    inspect_crate(archive_path, version)
    if options.consumer:
        run_consumer(archive_path, version)
    print(f"verified {archive_path.name}")


if __name__ == "__main__":
    main()
