"""Verify the loaded package against its exact release wheel."""

from __future__ import annotations

import json
import platform
import sys
import zipfile
from pathlib import Path

from scripts.benchmark_suite.release import load_release
from scripts.toolkit import sha256_file as file_hash
from scripts.toolkit import sha256_stream


def release_identity(build_path: Path) -> dict:
    import calc_flow
    from calc_flow import _native

    build = json.loads(build_path.read_text(encoding="utf-8"))
    native = Path(_native.__file__).resolve()
    if build.get("contract") == "benchmark-release-v1":
        build = load_release(build_path)
        wheel = Path(build["wheel_path"])
        native_sha256 = file_hash(native)
        if native_sha256 != build["native_sha256"]:
            raise ValueError(
                "installed native module does not match the benchmark release"
            )
        proof = {"native": str(native), "native_sha256": native_sha256, "build": build}
    else:
        from benchmarks.native_memory_profile import prove_release

        proof = prove_release(build, native)
        wheel = Path(build["wheel"])
        if file_hash(wheel) != build["wheel_sha256"]:
            raise ValueError("release wheel does not match its build record")
    package = Path(calc_flow.__file__).resolve().parent
    if native.parent != package:
        raise ValueError("Python and native modules came from different packages")
    verify_package_files(wheel, package.parent)
    return {
        **proof,
        "python_executable": sys.executable,
        "python_package": str(package),
        "python_version": platform.python_version(),
        "instrument_sha256": file_hash(Path(__file__)),
    }


def verify_package_files(wheel: Path, site: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("calc_flow/") or member.endswith("/"):
                continue
            with archive.open(member) as content:
                digest = sha256_stream(content)
            if file_hash(site / member) != digest:
                raise ValueError(
                    f"loaded package differs from the release wheel: {member}"
                )
