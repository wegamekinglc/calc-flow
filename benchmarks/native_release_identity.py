"""Verify the loaded package against its exact release wheel."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_identity(build_path: Path) -> dict:
    import calc_flow
    from benchmarks.native_memory_profile import prove_release
    from calc_flow import _native

    build = json.loads(build_path.read_text())
    native = Path(_native.__file__).resolve()
    proof = prove_release(build, native)
    wheel = Path(build["wheel"])
    if file_hash(wheel) != build["wheel_sha256"]:
        raise ValueError("release wheel does not match its build record")
    package = Path(calc_flow.__file__).resolve().parent
    if native.parent != package:
        raise ValueError("Python and native modules came from different packages")
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("calc_flow/") or member.endswith("/"):
                continue
            with archive.open(member) as content:
                digest = hashlib.file_digest(content, "sha256").hexdigest()
            if file_hash(package.parent / member) != digest:
                raise ValueError(
                    f"loaded package differs from the release wheel: {member}"
                )
    return {
        **proof,
        "python_executable": sys.executable,
        "python_package": str(package),
        "python_version": platform.python_version(),
        "instrument_sha256": file_hash(Path(__file__)),
    }
