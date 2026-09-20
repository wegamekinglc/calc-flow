"""Portable release-wheel provenance, independent of runner filesystem paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.profile_warm_stream import build, source_identity
from scripts.toolkit import FULL_SHA, sha256_file, wheel_native_sha256


async def build_release(source: Path, output: Path) -> None:
    if (await source_identity(source))["git_clean"] is not True:
        raise ValueError("benchmark release builds require a clean source checkout")
    await build(
        argparse.Namespace(source=source, output=output, target=output / "cargo")
    )
    manifest = json.loads((output / "build.json").read_text(encoding="utf-8"))
    manifest = {
        **manifest,
        "contract": "benchmark-release-v1",
        "wheel": Path(manifest["wheel"]).name,
    }
    (output / "release.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_release(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != "benchmark-release-v1"
        or manifest.get("build_profile") != "release"
        or manifest.get("git_clean") is not True
        or not FULL_SHA.fullmatch(manifest.get("git_sha", ""))
    ):
        raise ValueError("expected a clean, exact-SHA benchmark release manifest")
    name = manifest["wheel"]
    if Path(name).name != name:
        raise ValueError("wheel must be a basename inside the manifest directory")
    wheel = path.parent / name
    if (
        sha256_file(wheel) != manifest["wheel_sha256"]
        or wheel_native_sha256(wheel) != manifest["native_sha256"]
    ):
        raise ValueError("release wheel/native hash differs from its manifest")
    return {**manifest, "wheel_path": str(wheel.resolve())}
