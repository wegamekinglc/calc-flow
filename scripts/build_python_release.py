#!/usr/bin/env python3
"""Build and inspect the local-platform Calc Flow Python release artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

if __package__:
    from scripts.inspect_wheel import (
        inspect_sdist,
        inspect_studio_wheel,
        inspect_wheel,
    )
else:
    from inspect_wheel import inspect_sdist, inspect_studio_wheel, inspect_wheel


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class BuildStep:
    name: str
    command: tuple[str, ...]
    cwd: Path


def resolve_dist_dir(dist_dir: Path, root: Path = ROOT) -> Path:
    root = root.resolve()
    target = (root / "target").resolve()
    resolved = (
        (root / dist_dir).resolve()
        if not dist_dir.is_absolute()
        else dist_dir.resolve()
    )
    if resolved == target or not resolved.is_relative_to(target):
        raise ValueError(
            "release output must be a child directory of the repository "
            f"target tree: {target}"
        )
    return resolved


def prepare_dist_dir(
    dist_dir: Path,
    clean: bool,
    root: Path = ROOT,
) -> Path:
    resolved = resolve_dist_dir(dist_dir, root)
    if clean and resolved.exists():
        rmtree(resolved)
    if resolved.exists() and next(resolved.iterdir(), None) is not None:
        raise ValueError(
            f"release output directory is not empty: {resolved}; rerun with --clean"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_commands(root: Path, dist_dir: Path) -> tuple[BuildStep, ...]:
    root = root.resolve()
    dist_dir = dist_dir.resolve()
    web_ui = root / "web-ui"
    npm = "npm.cmd" if os.name == "nt" else "npm"
    return (
        BuildStep(
            "core wheel",
            (
                "uv",
                "run",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--out",
                str(dist_dir),
            ),
            root,
        ),
        BuildStep(
            "core source distribution",
            (
                "uv",
                "run",
                "maturin",
                "sdist",
                "--out",
                str(dist_dir),
            ),
            root,
        ),
        BuildStep("Studio frontend dependencies", (npm, "ci"), web_ui),
        BuildStep("Studio frontend", (npm, "run", "build"), web_ui),
        BuildStep(
            "Studio wheel",
            (
                "uv",
                "build",
                "--project",
                "web-ui/backend",
                "--wheel",
                "--out-dir",
                str(dist_dir),
            ),
            root,
        ),
    )


def _single_artifact(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(
            f"expected one {description}, found {[path.name for path in paths]}"
        )
    return paths[0]


def build_python_release(
    dist_dir: Path,
    *,
    clean: bool = False,
    root: Path = ROOT,
) -> tuple[Path, Path, Path]:
    output = prepare_dist_dir(dist_dir, clean, root)
    for step in build_commands(root, output):
        print(f"Building {step.name}...", flush=True)
        subprocess.run(step.command, cwd=step.cwd, check=True)

    core_wheel = _single_artifact(
        sorted(
            path
            for path in output.glob("calc_flow-*.whl")
            if not path.name.startswith("calc_flow_studio-")
        ),
        "core wheel",
    )
    sdist = _single_artifact(
        sorted(output.glob("calc_flow-*.tar.gz")),
        "core source distribution",
    )
    studio_wheel = _single_artifact(
        sorted(output.glob("calc_flow_studio-*.whl")),
        "Studio wheel",
    )
    inspect_wheel(core_wheel)
    inspect_sdist(sdist)
    inspect_studio_wheel(studio_wheel)
    return core_wheel, sdist, studio_wheel


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the local core wheel, core sdist, and Studio wheel under target/"
        )
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("target/python-release"),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove the validated output directory before building",
    )
    args = parser.parse_args()
    try:
        artifacts = build_python_release(args.dist_dir, clean=args.clean)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Python release build failed: {error}", file=sys.stderr)
        return 1
    print("Verified local Python release artifacts:")
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
