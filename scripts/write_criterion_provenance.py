"""Write comparable machine, dependency, and workload Criterion provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess  # nosec B404
from pathlib import Path
from typing import Any


def _command(*arguments: str) -> str:
    # Every invocation below uses a fixed executable and fixed flags. The only
    # variable is the repository path passed as one argv item; shell parsing is
    # disabled, so it cannot become command syntax.
    return subprocess.run(  # nosec B603  # nosemgrep
        arguments,
        check=True,
        capture_output=True,
        shell=False,
        text=True,
    ).stdout.strip()


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _machine_identity() -> dict[str, object]:
    cpu_model = platform.processor() or platform.machine()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        pass
    else:
        cpu_model = next(
            (
                line.partition(":")[2].strip()
                for line in cpuinfo.splitlines()
                if line.startswith("model name")
            ),
            cpu_model,
        )
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None:
        raise RuntimeError("logical CPU count is unavailable")
    return {
        "operating_system": platform.system().casefold(),
        "architecture": platform.machine().casefold(),
        "cpu_model": " ".join(cpu_model.casefold().split()),
        "logical_cpu_count": logical_cpu_count,
        "runner_name": os.environ.get("RUNNER_NAME", ""),
        "runner_os": os.environ.get("RUNNER_OS", ""),
        "runner_arch": os.environ.get("RUNNER_ARCH", ""),
    }


def _resolve_sources(repository: Path, sources: list[Path]) -> list[tuple[str, Path]]:
    resolved = []
    for source in sources:
        candidate = (repository / source).resolve()
        if not candidate.is_relative_to(repository):
            raise ValueError(f"benchmark source escapes repository: {source}")
        if not candidate.is_file():
            raise ValueError(f"benchmark source is missing: {source}")
        resolved.append((source.as_posix(), candidate))
    return resolved


def _git_identity(repository: Path) -> str:
    git_sha = _command("git", "-C", str(repository), "rev-parse", "HEAD^{commit}")
    if len(git_sha) != 40:
        raise ValueError("git did not return a lowercase full commit SHA")
    if any(character not in "0123456789abcdef" for character in git_sha):
        raise ValueError("git did not return a lowercase full commit SHA")
    tree_status = _command(
        "git",
        "-C",
        str(repository),
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tree_status:
        raise ValueError("Criterion provenance requires a clean tracked worktree")
    return git_sha


def _dependency_identity(cargo_lock: Path) -> dict[str, str]:
    return {
        "cargo_lock_sha256": _file_hash(cargo_lock),
        "rustc": _command("rustc", "-Vv"),
        "cargo": _command("cargo", "-V"),
    }


def build_provenance(root: Path, sources: list[Path]) -> dict[str, Any]:
    """Build provenance bound to exact benchmark source bytes."""
    repository = root.resolve()
    resolved_sources = _resolve_sources(repository, sources)
    cargo_lock = repository / "Cargo.lock"
    if not cargo_lock.is_file():
        raise ValueError("Cargo.lock is missing")
    git_sha = _git_identity(repository)
    dependency_identity = _dependency_identity(cargo_lock)
    workload_identity = {
        path: _file_hash(source) for path, source in sorted(resolved_sources)
    }
    machine_identity = _machine_identity()
    return {
        "schema": "calc-flow.criterion-provenance.v1",
        "git_sha": git_sha,
        "tracked_worktree_clean": True,
        "benchmarks": [Path(path).stem for path, _source in sorted(resolved_sources)],
        "machine_identity": machine_identity,
        "dependency_identity": dependency_identity,
        "workload_identity": workload_identity,
        "machine_fingerprint": _fingerprint(machine_identity),
        "dependency_fingerprint": _fingerprint(dependency_identity),
        "workload_fingerprint": _fingerprint(workload_identity),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", nargs="+", type=Path)
    options = parser.parse_args()
    document = build_provenance(options.root, options.sources)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
