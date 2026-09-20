"""Shared hashing, subprocess, JSON, and percentile helpers for scripts.

Flat scripts import this module as ``scripts.toolkit``; scripts that also
support direct ``python scripts/<name>.py`` execution fall back to a plain
``toolkit`` import because direct execution puts only ``scripts/`` on
``sys.path``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess  # nosec B404
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FULL_SHA = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def wheel_native_sha256(path: Path) -> str:
    with zipfile.ZipFile(path) as wheel:
        names = [
            name
            for name in wheel.namelist()
            if name.startswith("calc_flow/_native") and name.endswith((".so", ".pyd"))
        ]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one native module")
        with wheel.open(names[0]) as native:
            return hashlib.file_digest(native, "sha256").hexdigest()


def canonical_json(
    value: object, *, ensure_ascii: bool = False, allow_nan: bool = True
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )


def fingerprint_json(
    value: object, *, ensure_ascii: bool = False, allow_nan: bool = True
) -> str:
    return hashlib.sha256(
        canonical_json(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan).encode()
    ).hexdigest()


def command_output(arguments: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run one fixed-argv command without a shell and return stripped stdout."""
    completed = subprocess.run(  # nosec B603 B607  # nosemgrep
        list(arguments),
        check=True,
        capture_output=True,
        cwd=cwd,
        shell=False,
        text=True,
    )
    return completed.stdout.strip()


def git_output(root: Path, *arguments: str) -> str:
    # Callers pass fixed git subcommands with namespaced or validated refs.
    result = subprocess.run(  # nosec B603
        ("git", *arguments),
        cwd=root,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"git {arguments[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def require_executable(name: str) -> str:
    located = shutil.which(name)
    if located is None:
        raise RuntimeError(f"{name} executable is missing from PATH")
    path = Path(located)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"{name} executable must be an absolute regular file")
    return str(path)


def write_json(path: Path, payload: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile matching the Rust evidence harness."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cpu_affinity() -> list[int] | None:
    import psutil

    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        return None
    try:
        return process.cpu_affinity()
    except (psutil.Error, NotImplementedError):
        return None


def worker_environment() -> dict[str, Any]:
    import numpy as np
    import pyarrow as pa

    import calc_flow._native as native

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": _cpu_affinity(),
        "native_sha256": sha256_file(Path(native.__file__)),
        "tokio_worker_threads": os.environ.get("TOKIO_WORKER_THREADS"),
    }
