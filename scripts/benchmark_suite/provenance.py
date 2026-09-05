"""Bind the common measuring harness and dependency lock to every shard."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def harness_sha256() -> str:
    paths = [
        *ROOT.glob("scripts/benchmark_suite/*.py"),
        *ROOT.glob("scripts/benchmark_suite/*.mjs"),
        *ROOT.glob("benchmarks/engine_*.py"),
        ROOT / "benchmarks/warm_stream.py",
        ROOT / "benchmarks/rolling_indicator_comparison.py",
        ROOT / "benchmarks/requirements.lock",
        ROOT / "scripts/profile_warm_stream.py",
        ROOT / "scripts/verify_sql_datafusion_performance.py",
        ROOT / "scripts/verify_stream_lifecycle_evidence.py",
        ROOT / "scripts/write_criterion_provenance.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
