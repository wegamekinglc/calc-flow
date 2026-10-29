"""Snapshot maintained entry points and timer anchors without running a suite."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_suite.catalog import shard_cases, shards  # noqa: E402

TIMER = re.compile(
    r"perf_counter|Instant::now|elapsed\(|benchmark\(|iter_batched|\.iter\(|json.dumps|serde_json::|write_json|to_pyarrow|to_pylist|validated_all_rows"
)


def anchors(path):
    text = path.read_text()
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "anchors": [
            {"line": index, "text": line.strip()}
            for index, line in enumerate(text.splitlines(), 1)
            if TIMER.search(line)
        ],
    }


def python_boundaries(path):
    tree = ast.parse(path.read_text())
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [
            ast.unparse(call.func)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        ]
        if any("benchmark" in name or "perf_counter" in name for name in calls):
            rows.append(
                {
                    "function": node.name,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "calls": calls,
                }
            )
    return rows


def inventory():
    paths = sorted(
        {
            *ROOT.glob("benchmarks/*.py"),
            *ROOT.glob("scripts/benchmark_suite/*.py"),
            *ROOT.glob("crates/calc-flow/benches/*.rs"),
            *ROOT.glob(".github/workflows/*.yml"),
            ROOT / "scripts/release_performance.py",
            ROOT / "scripts/verify_perf_gates.py",
            ROOT / "scripts/verify_stream_lifecycle_evidence.py",
        }
    )
    targets = tomllib.loads((ROOT / "crates/calc-flow/Cargo.toml").read_text())["bench"]
    return {
        "kind": "static source inventory; no performance classification",
        "shards": shards(),
        "catalog_counts": {
            shard["id"]: len(shard_cases(shard))
            for shard in shards()
            if shard["family"] in ("engines", "warm")
        },
        "rust_targets": targets,
        "sources": {str(path.relative_to(ROOT)): anchors(path) for path in paths},
        "pytest_boundaries": {
            str(path.relative_to(ROOT)): python_boundaries(path)
            for path in ROOT.glob("benchmarks/test_*.py")
        },
        "backend_counts_32768": dict(
            Counter(
                case["backend"]
                for case in shard_cases({"family": "engines", "rows": 32768})
            )
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inventory(), indent=2) + "\n")
