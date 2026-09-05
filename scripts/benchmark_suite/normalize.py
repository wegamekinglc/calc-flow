"""Adapters for existing machine-readable benchmark formats; no log scraping."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _invalid_constant(value: str):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def pytest_rows(path: Path) -> dict:
    report = read_json(path)
    rows = {}
    for case in report["benchmarks"]:
        name = case["fullname"]
        if name in rows:
            raise ValueError(f"duplicate pytest benchmark: {name}")
        stats, extra = case["stats"], case.get("extra_info", {})
        rows[name] = {
            "samples": stats.get("data") or [stats["median"]],
            "rows": extra.get("input_rows", extra.get("table_rows")),
            "scope": extra.get("scope", "pytest-native-boundary"),
            "metadata": extra,
        }
    return rows


def criterion_rows(root: Path) -> dict:
    rows = {}
    for path in sorted(root.glob("**/new/benchmark.json")):
        descriptor = read_json(path)
        sample = read_json(path.with_name("sample.json"))
        iterations, times = sample["iters"], sample["times"]
        if len(times) != len(iterations) or not times:
            raise ValueError(f"invalid Criterion sample: {path}")
        name = descriptor["full_id"]
        if name in rows:
            raise ValueError(f"duplicate Criterion benchmark: {name}")
        rows[name] = {
            "samples": [
                elapsed / count / 1e9
                for elapsed, count in zip(times, iterations, strict=True)
            ],
            "rows": None,
            "scope": "criterion-native-boundary",
            "metadata": descriptor,
        }
    if not rows:
        raise ValueError("Criterion produced no benchmark samples")
    return rows


def vitest_rows(path: Path) -> dict:
    report = read_json(path)
    rows = {}
    for file in report["files"]:
        for group in file["groups"]:
            for case in group["benchmarks"]:
                name = case["name"]
                if name in rows:
                    raise ValueError(f"duplicate Vitest benchmark: {name}")
                if not case["samples"]:
                    raise ValueError(
                        "Vitest raw samples are required; enable includeSamples"
                    )
                rows[name] = {
                    "samples": [value / 1000 for value in case["samples"]],
                    "rows": None,
                    "scope": "vitest-native-boundary",
                    "metadata": {"group": group["fullName"]},
                }
    if not rows:
        raise ValueError("Vitest produced no benchmark samples")
    return rows
