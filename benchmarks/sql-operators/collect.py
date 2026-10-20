"""Reproduce DAL-389's added SQL cells with the maintained suite collector."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from scripts.benchmark_suite.catalog import (
    baseline_case_ids,
    comparison_kind,
    engine_cases,
)
from scripts.benchmark_suite.legacy import validate_sources
from scripts.benchmark_suite.measure import measure_case
from scripts.benchmark_suite.process import ROOT, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.report import ROUNDS, SAMPLES
from scripts.toolkit import sha256_file


async def collect(args: argparse.Namespace) -> int:
    started = time.monotonic()
    release = load_release(args.release.resolve())
    await validate_sources({"candidate": ROOT}, {"candidate": release})
    known = baseline_case_ids(args.baseline_source, {"family": "engines"})
    if known is None:
        raise ValueError("baseline catalog must be readable to identify new SQL cells")
    cases = [
        case
        for case in engine_cases()
        if case["backend"] in ("calc-flow-sql", "datafusion")
        and case["id"] not in known
    ]
    if len(cases) != 98:
        raise ValueError("expected DAL-389's 98 added SQL/DataFusion cells")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "DAL-389 added SQL cells; candidate coverage, no speedup claim",
        "argv": sys.argv,
        "release": release,
        "harness_sha256": harness_sha256(),
        "collector_sha256": sha256_file(Path(__file__)),
        "baseline_catalog_sha256": sha256_file(
            args.baseline_source / "scripts/benchmark_suite/catalog.py"
        ),
        "rounds": ROUNDS,
        "samples_per_round": SAMPLES,
        "budget_seconds": args.budget_seconds,
        "inventory": cases,
        "cases": [],
        "errors": [],
    }
    try:
        async with asyncio.timeout(args.budget_seconds):
            site = await install(release, output / "candidate")
            for index, case in enumerate(cases):
                print(f"{index + 1}/{len(cases)} {case['id']}", flush=True)
                report["cases"].append(
                    await measure_case(
                        case,
                        comparison_kind(case, known),
                        {"candidate": (site, ROOT)},
                        {"candidate": release},
                        output / f"case-{index}",
                    )
                )
                report["elapsed_seconds"] = time.monotonic() - started
                (output / "results.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
    except TimeoutError:
        report["errors"].append(
            "collection budget exhausted; remaining cells unmeasured"
        )
    finally:
        measured = {case["id"] for case in report["cases"]}
        report["unmeasured"] = [
            case["id"] for case in cases if case["id"] not in measured
        ]
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return int(
        bool(report["errors"] or report["unmeasured"])
        or any(case["status"] != "ok" for case in report["cases"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, required=True)
    raise SystemExit(asyncio.run(collect(parser.parse_args())))
