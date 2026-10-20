"""Reproduce DAL-394's added Polars cells with the maintained suite collector."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from scripts.benchmark_suite.catalog import baseline_case_ids, engine_cases
from scripts.benchmark_suite.measure import measure_case
from scripts.benchmark_suite.process import ROOT, install
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.report import ROUNDS, SAMPLES
from scripts.profile_warm_stream import source_identity
from scripts.toolkit import sha256_file


async def collect(args: argparse.Namespace) -> int:
    started = time.monotonic()
    release = load_release(args.release.resolve())
    source = await source_identity(ROOT)
    # These external references can reuse a sealed wheel with identical native
    # inputs. Keep its original build identity, separate from the new harness.
    if source["git_clean"] is not True or any(
        source[key] != release[key] for key in ("source_sha256", "cargo_lock_sha256")
    ):
        raise ValueError("expected clean harness source and matching wheel inputs")
    known = baseline_case_ids(args.baseline_source, {"family": "engines"})
    if known is None:
        raise ValueError(
            "baseline catalog must be readable to identify new Polars cells"
        )
    cases = [
        case
        for case in engine_cases()
        if case["backend"] in ("polars", "polars-1t") and case["id"] not in known
    ]
    if len(cases) != 84:
        raise ValueError("expected DAL-394's 84 added Polars cells")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "DAL-394 added Polars cells; external coverage, no speedup claim",
        "argv": sys.argv,
        "source": source,
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
                        "external",
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
