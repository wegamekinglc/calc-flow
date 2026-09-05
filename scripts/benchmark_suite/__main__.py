"""Run ``python -m scripts.benchmark_suite --help`` for the CI entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scripts.benchmark_suite.catalog import CONTRACT, get_shard, shards
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.report import render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    matrix = commands.add_parser("catalog")
    matrix.add_argument("--github-output", type=Path)
    refs = commands.add_parser("refs")
    refs.add_argument("--github-output", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--shard", required=True)
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--baseline-source", type=Path)
    run.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--github-summary", type=Path)
    summary.add_argument("--expected-base", required=True)
    summary.add_argument("--expected-head", required=True)
    return parser.parse_args()


async def run_shard(args: argparse.Namespace) -> int:
    from scripts.benchmark_suite.measure import measure_shard
    from scripts.benchmark_suite.release import load_release

    shard = get_shard(args.shard)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "shard": shard,
        "cases": [],
        "errors": [],
    }
    try:
        releases = {
            "baseline": load_release(args.baseline),
            "candidate": load_release(args.candidate),
        }
        report["releases"] = releases
        if shard["family"] in ("engines", "warm"):
            report = await measure_shard(shard, releases, output)
        else:
            from scripts.benchmark_suite.legacy import measure_legacy

            report = await measure_legacy(shard, releases, output, args.baseline_source)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    (output / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rendered = render_report(report["cases"], report["errors"])
    (output / "summary.md").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    bad_cases = [
        case
        for case in report["cases"]
        if case["status"] != "ok"
        or case.get("result", {}).get("verdict") == "regression"
    ]
    return int(bool(report["errors"] or bad_cases))


def main() -> int:
    args = parse_args()
    match args.operation:
        case "catalog":
            matrix = json.dumps({"include": shards()}, separators=(",", ":"))
            print(matrix)
            if args.github_output:
                with args.github_output.open("a", encoding="utf-8") as output:
                    output.write(f"matrix={matrix}\n")
        case "build":
            from scripts.benchmark_suite.release import build_release

            asyncio.run(build_release(args.source.resolve(), args.output.resolve()))
        case "refs":
            from scripts.benchmark_suite.refs import write_refs

            write_refs(args.github_output)
        case "worker":
            from scripts.benchmark_suite.worker import main as worker

            worker(args.root)
        case "run":
            return asyncio.run(run_shard(args))
        case "summarize":
            from scripts.benchmark_suite.aggregate import summarize

            return summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
