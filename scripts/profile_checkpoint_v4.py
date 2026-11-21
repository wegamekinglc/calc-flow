"""P3.7 checkpoint-on/off evidence drivers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts.profile_warm_stream import (
        _native_binary_identity,
        _settle_native_process,
        _verify_native_binaries,
        paired_summary,
    )
    from scripts.toolkit import sha256_file
except ImportError:  # direct execution puts only scripts/ on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.profile_warm_stream import (
        _native_binary_identity,
        _settle_native_process,
        _verify_native_binaries,
        paired_summary,
    )
    from scripts.toolkit import sha256_file

DIRECT_TEST = (
    "runtime::streaming::operator_task::performance_evidence::"
    "checkpoint_direct_evidence"
)
DIRECT_CONTRACT = "calc-flow.checkpoint-v4-direct/1"
DIRECT_CASES = ("1x1-dual", "6400x64-dual")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def paired_statistics(rounds: list[dict[str, Any]], gate: float) -> dict[str, Any]:
    per_round = []
    all_on = []
    all_off = []
    for index, evidence in enumerate(rounds, 1):
        pairs = evidence["pairs"]
        on = [sample["on_median_seconds"] for sample in pairs]
        off = [sample["off_median_seconds"] for sample in pairs]
        all_on.extend(on)
        all_off.extend(off)
        summary = paired_summary(on, off, rows=len(pairs))
        per_round.append(
            {
                "round": index,
                "on_off_median": summary["paired_speedup_median"] - 1,
                "on_off_ci95": [
                    boundary - 1 for boundary in summary["paired_speedup_ci95"]
                ],
                "summary": summary,
            }
        )
    pooled = paired_summary(all_on, all_off, rows=len(all_on))
    pooled_median = pooled["paired_speedup_median"] - 1
    intervals = [item["on_off_ci95"] for item in per_round]
    if (
        any(lower > gate for lower, _ in intervals)
        or pooled_median > gate
        and all(lower > gate for lower, _ in intervals)
    ):
        verdict = "failed"
    elif any(lower <= gate <= upper for lower, upper in intervals):
        verdict = "inconclusive"
    elif pooled_median > gate:
        verdict = "failed"
    else:
        verdict = "passed"
    return {
        "gate": gate,
        "metric": "median(on_median_30 / off_median_30 - 1)",
        "per_round": per_round,
        "pooled": {
            "on_off_median": pooled_median,
            "on_off_ci95": [boundary - 1 for boundary in pooled["paired_speedup_ci95"]],
            "summary": pooled,
        },
        "verdict": verdict,
    }


def validate_direct_evidence(
    evidence: dict[str, Any], args: argparse.Namespace
) -> None:
    if evidence.get("contract") != DIRECT_CONTRACT:
        raise ValueError("unexpected direct evidence contract")
    if evidence.get("case") != args.case:
        raise ValueError("direct evidence case differs from its request")
    if evidence.get("history_rows") != args.history_rows:
        raise ValueError("direct evidence history differs from its request")
    if evidence.get("recorder") != args.recorder:
        raise ValueError("direct evidence recorder differs from its request")
    pairs = evidence.get("pairs", [])
    if len(pairs) != args.pairs:
        raise ValueError("direct evidence pair count differs from its request")
    for pair in pairs:
        if pair.get("dispatches_per_arm") != args.dispatches:
            raise ValueError("direct evidence dispatch count differs from its request")
        for name in ("on_median_seconds", "off_median_seconds"):
            value = pair.get(name)
            if not isinstance(value, float) or value <= 0:
                raise ValueError(f"direct evidence contains an invalid {name}")
        if pair.get("output_digest") is None:
            raise ValueError("direct evidence lacks an output digest")


async def run_direct_round(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite direct evidence: {output}")
    environment = dict(
        os.environ,
        CALC_FLOW_CHECKPOINT_V4_DIRECT_OUTPUT=str(output),
        CALC_FLOW_CHECKPOINT_V4_DIRECT_CASE=args.case,
        CALC_FLOW_CHECKPOINT_V4_DIRECT_HISTORY=str(args.history_rows),
        CALC_FLOW_CHECKPOINT_V4_DIRECT_RECORDER=args.recorder,
        CALC_FLOW_CHECKPOINT_V4_DIRECT_PAIRS=str(args.pairs),
        CALC_FLOW_CHECKPOINT_V4_DIRECT_DISPATCHES=str(args.dispatches),
    )
    with output.with_suffix(".log").open("wb") as log:
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                str(args.binary),
                "--exact",
                DIRECT_TEST,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
                env=environment,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
        )
        process = None
        try:
            process = await asyncio.shield(creation)
            code = await process.wait()
        except BaseException as error:
            await _settle_native_process(creation, process, error)
            raise
    if code:
        raise RuntimeError(
            f"direct evidence binary exited {code}; see {output.with_suffix('.log')}"
        )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    validate_direct_evidence(evidence, args)
    return evidence


def direct_markdown(report: dict[str, Any]) -> str:
    statistics = report["statistics"]
    lines = [
        f"# Checkpoint v4 direct {report['configuration']['case']}",
        "",
        f"Verdict: **{statistics['verdict']}**; "
        f"gate: on/off - 1 <= {statistics['gate']:.3f}.",
        "",
        "| Round | On/off median | 95% interval |",
        "|-------|--------------:|--------------|",
    ]
    for item in statistics["per_round"]:
        low, high = item["on_off_ci95"]
        lines.append(
            f"| {item['round']} | {item['on_off_median']:.4%} | "
            f"[{low:.4%}, {high:.4%}] |"
        )
    pooled = statistics["pooled"]
    lines.extend(
        [
            f"| Pooled | {pooled['on_off_median']:.4%} | "
            f"[{pooled['on_off_ci95'][0]:.4%}, {pooled['on_off_ci95'][1]:.4%}] |",
            "",
            f"Wall time: {report['wall_seconds']:.3f} seconds within budget: "
            f"{report['within_budget']}. Binary: `{report['binary']['sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


async def direct(args: argparse.Namespace) -> None:
    started = time.monotonic()
    binary = _native_binary_identity(args.binary, "checkpoint-v4")
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"direct work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite direct report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rounds = []
    raw = []
    error = None
    try:
        for number in range(1, args.rounds + 1):
            path = work / f"round-{number}.json"
            evidence = await run_direct_round(args, path)
            rounds.append(evidence)
            raw.append({"path": str(path), "sha256": sha256_file(path)})
    except BaseException as exc:
        error = repr(exc)
    _verify_native_binaries([binary])
    elapsed = time.monotonic() - started
    configuration = {
        "case": args.case,
        "history_rows": args.history_rows,
        "recorder": args.recorder,
        "rounds": args.rounds,
        "pairs": args.pairs,
        "dispatches": args.dispatches,
        "budget_seconds": args.budget_seconds,
    }
    statistics = paired_statistics(rounds, 0.03) if rounds else None
    report = {
        "schema": "calc-flow.checkpoint-v4-direct-report.v1",
        "configuration": configuration,
        "binary": binary,
        "raw_rounds": raw,
        "statistics": statistics,
        "wall_seconds": elapsed,
        "within_budget": elapsed <= args.budget_seconds,
        "error": error,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if statistics is not None:
        output.with_suffix(".md").write_text(direct_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if error is not None:
        raise RuntimeError(error)
    if elapsed > args.budget_seconds:
        raise RuntimeError("direct acquisition exceeded its budget")
    if statistics["verdict"] != "passed":
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    direct_parser = commands.add_parser("direct")
    direct_parser.add_argument("--binary", type=Path, required=True)
    direct_parser.add_argument("--case", choices=DIRECT_CASES, required=True)
    direct_parser.add_argument("--history-rows", type=positive_int, required=True)
    direct_parser.add_argument("--recorder", choices=("absent",), default="absent")
    direct_parser.add_argument("--rounds", type=positive_int, default=2)
    direct_parser.add_argument("--pairs", type=positive_int, default=10)
    direct_parser.add_argument("--dispatches", type=positive_int, default=30)
    direct_parser.add_argument("--budget-seconds", type=positive_float, required=True)
    direct_parser.add_argument("--work-dir", type=Path, required=True)
    direct_parser.add_argument("--output", type=Path, required=True)
    direct_parser.set_defaults(handler=direct)
    return result


def main() -> None:
    args = parser().parse_args()
    args.binary = args.binary.resolve(strict=True)
    asyncio.run(args.handler(args))


if __name__ == "__main__":
    main()
