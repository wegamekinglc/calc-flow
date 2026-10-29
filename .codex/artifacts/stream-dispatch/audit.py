"""Audit every retained Arrow result, phase clock and fixed paired statistic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
from summarize import summarize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".codex/artifacts/sql-arrow-comparison"))
from common import CASES, check_rows, digest, encode, fixture, read  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_clocks(worker, observer, count):
    start, end = worker["execute_ns"]
    require(worker["warmup_ns"][1] <= observer[0], "warmup inside observer")
    require(observer[0] <= start <= end <= observer[1], "execute outside observer")
    require(worker["concat_ns"][0] >= end, "concat before execution complete")
    require(worker["output_ipc_ns"][0] >= worker["concat_ns"][1], "IPC before concat")
    require(worker["output_ipc_ns"][0] <= observer[1], "consumer ended before IPC")
    require(len(worker["batch_completed_ns"]) == count, "missing batch completion")
    require(all(start <= t <= end for t in worker["batch_completed_ns"]), "batch clock")
    require(
        worker["batch_completed_ns"] == sorted(worker["batch_completed_ns"]),
        "clock order",
    )
    require(worker["output_batch_rows"] == [16384 // count] * count, "batch shape")
    require(worker["close_ns"][0] >= worker["output_ipc_ns"][1], "close before output")


def audit_sample(directory, sample, seals):
    case = next(case for case in CASES[:2] if case.name == sample["case"])
    worker = read(directory / "worker.json")
    require(worker == sample["worker"], "worker metadata copy differs")
    require(
        worker["native_sha256"] == seals[sample["side"]]["native_sha256"], "native seal"
    )
    for name in ("input", "output"):
        require(
            digest(directory / f"{name}.arrow") == sample[f"{name}_sha256"], "IPC hash"
        )
    require(
        (directory / "input.arrow").read_bytes() == encode(fixture(case)[1]),
        "input or batch shape differs",
    )
    table = pa.ipc.open_stream(directory / "output.arrow").read_all()
    check_rows(case, table)
    count = 32768 // case.batch_rows
    require(
        [b.num_rows for b in table.to_batches()] == worker["output_batch_rows"],
        "IPC batch shape differs from callback outputs",
    )
    audit_clocks(worker, sample["observer_ns"], count)
    require(read(directory / "exit.json")["code"] == 0, "worker cleanup failed")
    elapsed = (sample["observer_ns"][1] - sample["observer_ns"][0]) / 1e6
    require(elapsed == sample["consumable_ms"], "complete-consumption clock differs")
    require(sample["input_rows_s"] == 32768 * 1000 / elapsed, "throughput differs")
    return table.num_rows


def main(args):
    samples = read(args.directory / "samples.json")
    seals = read(args.seals)
    expected = [
        (case.name, round_number, pair, side)
        for case in CASES[:2]
        for round_number in range(2)
        for pair in range(10)
        for side in (("base", "head") if pair % 2 == 0 else ("head", "base"))
    ]
    actual = [(s["case"], s["round"], s["pair"], s["side"]) for s in samples]
    require(actual == expected, "predeclared adjacent AB/BA inventory differs")
    rows = 0
    for sample in samples:
        name = f"{sample['case']}-r{sample['round']}-p{sample['pair']}-{sample['side']}"
        rows += audit_sample(args.directory / name, sample, seals)
    require(summarize(samples) == read(args.summary), "paired summary differs")
    print(
        json.dumps({"samples": len(samples), "rows_checked": rows, "audit": "passed"})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seals", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    main(parser.parse_args())
