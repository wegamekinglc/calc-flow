"""Verify the delivered DAL-417 samples without executing either engine."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(row: dict) -> tuple:
    return tuple(sorted(row.items()))


def audit_sample(sample: dict, case: str, context: tuple[Path, Path, dict]) -> None:
    results, directory, identity = context
    arrivals = read(directory / "arrivals.json")
    sent = read(directory / "send_times.json")
    actual = [arrival["row"] for arrival in arrivals]
    expected = read(directory / "expected.json")
    require(sorted(actual, key=canonical) == sorted(expected, key=canonical), "rows")
    require(len(sent) == 65, "send frame count")
    require(all(a <= b for a, b in zip(sent, sent[1:])), "send ordering")
    require(sample["correctness"] == "exact_all_rows", "correctness marker")
    require(sample["output_before_eof"], "output before EOF")
    require(sample["elapsed_s"] == arrivals[-1]["at"] - sent[0], "elapsed time")
    require(sample["latency_ms"]["min"] >= 0, "output latency")
    if case == "window":
        require(sample["finalization_latency_ms"]["min"] >= 0, "finalization")
    command = read(directory / "command.json")
    require(command["argv"][:3] == ["taskset", "-c", "4-7"], "CPU affinity")
    if sample["engine"] == "calc_flow":
        first_line = (directory / "stderr.log").read_text().splitlines()[0]
        require(
            json.loads(first_line)["native_sha256"] == identity["native_sha256"],
            "native identity",
        )
        require(sample["exit_code"] == -2, "calc-flow exit")
    else:
        require(sample["exit_code"] == 0, "Arroyo exit")
    expected_rows = 16384 if case == "stateless" else 512
    require(sample["output_rows"] == expected_rows, "output row count")


def audit_pair(
    samples: list[dict],
    case: str,
    gap: float,
    pair: int,
    context: tuple[Path, dict],
) -> float:
    results, identity = context
    matched = [
        sample
        for sample in samples
        if (sample["case"], sample["gap_s"], sample["pair"]) == (case, gap, pair)
    ]
    order = ["calc_flow", "arroyo"]
    expected_order = order if pair % 2 == 0 else order[::-1]
    require([sample["engine"] for sample in matched] == expected_order, "pair order")
    for sample in matched:
        name = f"{case}-gap{gap:g}-pair{pair}-{sample['engine']}"
        audit_sample(sample, case, (results, results / name, identity))
    by_engine = {sample["engine"]: sample for sample in matched}
    return (
        by_engine["arroyo"]["rows_per_second"]
        / by_engine["calc_flow"]["rows_per_second"]
    )


def audit_case(
    samples: list[dict], case: str, gap: float, results: Path, identity: dict
) -> dict:
    ratios = [
        audit_pair(samples, case, gap, pair, (results, identity)) for pair in range(5)
    ]
    return {
        "case": case,
        "gap_s": gap,
        "paired_arroyo_over_calc_flow": ratios,
        "median_ratio": statistics.median(ratios),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
    }


def main() -> None:
    if not __debug__:
        raise RuntimeError("Run the evidence audit without Python -O or -OO")
    root = Path(sys.argv[1])
    results = root / "results"
    identity = read(results / "identity.json")
    samples = read(results / "samples.json")
    require(len(samples) == 40, "sample count")
    require(len({sample["input_sha256"] for sample in samples}) == 1, "input identity")
    harness = root / "measure-measured.py"
    if not harness.exists():
        harness = Path(__file__).with_name("measure.py")
    require(
        hashlib.sha256(harness.read_bytes()).hexdigest() == identity["harness_sha256"],
        "harness identity",
    )
    pairs = [
        audit_case(samples, case, gap, results, identity)
        for case in ("stateless", "window")
        for gap in (0, 0.005)
    ]
    print(json.dumps({"samples": 40, "audit": "passed", "pairs": pairs}, indent=2))


if __name__ == "__main__":
    main()
