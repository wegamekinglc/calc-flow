"""Verify the delivered DAL-417 samples without executing either engine."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    if not __debug__:
        raise RuntimeError("Run the evidence audit without Python -O or -OO")
    root = Path(sys.argv[1])
    results = root / "results"
    identity = read(results / "identity.json")
    samples = read(results / "samples.json")
    assert len(samples) == 40
    assert len({sample["input_sha256"] for sample in samples}) == 1
    harness = root / "measure-measured.py"
    if not harness.exists():
        harness = Path(__file__).with_name("measure.py")
    assert (
        hashlib.sha256(harness.read_bytes()).hexdigest() == identity["harness_sha256"]
    )
    pairs = []
    for case in ("stateless", "window"):
        for gap in (0, 0.005):
            ratios = []
            for pair in range(5):
                matched = [
                    sample
                    for sample in samples
                    if (sample["case"], sample["gap_s"], sample["pair"])
                    == (case, gap, pair)
                ]
                order = ["calc_flow", "arroyo"]
                assert [sample["engine"] for sample in matched] == (
                    order if pair % 2 == 0 else order[::-1]
                )
                for sample in matched:
                    name = f"{case}-gap{gap:g}-pair{pair}-{sample['engine']}"
                    directory = results / name
                    arrivals = read(directory / "arrivals.json")
                    sent = read(directory / "send_times.json")
                    actual = [arrival["row"] for arrival in arrivals]
                    expected = read(directory / "expected.json")
                    canonical = lambda row: tuple(sorted(row.items()))  # noqa: E731
                    assert sorted(actual, key=canonical) == sorted(
                        expected, key=canonical
                    )
                    assert len(sent) == 65
                    assert all(a <= b for a, b in zip(sent, sent[1:]))
                    assert sample["correctness"] == "exact_all_rows"
                    assert sample["output_before_eof"]
                    assert sample["elapsed_s"] == arrivals[-1]["at"] - sent[0]
                    assert sample["latency_ms"]["min"] >= 0
                    if case == "window":
                        assert sample["finalization_latency_ms"]["min"] >= 0
                    command = read(directory / "command.json")
                    assert command["argv"][:3] == ["taskset", "-c", "4-7"]
                    if sample["engine"] == "calc_flow":
                        first_line = (
                            (directory / "stderr.log").read_text().splitlines()[0]
                        )
                        assert (
                            json.loads(first_line)["native_sha256"]
                            == identity["native_sha256"]
                        )
                        assert sample["exit_code"] == -2
                    else:
                        assert sample["exit_code"] == 0
                    assert sample["output_rows"] == (
                        16384 if case == "stateless" else 512
                    )
                by_engine = {sample["engine"]: sample for sample in matched}
                ratios.append(
                    by_engine["arroyo"]["rows_per_second"]
                    / by_engine["calc_flow"]["rows_per_second"]
                )
            pairs.append(
                {
                    "case": case,
                    "gap_s": gap,
                    "paired_arroyo_over_calc_flow": ratios,
                    "median_ratio": statistics.median(ratios),
                    "min_ratio": min(ratios),
                    "max_ratio": max(ratios),
                }
            )
    print(json.dumps({"samples": 40, "audit": "passed", "pairs": pairs}, indent=2))


if __name__ == "__main__":
    main()
