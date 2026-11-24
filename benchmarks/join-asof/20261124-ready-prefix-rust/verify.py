"""Verify archived worker output and reproduce statistics without running a worker."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from scripts.benchmark_suite.asof import CASES, _valid_sample
from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.toolkit import linear_percentile

ROOT = Path(__file__).resolve().parent
SIDES = ("baseline", "candidate")


def require(valid: bool, message: str) -> None:
    if not valid:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def report(line: bytes, tag: str) -> dict:
    name, payload = line.split(b" ", 1)
    data = json.loads(payload)
    require(name.decode() == tag, "worker report tag mismatch")
    require(data["validated_all_rows"] is (tag == "ASOF_READY"), "row oracle flag")
    require(_valid_sample(data, CASES["balanced_512"], "latest-per-key"), "ASOF oracle")
    return data


def worker_samples() -> list[dict]:
    lines = iter((ROOT / "workers.log").read_bytes().splitlines(keepends=True))
    samples = []
    for group in (1, 2):
        for pair in range(11):
            order = SIDES if pair == 0 or pair % 2 else SIDES[::-1]
            for side in order:
                header, ready, sample = (next(lines) for _ in range(3))
                filename = f"r{group}-{pair}-{side}.log"
                digest = sha256(ready + sample)
                require(header == f"WORKER {filename} {digest}\n".encode(), filename)
                report(ready, "ASOF_READY")
                observation = report(sample, "ASOF_SAMPLE")
                samples.append(
                    {"round": group, "pair": pair, "side": side, "data": observation}
                )
    require(next(lines, None) is None, "unexpected extra worker output")
    return samples


def pooled(arms: dict[str, list[float]]) -> dict:
    return {
        side: {"p50": statistics.median(times), "p95": linear_percentile(times, 0.95)}
        for side, times in arms.items()
    }


def summarize(samples: list[dict]) -> dict:
    arms = {side: [] for side in SIDES}
    rounds = []
    for group in (1, 2):
        times = {
            side: [
                row["data"]["seconds"] * 1e6
                for row in samples
                if row["round"] == group and row["side"] == side and row["pair"] > 0
            ]
            for side in SIDES
        }
        rounds.append(paired_round(times["baseline"], times["candidate"]))
        for side in SIDES:
            arms[side].extend(times[side])
    return {
        "pooled_us": pooled(arms),
        "rounds": rounds,
        "verdict": _paired_verdict(rounds),
    }


def main() -> None:
    identity = json.loads((ROOT / "provenance.json").read_bytes())
    artifacts = {
        "workers.log": "worker_logset_sha256",
        "resources.jsonl": "resources_jsonl_sha256",
        "outcome.json": "outcome_json_sha256",
        "protocol.md": "registered_protocol_sha256",
        "measure.py": "driver_sha256",
        "sources.sha256": "registered_file_hashes_manifest_sha256",
    }
    for name, key in artifacts.items():
        require(sha256((ROOT / name).read_bytes()) == identity[key], name)
    samples = worker_samples()
    partial = json.dumps(samples, indent=2, sort_keys=True) + "\n"
    require(sha256(partial.encode()) == identity["partial_json_sha256"], "partial.json")
    result = summarize(samples)
    observed = json.loads((ROOT / "outcome.json").read_bytes())
    require(all(observed[key] == value for key, value in result.items()), "outcome")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
