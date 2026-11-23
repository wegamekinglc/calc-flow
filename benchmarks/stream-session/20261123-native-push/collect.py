"""Collect the two release workers on one Linux CPU with the registered oracle."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from fixture import generate

from scripts.benchmark_suite.process import child_environment, command
from scripts.benchmark_suite.release_pairs import SIDES, pair_order
from scripts.benchmark_suite.report import ROUNDS, SAMPLES, comparison
from scripts.sparse_changelog_ci import _resource
from scripts.toolkit import linear_percentile, sha256_file, write_json

FIXTURE_HASHES = {
    "project.json": "57e36971d02b987132f4b6ab8f8fae4c9aa0f74de2c53660b2f3ead67eb6a5b6",
    "inputs.arrow": "90da6c27244354bacbc53983ca2d1f71be2654ee3b7943779843648197ebe253",
    "expected.arrow": (
        "c86c9e5c2141776e9881d7467da417f7fe4f19ebdd594df2e5339172cb2fe738"
    ),
}
PROOFS = ("all_columns_validated", "input_unchanged", "finished_empty")


def observations(proof: dict) -> list[int]:
    if (proof["history"], proof["appends"]) != (4096, 64):
        raise ValueError("worker used different input dimensions")
    if any(proof[field] is not True for field in PROOFS):
        raise ValueError("worker failed to verify every required property")
    return checked_nanoseconds(proof["nanoseconds"])


def checked_nanoseconds(samples: list[int]) -> list[int]:
    if len(samples) != 64 or any(type(n) is not int or n <= 0 for n in samples):
        raise ValueError("expected 64 positive single-push timings in nanoseconds")
    return samples


def verify_files(directory: Path, seals: dict[str, str]) -> None:
    for name, expected in seals.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"SHA-256 mismatch for {name}")


def schedule():
    for round_index in range(ROUNDS):
        for pair in range(-1, SAMPLES):
            for side in pair_order(max(0, pair)):
                yield round_index, pair, side


async def workers(output: Path, paths: dict, cpu: int) -> list[dict]:
    records = []
    env = child_environment(tokio_threads=1, polars_threads=1)
    for round_index, pair, side in schedule():
        worker_log = output / f"round{round_index}-pair{pair}-{side}.log"
        await command(
            ["taskset", "-c", str(cpu), str(paths[side]), str(output)],
            cwd=output,
            log=worker_log,
            env=env,
        )
        proof = json.loads(worker_log.read_text())
        observations(proof)
        records.append({"round": round_index, "pair": pair, "side": side, **proof})
        with (output / "workers.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(records[-1], separators=(",", ":")) + "\n")
    return records


def require_schedule(records: list[dict]) -> None:
    actual = [(row["round"], row["pair"], row["side"]) for row in records]
    if actual != list(schedule()):
        raise ValueError("incomplete or reordered process pairs or warmups")


def summarize(records: list[dict]) -> dict:
    require_schedule(records)
    blocks = {side: [[] for _ in range(ROUNDS)] for side in SIDES}
    pooled = {side: [] for side in SIDES}
    for record in records:
        if record["pair"] >= 0:
            side = record["side"]
            samples = observations(record)
            pooled[side].extend(samples)
            blocks[side][record["round"]].append(statistics.median(samples) / 1e9)
    stats = comparison(
        {
            "id": "native-session-push",
            "comparison": "interleaved",
            "correctness": True,
            **blocks,
        }
    )
    return {
        "block_median_seconds": stats,
        "pooled_nanoseconds": {
            side: {
                "count": len(values),
                "p50": statistics.median(values),
                "p95": linear_percentile(values, 0.95),
            }
            for side, values in pooled.items()
        },
    }


async def acquire(output: Path, paths: dict, seals: dict) -> None:
    generate(output)
    verify_files(output, FIXTURE_HASHES)
    for side in SIDES:
        verify_files(paths[side].parent, {paths[side].name: seals[side]})
    resources = output / "resources.jsonl"
    cpu = await _resource(resources, None)
    async with asyncio.TaskGroup() as group:
        pending = group.create_task(workers(output, paths, cpu))
        while not pending.done():
            await _resource(resources, cpu)
    write_json(output / "statistics.json", summarize(pending.result()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for side in SIDES:
        parser.add_argument(side, type=Path)
        parser.add_argument(f"{side}_sha256")
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {s: getattr(args, s).resolve() for s in SIDES}
    seals = {s: getattr(args, f"{s}_sha256") for s in SIDES}
    if len(set(seals.values())) != len(SIDES):
        raise ValueError("two distinct sealed executables are required")
    outcome = {
        "state": "stopped",
        "binary_sha256": seals,
        "kernel": platform.platform(),
        "python": sys.version,
    }
    try:
        asyncio.run(asyncio.wait_for(acquire(output, paths, seals), timeout=285))
        outcome["state"] = "completed"
    except BaseException as error:
        outcome["reason"] = repr(error)
        raise
    finally:
        outcome["controller_elapsed_seconds"] = time.monotonic() - started
        write_json(output / "outcome.json", outcome)


if __name__ == "__main__":
    main()
