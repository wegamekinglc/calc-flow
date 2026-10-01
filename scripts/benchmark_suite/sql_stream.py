"""Validate cumulative SQL snapshots against an independent prefix oracle."""

from __future__ import annotations

import hashlib
import math
import struct
from functools import lru_cache
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json

ROWS = 100_000
QUERY = (
    "SELECT key, SUM(value) AS total, COUNT(value) AS count, "
    "MIN(value) AS minimum, MAX(value) AS maximum FROM events GROUP BY key"
)
CASES = {
    "fixed_groups_1_batch": (1, 0, ROWS, False),
    "fixed_groups_10_batches": (10, 0, ROWS, False),
    "fixed_groups_100_batches": (100, 0, ROWS, False),
    "fixed_groups_100_batches_checkpoint_10": (100, 10, ROWS, False),
    "fixed_groups_100_batches_checkpoint_1": (100, 1, ROWS, False),
    "fixed_groups_1000_batches": (1000, 0, ROWS, False),
    "growing_groups_10k_100_batches": (100, 0, 10_000, True),
}


def _nullable(value: int | None) -> bytes:
    return struct.pack("<Bq", value is not None, value or 0)


@lru_cache(maxsize=5)
def expected_snapshot_digest(
    batches: int, rows: int = ROWS, unique_keys: bool = False
) -> str:
    """Hash each independently accumulated, sorted snapshot of the fixed input."""

    if (batches, rows, unique_keys) not in {
        (batch, count, unique) for batch, _, count, unique in CASES.values()
    }:
        raise ValueError("invalid SQL stream batch count")
    groups: dict[int | None, tuple[int, int, int | None, int | None]] = {}
    combined = hashlib.sha256()
    chunk = rows // batches
    for index in range(rows):
        key = (index if unique_keys else index % 64) if index % 101 else None
        value = index % 257 - 128 if index % 13 and key != 63 else None
        count, total, minimum, maximum = groups.get(key, (0, 0, None, None))
        if value is not None:
            count += 1
            total += value
            minimum = value if minimum is None else min(minimum, value)
            maximum = value if maximum is None else max(maximum, value)
        groups[key] = (count, total, minimum, maximum)
        if (index + 1) % chunk == 0:
            snapshot = hashlib.sha256()
            for group in sorted(groups, key=lambda item: (item is not None, item or 0)):
                count, total, minimum, maximum = groups[group]
                snapshot.update(_nullable(group))
                snapshot.update(struct.pack("<q", count))
                for scalar in (total if count else None, minimum, maximum):
                    snapshot.update(_nullable(scalar))
            combined.update(snapshot.digest())
    return combined.hexdigest()


def expected_output_rows(
    batches: int, rows: int = ROWS, unique_keys: bool = False
) -> int:
    if not unique_keys:
        return 65 * batches
    chunk = rows // batches
    return sum(end - (end + 100) // 101 + 1 for end in range(chunk, rows + 1, chunk))


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: object, expected: int | None = None) -> bool:
    return type(value) is int and value >= 0 and (expected is None or value == expected)


def _validate_sample(
    sample: object,
    batches: int,
    checkpoint: int,
    rows: int,
    unique_keys: bool,
    oracle: bool,
) -> None:
    if not isinstance(sample, dict):
        raise ValueError("invalid SQL stream observation")
    for field in ("seconds", "process_seconds", "prepare_seconds", "capture_seconds"):
        value = sample.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid SQL stream timing")
    if (
        sample["seconds"] <= 0
        or sample["process_seconds"] <= 0
        or sum(
            sample[field]
            for field in ("process_seconds", "prepare_seconds", "capture_seconds")
        )
        > sample["seconds"] * 1.01
    ):
        raise ValueError("invalid SQL stream phase durations")
    expected = {
        "input_rows": rows,
        "output_rows": expected_output_rows(batches, rows, unique_keys),
        "snapshot_count": batches,
        "checkpoint_count": batches // checkpoint if checkpoint else 0,
    }
    if any(not _integer(sample.get(field), value) for field, value in expected.items()):
        raise ValueError("invalid SQL stream row or checkpoint count")
    if (
        sample.get("validated_all_snapshots") is not True
        or sample.get("validated_recovery") is not oracle
        or sample.get("all_snapshots_sha256")
        != expected_snapshot_digest(batches, rows, unique_keys)
    ):
        raise ValueError("invalid SQL stream snapshot oracle")
    if (
        not _digest(sample.get("final_checkpoint_sha256"))
        or not _integer(sample.get("final_checkpoint_bytes"))
        or sample["final_checkpoint_bytes"] == 0
        or not _integer(sample.get("checkpoint_bytes"))
    ):
        raise ValueError("invalid SQL stream checkpoint evidence")
    if checkpoint:
        if sample["checkpoint_bytes"] < sample["final_checkpoint_bytes"]:
            raise ValueError("incomplete SQL stream checkpoint bytes")
    elif any(
        sample[field] != 0
        for field in ("checkpoint_bytes", "prepare_seconds", "capture_seconds")
    ):
        raise ValueError("unexpected SQL stream checkpoint work")


def sql_stream_rows(path: Path, *, minimum_samples: int = 20) -> dict:
    """Retain complete observations after validating every cumulative snapshot."""

    report = read_json(path)
    if (
        report.get("schema") != "calc-flow.sql-stream-aggregate.v1"
        or report.get("scope") != "warm-native-operator-cumulative-snapshots"
    ):
        raise ValueError("invalid SQL stream benchmark contract")
    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != len(CASES)
        or any(
            not isinstance(case, dict) or not isinstance(case.get("name"), str)
            for case in cases
        )
        or {case.get("name") for case in cases} != set(CASES)
    ):
        raise ValueError("incomplete SQL stream benchmark inventory")
    rows = {}
    for case in cases:
        batches, checkpoint, count, unique = CASES[case["name"]]
        maximum_groups = count - (count + 100) // 101 + 1 if unique else 65
        if (
            not _integer(case.get("rows"), count)
            or not _integer(case.get("batches"), batches)
            or not _integer(case.get("checkpoint_every"), checkpoint)
            or not _integer(case.get("maximum_groups"), maximum_groups)
            or case.get("unique_keys") is not unique
            or case.get("query") != QUERY
        ):
            raise ValueError("invalid SQL stream workload")
        samples = case.get("samples")
        if not isinstance(samples, list) or len(samples) < minimum_samples:
            raise ValueError("incomplete SQL stream observations")
        _validate_sample(case.get("oracle"), batches, checkpoint, count, unique, True)
        for sample in samples:
            _validate_sample(sample, batches, checkpoint, count, unique, False)
        if any(
            sample["final_checkpoint_sha256"]
            != case["oracle"]["final_checkpoint_sha256"]
            for sample in samples
        ):
            raise ValueError("unstable SQL stream checkpoint bytes")
        rows[f"stream_sql_aggregate/{case['name']}"] = {
            "samples": [sample["seconds"] for sample in samples],
            "rows": count,
            "scope": report["scope"],
            "metadata": {
                "oracle": case["oracle"],
                "observations": samples,
                "batches": batches,
                "checkpoint_every": checkpoint,
                "unique_keys": unique,
            },
        }
    return rows
