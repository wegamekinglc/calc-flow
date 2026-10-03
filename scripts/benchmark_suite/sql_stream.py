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
Workload = tuple[int, int, int, bool]
DecimalType = tuple[int, int, int]
DECIMAL_PRECISION = {32: (4, 9), 64: (12, 18), 128: (28, 38), 256: (60, 76)}


def _nullable(value: int | None) -> bytes:
    return struct.pack("<Bq", value is not None, value or 0)


def _input_row(index: int, unique_keys: bool) -> tuple[int | None, int | None]:
    key = (index if unique_keys else index % 64) if index % 101 else None
    value = index % 257 - 128 if index % 13 and key != 63 else None
    return key, value


def _accumulate(
    aggregate: tuple[int, int, int | None, int | None], value: int | None
) -> tuple[int, int, int | None, int | None]:
    count, total, minimum, maximum = aggregate
    if value is not None:
        count += 1
        total += value
        minimum = value if minimum is None else min(minimum, value)
        maximum = value if maximum is None else max(maximum, value)
    return count, total, minimum, maximum


def _typed_nullable(
    value: int | None, decimal: DecimalType | None, *, total: bool
) -> bytes:
    if decimal is None:
        return _nullable(value)
    bits, precision, scale = decimal
    if total:
        precision = min(precision + 10, DECIMAL_PRECISION[bits][1])
    return struct.pack("<HBbB", bits, precision, scale, value is not None) + (
        value or 0
    ).to_bytes(bits // 8, "little", signed=True)


def _snapshot_digest(
    groups: dict[int | None, tuple[int, int, int | None, int | None]],
    decimal: DecimalType | None = None,
) -> bytes:
    snapshot = hashlib.sha256()
    for group in sorted(groups, key=lambda item: (item is not None, item or 0)):
        count, total, minimum, maximum = groups[group]
        snapshot.update(_nullable(group))
        snapshot.update(struct.pack("<q", count))
        for index, scalar in enumerate((total if count else None, minimum, maximum)):
            snapshot.update(_typed_nullable(scalar, decimal, total=index == 0))
    return snapshot.digest()


@lru_cache(maxsize=45)
def expected_snapshot_digest(
    batches: int,
    rows: int = ROWS,
    unique_keys: bool = False,
    decimal: DecimalType | None = None,
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
        key, value = _input_row(index, unique_keys)
        groups[key] = _accumulate(groups.get(key, (0, 0, None, None)), value)
        if (index + 1) % chunk == 0:
            combined.update(_snapshot_digest(groups, decimal))
    return combined.hexdigest()


@lru_cache(maxsize=16)
def expected_recovery_digest(
    rows: int, unique_keys: bool, decimal: DecimalType | None
) -> str:
    groups: dict[int | None, tuple[int, int, int | None, int | None]] = {}
    recovery = hashlib.sha256()
    for index in range(rows + 1):
        key, value = _input_row(index, unique_keys)
        groups[key] = _accumulate(groups.get(key, (0, 0, None, None)), value)
        if index + 1 in (rows, rows + 1):
            recovery.update(_snapshot_digest(groups, decimal))
    return recovery.hexdigest()


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


def _valid_timing(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_timing(sample: dict) -> None:
    for field in ("seconds", "process_seconds", "prepare_seconds", "capture_seconds"):
        value = sample.get(field)
        if not _valid_timing(value):
            raise ValueError("invalid SQL stream timing")
    durations = sum(
        sample[field]
        for field in ("process_seconds", "prepare_seconds", "capture_seconds")
    )
    if (
        sample["seconds"] <= 0
        or sample["process_seconds"] <= 0
        or durations > sample["seconds"] * 1.01
    ):
        raise ValueError("invalid SQL stream phase durations")


def _validate_counts(sample: dict, workload: Workload) -> None:
    batches, checkpoint, rows, unique_keys = workload
    expected = {
        "input_rows": rows,
        "output_rows": expected_output_rows(batches, rows, unique_keys),
        "snapshot_count": batches,
        "checkpoint_count": batches // checkpoint if checkpoint else 0,
    }
    if any(not _integer(sample.get(field), value) for field, value in expected.items()):
        raise ValueError("invalid SQL stream row or checkpoint count")


def _validate_oracle(
    sample: dict,
    workload: Workload,
    decimal: DecimalType | None,
    *,
    oracle: bool,
    current: bool,
) -> None:
    batches, _, rows, unique_keys = workload
    if (
        sample.get("validated_all_snapshots") is not True
        or sample.get("validated_recovery") is not oracle
        or sample.get("all_snapshots_sha256")
        != expected_snapshot_digest(batches, rows, unique_keys, decimal)
    ):
        raise ValueError("invalid SQL stream snapshot oracle")
    if decimal is not None or current:
        expected = (
            expected_recovery_digest(rows, unique_keys, decimal) if oracle else None
        )
        if (
            "recovery_snapshots_sha256" not in sample
            or sample["recovery_snapshots_sha256"] != expected
        ):
            raise ValueError("invalid SQL stream recovery oracle")


def _validate_checkpoint_fields(sample: dict) -> None:
    if (
        not _digest(sample.get("final_checkpoint_sha256"))
        or not _integer(sample.get("final_checkpoint_bytes"))
        or sample["final_checkpoint_bytes"] == 0
        or not _integer(sample.get("checkpoint_bytes"))
    ):
        raise ValueError("invalid SQL stream checkpoint evidence")


def _expected_logical_bytes(workload: Workload, decimal: DecimalType | None) -> int:
    batches, _, rows, _ = workload
    chunk = rows // batches
    bitmap = (chunk + 7) // 8
    width = 8 if decimal is None else decimal[0] // 8
    validity = sum(
        bitmap
        * sum(
            (start + divisor - 1) // divisor * divisor < start + chunk
            for divisor in (101, 13)
        )
        for start in range(0, rows, chunk)
    )
    return rows * (8 + width) + validity


def _validate_current_checkpoint(
    sample: dict, workload: Workload, decimal: DecimalType | None
) -> int:
    state = sample.get("checkpoint_state")
    if not isinstance(state, dict) or type(state.get("layout")) is not int:
        raise ValueError("invalid SQL stream current checkpoint state")
    layout = state["layout"]
    if layout not in (3, 4):
        raise ValueError("invalid SQL stream current checkpoint layout")
    _, _, rows, unique = workload
    groups = rows - (rows + 100) // 101 + 1 if unique else 65
    segment = "group-state" if layout == 3 else "input-retained"
    expected = {
        "accounting": layout,
        "rows": groups if layout == 3 else rows,
        "columns": 5 if layout == 3 else 2,
        "logical_rows": rows,
        "logical_bytes": _expected_logical_bytes(workload, decimal),
    }
    if any(not _integer(state.get(field), value) for field, value in expected.items()):
        raise ValueError("invalid SQL stream current checkpoint census or ledger")
    if state.get("segment") != segment or state.get("segment_ids") != [
        "batch-metadata",
        "control",
        segment,
        "logical-schema",
    ]:
        raise ValueError("invalid SQL stream current checkpoint inventory")
    if (
        not _integer(state.get("total_bytes"))
        or state["total_bytes"] <= sample["final_checkpoint_bytes"]
        or not _digest(state.get("snapshot_sha256"))
    ):
        raise ValueError("invalid SQL stream current checkpoint bytes")
    return state["total_bytes"]


def _validate_checkpoint(
    sample: dict, workload: Workload, decimal: DecimalType | None, *, current: bool
) -> None:
    _validate_checkpoint_fields(sample)
    final_bytes = (
        _validate_current_checkpoint(sample, workload, decimal)
        if current
        else sample["final_checkpoint_bytes"]
    )
    if workload[1]:
        if sample["checkpoint_bytes"] < final_bytes:
            raise ValueError("incomplete SQL stream checkpoint bytes")
    elif any(
        sample[field] != 0
        for field in ("checkpoint_bytes", "prepare_seconds", "capture_seconds")
    ):
        raise ValueError("unexpected SQL stream checkpoint work")


def _validate_sample(
    sample: object,
    workload: Workload,
    decimal: DecimalType | None,
    *,
    oracle: bool,
    current: bool,
) -> None:
    if not isinstance(sample, dict):
        raise ValueError("invalid SQL stream observation")
    _validate_timing(sample)
    _validate_counts(sample, workload)
    _validate_oracle(sample, workload, decimal, oracle=oracle, current=current)
    _validate_checkpoint(sample, workload, decimal, current=current)


def _validate_inventory(cases: object) -> list[dict]:
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise ValueError("incomplete SQL stream benchmark inventory")
    if any(
        not isinstance(case, dict) or not isinstance(case.get("name"), str)
        for case in cases
    ):
        raise ValueError("incomplete SQL stream benchmark inventory")
    if {case["name"] for case in cases} != set(CASES):
        raise ValueError("incomplete SQL stream benchmark inventory")
    return cases


def _validate_workload(case: dict, workload: Workload) -> None:
    batches, checkpoint, count, unique = workload
    maximum_groups = count - (count + 100) // 101 + 1 if unique else 65
    expected = {
        "rows": count,
        "batches": batches,
        "checkpoint_every": checkpoint,
        "maximum_groups": maximum_groups,
    }
    if any(not _integer(case.get(field), value) for field, value in expected.items()):
        raise ValueError("invalid SQL stream workload")
    if case.get("unique_keys") is not unique or case.get("query") != QUERY:
        raise ValueError("invalid SQL stream workload")


def _case_row(
    case: dict,
    scope: str,
    minimum_samples: int,
    decimal: DecimalType | None,
    *,
    current: bool,
) -> dict:
    workload = CASES[case["name"]]
    batches, checkpoint, count, unique = workload
    _validate_workload(case, workload)
    samples = case.get("samples")
    if not isinstance(samples, list) or len(samples) < minimum_samples:
        raise ValueError("incomplete SQL stream observations")
    _validate_sample(
        case.get("oracle"), workload, decimal, oracle=True, current=current
    )
    for sample in samples:
        _validate_sample(sample, workload, decimal, oracle=False, current=current)
    if any(
        sample["final_checkpoint_sha256"] != case["oracle"]["final_checkpoint_sha256"]
        or sample["final_checkpoint_bytes"] != case["oracle"]["final_checkpoint_bytes"]
        for sample in samples
    ):
        raise ValueError("unstable SQL stream checkpoint bytes")
    if current and any(
        sample["checkpoint_state"] != case["oracle"]["checkpoint_state"]
        for sample in samples
    ):
        raise ValueError("unstable SQL stream current checkpoint state")
    return {
        "samples": [sample["seconds"] for sample in samples],
        "rows": count,
        "scope": scope,
        "metadata": {
            "oracle": case["oracle"],
            "observations": samples,
            "batches": batches,
            "checkpoint_every": checkpoint,
            "unique_keys": unique,
        },
    }


def _decimal_descriptor(decimal: DecimalType) -> dict:
    bits, precision, scale = decimal
    return {
        "name": f"decimal{bits}",
        "input": {"bits": bits, "precision": precision, "scale": scale},
        "total": {
            "bits": bits,
            "precision": min(precision + 10, DECIMAL_PRECISION[bits][1]),
            "scale": scale,
        },
    }


def _report_decimal(report: dict) -> DecimalType | None:
    if report.get("schema") == "calc-flow.sql-stream-aggregate.v1":
        if "value_type" in report:
            raise ValueError("undeclared SQL stream datatype")
        return None
    if (
        report.get("schema") == "calc-flow.sql-stream-aggregate.v3"
        and "value_type" not in report
    ):
        return None
    if report.get("schema") not in (
        "calc-flow.sql-stream-aggregate.v2",
        "calc-flow.sql-stream-aggregate.v3",
    ):
        raise ValueError("invalid SQL stream benchmark contract")
    return _validate_decimal_descriptor(report.get("value_type"))


def _validate_decimal_descriptor(value: object) -> DecimalType:
    if not isinstance(value, dict) or not isinstance(value.get("input"), dict):
        raise ValueError("invalid SQL stream decimal datatype")
    source = value["input"]
    bits, precision, scale = (
        source.get(field) for field in ("bits", "precision", "scale")
    )
    _validate_decimal_numbers(source)
    if bits not in DECIMAL_PRECISION or scale not in (-2, 2):
        raise ValueError("unsupported SQL stream decimal datatype")
    decimal = (bits, DECIMAL_PRECISION[bits][0], scale)
    if value != _decimal_descriptor(decimal):
        raise ValueError("invalid SQL stream decimal promotion")
    _validate_decimal_numbers(value["total"])
    return decimal


def _validate_decimal_numbers(value: dict) -> None:
    if any(
        type(value.get(field)) is not int for field in ("bits", "precision", "scale")
    ):
        raise ValueError("invalid SQL stream decimal promotion")


def _typed_case_row(
    case: dict, report: dict, minimum_samples: int, decimal: DecimalType | None
) -> dict:
    row = _case_row(
        case,
        report["scope"],
        minimum_samples,
        decimal,
        current=report["schema"] == "calc-flow.sql-stream-aggregate.v3",
    )
    if decimal is not None:
        row["metadata"]["value_type"] = report["value_type"]
    return row


def sql_stream_rows(path: Path, *, minimum_samples: int = 20) -> dict:
    """Retain complete observations after validating every cumulative snapshot."""

    report = read_json(path)
    decimal = _report_decimal(report)
    if report.get("scope") != "warm-native-operator-cumulative-snapshots":
        raise ValueError("invalid SQL stream benchmark contract")
    cases = _validate_inventory(report.get("cases"))
    prefix = (
        "stream_sql_aggregate"
        if decimal is None
        else f"stream_sql_aggregate/decimal{decimal[0]}_scale{decimal[2]}"
    )
    return {
        f"{prefix}/{case['name']}": _typed_case_row(
            case, report, minimum_samples, decimal
        )
        for case in cases
    }
