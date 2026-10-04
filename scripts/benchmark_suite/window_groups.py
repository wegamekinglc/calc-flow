"""Validate native window group workloads and complete output values."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from functools import lru_cache
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json

CASES = {
    (
        f"{keys}_{groups}_{'hopping' if hopping else 'tumbling'}_"
        f"{'five' if multiple else 'sum'}"
    ): {
        "rows": 100_000,
        "groups": groups,
        "keys": keys,
        "hopping": hopping,
        "multiple": multiple,
    }
    for keys in ("integer", "string", "composite")
    for groups in (4, 8192)
    for hopping in (False, True)
    for multiple in (False, True)
}


def _integer(value: int | None) -> bytes:
    return struct.pack("<Bq", int(value is not None), value or 0)


@lru_cache(maxsize=2)
def _expected_groups(groups: int) -> tuple:
    values = [[] for _ in range(groups)]
    for row in range(100_000):
        key = row % groups
        if key != 0 and row % 17 != 0:
            values[key].append(row % 97 - 48)
    return tuple(
        (sum(group), len(group), min(group, default=None), max(group, default=None))
        for group in values
    )


def _multiple(values: tuple) -> bytes:
    total, count, minimum, maximum = values
    average = (
        struct.pack("<Bd", 1, total / count) if count else struct.pack("<BQ", 0, 0)
    )
    return struct.pack("<Q", count) + _integer(minimum) + _integer(maximum) + average


def _row(config: dict, start: int, key: int, values: tuple) -> bytes:
    total, count, _, _ = values
    end = start + (20 if config["hopping"] else 10)
    record = struct.pack("<qq", start, end) + _integer(key if key else None)
    if config["keys"] == "composite":
        record += struct.pack("<q", 7)
    record += _integer(total if count else None)
    return record + _multiple(values) if config["multiple"] else record


@lru_cache(maxsize=24)
def _digest(keys: str, groups: int, hopping: bool, multiple: bool) -> str:
    config = {"keys": keys, "groups": groups, "hopping": hopping, "multiple": multiple}
    digest = hashlib.sha256()
    for start in (-10, 0) if hopping else (0,):
        for key, values in enumerate(_expected_groups(groups)):
            digest.update(_row(config, start, key, values))
    return digest.hexdigest()


def expected_digest(config: dict) -> str:
    return _digest(
        config["keys"], config["groups"], config["hopping"], config["multiple"]
    )


def _timing(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_timing(sample: object) -> None:
    if not isinstance(sample, dict):
        raise ValueError("invalid window observation")
    times = [
        sample.get(field) for field in ("seconds", "process_seconds", "end_seconds")
    ]
    if not all(_timing(value) for value in times):
        raise ValueError("invalid window timing")
    if not math.isclose(times[0], sum(times[1:]), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("invalid window phase timing")


def _validate_counts(sample: dict, config: dict) -> None:
    expected = {
        "process_output_rows": 0,
        "input_rows": config["rows"],
        "output_rows": config["groups"] * (2 if config["hopping"] else 1),
    }
    if any(
        type(sample.get(field)) is not int or sample[field] != count
        for field, count in expected.items()
    ):
        raise ValueError("invalid window input/output counters")


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _validate_checkpoint(sample: dict, *, recovery: bool) -> None:
    count, digest = sample.get("checkpoint_bytes"), sample.get("checkpoint_sha256")
    if type(count) is not int or count < 0:
        raise ValueError("invalid window checkpoint byte count")
    if not recovery:
        if count != 0 or digest is not None:
            raise ValueError("unexpected measured window checkpoint")
        return
    if count == 0 or not _sha(digest):
        raise ValueError("invalid window recovery checkpoint")


def _validate_sample(sample: object, config: dict, *, recovery: bool) -> None:
    _validate_timing(sample)
    _validate_counts(sample, config)
    _validate_checkpoint(sample, recovery=recovery)
    if (
        sample.get("validated_all_rows") is not True
        or sample.get("validated_recovery") is not recovery
    ):
        raise ValueError("invalid window value/recovery proof")
    if sample.get("sha256") != expected_digest(config):
        raise ValueError("invalid window complete output digest")


def _validate_config(config: object, expected: dict) -> None:
    if config != expected:
        raise ValueError("invalid window workload")
    if any(type(config[field]) is not type(value) for field, value in expected.items()):
        raise ValueError("invalid window workload field type")


def _validate_case(case: dict, minimum_samples: int) -> None:
    expected = CASES[case["name"]]
    _validate_config(case.get("config"), expected)
    samples = case.get("samples")
    if not isinstance(samples, list) or len(samples) < minimum_samples:
        raise ValueError("invalid window sample count")
    _validate_sample(case.get("oracle"), expected, recovery=True)
    for sample in samples:
        _validate_sample(sample, expected, recovery=False)


def _validate_inventory(cases: object) -> None:
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise ValueError("invalid window case inventory")
    if not all(
        isinstance(case, dict) and isinstance(case.get("name"), str) for case in cases
    ):
        raise ValueError("invalid window case descriptor")
    if {case["name"] for case in cases} != set(CASES):
        raise ValueError("incomplete or duplicate window case inventory")


def _validated_cases(report: object, minimum_samples: int) -> list[dict]:
    if not isinstance(report, dict):
        raise ValueError("invalid window report")
    if (
        report.get("schema") != "calc-flow.window-groups.v1"
        or report.get("scope") != "operator-input-and-finalization"
    ):
        raise ValueError("invalid window benchmark contract")
    cases = report.get("cases")
    _validate_inventory(cases)
    for case in cases:
        _validate_case(case, minimum_samples)
    return cases


def window_groups_rows(path: Path, *, minimum_samples: int = 20) -> dict:
    if type(minimum_samples) is not int or minimum_samples < 0:
        raise ValueError("invalid minimum window sample count")
    report = read_json(path)
    cases = _validated_cases(report, minimum_samples)
    return {
        f"stream_window_groups/{case['name']}": {
            "samples": [sample["seconds"] for sample in case["samples"]],
            "rows": case["config"]["rows"],
            "scope": report["scope"],
            "metadata": {
                "config": case["config"],
                "oracle": case["oracle"],
                "observations": case["samples"],
            },
        }
        for case in cases
    }
