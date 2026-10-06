"""Validate watermark eviction workloads and recovery probe values."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from functools import lru_cache
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json

CASES = {
    f"{mode}_{keys}": {
        "keys": keys,
        "mode": mode,
        "ticks": 1 if mode == "all_expired" else 32,
    }
    for keys in (4096, 65536)
    for mode in (
        "sparse_identity_held",
        "sparse_identity_removed",
        "none_expired",
        "all_expired",
    )
}


def expected_status(config: dict, tick: int) -> dict:
    keys, mode = config["keys"], config["mode"]
    evicted = keys if mode == "all_expired" else 0 if mode == "none_expired" else tick
    held = mode == "sparse_identity_held"
    frontier = keys if mode == "all_expired" else tick
    return {
        "retained_right_rows": keys - evicted,
        "identity_only_rows": evicted if held else 0,
        "state_rows": keys if held else keys - evicted,
        "evicted_right_rows": evicted,
        "left_watermark": frontier,
        "right_watermark": 0 if held else frontier,
        "output_watermark": -1 if held else frontier - 1,
        "pending_left_rows": 0,
        "emitted_left_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "right_accepted_rows": keys,
        "left_accepted_rows": 0,
    }


def _probe_time(key: int, keys: int, mode: str) -> int:
    if mode == "none_expired":
        return 1024 + key
    if mode == "all_expired":
        return keys + key
    return max(key, 32)


def _probe_record(key: int, keys: int, mode: str) -> bytes:
    matched = mode == "none_expired" or mode != "all_expired" and key >= 32
    value = matched and key % 17 != 0
    return struct.pack(
        "<qqBqBq",
        key,
        _probe_time(key, keys, mode),
        int(matched),
        key if matched else 0,
        int(value),
        key * 3 if value else 0,
    )


@lru_cache(maxsize=8)
def _probe_digest(keys: int, mode: str) -> str:
    digest = hashlib.sha256()
    for key in range(keys):
        digest.update(_probe_record(key, keys, mode))
    return digest.hexdigest()


def expected_probe_digest(config: dict) -> str:
    return _probe_digest(config["keys"], config["mode"])


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def _timing(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _validate_timing(sample: object, config: dict) -> None:
    if not isinstance(sample, dict) or not _timing(sample.get("seconds")):
        raise ValueError("invalid eviction timing")
    times = sample.get("tick_seconds", [])
    if not isinstance(times, list) or len(times) != config["ticks"]:
        raise ValueError("invalid eviction tick observations")
    if not all(_timing(value) for value in times):
        raise ValueError("invalid eviction tick observations")
    if not math.isclose(sum(times), sample["seconds"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("invalid eviction tick observations")


def _validate_status(status: object, config: dict, tick: int) -> None:
    if not isinstance(status, dict):
        raise ValueError("eviction status/frontier mismatch")
    expected = expected_status(config, tick)
    if any(
        type(status.get(field)) is not int or status[field] != value
        for field, value in expected.items()
    ):
        raise ValueError("eviction status/frontier mismatch")


def _validate_charge(status: dict, config: dict, previous_bytes: int) -> int:
    charged = status.get("state_bytes")
    if (
        not _count(charged)
        or charged > previous_bytes
        or bool(charged) != bool(status["state_rows"])
    ):
        raise ValueError("invalid eviction accounting")
    if config["mode"] == "none_expired" and charged != previous_bytes:
        raise ValueError("no-op changed accounting")
    return charged


def _validate_statuses(sample: dict, config: dict) -> None:
    statuses = sample.get("statuses", [])
    if not isinstance(statuses, list) or len(statuses) != config["ticks"]:
        raise ValueError("invalid eviction tick observations")
    previous_bytes = sample.get("before_state_bytes")
    if not _count(previous_bytes) or previous_bytes == 0:
        raise ValueError("invalid admission accounting")
    for tick, status in enumerate(statuses, 1):
        _validate_status(status, config, tick)
        previous_bytes = _validate_charge(status, config, previous_bytes)


def _validate_snapshot(sample: dict) -> None:
    if (
        type(sample.get("output_rows")) is not int
        or sample["output_rows"] != 0
        or sample.get("validated_status") is not True
        or not _count(sample.get("checkpoint_bytes"))
        or not _sha(sample.get("checkpoint_sha256"))
    ):
        raise ValueError("invalid eviction snapshot evidence")


def _validate_recovery_counts(sample: dict, config: dict) -> None:
    status = sample["statuses"][-1]
    captured_bytes = sample.get("checkpoint_state_bytes", status["state_bytes"])
    if (
        sample.get("validated_recovery") is not True
        or sample.get("probe_rows") != config["keys"]
        or type(sample.get("probe_rows")) is not int
        or sample.get("restored_state_rows") != status["state_rows"]
        or not _count(captured_bytes)
        or sample.get("restored_state_bytes") != captured_bytes
    ):
        raise ValueError("invalid eviction recovery/value oracle")


def _validate_recovery_values(sample: dict, config: dict) -> None:
    if sample.get("probe_sha256") != expected_probe_digest(config) or sample.get(
        "identity_duplicate_rejected"
    ) is not (config["mode"] == "sparse_identity_held"):
        raise ValueError("invalid eviction recovery/value oracle")


def _validate_sample(sample: object, config: dict, *, oracle: bool) -> None:
    _validate_timing(sample, config)
    _validate_statuses(sample, config)
    _validate_snapshot(sample)
    if oracle:
        _validate_recovery_counts(sample, config)
        _validate_recovery_values(sample, config)


def _validate_inventory(cases: object) -> None:
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise ValueError("incomplete or duplicate eviction inventory")
    if not all(isinstance(case, dict) for case in cases):
        raise ValueError("invalid eviction case")
    if {case.get("name") for case in cases} != set(CASES):
        raise ValueError("incomplete or duplicate eviction inventory")


def _validate_case(case: dict, minimum_samples: int) -> None:
    config = CASES[case["name"]]
    if (
        case.get("config") != config
        or not isinstance(case.get("samples"), list)
        or len(case["samples"]) < minimum_samples
    ):
        raise ValueError("invalid eviction workload/sample count")
    _validate_sample(case.get("oracle"), config, oracle=True)
    for sample in case["samples"]:
        _validate_sample(sample, config, oracle=False)


def _validated_cases(report: dict, minimum_samples: int) -> list[dict]:
    if (
        report.get("schema") != "calc-flow.asof-eviction.v1"
        or report.get("scope") != "operator-watermark-eviction"
    ):
        raise ValueError("invalid eviction benchmark contract")
    cases = report.get("cases", [])
    _validate_inventory(cases)
    for case in cases:
        _validate_case(case, minimum_samples)
    return cases


def asof_eviction_rows(path: Path, *, minimum_samples: int = 20) -> dict:
    report = read_json(path)
    cases = _validated_cases(report, minimum_samples)
    return {
        f"stream_asof_eviction/{case['name']}": {
            "samples": [sample["seconds"] for sample in case["samples"]],
            "rows": case["config"]["keys"],
            "scope": report["scope"],
            "metadata": {
                "config": case["config"],
                "oracle": case["oracle"],
                "observations": case["samples"],
            },
        }
        for case in cases
    }
