"""Validate admission-through-settlement ASOF benchmark observations."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.benchmark_suite.normalize import read_json

CASES = frozenset(
    {
        "admit_settle_100k",
        "eviction_ticks",
        "out_of_order_within_watermark",
        "composite_key",
    }
)
ROWS = 100_000


def asof_e2e_rows(path: Path) -> dict:
    """Keep timing and allocation samples for every validated ASOF workload."""

    report = read_json(path)
    cases = _validated_cases(report)
    return {
        f"stream_asof_e2e/{case['name']}": {
            "samples": [sample["seconds"] for sample in case["samples"]],
            "rows": ROWS,
            "scope": report["scope"],
            "metadata": {
                "oracle": case["oracle"],
                "observations": case["samples"],
            },
        }
        for case in cases
    }


def _validated_cases(report: dict) -> list[dict]:
    if (
        report.get("schema") != "calc-flow.asof-e2e.v1"
        or report.get("scope") != "operator-admission-settlement"
    ):
        raise ValueError("invalid ASOF e2e benchmark contract")
    cases = report.get("cases", [])
    if len(cases) != len(CASES) or {case.get("name") for case in cases} != CASES:
        raise ValueError("incomplete ASOF e2e inventory")
    for case in cases:
        _validate_case(case)
    return cases


def _validate_case(case: dict) -> None:
    if case.get("rows") != ROWS or len(case.get("samples", [])) < 20:
        raise ValueError("invalid ASOF e2e workload or sample count")
    for sample in (case.get("oracle"), *case["samples"]):
        if not _valid_sample(sample, case["name"]):
            raise ValueError("invalid ASOF e2e observation")


def _valid_sample(sample: object, case_name: str) -> bool:
    if not isinstance(sample, dict):
        return False
    seconds = sample.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        return False
    if (
        sample.get("output_rows") != ROWS
        or sample.get("validated_all_rows") is not True
    ):
        return False
    return _valid_right_rows(sample, case_name) and _valid_allocations(sample)


def _valid_right_rows(sample: dict, case_name: str) -> bool:
    evicted = sample.get("evicted_right_rows")
    retained = sample.get("retained_right_rows")
    if type(evicted) is not int or type(retained) is not int:
        return False
    if evicted < 0 or retained < 0 or evicted + retained != ROWS:
        return False
    return case_name != "eviction_ticks" or evicted > 0


def _valid_allocations(sample: dict) -> bool:
    return all(
        type(sample.get(field)) is int and sample[field] >= 0
        for field in (
            "allocation_total_bytes",
            "allocation_peak_bytes",
            "allocation_count",
        )
    )


def validate_matched_native(observation: dict) -> dict:
    """Validate the fixed million-row compiled ASOF replay and its evidence."""
    return validate_matched_observation(observation, "asof_join")


def validate_matched_correctness(observation: dict) -> dict:
    """Validate the fixed 3073-row actual-dispatch correctness preflight."""
    return _validate_matched_observation(
        observation, "asof_join", rows=3073, check_only=True
    )


def validate_matched_observation(observation: dict, scenario: str) -> dict:
    """Validate the common sealed-fixture observation used by ASOF and Join."""
    return _validate_matched_observation(
        observation, scenario, rows=1_000_000, check_only=False
    )


def _validate_matched_observation(
    observation: dict, scenario: str, *, rows: int, check_only: bool
) -> dict:
    row_field = "rows_per_side" if scenario == "asof_join" else "rows"
    if (
        observation.get("schema") != "calc-flow.matched-native-observation.v1"
        or observation.get("scenario") != scenario
        or observation.get("check_only") is not check_only
        or type(observation.get("checkpoint_enabled")) is not bool
        or type(observation.get("nonterminal_epochs")) is not int
        or observation["nonterminal_epochs"] != 0
        or type(observation.get(row_field)) is not int
        or observation[row_field] != rows
        or type(observation.get("batch_rows")) is not int
        or observation["batch_rows"] != 1024
        or observation.get("output_budget") != {"max_rows": 1024, "max_bytes": 64 << 20}
    ):
        raise ValueError("invalid matched native workload contract")
    for field in (
        "setup_seconds",
        "processing_seconds",
        "terminal_seconds",
        "oracle_seconds",
        "cleanup_seconds",
    ):
        seconds = observation.get(field)
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"invalid matched native clock: {field}")
    if (
        observation.get("validated_all_rows") is not True
        or observation.get("owned_cleanup_complete") is not True
        or type(observation.get("output_rows")) is not int
        or observation["output_rows"] != rows
        or type(observation.get("assembled_output_rows")) is not int
        or observation["assembled_output_rows"] != rows
        or type(observation.get("assembled_output_batches")) is not int
        or observation["assembled_output_batches"] <= 0
    ):
        raise ValueError("missing matched native oracle or cleanup evidence")
    _validate_projection_identity(observation, scenario)
    for field in ("post_processing_status", "callback_counts", "node_callback_counts"):
        if not isinstance(observation.get(field), dict) or not observation[field]:
            raise ValueError(f"missing native callback evidence: {field}")
    cpus = observation.get("observed_cpu_affinity")
    if (
        not isinstance(cpus, list)
        or not cpus
        or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
        or cpus != sorted(set(cpus))
    ):
        raise ValueError("invalid observed process affinity")
    environment = observation.get("observed_thread_environment")
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(key), str)
        or not environment[key].isdigit()
        or int(environment[key]) <= 0
        for key in (
            "TOKIO_WORKER_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "RAYON_NUM_THREADS",
            "POLARS_MAX_THREADS",
        )
    ):
        raise ValueError("missing observed thread environment")
    if type(observation.get("process_id")) is not int or observation["process_id"] <= 0:
        raise ValueError("missing observed process identity")
    return {
        **observation,
        "oracle_passed": True,
        "cleanup_complete": True,
        "lifecycle": "completed",
        "nonterminal_epochs": [],
        "processing_checkpoint_free": True,
        "processing_settled": True,
    }


def _validate_projection_identity(observation: dict, scenario: str) -> None:
    from scripts.toolkit import fingerprint_json

    for field in (
        "project_sha256",
        "inputs_sha256",
        "events_sha256",
        "shared_partial_order_sha256",
        "canonical_native_events_sha256",
        "output_contract_sha256",
        "compiled_plan_fingerprint",
        "compiled_projection_sha256",
    ):
        digest = observation.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid matched native identity: {field}")
    projection = observation.get("compiled_projection")
    if (
        not isinstance(projection, list)
        or not projection
        or fingerprint_json(projection, allow_nan=False)
        != observation["compiled_projection_sha256"]
    ):
        raise ValueError("compiled projection identity mismatch")
    expected_kind = "stream_asof_join" if scenario == "asof_join" else "stream_join"
    if (
        sum(
            node.get("kind") == expected_kind
            for node in projection
            if isinstance(node, dict)
        )
        != 1
    ):
        raise ValueError("compiled projection omits the matched join")
