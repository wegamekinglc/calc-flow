"""Fixed Join/ASOF acceptance inventory and fail-closed evidence validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from scripts.benchmark_suite.report import ROUNDS, SAMPLES

MODES = ("off", "low", "on")
RUNTIME_ROWS = (1, 2, 3, 6, 7)


@dataclass(frozen=True, slots=True)
class AcceptanceCase:
    row: int
    workload: str
    rows: int
    batch_rows: int
    mode: str
    threads: int = 1

    @property
    def id(self) -> str:
        return (
            f"row-{self.row}/{self.workload}/{self.rows}/batch-{self.batch_rows}/"
            f"{self.mode}/cpu-{self.threads}"
        )

    @property
    def sides(self) -> tuple[str, ...]:
        return ("candidate",) if self.mode == "off" else ("baseline", "candidate")

    @property
    def metric(self) -> str:
        if self.row == 7 and self.workload == "asof":
            return "settled_processing_seconds"
        return {
            "join_capture": "snapshot_seconds",
            "join_restore": "restore_seconds",
            "join_handler_compacting": "handler_seconds",
            "join_handler_steady": "handler_seconds",
            "join_small_batch": "handler_seconds",
            "asof_operator": "processing_seconds",
        }.get(self.workload, "processing_seconds")


def acceptance_inventory() -> tuple[AcceptanceCase, ...]:
    """List each selected size, mode, thread setting, and necessary control."""

    runtime = _runtime_inventory()
    overhead = tuple(
        AcceptanceCase(7, workload, 1_000_000, 1024, mode)
        for workload in ("asof", "projection", "asof_operator")
        for mode in (MODES if workload != "asof_operator" else ("off", "low"))
    )
    components = tuple(
        AcceptanceCase(row, workload, 1_000_000, batch_rows, "enabled")
        for row, workload, batch_rows in (
            (4, "join_capture", 64_000),
            (4, "join_restore", 64_000),
            (5, "join_handler_compacting", 500),
            (5, "join_handler_steady", 500),
        )
    )
    small_batches = tuple(
        AcceptanceCase(8, "join_small_batch", 1_000_000, 1024, mode)
        for mode in ("off", "enabled")
    )
    return (*runtime, *components, *overhead, *small_batches)


def _runtime_inventory() -> tuple[AcceptanceCase, ...]:
    return tuple(
        AcceptanceCase(row, workload, rows, 64_000, mode, threads)
        for row, workload, sizes, configurations in (
            (1, "join_sealed", (1_000_000,), (1, 32)),
            (2, "join_sealed", (10_000_000,), (1, 32)),
            (3, "join_retained", (1_000_000, 4_000_000), (1,)),
            (6, "asof", (1_000_000,), (1, 32)),
        )
        for rows in sizes
        for threads in configurations
        for mode in MODES
    )


def checkpoint_settings(mode: str) -> dict:
    """Return semantic settings; enabled legacy main omits the new off flag."""

    if mode not in MODES:
        raise ValueError("runtime checkpoint mode must be off, low, or on")
    return {
        "checkpointing": mode != "off",
        "checkpoint_interval_millis": {
            "off": None,
            "low": 86_400_000,
            "on": 100,
        }[mode],
    }


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_measurement(case: AcceptanceCase, side: str, record: dict) -> None:
    rounds = record.get("rounds")
    _validate_rounds(case, side, rounds)
    warmups = record.get("warmups")
    if not isinstance(warmups, list) or len(warmups) != ROUNDS:
        raise ValueError(f"{case.id}/{side}: expected one warmup per round")
    for sample in warmups:
        validate_observation(case, side, sample)
    for samples in rounds:
        for sample in samples:
            validate_observation(case, side, sample)


def _validate_rounds(case: AcceptanceCase, side: str, rounds: object) -> None:
    if (
        not isinstance(rounds, list)
        or len(rounds) != ROUNDS
        or any(not isinstance(items, list) or len(items) != SAMPLES for items in rounds)
    ):
        raise ValueError(f"{case.id}/{side}: expected two complete rounds of ten")


def validate_observation(case: AcceptanceCase, side: str, sample: dict) -> None:
    """Reject invalid warmups and samples immediately at the controller boundary."""
    if side not in case.sides:
        raise ValueError(f"{case.id}: unsupported revision arm")
    _validate_sample_metric(case, side, sample)
    _validate_lifecycle(case, side, sample)
    if case.workload not in ("join_sealed", "join_retained", "asof", "projection"):
        return
    _validate_runtime_observation(case, side, sample)
    _validate_disabled_observation(case, side, sample)
    if case.row == 7 and case.workload == "asof":
        _validate_settled_asof(case, side, sample)


def _validate_sample_metric(case: AcceptanceCase, side: str, sample: dict) -> None:
    if (
        not isinstance(sample, dict)
        or not _positive(sample.get(case.metric))
        or sample.get("oracle_passed") is not True
        or sample.get("cleanup_complete") is not True
    ):
        raise ValueError(f"{case.id}/{side}: missing metric or valid evidence")


def _validate_lifecycle(case: AcceptanceCase, side: str, sample: dict) -> None:
    lifecycle = sample.get("lifecycle")
    if lifecycle != "completed" and not (
        side == "baseline"
        and lifecycle == "failed"
        and isinstance(sample.get("lifecycle_error"), str)
        and sample["lifecycle_error"]
    ):
        raise ValueError(f"{case.id}/{side}: invalid lifecycle evidence")


def _validate_runtime_observation(
    case: AcceptanceCase, side: str, sample: dict
) -> None:
    if (
        sample.get("mode") != case.mode
        or sample.get("oracle", {}).get("rows") != case.rows
        or not isinstance(sample.get("checkpoint_observations"), list)
        or len(sample["checkpoint_observations"]) < 2
        or type(sample.get("processing_checkpoint_free")) is not bool
        or sample.get("observer_policy")
        != "all-modes-status-every-10ms-plus-final-frontier-1ms"
    ):
        raise ValueError(f"{case.id}/{side}: incomplete runtime evidence")


def _validate_disabled_observation(
    case: AcceptanceCase, side: str, sample: dict
) -> None:
    if case.mode == "off" and (
        not sample["processing_checkpoint_free"]
        or sample.get("checkpoint_evidence", {}).get("state_file_count") != 0
        or sample.get("checkpoint_evidence", {}).get("coverage") != "disabled"
    ):
        raise ValueError(
            f"{case.id}/{side}: disabled mode produced checkpoint activity"
        )


def _validate_settled_asof(case: AcceptanceCase, side: str, sample: dict) -> None:
    if (
        sample.get("processing_settled") is not True
        or not _positive(sample.get("processing_seconds"))
        or sample["settled_processing_seconds"] < sample["processing_seconds"]
        or not isinstance(sample.get("match_identity"), dict)
        or not isinstance(sample.get("dispatch_trace"), dict)
        or sample["dispatch_trace"].get("schema") != "calc-flow.asof-dispatch-trace.v1"
    ):
        raise ValueError(f"{case.id}/{side}: ASOF processing is not settled")


def validate_complete_results(report: dict) -> None:
    """Reject omitted arms, placeholders, missing metrics, and partial samples."""

    cases = acceptance_inventory()
    expected = {(case.id, side): case for case in cases for side in case.sides}
    records = report.get("measurements")
    if not isinstance(records, list):
        raise ValueError("missing measurements for the complete eight-row inventory")
    actual = _measurement_records(records, expected)
    missing = set(expected) - set(actual)
    if missing:
        raise ValueError(f"missing measurements: {sorted(missing)}")
    for key, record in actual.items():
        _validate_measurement(expected[key], key[1], record)
    validate_observation_references(records)


def _measurement_records(records: list, expected: dict) -> dict:
    actual = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid acceptance measurement record")
        key = (record.get("case"), record.get("side"))
        if key not in expected or key in actual:
            raise ValueError("duplicate or unexpected acceptance measurement")
        actual[key] = record
    return actual


def validate_observation_references(records: list[dict]) -> None:
    """Require shared metrics to identify the same physical raw observation."""
    cases = {case.id: case for case in acceptance_inventory()}
    references = {}
    physical = {}
    for record in records:
        case = cases[record["case"]]
        members = _group_members(case, cases)
        for round_index in range(ROUNDS):
            observations = [
                ("warmup", record["warmups"][round_index]),
                *enumerate(record["rounds"][round_index]),
            ]
            for index, sample in observations:
                observation_id, digest = _raw_observation_identity(sample, members)
                slot = (tuple(members), record["side"], round_index, index)
                prior = physical.setdefault(slot, (observation_id, digest))
                if prior != (observation_id, digest):
                    raise ValueError(
                        "shared metric arms reference different physical observations"
                    )
                identity = (slot, digest)
                if references.setdefault(observation_id, identity) != identity:
                    raise ValueError(
                        "one raw observation was reused as independent samples"
                    )


def _group_members(case: AcceptanceCase, cases: dict) -> list[str]:
    return (
        sorted(
            item.id
            for item in cases.values()
            if item.row == case.row and case.row in (4, 5)
        )
        if case.row in (4, 5)
        else [case.id]
    )


def _raw_observation_identity(sample: dict, members: list) -> tuple[str, str]:
    observation_id = sample.get("observation_id")
    digest = sample.get("raw_observation_sha256")
    if (
        not isinstance(observation_id, str)
        or not observation_id
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or sorted(sample.get("group_member_ids", [])) != members
    ):
        raise ValueError("missing exact raw observation and shared-member references")
    return observation_id, digest


def matched_runtime_identity(fixture: dict, mode: str, cpu_affinity: list[int]) -> dict:
    """Identify shared work without asserting observed cross-ingress dispatch order."""
    return {
        "project_sha256": fixture["project"]["sha256"],
        "inputs_sha256": fixture["inputs_sha256"],
        "shared_partial_order_sha256": fixture["shared_partial_order_sha256"],
        "output_contract_sha256": fixture["expected_output"]["sha256"],
        "compiled_plan_fingerprint": fixture["compiled_plan_fingerprint"],
        "rows_per_side": fixture["rows_per_side"],
        "checkpoint_enabled": fixture["checkpoint_enabled"],
        "batch_rows": fixture["batch_rows"],
        "mode": mode,
        "cpu_affinity": list(cpu_affinity),
        "processing_boundary": "final-output-projection-watermark-completed",
        "output_frontier": "minimum-input-watermark-minus-one-microsecond",
    }


def matched_native_identity(native: dict, mode: str, cpu_affinity: list[int]) -> dict:
    identity = {
        key: native[key]
        for key in (
            "project_sha256",
            "inputs_sha256",
            "shared_partial_order_sha256",
            "output_contract_sha256",
            "compiled_plan_fingerprint",
            "checkpoint_enabled",
            "rows_per_side",
            "batch_rows",
        )
    }
    return {
        **identity,
        "mode": mode,
        "cpu_affinity": list(cpu_affinity),
        "processing_boundary": "final-output-projection-watermark-completed",
        "output_frontier": "minimum-input-watermark-minus-one-microsecond",
    }


def validate_dispatch_replay(end_to_end: dict, native: dict) -> dict:
    """Bind the verified recorded callback replay to its exact source observation."""
    source_id = end_to_end.get("observation_id")
    raw_digest = end_to_end.get("raw_observation_sha256")
    trace_digest = end_to_end.get("dispatch_trace_envelope_sha256")
    trace = end_to_end.get("dispatch_trace", {})
    replay = native.get("dispatch_trace")
    if (
        not _valid_replay_source(source_id, raw_digest, trace_digest, native)
        or not _valid_replay_binding(source_id, raw_digest, trace_digest, native)
        or not _valid_source_trace(trace, end_to_end)
    ):
        raise ValueError("ASOF replay does not verify this exact source observation")
    _validate_trace_witness(trace, replay)
    return {
        "status": "recorded-dispatch-chain-replayed",
        "source_observation_id": source_id,
        "source_raw_sha256": raw_digest,
        "trace_sha256": trace_digest,
        "native_executable_sha256": native["native_executable_sha256"],
        "claim": (
            "Same recorded ASOF ingress/callback chain, ingress/final-output "
            "batch geometry, per-node aggregate data counts and ASOF state; "
            "instrumented component residual."
        ),
    }


def _validate_trace_witness(trace: dict, replay: dict) -> None:
    fields = (
        "schema",
        "node_id",
        "compiled_plan_fingerprint",
        "capacity",
        "record_count",
        "overflowed",
        "terminal_excluded",
        "allow_barriers",
        "events",
        "callback_counts",
        "completed_frontiers",
        "operator_data_counts",
        "asof_state",
        "output_record_batch_rows",
        "output_node_id",
        "required_output_frontier_micros",
    )
    if not isinstance(replay, dict) or any(
        field not in trace or replay.get(field) != trace[field] for field in fields
    ):
        raise ValueError("ASOF replay witness differs from the source state or trace")
    if any(
        not isinstance(trace[field], expected) or not trace[field]
        for field, expected in (
            ("events", list),
            ("callback_counts", dict),
            ("completed_frontiers", dict),
            ("operator_data_counts", dict),
            ("asof_state", dict),
            ("output_record_batch_rows", list),
        )
    ):
        raise ValueError("ASOF replay has an empty state or trace witness")


def _valid_replay_source(source_id, raw_digest, trace_digest, native) -> bool:
    return (
        isinstance(source_id, str)
        and bool(source_id)
        and not any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in (
                raw_digest,
                trace_digest,
                native.get("native_executable_sha256"),
            )
        )
    )


def _valid_replay_binding(source_id, raw_digest, trace_digest, native) -> bool:
    return (
        native.get("dispatch_trace_verified") is True
        and native.get("dispatch_replay_state_verified") is True
        and native.get("replay_source_observation_id") == source_id
        and native.get("replay_source_raw_sha256") == raw_digest
        and native.get("replay_trace_sha256") == trace_digest
    )


def _valid_source_trace(trace, end_to_end) -> bool:
    return (
        isinstance(trace, dict)
        and trace.get("schema") == "calc-flow.asof-dispatch-trace.v1"
        and trace.get("overflowed") is False
        and trace.get("terminal_excluded") is True
        and trace.get("allow_barriers") is False
        and trace.get("compiled_plan_fingerprint")
        == end_to_end.get("match_identity", {}).get("compiled_plan_fingerprint")
    )


def matched_overhead(
    end_to_end: dict,
    native: dict,
    projection_seconds: float,
    *,
    scheduling_proof: dict | None = None,
) -> dict:
    """Retain the signed instrumented residual with verified recorded replay."""
    identity = _matched_component_identity(end_to_end, native)
    _validate_overhead_checkpoint(identity, end_to_end, native)
    _validate_overhead_observations(end_to_end, native, projection_seconds)
    _validate_native_projection(native)
    overhead = end_to_end["seconds"] - native["seconds"]
    target = 2 * projection_seconds
    replay_proof = (
        validate_dispatch_replay(end_to_end, native)
        if native.get("dispatch_trace_verified") is True
        else None
    )
    equivalent = replay_proof is not None
    _validate_scheduling_proof(scheduling_proof, end_to_end, native, identity)
    if overhead < 0 or not equivalent:
        status = "inconclusive"
    elif overhead <= target:
        status = "observed-within-target"
    else:
        status = "observed-above-target"
    return {
        "end_to_end_seconds": end_to_end["seconds"],
        "operator_seconds": native["seconds"],
        "overhead_seconds": overhead,
        "projection_seconds": projection_seconds,
        "target_seconds": target,
        "target_status": status,
        "schedule_status": "recorded-dispatch-chain-replayed"
        if equivalent
        else "schedule-dependent-estimate",
        "dispatch_replay_proof": replay_proof,
        "scheduling_proof": scheduling_proof,
        "canonical_native_events_sha256": native["canonical_native_events_sha256"],
        "compiled_projection_sha256": native["compiled_projection_sha256"],
    }


def _validate_overhead_checkpoint(identity, end_to_end, native) -> None:
    if (
        identity["checkpoint_enabled"] is not (identity["mode"] != "off")
        or identity["mode"] not in ("off", "low")
    ) or any(
        sample.get("nonterminal_epochs") != []
        or sample.get("processing_checkpoint_free") is not True
        for sample in (end_to_end, native)
    ):
        raise ValueError(
            "checkpoint activity or uncertainty cannot be treated as harness overhead"
        )


def _validate_overhead_observations(end_to_end, native, projection_seconds) -> None:
    if not _positive(projection_seconds) or any(
        not _positive(sample.get("seconds"))
        or sample.get("oracle_passed") is not True
        or sample.get("processing_settled") is not True
        for sample in (end_to_end, native)
    ):
        raise ValueError("ASOF overhead requires settled, oracle-checked measurements")


def _validate_native_projection(native) -> None:
    if (
        not isinstance(native.get("compiled_projection"), list)
        or not native["compiled_projection"]
        or not isinstance(native.get("compiled_projection_sha256"), str)
        or len(native["compiled_projection_sha256"]) != 64
        or not isinstance(native.get("canonical_native_events_sha256"), str)
        or len(native["canonical_native_events_sha256"]) != 64
    ):
        raise ValueError(
            "ASOF native component lacks ordinary-compiler projection "
            "and schedule attestation"
        )


def _validate_scheduling_proof(scheduling_proof, end_to_end, native, identity) -> None:
    if (
        isinstance(scheduling_proof, dict)
        and scheduling_proof.get("status") == "verified"
    ):
        expected_proof = {
            "product_commit": end_to_end["product_commit"],
            "native_executable_sha256": native.get("native_executable_sha256"),
            "project_sha256": identity["project_sha256"],
            "compiled_projection_sha256": native["compiled_projection_sha256"],
            "batch_rows": identity["batch_rows"],
            "checkpoint_enabled": identity["checkpoint_enabled"],
        }
        if (
            any(
                scheduling_proof.get(key) != value
                for key, value in expected_proof.items()
            )
            or re.fullmatch(
                r"[0-9a-f]{64}", str(scheduling_proof.get("artifact_sha256", ""))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(native.get("native_executable_sha256", ""))
            )
            is None
        ):
            raise ValueError(
                "ASOF scheduling proof does not match this release "
                "and workload contract"
            )


def _matched_component_identity(end_to_end: dict, native: dict) -> dict:
    _validate_product_revision(end_to_end, native)
    identity = end_to_end.get("match_identity")
    if (
        not isinstance(identity, dict)
        or not identity
        or identity != native.get("match_identity")
    ):
        raise ValueError("ASOF overhead requires matching component identities")
    required = (
        "project_sha256",
        "inputs_sha256",
        "shared_partial_order_sha256",
        "output_contract_sha256",
        "compiled_plan_fingerprint",
        "checkpoint_enabled",
        "rows_per_side",
        "batch_rows",
        "mode",
        "cpu_affinity",
        "processing_boundary",
        "output_frontier",
    )
    if any(field not in identity for field in required):
        raise ValueError("ASOF overhead requires complete matching identities")
    return identity


def _validate_product_revision(end_to_end: dict, native: dict) -> None:
    product_commit = end_to_end.get("product_commit")
    if (
        not isinstance(product_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", product_commit) is None
        or product_commit != native.get("product_commit")
    ):
        raise ValueError("ASOF overhead requires the same exact product revision")
