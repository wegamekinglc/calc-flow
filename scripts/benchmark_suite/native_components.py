"""Validate fixed-size native Join component observations."""

from __future__ import annotations

import math


def validate_join_dictionary_refusal(observation: dict) -> dict:
    """Validate the bounded refusal control without performance denominators."""
    _require_refusal_fields(
        observation,
        {
            "schema": "calc-flow.join-dictionary-refusal.v1",
            "case": "join_dictionary_refusal_retention",
            "check_only": True,
            "admission_batch_rows": [2, 1],
            "left_retained_rows": 2,
            "right_retained_rows": 1,
            "left_retained_bytes": 228,
            "right_retained_bytes": 114,
            "full_v1_row_charges": [114, 114, 114],
            "output_keys": [["a", "a"]],
            "state_limit_failures": 0,
            "match_limit_failures": 0,
            "dictionary_grant_refusals": 1,
            "needs_rebuild_routes": 1,
            "legacy_key_encodings": 3,
            "sql_probe_table_builds": 1,
            "rebuild_attempts": 0,
            "rebuild_completions": 0,
            "owned_cleanup_complete": True,
        },
    )
    pool = observation.get("pool")
    _require_refusal_fields(
        pool,
        {
            "last_rejected_consumer": "sql-incremental:stream-join-dictionary",
            "failed_dictionary_grows": 1,
            "reserved_bytes": 0,
        },
    )
    for field in ("admission_batch_rows", "full_v1_row_charges"):
        if any(type(value) is not int for value in observation[field]):
            raise ValueError(f"invalid dictionary refusal integer list: {field}")
    _positive_refusal_counts(
        pool,
        (
            "successful_sql_grows",
            "last_rejected_bytes",
        ),
    )
    if observation["dictionary_grant_refusals"] != pool["failed_dictionary_grows"]:
        raise ValueError("dictionary refusal counter lacks matching pool evidence")
    return dict(observation)


def _require_refusal_fields(observation: dict, expected: dict) -> None:
    if not isinstance(observation, dict):
        raise ValueError("missing dictionary refusal evidence")
    for field, value in expected.items():
        actual = observation.get(field)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"invalid dictionary refusal evidence: {field}")


def _positive_refusal_counts(observation: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = observation.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"missing dictionary refusal work: {field}")


HISTORY_PHASES = (
    *(
        (name, "fresh")
        for name in (
            "fresh_admission",
            "fresh_prepare",
            "fresh_prepare_repeat",
            "fresh_capture",
            "fresh_prepare_clean",
            "fresh_capture_clean",
            "fresh_released",
        )
    ),
    *(
        (name, "history")
        for name in (
            "history_checkpoint_1",
            "history_checkpoint_2",
            "history_checkpoint_3",
            "history_checkpoint_4",
            "history_replace",
            "history_prepare_clean",
            "history_capture",
            "history_source_released",
        )
    ),
    ("restore_old", "restore_old"),
    ("restore_new", "restore_new"),
    ("history_snapshots_released", "history"),
    ("history_released", "history"),
)


def validate_join_checkpoint_history(observation: dict) -> dict:
    """Require the fixed row3 geometry, full-state oracle and exact pool evidence."""
    _validate_history_header(observation)
    unbounded = observation["schema"] == "calc-flow.join-checkpoint-history.v2"
    _validate_history_geometry(observation, unbounded=unbounded)
    fresh, history = observation.get("fresh"), observation.get("history")
    if not isinstance(fresh, dict) or not isinstance(history, dict):
        raise ValueError("missing fresh or retained-history evidence")
    oracle = _history_oracle(fresh.get("oracle"))
    if fresh.get("repeated_capture_segments_equal") is not True:
        raise ValueError("missing repeated preparation and capture proof")
    for item in (fresh, history):
        _history_count(item.get("output_rows"), 4_000_000, "output rows")
        if _history_oracle(item.get("oracle")) != oracle:
            raise ValueError("history full-state oracle differs")
    before = _validate_history_versions(history)
    _validate_history_alias(history, before)
    _validate_history_restores(observation, oracle)
    _history_reservations(observation, unbounded=unbounded)
    return dict(observation)


def _history_count(value, expected: int, field: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"invalid checkpoint history {field}")


def _history_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _history_oracle(oracle: dict) -> dict:
    if not isinstance(oracle, dict) or oracle.get("validated_all_rows") is not True:
        raise ValueError("missing complete read-only state oracle")
    for field, count in (("left_rows", 4_000_000), ("right_rows", 64)):
        _history_count(oracle.get(field), count, field)
    if any(
        not _history_digest(oracle.get(field))
        for field in ("left_sha256", "right_sha256")
    ):
        raise ValueError("invalid complete state digest")
    return oracle


def _history_snapshot(
    snapshot: dict, *, epoch: int, rows: int, replacement: bool
) -> None:
    if not isinstance(snapshot, dict):
        raise ValueError("missing checkpoint history snapshot")
    for field, count in (
        ("epoch", epoch),
        ("left_rows", rows),
        ("right_rows", 64),
        ("base_epoch", 4 if replacement else 0),
        ("left_watermark_micros", 1_767_288_099_000_000 + (rows == 4_000_000)),
    ):
        _history_count(snapshot.get(field), count, field)
    if snapshot.get("right_watermark_micros") is not None:
        raise ValueError("retained dimension was sealed")
    _validate_history_native_indexes(snapshot)
    inventory, names = _history_segment_inventory(snapshot)
    _validate_history_segment_versions(snapshot, inventory, names, epoch, replacement)


def _history_pool(sample: dict, *, unbounded: bool = False) -> None:
    if not isinstance(sample, dict):
        raise ValueError("missing reservation pool observation")
    if unbounded:
        if "limit_bytes" not in sample or sample["limit_bytes"] is not None:
            raise ValueError(
                "unbounded reservation pool requires an explicit null limit"
            )
    else:
        _history_count(sample.get("limit_bytes"), 1 << 30, "pool limit")
    _validate_history_pool_counters(sample, unbounded=unbounded)
    _validate_history_rejection(sample)


def _history_reservations(observation: dict, *, unbounded: bool = False) -> None:
    pools, phases = observation.get("pools"), observation.get("reservation_phases")
    pool_names = ("fresh", "history", "restore_old", "restore_new")
    if (
        not isinstance(pools, list)
        or len(pools) != len(pool_names)
        or not isinstance(phases, list)
        or len(phases) != len(HISTORY_PHASES)
    ):
        raise ValueError("missing fixed reservation phases or distinct pools")
    previous = {}
    for item, (phase, name) in zip(phases, HISTORY_PHASES, strict=True):
        _validate_reservation_phase(item, phase, name, previous, unbounded=unbounded)
        previous[name] = item
    for item, name in zip(pools, pool_names, strict=True):
        _validate_final_history_pool(item, name, previous, unbounded=unbounded)


def validate_native_component(observation: dict) -> dict:
    """Validate the fixed-size native checkpoint measurement boundary."""
    if observation.get("scenario") == "join_sealed":
        return validate_small_batch(observation)
    if (
        observation.get("schema") != "calc-flow.native-component.v1"
        or observation.get("checkpoint_enabled") is not True
        or observation.get("check_only", False) is not False
    ):
        raise ValueError("invalid native component contract")
    if observation.get("case") == "join_compaction_1m":
        return _validate_compaction(observation)
    if observation.get("case") != "join_capture_restore_1m":
        raise ValueError("invalid native component case")
    _validate_capture_rows(observation)
    for field in (
        "snapshot_seconds",
        "preparation_seconds",
        "capture_seconds",
        "restore_seconds",
    ):
        _positive_seconds(observation, field)
    _validate_lifecycle(observation)
    _validate_capture_snapshot(observation)
    return {
        **observation,
        "capture_us_per_row": observation["snapshot_seconds"]
        * 1_000_000
        / observation["captured_dirty_rows"],
        "restore_us_per_row": observation["restore_seconds"]
        * 1_000_000
        / observation["restored_live_rows"],
    }


def _positive_seconds(observation: dict, field: str) -> float:
    value = observation.get(field)
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid native component clock: {field}")
    return value


def _validate_compaction(observation: dict) -> dict:
    if (
        observation.get("resident_rows") != 1_000_000
        or observation.get("incoming_rows") != 500
    ):
        raise ValueError("invalid native compaction row denominator")
    if any(
        observation.get(field) is not True
        for field in ("outputs_equal", "state_equal", "restore_output_equivalent")
    ):
        raise ValueError("missing native compaction equivalence proof")
    for mode in ("armed", "steady"):
        _validate_compaction_sample(observation.get(mode), mode)
    return {
        **observation,
        "handler_ratio": observation["armed"]["handler_seconds"]
        / observation["steady"]["handler_seconds"],
    }


def _validate_compaction_sample(sample: dict, mode: str) -> None:
    _validate_compaction_rows(sample)
    _validate_lifecycle(sample)
    for field in ("preparation_seconds", "handler_seconds", "combined_seconds"):
        _positive_seconds(sample, field)
    if (
        sample["combined_seconds"] + 1e-12
        < sample["preparation_seconds"] + sample["handler_seconds"]
    ):
        raise ValueError("combined compaction clock omits work")
    _validate_compaction_witness(sample, mode)
    if "schedule" in sample:
        validate_serial_compaction_schedule(sample)


def validate_serial_compaction_schedule(sample: dict) -> None:
    """Check real preparation polling and serialized dispatch, never infer overlap."""
    schedule = sample.get("schedule", {})
    if not isinstance(schedule, dict):
        raise ValueError("invalid serialized compaction schedule")
    _validate_serial_compaction_owner(schedule)
    times = _serial_compaction_times(schedule)
    _validate_serial_compaction_clocks(sample, times)


def _validate_serial_compaction_owner(schedule: dict) -> None:
    if (
        schedule.get("owner") != "single-exclusive-operator"
        or schedule.get("during_compaction_status") != "unmeasured"
        or type(schedule.get("preparation_first_poll_pending")) is not bool
        or not isinstance(schedule.get("reason"), str)
        or not schedule["reason"]
    ):
        raise ValueError("invalid serialized compaction schedule")


def _serial_compaction_times(schedule: dict) -> list[int]:
    fields = (
        "preparation_started_ns",
        "handler_ready_ns",
        "preparation_completed_ns",
        "handler_started_ns",
        "handler_completed_ns",
    )
    times = [schedule.get(field) for field in fields]
    if any(type(value) is not int or value < 0 for value in times):
        raise ValueError("invalid serialized compaction schedule")
    if times != sorted(times) or times[0] != 0 or times[3] == times[4]:
        raise ValueError("invalid serialized compaction schedule")
    return times


def _validate_serial_compaction_clocks(sample: dict, times: list[int]) -> None:
    queue = sample.get("queue_seconds")
    if (
        type(queue) not in (float, int)
        or not math.isfinite(queue)
        or not math.isclose(queue, (times[3] - times[1]) / 1e9, abs_tol=1e-12)
        or sample["combined_seconds"] < times[4] / 1e9
    ):
        raise ValueError("compaction queue/combined clocks disagree with schedule")


def validate_join_layout_controls(observation: dict) -> dict:
    """Require each actual Join/ASOF layout execution and its full oracle."""
    expected, controls = _layout_control_inventory(observation)
    seen = set()
    for control in controls:
        if not isinstance(control, dict):
            raise ValueError("invalid native layout control")
        identity = (control.get("operator"), control.get("layout"))
        if identity not in expected or identity in seen:
            raise ValueError("unexpected or repeated native layout control")
        seen.add(identity)
        _require_refusal_fields(control, _expected_layout_counts(*identity))
    return dict(observation)


def _layout_control_inventory(observation: dict) -> tuple[set[tuple[str, str]], list]:
    expected = {
        (operator, layout)
        for operator in ("join", "asof")
        for layout in ("single_record", "multi_record", "unordered", "wide_key")
    }
    controls = observation.get("controls")
    if (
        observation.get("schema") != "calc-flow.join-layout-controls.v1"
        or observation.get("check_only") is not True
        or not isinstance(controls, list)
        or len(controls) != len(expected)
    ):
        raise ValueError("missing native layout control inventory")
    return expected, controls


def _expected_layout_counts(operator: str, layout: str) -> dict:
    geometry = [7, 9, 8] if layout == "multi_record" else [24]
    key_width = 1024 if layout == "wide_key" else 8
    unordered = layout == "unordered"
    return {
        "data_callbacks": 2,
        "input_rows": 48,
        "output_rows": 144 if unordered and operator == "join" else 24,
        "event_record_batch_rows": [geometry, geometry],
        "event_record_batch_count": len(geometry),
        "timestamp_descents_per_event": 23 if unordered else 0,
        "per_key_timestamp_descents": 20 if unordered else 0,
        "unique_keys": 4 if unordered else 24,
        "key_bytes_per_event": 24 * key_width,
        "max_key_bytes": key_width,
        "validated_all_rows": True,
        "input_unchanged": True,
        "owned_cleanup_complete": True,
    }


def _validate_lifecycle(observation: dict) -> None:
    for field in ("setup_seconds", "oracle_seconds", "cleanup_seconds"):
        _positive_seconds(observation, field)
    batches = observation.get("admission_batch_rows")
    if (
        not isinstance(batches, list)
        or any(type(rows) is not int or not 0 < rows <= 64_000 for rows in batches)
        or sum(batches) != 1_000_000
    ):
        raise ValueError("invalid native admission batch geometry")
    if observation.get("owned_cleanup_complete") is not True:
        raise ValueError("native component did not settle owned work")


def validate_small_batch(observation: dict) -> dict:
    """Keep full 1024-row Join handlers separate from the 576-row final tail."""
    from statistics import median

    from scripts.benchmark_suite.asof_e2e import validate_matched_observation
    from scripts.toolkit import linear_percentile

    validated = validate_matched_observation(observation, "join_sealed")
    if not _has_row_count(observation, "dimension_rows", 64):
        raise ValueError("invalid static Join dimension size")
    samples = observation.get("handler_samples")
    if not isinstance(samples, list) or len(samples) != 977:
        raise ValueError("invalid static Join callback count")
    for index, sample in enumerate(samples):
        _validate_small_handler(sample, index)
    if sum(sample["seconds"] for sample in samples) > observation["processing_seconds"]:
        raise ValueError("Join handler clocks exceed enclosing processing clock")
    seconds = [sample["seconds"] for sample in samples[:-1]]
    return {
        **validated,
        "handler_seconds": median(seconds),
        "handler_median_us": median(seconds) * 1_000_000,
        "handler_p95_us": linear_percentile(seconds, 0.95) * 1_000_000,
        "handler_p99_us": linear_percentile(seconds, 0.99) * 1_000_000,
        "full_handler_samples": samples[:-1],
        "partial_handler_samples": samples[-1:],
    }


def _validate_history_geometry(observation: dict, *, unbounded: bool = False) -> None:
    _validate_history_pool_policy(observation, unbounded=unbounded)
    for field, expected in (
        ("left_rows", 4_000_000),
        ("dimension_rows", 64),
        ("resident_rows", 4_000_064),
        ("batch_rows", 64_000),
    ):
        _history_count(observation.get(field), expected, field)
    expected_spec = {
        "join_type": "inner",
        "left_keys": ["symbol"],
        "right_keys": ["symbol"],
        "left_event_time": "event_time",
        "right_event_time": "event_time",
        "bounds": {"before_micros": 62_499_000_001, "after_micros": 0},
        "limits": {
            "max_state_rows_per_side": 4_000_064,
            "max_state_bytes_per_side": 4_096_065_536,
            "max_matches_per_input_batch": 64_064,
        },
        "left_prefix": "quote",
        "right_prefix": "dimension",
    }
    if observation.get("join_spec") != expected_spec:
        raise ValueError("checkpoint history Join specification differs from row3")
    for section in ("limits", "bounds"):
        for field, expected in expected_spec[section].items():
            _history_count(observation["join_spec"][section][field], expected, field)
    expected_fixture = {
        "entities": 64,
        "base_micros": 1_767_225_600_000_000,
        "left_schema": [
            ["event_time", "timestamp_us_utc", False],
            ["sequence", "uint64", False],
            ["symbol", "string", False],
            ["price", "float64", False],
        ],
        "right_schema": [
            ["symbol", "string", False],
            ["factor", "float64", False],
            ["sequence", "uint64", False],
            ["event_time", "timestamp_us_utc", False],
        ],
    }
    if observation.get("fixture") != expected_fixture:
        raise ValueError("checkpoint history fixture differs from row3")


def _validate_history_versions(history: dict) -> dict:
    _history_count(history.get("dirty_epochs"), 4, "dirty epochs")
    checkpoints = history.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 4:
        raise ValueError("missing four dirty checkpoint epochs")
    for index, checkpoint in enumerate(checkpoints):
        _history_snapshot(
            checkpoint, epoch=index + 1, rows=3_999_997 + index, replacement=False
        )
    before, after = history.get("before"), history.get("after")
    _history_snapshot(after, epoch=5, rows=4_000_000, replacement=True)
    if before != checkpoints[-1] or before["base_sha256"] == after["base_sha256"]:
        raise ValueError("missing held-history base replacement")
    if (
        history.get("old_segments_unchanged") is not True
        or history.get("old_segments_after") != before["segments"]
    ):
        raise ValueError("held old-history segment bytes changed")
    return before


def _validate_history_alias(history: dict, before: dict) -> None:
    alias = _history_alias(history)
    _history_count(alias.get("snapshot_epoch"), 4, "segment alias epoch")
    member = next(
        (item for item in before["segments"] if item["name"] == alias.get("name")), None
    )
    if (
        member is None
        or not member["name"].startswith("left-payload-")
        or alias.get("sha256") != member["sha256"]
    ):
        raise ValueError("held segment alias differs from E4 inventory")
    _history_count(alias.get("bytes"), member["bytes"], "held segment bytes")


def _validate_snapshot_segments(observation: dict) -> None:
    segments = observation.get("snapshot_segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("missing native snapshot segments")
    names = set()
    total = 0
    for segment in segments:
        name, size = _validated_segment(
            segment, names, "invalid native snapshot segment"
        )
        names.add(name)
        total += size
    if (
        type(observation.get("snapshot_bytes")) is not int
        or total != observation["snapshot_bytes"]
    ):
        raise ValueError("native snapshot segment bytes do not match total")


def _validate_compaction_witness(sample: dict, mode: str) -> None:
    witness = sample.get("compaction_witness")
    if (
        not isinstance(witness, dict)
        or witness.get("setup_dirty_epochs") != 4
        or witness.get("layout_version") not in (1, 2)
    ):
        raise ValueError("invalid native compaction witness")
    before, after = (witness.get(f"{cut}_base_sha256") for cut in ("before", "after"))
    for hashes in (before, after):
        _validate_compaction_bases(hashes)
    before_deltas, after_deltas = (
        witness.get(f"{cut}_deltas") for cut in ("before", "after")
    )
    for deltas in (before_deltas, after_deltas):
        _validate_compaction_deltas(deltas)
    _validate_compaction_change(
        witness, mode, (before, after), (before_deltas, after_deltas)
    )


def _validate_history_header(observation: dict) -> None:
    if (
        not isinstance(observation, dict)
        or observation.get("schema")
        not in (
            "calc-flow.join-checkpoint-history.v1",
            "calc-flow.join-checkpoint-history.v2",
        )
        or any(
            observation.get(field) != expected
            for field, expected in (
                ("case", "join_checkpoint_history_4m"),
                ("outcome", "complete"),
            )
        )
        or any(
            observation.get(field) is not True
            for field in (
                "check_only",
                "checkpoint_enabled",
                "owned_cleanup_complete",
            )
        )
    ):
        raise ValueError("incomplete fixed checkpoint history diagnostic")


def _validate_history_restores(observation: dict, oracle: dict) -> None:
    restores = observation.get("restores")
    if not isinstance(restores, list) or len(restores) != 2:
        raise ValueError("missing both read-only restore oracles")
    for item, name in zip(restores, ("old_history", "replacement"), strict=True):
        if (
            not isinstance(item, dict)
            or item.get("name") != name
            or _history_oracle(item.get("oracle")) != oracle
        ):
            raise ValueError("restored full-state oracle differs")


def _history_segment_inventory(snapshot: dict) -> tuple[list, set]:
    inventory = snapshot.get("segments")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("missing checkpoint history segments")
    names = set()
    for item in inventory:
        name, size = _validated_segment(
            item, names, "invalid checkpoint history segment"
        )
        names.add(name)
    _history_count(
        snapshot.get("segment_bytes"),
        sum(item["bytes"] for item in inventory),
        "segment bytes",
    )
    return inventory, names


def _validate_history_segment_versions(snapshot, inventory, names, epoch, replacement):
    base = {
        item["name"]: item["sha256"]
        for item in inventory
        if item["name"].endswith("-base")
    }
    expected_deltas = _history_delta_names(epoch, replacement)
    if (
        set(base) != {"left-base", "right-base"}
        or snapshot.get("base_sha256") != base
        or _history_deltas_differ(snapshot, names, expected_deltas)
    ):
        raise ValueError("checkpoint history base/delta inventory differs")


def _validate_history_pool_counters(sample: dict, *, unbounded: bool) -> None:
    fields = (
        "reserved_bytes",
        "peak_reserved_bytes",
        "phase_peak_reserved_bytes",
        "successful_grows",
        "successful_shrinks",
        "failed_grows",
    )
    if any(type(sample.get(field)) is not int or sample[field] < 0 for field in fields):
        raise ValueError("invalid reservation pool counters")
    _validate_history_pool_peak(sample, unbounded=unbounded)
    if unbounded and sample["failed_grows"] != 0:
        raise ValueError("unbounded reservation pool reported a rejected reservation")


def _validate_history_rejection(sample: dict) -> None:
    rejection = sample.get("last_rejection")
    if sample["failed_grows"] == 0:
        if rejection is not None:
            raise ValueError("rejection evidence without a failed reservation")
    elif not _valid_history_rejection(rejection, sample["peak_reserved_bytes"]):
        raise ValueError("missing or invalid last reservation rejection")


def _validate_reservation_phase(item, phase, name, previous, *, unbounded):
    _history_pool(item, unbounded=unbounded)
    if item.get("phase") != phase or item.get("pool_id") != name:
        raise ValueError("reservation phase or pool identity differs")
    _validate_reservation_progress(item, name, previous)
    if (
        phase in ("history_source_released", "history_snapshots_released")
        and item["reserved_bytes"] == 0
    ):
        raise ValueError("held snapshot or segment alias lost its reservation")
    if (
        phase == "history_snapshots_released"
        and item["phase_peak_reserved_bytes"] <= item["reserved_bytes"]
    ):
        raise ValueError(
            "snapshot credit did not refund before the final segment alias"
        )


def _validate_capture_snapshot(observation: dict) -> None:
    if observation["snapshot_seconds"] + 1e-12 < (
        observation["preparation_seconds"] + observation["capture_seconds"]
    ):
        raise ValueError("snapshot clock omits required preparation or capture")
    if (
        observation.get("validated_all_segments") is not True
        or observation.get("restore_output_equivalent") is not True
        or type(observation.get("snapshot_layout_version")) is not int
        or observation["snapshot_layout_version"] not in (1, 2)
    ):
        raise ValueError("missing native snapshot validation proof")
    _validate_snapshot_segments(observation)


def _validate_compaction_rows(sample: dict) -> None:
    if not isinstance(sample, dict) or any(
        type(sample.get(field)) is not int or sample[field] != rows
        for field, rows in (
            ("pre_handler_retained_rows", 1_000_000),
            ("post_handler_retained_rows", 1_000_500),
            ("replay_output_rows", 1_000_500),
        )
    ):
        raise ValueError("invalid native compaction actual rows")


def _validate_small_handler(sample: dict, index: int) -> None:
    if (
        not isinstance(sample, dict)
        or not _has_row_count(sample, "rows", 1024 if index < 976 else 576)
        or not _has_row_count(sample, "left_retained_rows", 0)
        or not _has_row_count(sample, "right_retained_rows", 64)
    ):
        raise ValueError("invalid static Join handler geometry or retention")
    _positive_seconds(sample, "seconds")


def _validate_history_pool_policy(observation: dict, *, unbounded: bool) -> None:
    if unbounded:
        if (
            observation.get("pool_policy") != "unbounded"
            or "pool_limit_bytes" not in observation
            or observation["pool_limit_bytes"] is not None
        ):
            raise ValueError(
                "unbounded checkpoint history requires an explicit pool policy"
            )
    else:
        if observation.get("pool_policy", "bounded") != "bounded":
            raise ValueError("legacy checkpoint history requires a bounded pool policy")
        _history_count(observation.get("pool_limit_bytes"), 1 << 30, "pool_limit_bytes")


def _history_alias(history: dict) -> dict:
    alias = history.get("alias_ownership")
    if (
        not isinstance(alias, dict)
        or alias.get("kind") != "StateSegment"
        or alias.get("pool_id") != "history"
        or alias.get("has_owner") is not True
        or alias.get("credit_scope")
        != "per_segment_backing_plus_shared_capture_controls"
    ):
        raise ValueError("missing owned source-pool segment alias")
    return alias


def _validate_compaction_bases(hashes: dict) -> None:
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(name, str)
            or not name.endswith("-base")
            or not _history_digest(digest)
            for name, digest in hashes.items()
        )
    ):
        raise ValueError("invalid native compaction base inventory")


def _validate_compaction_deltas(deltas: list) -> None:
    if (
        not isinstance(deltas, list)
        or any(not isinstance(name, str) or "-delta-" not in name for name in deltas)
        or len(deltas) != len(set(deltas))
    ):
        raise ValueError("invalid native compaction delta inventory")


def _validate_history_native_indexes(snapshot: dict) -> None:
    native_index = snapshot.get("native_index_present")
    if (
        not isinstance(native_index, dict)
        or set(native_index) != {"left", "right"}
        or any(type(present) is not bool for present in native_index.values())
    ):
        raise ValueError("missing native index presence observation")


def _validate_final_history_pool(item, name, previous, *, unbounded):
    _history_pool(item, unbounded=unbounded)
    if (
        item.get("pool_id") != name
        or item["reserved_bytes"] != 0
        or item
        != {key: value for key, value in previous[name].items() if key != "phase"}
    ):
        raise ValueError("pool identity, final refund or final counters differ")


def _validate_capture_rows(observation: dict) -> None:
    for field in (
        "resident_rows",
        "captured_dirty_rows",
        "restored_live_rows",
        "replay_output_rows",
    ):
        if type(observation.get(field)) is not int or observation[field] != 1_000_000:
            raise ValueError(f"invalid actual row denominator: {field}")


def _validate_compaction_change(witness, mode, bases, deltas):
    before, after = bases
    before_deltas, after_deltas = deltas
    if mode == "armed" and (before == after or len(before_deltas) <= len(after_deltas)):
        raise ValueError("armed preparation did not replace the base and fold deltas")
    if mode == "steady" and (
        before != after
        or witness.get("before_base_epoch") != witness.get("after_base_epoch")
    ):
        raise ValueError("steady preparation unexpectedly changed the base")


def _history_delta_names(epoch: int, replacement: bool) -> list[str]:
    expected_deltas = (
        []
        if replacement
        else sorted(
            [f"left-delta-{i}" for i in range(1, epoch + 1)] + ["right-delta-1"]
        )
    )
    return expected_deltas


def _validate_history_pool_peak(sample: dict, *, unbounded: bool) -> None:
    if (
        not 0
        <= sample["reserved_bytes"]
        <= sample["phase_peak_reserved_bytes"]
        <= sample["peak_reserved_bytes"]
        or not unbounded
        and sample["peak_reserved_bytes"] > 1 << 30
        or sample["peak_reserved_bytes"] == 0
        or sample["successful_grows"] == 0
    ):
        raise ValueError("invalid exact reservation peak")


def _validate_reservation_progress(item: dict, name: str, previous: dict) -> None:
    if name in previous and any(
        item[field] < previous[name][field]
        for field in (
            "peak_reserved_bytes",
            "successful_grows",
            "successful_shrinks",
            "failed_grows",
        )
    ):
        raise ValueError("reservation aggregate counters went backward")


def _has_row_count(sample: dict, field: str, rows: int) -> bool:
    return type(sample.get(field)) is int and sample[field] == rows


def _validated_segment(segment: dict, names: set, message: str) -> tuple[str, int]:
    if not isinstance(segment, dict):
        raise ValueError(message)
    name, size = segment.get("name"), segment.get("bytes")
    if not _valid_segment_size(name, size, names) or not _history_digest(
        segment.get("sha256")
    ):
        raise ValueError(message)
    return name, size


def _valid_segment_size(name, size, names: set) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name not in names
        and type(size) is int
        and size > 0
    )


def _valid_history_rejection(rejection: dict, peak: int) -> bool:
    return (
        isinstance(rejection, dict)
        and _valid_rejected_reservation(rejection, peak)
        and isinstance(rejection.get("consumer_name"), str)
        and 0 < len(rejection["consumer_name"]) <= 256
    )


def _valid_rejected_reservation(rejection: dict, peak: int) -> bool:
    return (
        type(rejection.get("additional_bytes")) is int
        and rejection["additional_bytes"] > 0
        and type(rejection.get("reserved_bytes")) is int
        and 0 <= rejection["reserved_bytes"] <= peak
        and rejection.get("limit_bytes") == 1 << 30
        and rejection["additional_bytes"] + rejection["reserved_bytes"] > 1 << 30
    )


def _history_deltas_differ(snapshot: dict, names: set, expected: list[str]) -> bool:
    return (
        snapshot.get("delta_names") != expected
        or sorted(name for name in names if "-delta-" in name) != expected
    )
