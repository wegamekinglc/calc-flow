"""Validate fixed-size native Join component observations."""

from __future__ import annotations

import math


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
    for field in (
        "resident_rows",
        "captured_dirty_rows",
        "restored_live_rows",
        "replay_output_rows",
    ):
        if type(observation.get(field)) is not int or observation[field] != 1_000_000:
            raise ValueError(f"invalid actual row denominator: {field}")
    for field in (
        "snapshot_seconds",
        "preparation_seconds",
        "capture_seconds",
        "restore_seconds",
    ):
        _positive_seconds(observation, field)
    _validate_lifecycle(observation)
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
    segments = observation.get("snapshot_segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("missing native snapshot segments")
    names = set()
    total = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("invalid native snapshot segment")
        name, size, digest = (segment.get(key) for key in ("name", "bytes", "sha256"))
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid native snapshot segment")
        names.add(name)
        total += size
    if (
        type(observation.get("snapshot_bytes")) is not int
        or total != observation["snapshot_bytes"]
    ):
        raise ValueError("native snapshot segment bytes do not match total")
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
        sample = observation.get(mode)
        if not isinstance(sample, dict) or any(
            type(sample.get(field)) is not int or sample[field] != rows
            for field, rows in (
                ("pre_handler_retained_rows", 1_000_000),
                ("post_handler_retained_rows", 1_000_500),
                ("replay_output_rows", 1_000_500),
            )
        ):
            raise ValueError("invalid native compaction actual rows")
        _validate_lifecycle(sample)
        for field in ("preparation_seconds", "handler_seconds", "combined_seconds"):
            _positive_seconds(sample, field)
        if (
            sample["combined_seconds"] + 1e-12
            < sample["preparation_seconds"] + sample["handler_seconds"]
        ):
            raise ValueError("combined compaction clock omits work")
        witness = sample.get("compaction_witness")
        if (
            not isinstance(witness, dict)
            or witness.get("setup_dirty_epochs") != 4
            or witness.get("layout_version") not in (1, 2)
        ):
            raise ValueError("invalid native compaction witness")
        before, after = (
            witness.get(f"{cut}_base_sha256") for cut in ("before", "after")
        )
        for hashes in (before, after):
            if (
                not isinstance(hashes, dict)
                or not hashes
                or any(
                    not isinstance(name, str)
                    or not name.endswith("-base")
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    for name, digest in hashes.items()
                )
            ):
                raise ValueError("invalid native compaction base inventory")
        before_deltas, after_deltas = (
            witness.get(f"{cut}_deltas") for cut in ("before", "after")
        )
        for deltas in (before_deltas, after_deltas):
            if (
                not isinstance(deltas, list)
                or any(
                    not isinstance(name, str) or "-delta-" not in name
                    for name in deltas
                )
                or len(deltas) != len(set(deltas))
            ):
                raise ValueError("invalid native compaction delta inventory")
        if mode == "armed" and (
            before == after or len(before_deltas) <= len(after_deltas)
        ):
            raise ValueError(
                "armed preparation did not replace the base and fold deltas"
            )
        if mode == "steady" and (
            before != after
            or witness.get("before_base_epoch") != witness.get("after_base_epoch")
        ):
            raise ValueError("steady preparation unexpectedly changed the base")
    return {
        **observation,
        "handler_ratio": observation["armed"]["handler_seconds"]
        / observation["steady"]["handler_seconds"],
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
    if (
        type(observation.get("dimension_rows")) is not int
        or observation["dimension_rows"] != 64
    ):
        raise ValueError("invalid static Join dimension size")
    samples = observation.get("handler_samples")
    if not isinstance(samples, list) or len(samples) != 977:
        raise ValueError("invalid static Join callback count")
    for index, sample in enumerate(samples):
        if (
            not isinstance(sample, dict)
            or type(sample.get("rows")) is not int
            or sample["rows"] != (1024 if index < 976 else 576)
            or type(sample.get("left_retained_rows")) is not int
            or sample["left_retained_rows"] != 0
            or type(sample.get("right_retained_rows")) is not int
            or sample["right_retained_rows"] != 64
        ):
            raise ValueError("invalid static Join handler geometry or retention")
        _positive_seconds(sample, "seconds")
    if sum(sample["seconds"] for sample in samples) > observation["processing_seconds"]:
        raise ValueError("Join handler clocks exceed enclosing processing clock")
    seconds = [sample["seconds"] for sample in samples[:-1]]
    return {
        **validated,
        "handler_seconds": median(seconds),
        "handler_median_us": median(seconds) * 1_000_000,
        "handler_p95_us": linear_percentile(seconds, 0.95) * 1_000_000,
        "full_handler_samples": samples[:-1],
        "partial_handler_samples": samples[-1:],
    }
