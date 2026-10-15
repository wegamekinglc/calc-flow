from __future__ import annotations

import unittest
from copy import deepcopy

from scripts.benchmark_suite.native_components import validate_native_component


def dictionary_refusal_observation():
    return {
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
        "pool": {
            "failed_dictionary_grows": 1,
            "successful_sql_grows": 1,
            "last_rejected_consumer": "sql-incremental:stream-join-dictionary",
            "last_rejected_bytes": 1024,
            "reserved_bytes": 0,
        },
        "owned_cleanup_complete": True,
    }


class DictionaryRefusalComponentTests(unittest.TestCase):
    def test_dictionary_refusal_requires_exact_route_and_rebuild_counters(self):
        from scripts.benchmark_suite.native_components import (
            validate_join_dictionary_refusal,
        )

        observation = dictionary_refusal_observation()
        for field, expected in (
            ("dictionary_grant_refusals", 1),
            ("needs_rebuild_routes", 1),
            ("rebuild_attempts", 0),
            ("rebuild_completions", 0),
        ):
            for value in (None, True, -1, expected + 1):
                invalid = {**observation, field: value}
                if field == "dictionary_grant_refusals":
                    invalid["pool"] = {
                        **observation["pool"],
                        "failed_dictionary_grows": value,
                    }
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    validate_join_dictionary_refusal(invalid)

    def test_dictionary_refusal_rejects_boolean_pool_counts(self):
        from scripts.benchmark_suite.native_components import (
            validate_join_dictionary_refusal,
        )

        observation = dictionary_refusal_observation()
        for field in (
            "failed_dictionary_grows",
            "successful_sql_grows",
            "last_rejected_bytes",
            "reserved_bytes",
        ):
            invalid = deepcopy(observation)
            invalid["pool"][field] = field != "reserved_bytes"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_join_dictionary_refusal(invalid)

    def test_dictionary_refusal_requires_retention_and_actual_pool_witness(self):
        from scripts.benchmark_suite import native_components

        observation = dictionary_refusal_observation()
        original = deepcopy(observation)
        result = native_components.validate_join_dictionary_refusal(observation)
        self.assertEqual(result, original)
        self.assertEqual(observation, original)
        changes = (
            ("check_only", False),
            ("admission_batch_rows", [2, True]),
            ("left_retained_rows", 0),
            ("right_retained_bytes", 113),
            ("full_v1_row_charges", [113, 114, 114]),
            ("output_keys", []),
            ("state_limit_failures", 1),
            ("dictionary_grant_refusals", 0),
            ("legacy_key_encodings", 0),
            ("sql_probe_table_builds", 0),
            ("owned_cleanup_complete", False),
        )
        for field, value in changes:
            with self.subTest(field=field), self.assertRaises(ValueError):
                native_components.validate_join_dictionary_refusal(
                    {**observation, field: value}
                )

    def test_dictionary_refusal_rejects_missing_or_forged_pool_evidence(self):
        from scripts.benchmark_suite import native_components

        observation = dictionary_refusal_observation()
        for field, value in (
            ("failed_dictionary_grows", 0),
            ("successful_sql_grows", 0),
            ("last_rejected_consumer", "forced-sql-flag"),
            ("last_rejected_bytes", 0),
            ("reserved_bytes", 1),
        ):
            invalid = deepcopy(observation)
            invalid["pool"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                native_components.validate_join_dictionary_refusal(invalid)
        for field in ("pool", "dictionary_grant_refusals", "full_v1_row_charges"):
            invalid = deepcopy(observation)
            invalid.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                native_components.validate_join_dictionary_refusal(invalid)


def capture_observation():
    return {
        "schema": "calc-flow.native-component.v1",
        "case": "join_capture_restore_1m",
        "checkpoint_enabled": True,
        "resident_rows": 1_000_000,
        "captured_dirty_rows": 1_000_000,
        "restored_live_rows": 1_000_000,
        "snapshot_seconds": 0.05,
        "preparation_seconds": 0.04,
        "capture_seconds": 0.005,
        "restore_seconds": 0.1,
        "snapshot_layout_version": 2,
        "snapshot_bytes": 128,
        "snapshot_segments": [{"name": "left-base", "bytes": 128, "sha256": "a" * 64}],
        "validated_all_segments": True,
        "restore_output_equivalent": True,
        "replay_output_rows": 1_000_000,
        "admission_batch_rows": [64_000] * 7 + [52_000] + [64_000] * 7 + [52_000],
        "setup_seconds": 0.2,
        "oracle_seconds": 0.3,
        "cleanup_seconds": 0.01,
        "owned_cleanup_complete": True,
    }


def compaction_observation():
    def sample(armed):
        return {
            "pre_handler_retained_rows": 1_000_000,
            "post_handler_retained_rows": 1_000_500,
            "replay_output_rows": 1_000_500,
            "admission_batch_rows": [64_000] * 7
            + [51_997]
            + [64_000] * 7
            + [51_997]
            + [1] * 6,
            "setup_seconds": 0.2,
            "oracle_seconds": 0.3,
            "cleanup_seconds": 0.01,
            "owned_cleanup_complete": True,
            "preparation_seconds": 0.02,
            "handler_seconds": 0.0015 if armed else 0.001,
            "combined_seconds": 0.03,
            "compaction_witness": {
                "setup_dirty_epochs": 4,
                "layout_version": 2,
                "before_base_epoch": 0 if armed else 4,
                "after_base_epoch": 4,
                "before_base_sha256": {"left-base": ("a" if armed else "b") * 64},
                "after_base_sha256": {"left-base": "b" * 64},
                "before_deltas": [
                    "left-delta-1",
                    "left-delta-2",
                    "left-delta-3",
                    "left-delta-4",
                ]
                if armed
                else [],
                "after_deltas": ["left-delta-5"],
            },
        }

    return {
        "schema": "calc-flow.native-component.v1",
        "case": "join_compaction_1m",
        "checkpoint_enabled": True,
        "resident_rows": 1_000_000,
        "incoming_rows": 500,
        "armed": sample(True),
        "steady": sample(False),
        "outputs_equal": True,
        "state_equal": True,
        "restore_output_equivalent": True,
    }


class NativeComponentTests(unittest.TestCase):
    def test_compaction_primary_ratio_is_handler_only_and_requires_witness(self):
        observation = compaction_observation()
        original = deepcopy(observation)
        result = validate_native_component(observation)
        self.assertEqual(result["handler_ratio"], 1.5)
        self.assertEqual(observation, original)
        for field, value in (
            ("resident_rows", 1_000_006),
            ("incoming_rows", 499),
            ("state_equal", False),
            ("check_only", True),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_native_component({**observation, field: value})
        for mode in ("armed", "steady"):
            invalid = deepcopy(observation)
            invalid[mode]["compaction_witness"]["after_base_sha256"] = {
                "left-base": "a" * 64
            }
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_native_component(invalid)
        for mode in ("armed", "steady"):
            for field, value in (
                ("pre_handler_retained_rows", 1_000_000.0),
                ("owned_cleanup_complete", False),
                ("admission_batch_rows", [500_000, 500_000]),
                ("cleanup_seconds", None),
            ):
                invalid = deepcopy(observation)
                invalid[mode][field] = value
                with (
                    self.subTest(mode=mode, field=field),
                    self.assertRaises(ValueError),
                ):
                    validate_native_component(invalid)

    def test_capture_uses_actual_rows_and_includes_preparation(self):
        observation = capture_observation()
        original = deepcopy(observation)

        result = validate_native_component(observation)

        self.assertEqual(result["capture_us_per_row"], 0.05)
        self.assertEqual(result["restore_us_per_row"], 0.1)
        self.assertEqual(observation, original)
        for field, value in (
            ("resident_rows", 999_999),
            ("captured_dirty_rows", 1_000_006),
            ("restored_live_rows", 500_000),
            ("snapshot_seconds", 0.005),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_native_component({**observation, field: value})

    def test_capture_rejects_missing_or_inconsistent_recovery_proof(self):
        observation = capture_observation()
        for field, value in (
            ("validated_all_segments", False),
            ("restore_output_equivalent", False),
            ("snapshot_layout_version", 3),
            ("snapshot_bytes", 129),
            ("snapshot_segments", []),
            (
                "snapshot_segments",
                [{"name": "left-base", "bytes": 128, "sha256": "bad"}],
            ),
            ("snapshot_segments", observation["snapshot_segments"] * 2),
            ("replay_output_rows", 999_999),
            ("replay_output_rows", 1_000_000.0),
            ("admission_batch_rows", [500_000, 500_000]),
            ("admission_batch_rows", [64_000]),
            ("owned_cleanup_complete", False),
            ("setup_seconds", None),
            ("oracle_seconds", -1),
            ("cleanup_seconds", 0),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_native_component({**observation, field: value})


def matched_observation(scenario="asof_join"):
    from scripts.toolkit import fingerprint_json

    projection = [
        {
            "node_id": "join",
            "kind": "stream_asof_join" if scenario == "asof_join" else "stream_join",
            "input_fields": {"left": ["key"], "right": ["key"]},
            "output_fields": ["sequence", "value"],
            "configuration": {},
        }
    ]
    observation = {
        "schema": "calc-flow.matched-native-observation.v1",
        "scenario": scenario,
        "check_only": False,
        "checkpoint_enabled": False,
        "nonterminal_epochs": 0,
        "rows_per_side": 1_000_000,
        "rows": 1_000_000,
        "dimension_rows": 64,
        "batch_rows": 1024,
        "output_budget": {"max_rows": 1024, "max_bytes": 64 << 20},
        "setup_seconds": 0.1,
        "processing_seconds": 0.2,
        "terminal_seconds": 0.01,
        "oracle_seconds": 0.1,
        "cleanup_seconds": 0.01,
        "output_rows": 1_000_000,
        "assembled_output_rows": 1_000_000,
        "assembled_output_batches": 977,
        "validated_all_rows": True,
        "owned_cleanup_complete": True,
        "compiled_projection": projection,
        "compiled_projection_sha256": fingerprint_json(projection),
        "post_processing_status": {"join": {"pending_left_rows": 0}},
        "callback_counts": {"data": 2931},
        "node_callback_counts": {"join": {"data": 1954}},
        "observed_cpu_affinity": [0],
        "observed_thread_environment": {
            key: "1"
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
        },
        "process_id": 42,
        **{
            key: "a" * 64
            for key in (
                "project_sha256",
                "inputs_sha256",
                "events_sha256",
                "shared_partial_order_sha256",
                "canonical_native_events_sha256",
                "output_contract_sha256",
                "compiled_plan_fingerprint",
            )
        },
    }
    if scenario == "join_sealed":
        observation["handler_samples"] = [
            {
                "rows": 1024,
                "seconds": 0.0001,
                "left_retained_rows": 0,
                "right_retained_rows": 64,
            }
            for _ in range(976)
        ] + [
            {
                "rows": 576,
                "seconds": 0.00008,
                "left_retained_rows": 0,
                "right_retained_rows": 64,
            }
        ]
    return observation


class MatchedNativeTests(unittest.TestCase):
    def test_tiny_correctness_replay_is_separate_from_measurement_samples(self):
        from scripts.benchmark_suite.asof_e2e import (
            validate_matched_correctness,
            validate_matched_native,
        )

        observation = {
            **matched_observation(),
            "check_only": True,
            "rows_per_side": 3073,
            "output_rows": 3073,
            "assembled_output_rows": 3073,
            "assembled_output_batches": 4,
        }
        self.assertTrue(validate_matched_correctness(observation)["check_only"])
        with self.assertRaises(ValueError):
            validate_matched_native(observation)
        with self.assertRaises(ValueError):
            validate_matched_correctness(matched_observation())
        with self.assertRaises(ValueError):
            validate_matched_correctness({**observation, "rows_per_side": 2049})

    def test_asof_requires_matched_identity_scope_and_clean_lifecycle(self):
        from scripts.benchmark_suite.asof_e2e import validate_matched_native

        observation = matched_observation()
        original = deepcopy(observation)
        result = validate_matched_native(observation)
        self.assertEqual(result["processing_seconds"], 0.2)
        self.assertEqual(result["nonterminal_epochs"], [])
        self.assertTrue(result["processing_checkpoint_free"])
        self.assertTrue(result["processing_settled"])

        self.assertIsNot(result, observation)
        self.assertEqual(observation, original)
        for key, value in (
            ("rows_per_side", 100_000),
            ("check_only", True),
            ("nonterminal_epochs", 1),
            ("owned_cleanup_complete", False),
            ("compiled_projection_sha256", "b" * 64),
            ("terminal_seconds", None),
            ("observed_cpu_affinity", []),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_matched_native({**observation, key: value})

    def test_join_separates_full_handlers_and_partial_tail(self):
        observation = matched_observation("join_sealed")
        original = deepcopy(observation)
        result = validate_native_component(observation)
        self.assertEqual(result["handler_median_us"], 100.0)
        self.assertEqual(result["handler_p95_us"], 100.0)
        self.assertEqual(
            result["partial_handler_samples"], observation["handler_samples"][-1:]
        )
        self.assertEqual(observation, original)
        for key, value in (("rows", 576), ("left_retained_rows", 1), ("seconds", 0)):
            bad = deepcopy(observation)
            bad["handler_samples"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_component(bad)


def history_observation():
    from scripts.benchmark_suite.native_components import HISTORY_PHASES

    def pool(name, current=0):
        return {
            "pool_id": name,
            "limit_bytes": 1 << 30,
            "reserved_bytes": current,
            "peak_reserved_bytes": 900_000_000,
            "phase_peak_reserved_bytes": 800_000_000,
            "successful_grows": 20,
            "successful_shrinks": 20,
            "failed_grows": 0,
            "last_rejection": None,
        }

    def snapshot(epoch, left_rows, replacement=False):
        suffix = "b" if replacement else "a"
        segments = [
            {"name": "left-base", "bytes": 1, "sha256": suffix * 64},
            {"name": "right-base", "bytes": 1, "sha256": "c" * 64},
            {"name": "left-payload-" + "e" * 64, "bytes": 1, "sha256": "e" * 64},
        ]
        names = (
            []
            if replacement
            else sorted(
                [f"left-delta-{i}" for i in range(1, epoch + 1)] + ["right-delta-1"]
            )
        )
        segments.extend(
            {"name": name, "bytes": 1, "sha256": "d" * 64} for name in names
        )
        return {
            "epoch": epoch,
            "left_rows": left_rows,
            "right_rows": 64,
            "left_watermark_micros": 1_767_288_099_000_000 + (left_rows == 4_000_000),
            "right_watermark_micros": None,
            "native_index_present": {"left": True, "right": False},
            "base_epoch": 4 if replacement else 0,
            "base_sha256": {
                s["name"]: s["sha256"] for s in segments if s["name"].endswith("-base")
            },
            "delta_names": names,
            "segments": segments,
            "segment_bytes": len(segments),
        }

    oracle = {
        "validated_all_rows": True,
        "left_rows": 4_000_000,
        "right_rows": 64,
        "left_sha256": "1" * 64,
        "right_sha256": "2" * 64,
    }
    checkpoints = [snapshot(i + 1, 3_999_997 + i) for i in range(4)]
    before = checkpoints[-1]
    after = snapshot(5, 4_000_000, True)
    phases = [
        {
            "phase": phase,
            **pool(
                name,
                123
                if phase in ("history_source_released", "history_snapshots_released")
                else 0,
            ),
        }
        for phase, name in HISTORY_PHASES
    ]
    return {
        "schema": "calc-flow.join-checkpoint-history.v1",
        "case": "join_checkpoint_history_4m",
        "outcome": "complete",
        "check_only": True,
        "checkpoint_enabled": True,
        "left_rows": 4_000_000,
        "dimension_rows": 64,
        "resident_rows": 4_000_064,
        "batch_rows": 64_000,
        "pool_limit_bytes": 1 << 30,
        "join_spec": {
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
        },
        "fixture": {
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
        },
        "fresh": {
            "oracle": oracle,
            "repeated_capture_segments_equal": True,
            "output_rows": 4_000_000,
        },
        "history": {
            "dirty_epochs": 4,
            "checkpoints": checkpoints,
            "before": before,
            "after": after,
            "old_segments_after": deepcopy(before["segments"]),
            "old_segments_unchanged": True,
            "alias_ownership": {
                "kind": "StateSegment",
                "pool_id": "history",
                "snapshot_epoch": 4,
                "name": "left-payload-" + "e" * 64,
                "bytes": 1,
                "sha256": "e" * 64,
                "has_owner": True,
                "credit_scope": "per_segment_backing_plus_shared_capture_controls",
            },
            "oracle": oracle,
            "output_rows": 4_000_000,
        },
        "restores": [
            {"name": name, "oracle": oracle} for name in ("old_history", "replacement")
        ],
        "reservation_phases": phases,
        "pools": [
            pool(name) for name in ("fresh", "history", "restore_old", "restore_new")
        ],
        "owned_cleanup_complete": True,
    }


class JoinCheckpointHistoryTests(unittest.TestCase):
    def validate(self, observation):
        from scripts.benchmark_suite.native_components import (
            validate_join_checkpoint_history,
        )

        return validate_join_checkpoint_history(observation)

    def test_history_pool_requires_consistent_bounded_rejection_evidence(self):
        from scripts.benchmark_suite.native_components import _history_pool

        sample = history_observation()["pools"][0]
        sample["failed_grows"] = 1
        with self.assertRaises(ValueError):
            _history_pool(sample)
        sample["last_rejection"] = {
            "additional_bytes": 1 << 30,
            "reserved_bytes": 123,
            "limit_bytes": 1 << 30,
            "consumer_name": "join-checkpoint",
        }
        _history_pool(sample)
        for field, value in (
            ("additional_bytes", 0),
            ("reserved_bytes", -1),
            ("limit_bytes", 123),
            ("consumer_name", "x" * 257),
        ):
            invalid = deepcopy(sample)
            invalid["last_rejection"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _history_pool(invalid)

    def test_history_accepts_complete_fixed_evidence_without_mutating_it(self):
        observation = history_observation()
        original = deepcopy(observation)
        self.assertEqual(self.validate(observation), observation)
        self.assertEqual(observation, original)

    def unbounded_observation(self):
        observation = history_observation()
        observation.update(
            schema="calc-flow.join-checkpoint-history.v2",
            pool_policy="unbounded",
            pool_limit_bytes=None,
        )
        for sample in observation["pools"] + observation["reservation_phases"]:
            sample.update(
                limit_bytes=None,
                peak_reserved_bytes=2 << 30,
                phase_peak_reserved_bytes=(1 << 30) + 123,
            )
        return observation

    def test_unbounded_history_accepts_peaks_above_one_gib_without_mutation(self):
        observation = self.unbounded_observation()
        original = deepcopy(observation)
        self.assertEqual(self.validate(observation), observation)
        self.assertEqual(observation, original)

    def test_unbounded_history_requires_explicit_policy_and_null_limits(self):
        mutations = (
            lambda d: d.pop("pool_policy"),
            lambda d: d.update(pool_policy="bounded"),
            lambda d: d.pop("pool_limit_bytes"),
            *(
                lambda d, value=value: d.update(pool_limit_bytes=value)
                for value in (0, 1 << 30, (1 << 64) - 1, False, "null")
            ),
            lambda d: d.update(schema="calc-flow.join-checkpoint-history.v3"),
        )
        for index, mutate in enumerate(mutations):
            invalid = self.unbounded_observation()
            mutate(invalid)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(invalid)
        for collection in ("pools", "reservation_phases"):
            invalid = self.unbounded_observation()
            invalid[collection][0].pop("limit_bytes")
            with self.subTest(collection=collection), self.assertRaises(ValueError):
                self.validate(invalid)
            for limit in (0, 1 << 30, (1 << 64) - 1, False, "null"):
                invalid = self.unbounded_observation()
                invalid[collection][0]["limit_bytes"] = limit
                with (
                    self.subTest(collection=collection, limit=limit),
                    self.assertRaises(ValueError),
                ):
                    self.validate(invalid)

    def test_unbounded_history_retains_reservation_oracle_and_cleanup_checks(self):
        mutations = (
            lambda d: d["pools"][0].update(failed_grows=1),
            lambda d: d["pools"][0].update(last_rejection={}),
            lambda d: d["pools"][0].update(reserved_bytes=1),
            lambda d: d["reservation_phases"][0].update(
                phase_peak_reserved_bytes=(2 << 30) + 1
            ),
            lambda d: d["reservation_phases"][0].pop("peak_reserved_bytes"),
            lambda d: d["history"]["oracle"].update(validated_all_rows=False),
            lambda d: d["history"]["alias_ownership"].update(has_owner=False),
            lambda d: d.update(owned_cleanup_complete=False),
        )
        for index, mutate in enumerate(mutations):
            invalid = self.unbounded_observation()
            mutate(invalid)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(invalid)

    def test_legacy_history_cannot_claim_an_unbounded_pool_policy(self):
        invalid = history_observation()
        invalid["pool_policy"] = "unbounded"
        with self.assertRaises(ValueError):
            self.validate(invalid)

    def test_history_requires_explicit_boolean_native_index_observations(self):
        for value in (
            None,
            {"left": True},
            {"left": 1, "right": False},
            {"left": False, "right": False, "other": False},
        ):
            invalid = history_observation()
            invalid["history"]["checkpoints"][0]["native_index_present"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(invalid)

    def test_history_requires_snapshot_refund_before_the_final_segment_alias(self):
        invalid = history_observation()
        sample = next(
            item
            for item in invalid["reservation_phases"]
            if item["phase"] == "history_snapshots_released"
        )
        sample["phase_peak_reserved_bytes"] = sample["reserved_bytes"]
        with self.assertRaises(ValueError):
            self.validate(invalid)

    def test_history_requires_a_real_owned_e4_segment_alias(self):
        for field, value in (
            ("name", "missing"),
            ("sha256", "f" * 64),
            ("bytes", 2),
            ("has_owner", False),
            ("pool_id", "restore_new"),
            ("credit_scope", "segment_bytes"),
            ("credit_scope", "shared_encoder_owner"),
            ("snapshot_epoch", 5),
        ):
            invalid = history_observation()
            invalid["history"]["alias_ownership"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(invalid)

    def test_history_rejects_reduced_fixture_changed_limits_and_false_completion(self):
        for field, value in (
            ("left_rows", 3072),
            ("resident_rows", 4_000_000),
            ("outcome", "failed"),
            ("check_only", False),
            ("checkpoint_enabled", False),
            ("owned_cleanup_complete", False),
        ):
            invalid = history_observation()
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(invalid)
        invalid = history_observation()
        invalid["join_spec"]["limits"]["max_state_bytes_per_side"] += 1
        with self.assertRaises(ValueError):
            self.validate(invalid)

    def test_history_rejects_wrong_progress_state_restore_and_old_segment_witness(self):
        mutations = (
            lambda d: d["history"]["checkpoints"][0].update(
                left_watermark_micros=1_767_288_099_000_001
            ),
            lambda d: d["history"]["before"].update(right_watermark_micros=1),
            lambda d: d["history"].update(dirty_epochs=3),
            lambda d: d["history"].update(after=deepcopy(d["history"]["before"])),
            lambda d: d["history"]["old_segments_after"][0].update(sha256="e" * 64),
            lambda d: d["restores"][0].update(
                oracle={**d["fresh"]["oracle"], "left_sha256": "f" * 64}
            ),
            lambda d: d["history"]["oracle"].update(validated_all_rows=False),
        )
        for index, mutate in enumerate(mutations):
            invalid = history_observation()
            mutate(invalid)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(invalid)

    def test_history_rejects_missing_peaks_shared_pool_recount_and_incomplete_refunds(
        self,
    ):
        mutations = (
            lambda d: d["pools"][0].pop("peak_reserved_bytes"),
            lambda d: d["pools"][0].update(reserved_bytes=1),
            lambda d: d["pools"][0].update(peak_reserved_bytes=(1 << 30) + 1),
            lambda d: d["pools"][1].update(pool_id="fresh"),
            lambda d: d["reservation_phases"].pop(),
            lambda d: d["reservation_phases"][0].update(
                phase_peak_reserved_bytes=950_000_000
            ),
            lambda d: d["pools"][0].update(failed_grows=-1),
        )
        for index, mutate in enumerate(mutations):
            invalid = history_observation()
            mutate(invalid)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(invalid)


if __name__ == "__main__":
    unittest.main()
