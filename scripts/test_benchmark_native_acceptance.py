from __future__ import annotations

import unittest
from copy import deepcopy

from scripts.benchmark_suite.native_components import validate_native_component


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


if __name__ == "__main__":
    unittest.main()
