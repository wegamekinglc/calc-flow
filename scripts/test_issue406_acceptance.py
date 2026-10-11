from __future__ import annotations

import copy
import unittest


class AcceptanceInventoryTests(unittest.TestCase):
    def test_inventory_preserves_every_target_and_true_retained_sizes(self):
        from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory

        cases = acceptance_inventory()
        self.assertEqual({case.row for case in cases}, set(range(1, 9)))
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assert_retained_inventory(cases)
        self.assertTrue(all(case.rows == 1_000_000 for case in cases if case.row == 7))

    def assert_retained_inventory(self, cases):
        retained = [case for case in cases if case.row == 3]
        self.assertEqual({case.rows for case in retained}, {1_000_000, 4_000_000})
        self.assertEqual({case.workload for case in retained}, {"join_retained"})
        self.assertTrue(all(case.batch_rows == 64_000 for case in retained))

    def test_checkpoint_low_is_enabled_and_main_keeps_all_supported_cells(self):
        from scripts.benchmark_suite.issue406_acceptance import (
            acceptance_inventory,
            checkpoint_settings,
        )

        self.assertEqual(
            checkpoint_settings("off"),
            {"checkpointing": False, "checkpoint_interval_millis": None},
        )
        self.assertEqual(
            checkpoint_settings("low"),
            {"checkpointing": True, "checkpoint_interval_millis": 86_400_000},
        )
        self.assertEqual(checkpoint_settings("on")["checkpoint_interval_millis"], 100)
        cases = acceptance_inventory()
        for case in cases:
            expected = (
                ("candidate",) if case.mode == "off" else ("baseline", "candidate")
            )
            self.assertEqual(case.sides, expected)
        self.assertEqual(
            {case.row for case in cases if "baseline" in case.sides}, set(range(1, 9))
        )

    def test_eight_paper_rows_are_not_complete_metric_evidence(self):
        from scripts.benchmark_suite.issue406_acceptance import (
            validate_complete_results,
        )

        report = {"rows": [{"row": row, "status": "pending"} for row in range(1, 9)]}
        with self.assertRaisesRegex(ValueError, "missing measurements"):
            validate_complete_results(report)


class AcceptanceObservationReferenceTests(unittest.TestCase):
    @staticmethod
    def shared_records():
        from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory

        cases = [case for case in acceptance_inventory() if case.row == 4]
        members = [case.id for case in cases]

        def sample(index):
            return {
                "observation_id": f"raw/{index}",
                "raw_observation_sha256": "a" * 64,
                "group_member_ids": members,
            }

        return [
            {
                "case": case.id,
                "side": "candidate",
                "warmups": [sample("warm0"), sample("warm1")],
                "rounds": [
                    [sample(f"{round_index}/{index}") for index in range(10)]
                    for round_index in range(2)
                ],
            }
            for case in cases
        ]

    def test_shared_metrics_retain_one_physical_observation(self):
        from scripts.benchmark_suite.issue406_acceptance import (
            validate_observation_references,
        )

        records = self.shared_records()
        validate_observation_references(records)
        records[1]["rounds"][0][0]["raw_observation_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different physical"):
            validate_observation_references(records)

    def test_raw_observation_cannot_be_repeated_as_an_independent_sample(self):
        from scripts.benchmark_suite.issue406_acceptance import (
            validate_observation_references,
        )

        records = self.shared_records()
        for record in records:
            record["rounds"][0][1] = copy.deepcopy(record["rounds"][0][0])
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_observation_references(records)


class MatchedOverheadTests(unittest.TestCase):
    def setUp(self):
        self.identity = {
            "project_sha256": "a" * 64,
            "inputs_sha256": "b" * 64,
            "shared_partial_order_sha256": "c" * 64,
            "output_contract_sha256": "d" * 64,
            "compiled_plan_fingerprint": "e" * 64,
            "rows_per_side": 1_000_000,
            "checkpoint_enabled": False,
            "batch_rows": 1024,
            "mode": "off",
            "cpu_affinity": [2],
            "processing_boundary": "final-output-projection-watermark-completed",
            "output_frontier": "minimum-input-watermark-minus-one-microsecond",
        }

    def _sample(self, seconds, *, replay=False):
        self.identity["checkpoint_enabled"] = self.identity["mode"] != "off"
        sample = {
            "seconds": seconds,
            "match_identity": copy.deepcopy(self.identity),
            "oracle_passed": True,
            "processing_settled": True,
            "nonterminal_epochs": [],
            "processing_checkpoint_free": True,
            "product_commit": "1" * 40,
            "compiled_projection": [{"node_id": "asof"}],
            "compiled_projection_sha256": "f" * 64,
            "canonical_native_events_sha256": "a" * 64,
            "native_executable_sha256": "b" * 64,
            "observation_id": "source-observation",
            "raw_observation_sha256": "7" * 64,
            "dispatch_trace_envelope_sha256": "8" * 64,
            "dispatch_trace": {
                "schema": "calc-flow.asof-dispatch-trace.v1",
                "overflowed": False,
                "terminal_excluded": True,
                "allow_barriers": False,
                "compiled_plan_fingerprint": self.identity["compiled_plan_fingerprint"],
                "node_id": "asof",
                "capacity": 8192,
                "record_count": 1,
                "events": [{"kind": "watermark", "ingress": "left", "micros": 10}],
                "callback_counts": {"data": 2, "ingress_progress": 2, "watermark": 1},
                "completed_frontiers": {"result": 9},
                "operator_data_counts": {"asof": {"input_rows": 2}},
                "asof_state": {"output_watermark_micros": 9},
                "output_record_batch_rows": [[1]],
                "output_node_id": "result",
                "required_output_frontier_micros": 9,
            },
        }
        if replay:
            sample.update(
                dispatch_trace_verified=True,
                dispatch_replay_state_verified=True,
                replay_source_observation_id="source-observation",
                replay_source_raw_sha256="7" * 64,
                replay_trace_sha256="8" * 64,
            )
        return sample

    def _proof(self):
        return {
            "status": "verified",
            "artifact_sha256": "b" * 64,
            "product_commit": "1" * 40,
            "native_executable_sha256": "b" * 64,
            "project_sha256": self.identity["project_sha256"],
            "compiled_projection_sha256": "f" * 64,
            "batch_rows": self.identity["batch_rows"],
            "checkpoint_enabled": self.identity["mode"] != "off",
        }

    def test_matched_overhead_retains_raw_signed_residual(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        result = matched_overhead(
            self._sample(0.2),
            self._sample(0.15, replay=True),
            0.03,
            scheduling_proof=self._proof(),
        )
        self.assertAlmostEqual(result["overhead_seconds"], 0.05)
        self.assertAlmostEqual(result["target_seconds"], 0.06)
        self.assertEqual(result["target_status"], "observed-within-target")
        self.assertEqual(result["schedule_status"], "recorded-dispatch-chain-replayed")
        result = matched_overhead(self._sample(0.1), self._sample(0.15), 0.03)
        self.assertAlmostEqual(result["overhead_seconds"], -0.05)
        self.assertEqual(result["target_status"], "inconclusive")

    def test_unmatched_or_unsettled_work_cannot_supply_overhead(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        for field, value in (
            ("rows_per_side", 100_000),
            ("shared_partial_order_sha256", "e" * 64),
        ):
            with self.subTest(field=field):
                native = self._sample(0.15)
                native["match_identity"][field] = value
                with self.assertRaisesRegex(ValueError, "matching"):
                    matched_overhead(self._sample(0.2), native, 0.03)
        for field in ("oracle_passed", "processing_settled"):
            native = self._sample(0.15)
            native[field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                matched_overhead(self._sample(0.2), native, 0.03)

    def test_unproved_schedule_stays_inconclusive_and_overlap_blocks_residual(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        result = matched_overhead(self._sample(0.2), self._sample(0.15), 0.03)
        self.assertEqual(result["target_status"], "inconclusive")
        self.assertEqual(result["schedule_status"], "schedule-dependent-estimate")
        sample = self._sample(0.2)
        sample["processing_checkpoint_free"] = False
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            matched_overhead(sample, self._sample(0.15), 0.03)
        sample = self._sample(0.15)
        sample["product_commit"] = "2" * 40
        with self.assertRaisesRegex(ValueError, "revision"):
            matched_overhead(self._sample(0.2), sample, 0.03)

    def test_verified_observable_rendezvous_does_not_prove_equal_timed_work(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        proof = {
            **self._proof(),
            "rendezvous_equivalent": True,
            "callback_traces_identical": False,
            "claim": "observable rendezvous equivalence; no CPU-work equivalence claim",
        }
        result = matched_overhead(
            self._sample(0.2), self._sample(0.15), 0.03, scheduling_proof=proof
        )
        self.assertEqual(result["target_status"], "inconclusive")
        self.assertEqual(result["schedule_status"], "schedule-dependent-estimate")

    def test_actual_replay_requires_exact_source_observation_and_trace_hash(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        for field, value in (
            ("replay_source_observation_id", "another-source"),
            ("replay_source_raw_sha256", "0" * 64),
            ("replay_trace_sha256", "0" * 64),
        ):
            native = self._sample(0.15, replay=True)
            native[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "replay"),
            ):
                matched_overhead(self._sample(0.2), native, 0.03)

    def test_replay_rejects_missing_or_corrupted_returned_state_and_trace(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        for field in (
            "events",
            "callback_counts",
            "completed_frontiers",
            "operator_data_counts",
            "asof_state",
            "output_record_batch_rows",
        ):
            native = self._sample(0.15, replay=True)
            native["dispatch_trace"][field] = None
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "replay"),
            ):
                matched_overhead(self._sample(0.2), native, 0.03)
        for field in ("dispatch_trace", "dispatch_replay_state_verified"):
            native = self._sample(0.15, replay=True)
            native.pop(field)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "replay"),
            ):
                matched_overhead(self._sample(0.2), native, 0.03)

    def test_stale_or_wrong_context_schedule_proof_is_rejected(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        for key, value in (
            ("product_commit", "2" * 40),
            ("native_executable_sha256", "c" * 64),
            ("batch_rows", 64_000),
            ("checkpoint_enabled", True),
            ("compiled_projection_sha256", "c" * 64),
        ):
            proof = self._proof()
            proof[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "scheduling proof"),
            ):
                matched_overhead(
                    self._sample(0.2), self._sample(0.15), 0.03, scheduling_proof=proof
                )

    def test_enabled_low_requires_no_periodic_epoch_and_on_has_no_residual(self):
        from scripts.benchmark_suite.issue406_acceptance import matched_overhead

        self.identity["mode"] = "low"
        matched_overhead(
            self._sample(0.2),
            self._sample(0.15),
            0.03,
            scheduling_proof=self._proof(),
        )
        sample = self._sample(0.2)
        sample["nonterminal_epochs"] = [1]
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            matched_overhead(sample, self._sample(0.15), 0.03)
        self.identity["mode"] = "on"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            matched_overhead(
                self._sample(0.2),
                self._sample(0.15),
                0.03,
                scheduling_proof=self._proof(),
            )


if __name__ == "__main__":
    unittest.main()
