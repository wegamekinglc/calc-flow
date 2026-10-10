from __future__ import annotations

import copy
import unittest

from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory


def selected(row, workload=None, mode="off", rows=1_000_000, threads=1):
    return next(
        case
        for case in acceptance_inventory()
        if case.row == row
        and (workload is None or case.workload == workload)
        and case.mode == mode
        and case.rows == rows
        and case.threads == threads
    )


def record(case, seconds, *, side="candidate", lifecycle="completed", epochs=None):
    sample = {
        case.metric: seconds,
        "processing_seconds": seconds,
        "oracle_passed": True,
        "cleanup_complete": True,
        "lifecycle": lifecycle,
        "lifecycle_error": "baseline EOF failure" if lifecycle == "failed" else None,
        "mode": case.mode,
        "oracle": {"rows": case.rows},
        "checkpoint_observations": [{}, {}],
        "processing_checkpoint_free": case.mode != "on",
        "observer_policy": "all-modes-status-every-10ms-plus-final-frontier-1ms",
        "checkpoint_evidence": {
            "state_file_count": 0,
            "coverage": "disabled" if case.mode == "off" else "enabled",
        },
        "advancing_processing_epochs": [] if epochs is None else epochs,
    }
    return {
        "case": case.id,
        "side": side,
        "rounds": [[copy.deepcopy(sample) for _ in range(10)] for _ in range(2)],
    }


def report(records):
    return {
        "measurements": records,
        "assessment": {"compatibility_verified": True, "environment_suitable": True},
        "settlement": {"forced_kill_pids": []},
    }


class AcceptanceReportTests(unittest.TestCase):
    def test_partial_report_keeps_all_eight_targets_without_fabricating_main_off(self):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance

        case = selected(1)
        summary = summarize_acceptance(report([record(case, 0.04)]))
        self.assertEqual([row["row"] for row in summary["targets"]], list(range(1, 9)))
        result = next(item for item in summary["cases"] if item["case"] == case.id)
        self.assertEqual(result["baseline"]["status"], "unsupported")
        self.assertEqual(result["revision_comparison"]["verdict"], "candidate-only")
        self.assertFalse(summary["evidence_complete"])

    def test_revision_gate_uses_existing_two_round_rule_and_invalid_main_is_diagnostic(
        self,
    ):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance

        case = selected(4, "join_capture", mode="enabled")
        records = [record(case, 1.0, side="baseline"), record(case, 1.2)]
        summary = summarize_acceptance(report(records))
        result = next(item for item in summary["cases"] if item["case"] == case.id)
        self.assertEqual(result["revision_comparison"]["verdict"], "regression")
        records[0] = record(case, 1.0, side="baseline", lifecycle="failed")
        result = next(
            item
            for item in summarize_acceptance(report(records))["cases"]
            if item["case"] == case.id
        )
        self.assertEqual(result["revision_comparison"]["verdict"], "diagnostic")
        self.assertEqual(
            result["revision_comparison"]["maintained_verdict"], "regression"
        )

    def test_mode_ratio_of_medians_leaves_zero_epoch_periodic_impact_unmeasured(self):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance

        off, on = selected(1), selected(1, mode="on")
        summary = summarize_acceptance(report([record(off, 0.04), record(on, 0.05)]))
        row = summary["targets"][0]
        mode = next(
            item
            for item in row["mode_effects"]
            if item["mode"] == "on" and item["threads"] == 1
        )
        self.assertAlmostEqual(mode["throughput_ratio"], 0.8)
        self.assertEqual(mode["configuration_target_status"], "observed-within-target")
        self.assertEqual(mode["periodic_impact"], "unmeasured")
        self.assertNotIn("paired_intervals", mode)

    def test_retention_scaling_uses_group_medians_without_cross_group_pairing(self):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance

        small, large = (
            selected(3, "join_retained"),
            selected(3, "join_retained", rows=4_000_000),
        )
        summary = summarize_acceptance(report([record(small, 1.0), record(large, 4.3)]))
        scaling = next(
            item for item in summary["targets"][2]["scaling"] if item["mode"] == "off"
        )
        self.assertAlmostEqual(scaling["ratio_of_p50"], 4.3)
        self.assertEqual(scaling["target_status"], "observed-within-target")
        self.assertNotIn("paired_intervals", scaling)

    def test_main_retained_scaling_is_present_as_lifecycle_diagnostic(self):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance

        small = selected(3, "join_retained", mode="low")
        large = selected(3, "join_retained", mode="low", rows=4_000_000)
        summary = summarize_acceptance(
            report(
                [
                    record(small, 2.0, side="baseline", lifecycle="failed"),
                    record(large, 9.0, side="baseline", lifecycle="failed"),
                ]
            )
        )
        scaling = next(
            item for item in summary["targets"][2]["scaling"] if item["mode"] == "low"
        )
        self.assertEqual(scaling["baseline"]["status"], "diagnostic")
        self.assertAlmostEqual(scaling["baseline"]["ratio_of_p50"], 4.5)

    def test_primary_table_contains_fresh_main_scaling_compaction_and_row_units(self):
        from scripts.benchmark_suite.issue406_report import (
            render_acceptance_markdown,
            summarize_acceptance,
        )

        small = selected(3, "join_retained", mode="low")
        large = selected(3, "join_retained", mode="low", rows=4_000_000)
        armed = selected(5, "join_handler_compacting", mode="enabled")
        steady = selected(5, "join_handler_steady", mode="enabled")
        capture = selected(4, "join_capture", mode="enabled")
        summary = summarize_acceptance(
            report(
                [
                    record(small, 2.0, side="baseline"),
                    record(large, 9.0, side="baseline"),
                    record(armed, 0.004, side="baseline"),
                    record(steady, 0.001, side="baseline"),
                    record(capture, 0.05),
                    record(capture, 1.8, side="baseline"),
                ]
            )
        )
        snapshot = summary["targets"][3]["absolute"][0]
        self.assertEqual(snapshot["denominator_rows"], 1_000_000)
        self.assertAlmostEqual(snapshot["candidate"]["p50"], 0.05)
        self.assertEqual(snapshot["target_status"], "observed-within-target")
        self.assertAlmostEqual(snapshot["baseline"]["p50"], 1.8)
        primary = render_acceptance_markdown(summary).split(
            "Main has no checkpoint-off arm."
        )[0]
        self.assertIn("Fresh main (explicit policy)", primary)
        self.assertIn("low: ratio of p50 4.5 (ok)", primary)
        self.assertIn("Enabled ratio of p50 4 (ok)", primary)
        self.assertIn("enabled/join_capture: p50 1.8", primary)
        self.assertIn("microseconds/row", primary)

    def test_forced_kill_preserves_paired_diagnostic_but_prevents_revision_verdict(
        self,
    ):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance

        case = selected(4, "join_capture", mode="enabled")
        data = report([record(case, 1.0, side="baseline"), record(case, 1.2)])
        data["settlement"]["forced_kill_pids"] = [123]
        summary = summarize_acceptance(data)
        comparison = next(
            item["revision_comparison"]
            for item in summary["cases"]
            if item["case"] == case.id
        )
        self.assertFalse(summary["assessment"]["environment_suitable"])
        self.assertEqual(comparison["verdict"], "inconclusive")
        self.assertEqual(comparison["maintained_verdict"], "regression")

    def test_overhead_is_signed_difference_of_group_medians_not_index_paired_residual(
        self,
    ):
        from scripts.benchmark_suite.issue406_report import summarize_acceptance

        cases = {
            name: selected(7, name) for name in ("asof", "asof_operator", "projection")
        }
        identity = {
            "project_sha256": "a" * 64,
            "inputs_sha256": "b" * 64,
            "shared_partial_order_sha256": "c" * 64,
            "output_contract_sha256": "d" * 64,
            "compiled_plan_fingerprint": "e" * 64,
            "rows_per_side": 1_000_000,
            "checkpoint_enabled": False,
            "batch_rows": 1024,
            "mode": "off",
            "cpu_affinity": [0],
            "processing_boundary": "final-output-projection-watermark-completed",
            "output_frontier": "minimum-input-watermark-minus-one-microsecond",
        }
        records = [record(case, 1.0) for case in cases.values()]
        sequences = ([1.0] * 11 + [10.0] * 9, [0.5] * 9 + [9.0] * 11, [0.1] * 20)
        for item, values in zip(records, sequences, strict=True):
            for sample, value in zip(
                [sample for group in item["rounds"] for sample in group],
                values,
                strict=True,
            ):
                sample.update(
                    processing_seconds=value,
                    settled_processing_seconds=value,
                    product_commit="1" * 40,
                    processing_settled=True,
                    match_identity=copy.deepcopy(identity),
                    nonterminal_epochs=[],
                    compiled_projection=[{"node_id": "asof"}],
                    compiled_projection_sha256="f" * 64,
                    canonical_native_events_sha256="a" * 64,
                    native_executable_sha256="b" * 64,
                )
            item["warmups"] = [copy.deepcopy(group[0]) for group in item["rounds"]]
        sources = [*records[0]["warmups"], *sum(records[0]["rounds"], [])]
        replays = [*records[1]["warmups"], *sum(records[1]["rounds"], [])]
        for index, (source, replay) in enumerate(zip(sources, replays, strict=True)):
            source.update(
                observation_id=f"source/{index}",
                raw_observation_sha256=f"{index + 1:064x}",
                dispatch_trace_envelope_sha256=f"{index + 101:064x}",
                dispatch_trace={
                    "schema": "calc-flow.asof-dispatch-trace.v1",
                    "overflowed": False,
                    "terminal_excluded": True,
                    "allow_barriers": False,
                    "compiled_plan_fingerprint": "e" * 64,
                    "node_id": "asof",
                    "capacity": 8192,
                    "record_count": 1,
                    "events": [{"kind": "watermark", "ingress": "left", "micros": 10}],
                    "callback_counts": {
                        "data": 2,
                        "ingress_progress": 2,
                        "watermark": 1,
                    },
                    "completed_frontiers": {"result": 9},
                    "operator_data_counts": {"asof": {"input_rows": 2}},
                    "asof_state": {"output_watermark_micros": 9},
                    "output_record_batch_rows": [[1]],
                    "output_node_id": "result",
                    "required_output_frontier_micros": 9,
                },
            )
            replay.update(
                dispatch_trace_verified=True,
                dispatch_replay_state_verified=True,
                dispatch_trace=copy.deepcopy(source["dispatch_trace"]),
                replay_source_observation_id=source["observation_id"],
                replay_source_raw_sha256=source["raw_observation_sha256"],
                replay_trace_sha256=source["dispatch_trace_envelope_sha256"],
            )
        data = report(records)
        data["scheduling_proofs"] = {
            "candidate/off": {
                "status": "verified",
                "artifact_sha256": "c" * 64,
                "product_commit": "1" * 40,
                "native_executable_sha256": "b" * 64,
                "project_sha256": "a" * 64,
                "compiled_projection_sha256": "f" * 64,
                "batch_rows": 1024,
                "checkpoint_enabled": False,
            }
        }
        summary = summarize_acceptance(data)
        overhead = next(
            item
            for item in summary["targets"][6]["overhead"]
            if item["side"] == "candidate" and item["mode"] == "off"
        )
        self.assertAlmostEqual(overhead["overhead_seconds"], -8.0)
        self.assertEqual(overhead["target_status"], "inconclusive")
        self.assertAlmostEqual(overhead["target_seconds"], 0.2)
        self.assertEqual(overhead["semantic_replays_validated"], 22)
        self.assertNotIn("paired_intervals", overhead)
        replays[-1]["replay_source_observation_id"] = sources[0]["observation_id"]
        invalid = next(
            item
            for item in summarize_acceptance(data)["targets"][6]["overhead"]
            if item["side"] == "candidate" and item["mode"] == "off"
        )
        self.assertEqual(invalid["target_status"], "inconclusive")
        self.assertIn("exact source observation", invalid["errors"][0])

    def test_markdown_exposes_stopped_budget_and_environment_context(self):
        from scripts.benchmark_suite.issue406_report import (
            render_acceptance_markdown,
            summarize_acceptance,
        )

        data = report([])
        data.update(state="stopped", budget_satisfied=False)
        data["assessment"]["environment_suitable"] = False
        markdown = render_acceptance_markdown(summarize_acceptance(data))
        self.assertIn("Run state: stopped", markdown)
        self.assertIn("budget satisfied: False", markdown)
        self.assertIn("environment suitable: False", markdown)
        self.assertIn("Incomplete observations", markdown)

    def test_mode_markdown_exposes_lifecycle_costs_and_full_evidence_links(self):
        from scripts.benchmark_suite.issue406_report import (
            render_acceptance_markdown,
            summarize_acceptance,
        )

        enabled = record(selected(1, mode="low"), 0.05)
        for group in enabled["rounds"]:
            for sample in group:
                sample.update(terminal_seconds=0.002, full_cycle_seconds=0.055)
                sample["checkpoint_evidence"]["state_bytes"] = 1024
        native = record(selected(4, "join_capture", mode="enabled"), 0.05)
        for group in native["rounds"]:
            for sample in group:
                sample.update(setup_seconds=0.1, oracle_seconds=0.003)
        summary = summarize_acceptance(
            report([record(selected(1), 0.04), enabled, native])
        )
        markdown = render_acceptance_markdown(summary)
        self.assertIn("EOF p50/p95 [interval], ms", markdown)
        self.assertIn("p50 2, p95 2", markdown)
        self.assertIn("p50 55, p95 55", markdown)
        self.assertIn("p50 1024, p95 1024", markdown)
        self.assertIn("[Machine-readable phase costs](summary.json)", markdown)
        self.assertIn("[final budget ledger](budget.json)", markdown)
        phases = next(
            item["candidate"]["phases"]
            for item in summary["cases"]
            if item["case"] == native["case"]
        )
        self.assertEqual(phases["setup_seconds"]["p50"], 0.1)
        self.assertEqual(phases["oracle_seconds"]["p50"], 0.003)


if __name__ == "__main__":
    unittest.main()
