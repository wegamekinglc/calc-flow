"""Keep pure CASE masks, immutable Arrow fixtures and fixed paired evidence."""

from __future__ import annotations

import asyncio
import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa

from scripts import test_sum_columnar_keys as controls
from scripts.benchmark_suite import sum_case_projection as study
from scripts.benchmark_suite import sum_columnar_keys as shared


def case_record(seed=False):
    record = controls.trade_record(seed)
    schema = record.schema.with_metadata({b"fixture": b"case-unit"})
    index = schema.get_field_index("side")
    schema = schema.set(
        index, schema.field(index).with_metadata({b"role": b"case-mask"})
    )
    return pa.RecordBatch.from_arrays(record.columns, schema=schema)


def write_source(path):
    seed, business = case_record(True), case_record()
    with pa.ipc.new_stream(path, seed.schema) as writer:
        writer.write_batch(seed)
        for _ in range(20):
            writer.write_batch(business)


def expected_digests(side):
    sign = 1 if side == "buy" else -1
    return [
        shared.evidence.json_hash(
            [
                [
                    account,
                    stock,
                    (sign * index * ((1 << 63) - 1) + (1 << 63)) % (1 << 64)
                    - (1 << 63),
                ]
                for account, stock in sorted(controls.KEYS)
            ]
        )
        for index in range(21)
    ]


def buffer_identity(column):
    return tuple(
        None if buffer is None else (buffer.address, buffer.size)
        for buffer in column.buffers()
    )


def protocol():
    return {
        "slug": study.SLUG,
        "baseline_revision": study.BASELINE,
        "candidate_revision": "a" * 40,
        "budget_seconds": 600,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "cases": list(study.CASES),
        "control_fixtures": study.CONTROLS,
        "profile_route_anchors": [],
        "tool_source_sha256": shared.helper_seals(study.CONTRACT),
    }


def current_builds():
    return {
        **controls.builds(),
        "baseline": {**controls.builds()["baseline"], "revision": study.BASELINE},
    }


def case_publication(directory):
    context, acquisition, summary, raw = controls.publication_input(
        directory, controls.profiles()
    )
    cases = [
        {**old, **case} for old, case in zip(summary["cases"], study.CASES, strict=True)
    ]
    acquisition["cases"] = summary["cases"] = cases
    collection = replace(context.collection, builds=current_builds())
    context = shared.StudyFollowup(
        collection, protocol(), "protocol", "case-projection", None, study.CONTRACT
    )
    return context, acquisition, summary, raw


class UniformCaseFixtureTests(unittest.TestCase):
    def test_prepare_fixture_uses_pure_side_for_seed_and_every_business_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "fixture").mkdir()
            source = output / "fixture/stock.arrow"
            write_source(source)
            original = source.read_bytes()
            collection = shared.maintained.CollectionContext(
                SimpleNamespace(output=output), {}, {}, 0, float("inf")
            )
            measurement = shared.maintained.MeasurementContext(collection, {})
            contract = replace(
                shared.COLUMNAR,
                controls={
                    "uniform-" + side: {"copies": 1, "side": side}
                    for side in ("buy", "sell")
                },
            )
            with patch.object(
                shared.maintained, "prepare_fixture", return_value=measurement
            ):
                generated = shared.prepare_fixtures(collection, contract)
            self.assertEqual(source.read_bytes(), original)
            for side in ("buy", "sell"):
                with self.subTest(side=side):
                    path = output / ("fixture/uniform-" + side + ".arrow")
                    manifest = generated.fixture["uniform-" + side]
                    self.assertEqual(
                        manifest["input_sha256"], shared.maintained.sha(path)
                    )
                    self.assertEqual(manifest["batch_rows"], [4] * 21)
                    self.assertEqual(manifest["snapshot_rows"], [4] * 21)
                    self.assertEqual(
                        manifest["snapshot_sha256"], expected_digests(side)
                    )
                    with pa.ipc.open_stream(path) as reader:
                        self.assertTrue(
                            reader.schema.equals(
                                case_record().schema, check_metadata=True
                            )
                        )
                        for record in reader:
                            self.assertEqual(
                                record.column("side").to_pylist(), [side] * 4
                            )

    def test_side_replacement_preserves_fields_metadata_and_other_buffers(self):
        record = case_record()
        before = record.to_pydict()
        schema = record.schema
        buffers = [buffer_identity(column) for column in record.columns]
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                changed = shared.control_side(record, side)
                self.assertTrue(changed.schema.equals(schema, check_metadata=True))
                for index, column in enumerate(record.columns):
                    if schema.field(index).name != "side":
                        self.assertEqual(changed.column(index), column)
                        self.assertEqual(
                            buffer_identity(changed.column(index)), buffers[index]
                        )
                self.assertEqual(changed.column("side").to_pylist(), [side] * 40)
                self.assertEqual(record.to_pydict(), before)
                self.assertTrue(record.schema.equals(schema, check_metadata=True))
                self.assertEqual(
                    [buffer_identity(column) for column in record.columns], buffers
                )
        self.assertIs(shared.control_side(record, None), record)

    def test_side_replacement_preserves_large_string_type_and_nullable_field(self):
        record = case_record()
        field = pa.field("side", pa.large_string(), metadata={b"exact": b"field"})
        record = record.set_column(3, field, record.column(3).cast(pa.large_string()))
        changed = shared.control_side(record, "sell")
        self.assertEqual(changed.schema.field(3), field)
        self.assertTrue(changed.schema.equals(record.schema, check_metadata=True))
        self.assertEqual(changed.column(3).type, pa.large_string())

    def test_unknown_sides_and_missing_fields_fail_without_mutating_input(self):
        record = case_record()
        before = record.to_pydict()
        for side in ("order", "", 1, True):
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, "side"):
                shared.control_side(record, side)
        with self.assertRaisesRegex(ValueError, "side field"):
            shared.control_side(record.select(["account", "stock"]), "buy")
        self.assertEqual(record.to_pydict(), before)

    def test_default_and_explicit_none_keep_identical_legacy_ipc_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.arrow"
            write_source(source)
            for copies in (1, 2):
                with self.subTest(copies=copies):
                    default, explicit = (
                        root / f"default-{copies}.arrow",
                        root / f"none-{copies}.arrow",
                    )
                    first = shared.control_fixture(source, default, copies)
                    second = shared.control_fixture(source, explicit, copies, side=None)
                    self.assertEqual(default.read_bytes(), explicit.read_bytes())
                    self.assertEqual(first, second)

    def test_uniform_fixture_changes_only_side_after_the_legacy_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, baseline = root / "source.arrow", root / "legacy.arrow"
            write_source(source)
            original = source.read_bytes()
            shared.control_fixture(source, baseline, 1)
            with pa.ipc.open_stream(baseline) as reader:
                legacy = list(reader)
            for side in ("buy", "sell"):
                target = root / (side + ".arrow")
                manifest = shared.control_fixture(source, target, 1, side=side)
                self.assertEqual(manifest["fixture"], "uniform-" + side)
                self.assertIn("on seed and business", manifest["mapping"])
                with pa.ipc.open_stream(target) as reader:
                    records = list(reader)
                for old, new in zip(legacy, records, strict=True):
                    self.assertTrue(old.schema.equals(new.schema, check_metadata=True))
                    for name in old.schema.names:
                        if name != "side":
                            self.assertEqual(old.column(name), new.column(name))
            self.assertEqual(source.read_bytes(), original)

    def test_failed_uniform_fixture_removes_only_owned_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (
                Path(directory) / "source.arrow",
                Path(directory) / "out.arrow",
            )
            write_source(source)
            original = source.read_bytes()
            for options in ({"side": "order"}, {"side": "sell", "deadline": 0}):
                with (
                    self.subTest(options=options),
                    self.assertRaises((ValueError, TimeoutError)),
                ):
                    shared.control_fixture(source, target, 1, **options)
                self.assertFalse(target.exists())
            with self.assertRaises(FileExistsError):
                shared.control_fixture(source, source, 1, side="buy")
            self.assertEqual(source.read_bytes(), original)


class CaseProjectionContractTests(unittest.TestCase):
    def test_five_cases_separate_uniform_masks_without_adding_a_sixth_case(self):
        self.assertEqual(study.BASELINE, "dc8393d2f33771706b32492e9c4fa5436c6baf12")
        self.assertEqual(study.CONTRACT.phase, "case-projection")
        self.assertEqual(
            [case["id"] for case in study.CASES],
            [
                "stock-stream-1cpu",
                "stock-stream-4cpu",
                "sparse-stream-1cpu",
                "uniform-buy-stream-1cpu",
                "uniform-sell-stream-1cpu",
            ],
        )
        self.assertEqual(study.CASES[1]["cpus"], [0, 2, 4, 6])
        self.assertEqual(set(study.CONTROLS), {"uniform-buy", "uniform-sell"})
        self.assertTrue(all(spec["copies"] == 1 for spec in study.CONTROLS.values()))
        self.assertEqual(
            [spec["side"] for spec in study.CONTROLS.values()], ["buy", "sell"]
        )

    def test_protocol_rejects_changed_counts_cases_masks_baseline_and_anchors(self):
        study.validate_protocol(protocol())
        for change in (
            {"budget_seconds": 601},
            {"rounds": 3},
            {"pairs_per_round": 11},
            {"warmups_per_arm_per_round": 2},
            {"cases": list(study.CASES)[:-1]},
            {"control_fixtures": shared.CONTROL_FIXTURES},
            {"baseline_revision": "1fcafde11a247b1be275d3cb6ab41a59476d6832"},
            {"candidate_revision": study.BASELINE},
            {"profile_route_anchors": list(shared.PROFILE_ANCHORS)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                study.validate_protocol({**protocol(), **change})

    def test_all_thirteen_helpers_are_required_without_mutating_old_contracts(self):
        self.assertEqual(len(study.CONTRACT.tools), 13)
        shared.validate_tool_seals(protocol(), study.CONTRACT)
        with self.assertRaisesRegex(ValueError, "inventory"):
            shared.validate_tool_seals(
                {**protocol(), "tool_source_sha256": shared.helper_seals()},
                study.CONTRACT,
            )
        old = json.dumps([shared.CASES, shared.CONTROL_FIXTURES])
        with patch.object(shared, "main") as main:
            study.main()
        main.assert_called_once_with(contract=study.CONTRACT)
        self.assertEqual(json.dumps([shared.CASES, shared.CONTROL_FIXTURES]), old)
        self.assertEqual(shared.COLUMNAR.slug, shared.SLUG)

    def test_old_or_stale_products_and_incomplete_profiles_are_rejected(self):
        shared.validate_products(current_builds(), protocol(), study.CONTRACT)
        for change in (
            controls.builds(),
            {
                **current_builds(),
                "candidate": {
                    **current_builds()["candidate"],
                    "example_recompiled": False,
                },
            },
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                shared.validate_products(change, protocol(), study.CONTRACT)
        shared.validate_profiles(controls.profiles(), current_builds())
        with self.assertRaises(ValueError):
            shared.validate_profiles(controls.profiles()[:1], current_builds())
        with self.assertRaises(ValueError):
            shared.validate_profiles(
                [
                    controls.profiles()[0],
                    {
                        **controls.profiles()[1],
                        "samples_per_window": [1, 1, 0, 1, 1],
                    },
                ],
                current_builds(),
            )

    def test_one_dense_improvement_and_both_uniform_regression_controls_matter(self):
        cases = [
            {**case, "statistics": {"verdict": "inconclusive"}} for case in study.CASES
        ]
        cases[3]["statistics"]["verdict"] = "improved"
        self.assertFalse(
            shared.performance_acceptance("completed", cases, study.CONTRACT)
        )
        cases[1]["statistics"]["verdict"] = "improved"
        self.assertTrue(
            shared.performance_acceptance("completed", cases, study.CONTRACT)
        )
        for index in (3, 4):
            changed = [
                {**case, "statistics": dict(case["statistics"])} for case in cases
            ]
            changed[index]["statistics"]["verdict"] = "regression"
            self.assertFalse(
                shared.performance_acceptance("completed", changed, study.CONTRACT)
            )
        self.assertFalse(
            shared.performance_acceptance("stopped", cases, study.CONTRACT)
        )

    def test_new_context_keeps_zero_prior_and_avoids_legacy_columnar_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol()))
            args = SimpleNamespace(
                protocol=path, budget_seconds=1, prior_measurement_seconds=9
            )
            collection = shared.maintained.CollectionContext(
                args, current_builds(), {}, 0, 600
            )
            with (
                patch.object(shared, "validate_receipt_seals"),
                patch.object(
                    shared.maintained, "collection_context", return_value=collection
                ) as make,
                patch.object(shared.evidence, "source_equivalence"),
                patch.object(shared, "validate_symbol_anchors") as anchors,
            ):
                context = shared.prepare_context(args, 0, study.CONTRACT)
            anchors.assert_not_called()
            self.assertEqual(make.call_args.args[0].budget_seconds, 600)
            self.assertEqual(make.call_args.args[0].prior_measurement_seconds, 0)
            self.assertEqual(
                context.collection.scope["profile_sides"], ("baseline", "candidate")
            )
            self.assertEqual(context.collection.scope["cases"], study.CASES)
        invalid = controls.builds()
        invalid["baseline"]["compiled_columnar_anchors"] = [shared.PROFILE_ANCHORS[0]]
        with self.assertRaisesRegex(ValueError, "baseline"):
            shared.validate_symbol_anchors(invalid)


class CaseProjectionPublicationTests(unittest.TestCase):
    def test_cancelled_case_collection_still_finalizes_and_publishes_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, _, _ = case_publication(directory)
            with (
                patch.object(shared, "prepare_context", return_value=context),
                patch.object(shared, "plan_and_machine", return_value=({}, {})),
                patch.object(shared, "collect", side_effect=asyncio.CancelledError()),
                patch.object(shared.maintained, "finalize_acquisition") as finalize,
                patch.object(shared, "publish") as publish,
                self.assertRaises(asyncio.CancelledError),
            ):
                shared.run(context.collection.args, study.CONTRACT)
            finalize.assert_called_once()
            self.assertEqual(publish.call_args.args[1]["status"], "stopped")
            self.assertIn("CancelledError", publish.call_args.args[1]["failure"])

    def test_late_publication_keeps_original_deadline_and_rejects_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = case_publication(directory)
            ticks = iter((1, 1, 601))
            with (
                patch.object(
                    shared.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    shared.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(
                    shared.time, "monotonic", side_effect=lambda: next(ticks, 602)
                ),
                self.assertRaisesRegex(TimeoutError, "publication deadline"),
            ):
                shared.publish(context, acquisition, {}, {})
            destination = context.collection.args.publish_directory
            manifest = json.loads((destination / "evidence-manifest.json").read_text())
            complete = json.loads(
                gzip.decompress(
                    (destination / "complete-evidence.json.gz").read_bytes()
                )
            )
            self.assertEqual(manifest["status"], "stopped")
            self.assertFalse(manifest["performance_acceptance"])
            self.assertEqual(complete["acquisition"]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
