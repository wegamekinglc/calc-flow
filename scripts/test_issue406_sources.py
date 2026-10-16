from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite import issue406_harness as harness
from scripts.benchmark_suite import issue406_milestone as milestone
from scripts.benchmark_suite import issue406_milestone_report as reporting
from scripts.benchmark_suite import issue406_run as collector
from scripts.benchmark_suite import issue406_sources as sources
from scripts.test_benchmark_rust_harness import checkout
from scripts.test_issue406_harness import commit, fixtures
from scripts.test_issue406_milestone import plan
from scripts.toolkit import command_output, git_output, sha256_file


def seal_sources(current, execution):
    diff = Path(current["build_execution_diff"]["path"])
    diff.write_text(
        sources.source_diff(
            execution, current["build_harness_git_sha"], current["execution_git_sha"]
        )
    )
    files = {
        name: sha256_file(Path(name))
        for root in (Path(current["build_harness_source"]), execution)
        for name in sources.python_sources(root)
    }
    digest = sha256_file(diff)
    return {**current, "build_execution_diff": {"path": str(diff), "sha256": digest}}, {
        **files,
        str(diff): digest,
    }


def source_fixture(root):
    build = root / "build"
    checkout(build, b"native fixture", lock="version = 4\npackage = []\n")
    for name in (
        *sources.RECIPE_INPUTS,
        "benchmarks/engine_stream.py",
        "benchmarks/interval_acceptance.py",
    ):
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original tool\n")
    commit(build)
    original = git_output(build, "rev-parse", "HEAD")
    execution = root / "execution"
    command_output(["git", "clone", "--shared", "--quiet", str(build), str(execution)])
    (execution / "benchmarks/engine_stream.py").write_text(
        "# interval cap: 6 * batch + 768\n"
    )
    commit(execution)
    revision = git_output(execution, "rev-parse", "HEAD")
    current = {
        **plan(),
        "source_policy": sources.POLICY,
        "build_harness_source": str(build),
        "build_harness_git_sha": original,
        "harness_git_sha": revision,
        "execution_git_sha": revision,
        "preparation_main_git_sha": revision,
        "build_execution_diff": {"path": str(root / "sources.patch")},
    }
    current, files = seal_sources(current, execution)
    return current, files, build, execution


class SourcePolicyTests(unittest.TestCase):
    def test_explicit_source_policy_accepts_distinct_build_and_execution(self):
        current = {
            **plan(),
            "source_policy": "calc-flow.issue406-build-execution.v1",
            "build_harness_git_sha": "e" * 40,
            "build_harness_source": str(Path.cwd() / "original/build-source"),
            "build_execution_diff": {"path": "sources.patch", "sha256": "f" * 64},
            "preparation_main_git_sha": "c" * 40,
        }
        milestone.validate_envelope(current)

    def test_same_sha_default_and_historical_policy_stay_strict(self):
        milestone.validate_envelope(plan())
        for change in (
            {"build_harness_git_sha": "e" * 40},
            {"source_policy": "unknown"},
            {"execution_git_sha": "e" * 40},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                milestone.validate_envelope({**plan(), **change})
        historical = {
            "schema": "calc-flow.issue406-fresh.v3",
            "build_harness_git_sha": "e" * 40,
        }
        with patch.object(sources, "_validate_checkout") as check:
            # Historical dispatch never opts into the new policy implicitly.
            sources.validate_binding(historical, Path("unused"), {})
            check.assert_not_called()

    def test_real_separate_checkouts_pass_the_collector_inventory_gate(self):
        with TemporaryDirectory() as raw:
            current, files, _build, execution = source_fixture(Path(raw))
            frozen = {**current, "sealed_files": files}
            before = copy.deepcopy(frozen)
            with (
                patch.object(collector, "ROOT", execution),
                patch.object(collector, "_validate_plan_cpus"),
            ):
                self.assertEqual(collector._validate_plan_inventory(frozen), files)
            self.assertEqual(frozen, before)

    def test_missing_or_forged_build_and_execution_sources_are_rejected(self):
        with TemporaryDirectory() as raw:
            current, files, _build, execution = source_fixture(Path(raw))
            for change in (
                {"build_harness_source": str(Path(raw) / "missing")},
                {"build_harness_source": str(execution)},
                {"build_harness_git_sha": "f" * 40},
                {"execution_git_sha": "f" * 40},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    sources.validate_binding({**current, **change}, execution, files)
            for key in (
                "build_harness_source",
                "build_execution_diff",
                "preparation_main_git_sha",
            ):
                missing = {
                    name: value for name, value in current.items() if name != key
                }
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    milestone.validate_envelope(missing)

    def test_missing_wrong_or_untracked_python_fixture_tools_are_rejected(self):
        with TemporaryDirectory() as raw:
            current, files, build, execution = source_fixture(Path(raw))
            for root in (build, execution):
                name = str(root / "benchmarks/interval_acceptance.py")
                for invalid in (
                    {key: value for key, value in files.items() if key != name},
                    {**files, name: "0" * 64},
                ):
                    with (
                        self.subTest(source=root, seal=invalid.get(name)),
                        self.assertRaisesRegex(ValueError, "Python or fixture"),
                    ):
                        sources.validate_binding(current, execution, invalid)
            injected = execution / "benchmarks/untracked.py"
            injected.write_text(
                "# cannot enter a clean SHA through an untracked module\n"
            )
            with self.assertRaisesRegex(ValueError, "tracked and sealed"):
                sources.validate_binding(
                    current, execution, {**files, str(injected): sha256_file(injected)}
                )

    def test_changed_compiled_inputs_or_build_recipe_cannot_reuse_receipts(self):
        for name in ("python/changed.py", *sources.RECIPE_INPUTS[:2]):
            with self.subTest(name=name), TemporaryDirectory() as raw:
                current, _files, _build, execution = source_fixture(Path(raw))
                path = execution / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# changed\n")
                commit(execution)
                revision = git_output(execution, "rev-parse", "HEAD")
                current, files = seal_sources(
                    {
                        **current,
                        "harness_git_sha": revision,
                        "execution_git_sha": revision,
                        "preparation_main_git_sha": revision,
                    },
                    execution,
                )
                with self.assertRaisesRegex(
                    ValueError, "compiled inputs or original build recipe"
                ):
                    sources.validate_binding(current, execution, files)

    def test_self_consistently_sealed_but_forged_diff_is_rejected(self):
        with TemporaryDirectory() as raw:
            current, files, _build, execution = source_fixture(Path(raw))
            name = current["build_execution_diff"]["path"]
            with self.assertRaisesRegex(ValueError, "diff is not sealed"):
                sources.validate_binding(
                    current,
                    execution,
                    {key: value for key, value in files.items() if key != name},
                )
            Path(name).write_text("forged diff")
            digest = sha256_file(Path(name))
            forged = {
                **current,
                "build_execution_diff": {"path": name, "sha256": digest},
            }
            with self.assertRaisesRegex(ValueError, "actual commits"):
                sources.validate_binding(forged, execution, {**files, name: digest})

    def test_native_metadata_still_binds_product_build_and_actual_binary(self):
        current = {**plan(), "source_policy": sources.POLICY}
        descriptor = {"sha256": "d" * 64}
        original = {
            "harness_git_sha": "c" * 40,
            "product_commit": "a" * 40,
            "binary_sha256": "d" * 64,
        }
        sources.validate_native_identity(current, original, descriptor, "a" * 40)
        for key in original:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                sources.validate_native_identity(
                    current,
                    {**original, key: "f" * len(original[key])},
                    descriptor,
                    "a" * 40,
                )

    def test_independent_original_harness_validates_attestation_without_relabelling(
        self,
    ):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            product, build = fixtures(root)
            output = root / "attestation"
            with (
                patch.object(harness, "ROOT", build),
                harness.fresh_acceptance_harness(product, output, "stream_asof_e2e"),
            ):
                pass
            execution = root / "execution"
            command_output(
                ["git", "clone", "--shared", "--quiet", str(build), str(execution)]
            )
            (execution / "new_tool.py").write_text("# new execution tool\n")
            commit(execution)
            current = {
                **plan(),
                "source_policy": sources.POLICY,
                "build_harness_source": str(build),
                "build_harness_git_sha": git_output(build, "rev-parse", "HEAD"),
            }
            with patch.object(harness, "ROOT", execution):
                record = collector._validate_source_attestation(
                    current, output, "stream_asof_e2e"
                )
                self.assertEqual(
                    record["harness_git_sha"], current["build_harness_git_sha"]
                )
                missing = {
                    key: value
                    for key, value in current.items()
                    if key != "build_harness_source"
                }
                with self.assertRaisesRegex(ValueError, "attestation"):
                    collector._validate_source_attestation(
                        missing, output, "stream_asof_e2e"
                    )
                path = output / "build-stream_asof_e2e.acceptance-harness.json"
                for change in (
                    {"effective_files": {}},
                    {"product_git_sha": "f" * 40},
                    {"harness_git_sha": "f" * 40},
                ):
                    path.write_text(json.dumps({**record, **change}))
                    with (
                        self.subTest(change=change),
                        self.assertRaisesRegex(ValueError, "attestation"),
                    ):
                        collector._validate_source_attestation(
                            current, output, "stream_asof_e2e"
                        )

    def test_source_roles_survive_normal_and_aborted_report_identity(self):
        current = {
            **plan(),
            "source_policy": sources.POLICY,
            "build_harness_git_sha": "e" * 40,
            "build_harness_source": "/original",
            "build_execution_diff": {"path": "sources.patch", "sha256": "f" * 64},
            "preparation_main_git_sha": "c" * 40,
        }
        for report in (
            milestone.report_identity(current),
            reporting.aborted_summary(current),
        ):
            with self.subTest(report=report.get("schema")):
                for key in sources.REPORT_FIELDS:
                    self.assertEqual(report[key], current[key])
        rendered = reporting.render(reporting.aborted_summary(current))
        self.assertIn("Build tool `" + "e" * 40 + "`", rendered)
        self.assertIn("execution tool `" + "c" * 40 + "`", rendered)


class PreparationSourceGuardTests(unittest.TestCase):
    def test_failed_source_or_main_guard_never_creates_an_anchor(self):
        for failure in (ValueError("wrong execution"), ValueError("main changed")):
            with self.subTest(failure=failure), TemporaryDirectory() as raw:
                anchor = Path(raw) / "anchor.json"
                argv = [
                    "issue406_run",
                    "--begin-preparation",
                    str(anchor),
                    "--expected-execution-sha",
                    "c" * 40,
                    "--expected-main-sha",
                    "d" * 40,
                ]
                with (
                    patch("sys.argv", argv),
                    patch.object(sources, "preparation_guard", side_effect=failure),
                    self.assertRaises(ValueError),
                ):
                    collector.main()
                self.assertFalse(anchor.exists())

    def test_guard_checks_the_actual_checkout_remote_and_main_fixture_inputs(self):
        with TemporaryDirectory() as raw:
            current, _files, _build, execution = source_fixture(Path(raw))
            revision = current["execution_git_sha"]
            real = sources.git_output

            def git(source, *args):
                return (
                    revision + "\trefs/heads/main"
                    if args[0] == "ls-remote"
                    else real(source, *args)
                )

            with patch.object(sources, "git_output", side_effect=git):
                identity = sources.preparation_guard(execution, revision, revision)
                with self.assertRaisesRegex(ValueError, "checkout"):
                    sources.preparation_guard(execution, "f" * 40, revision)
                with self.assertRaisesRegex(ValueError, "main/source"):
                    sources.preparation_guard(execution, revision, "f" * 40)
            with self.assertRaisesRegex(ValueError, "fixture tools"):
                sources._validate_main_inputs(
                    execution, revision, current["build_harness_git_sha"]
                )
            sources.validate_anchor({**current}, {"source_guard": identity})
            with self.assertRaisesRegex(ValueError, "verified source guard"):
                sources.validate_anchor(current, {})

    def test_successful_guard_is_recorded_before_exclusive_anchor_creation(self):
        identity = {"execution_git_sha": "c" * 40, "remote_main_git_sha": "d" * 40}
        with TemporaryDirectory() as raw:
            anchor = Path(raw) / "anchor.json"
            argv = [
                "issue406_run",
                "--begin-preparation",
                str(anchor),
                "--expected-execution-sha",
                "c" * 40,
                "--expected-main-sha",
                "d" * 40,
            ]
            with (
                patch("sys.argv", argv),
                patch.object(sources, "preparation_guard", return_value=identity),
            ):
                self.assertEqual(collector.main(), 0)
                self.assertEqual(
                    json.loads(anchor.read_text())["source_guard"], identity
                )
                before = anchor.read_bytes()
                with self.assertRaises(FileExistsError):
                    collector.main()
                self.assertEqual(anchor.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
