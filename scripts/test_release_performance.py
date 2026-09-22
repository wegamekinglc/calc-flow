from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch


def _specialized_context(root: Path) -> dict:
    sides = ("baseline", "candidate")
    return {
        "roots": dict.fromkeys(sides, root),
        "sites": dict.fromkeys(sides, root),
        "releases": {
            side: {"native_sha256": "a" * 64, "git_sha": "b" * 40} for side in sides
        },
        "binaries": {
            side: {"allocation_regression": root / "allocation-binary"}
            for side in sides
        },
    }


class ReleasePerformanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_preparation_failure_writes_incomparable_summary_and_nonzero_exit(
        self,
    ):
        from scripts.release_performance import run_gate

        with TemporaryDirectory() as raw:
            root = Path(raw)
            options = argparse.Namespace(
                output=root, baseline_source=root / "base", allow_dependency_drift=False
            )
            with patch(
                "scripts.release_performance.prepare",
                new=AsyncMock(side_effect=RuntimeError("build exited 7")),
            ):
                self.assertEqual(await run_gate(options), 1)
            report = json.loads((root / "results.json").read_text())
            self.assertIn("build exited 7", report["errors"][0])
            self.assertIn("incomparable", (root / "summary.md").read_text())

    def test_inventory_drift_is_an_evidence_error(self):
        from scripts.release_performance import matching_inventory

        for candidate in (["case", "case"], ["other"], []):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                matching_inventory({"baseline": ["case"], "candidate": candidate})
        self.assertEqual(
            matching_inventory({"baseline": ["case"], "candidate": ["case"]}), ["case"]
        )

    def test_release_outcomes_explain_downstream_skips_without_hiding_failures(self):
        from scripts.release_performance import acceptance_summary

        results = {
            "performance": {"outcome": "failure", "conclusion": "failure"},
            "security": {"outcome": "skipped", "conclusion": "skipped"},
            "soak": {"outcome": "skipped", "conclusion": "skipped"},
        }
        rendered = acceptance_summary(results)
        self.assertIn("performance: outcome=failure", rendered)
        self.assertIn("security: outcome=skipped", rendered)
        self.assertIn("prior step performance", rendered)
        self.assertIn("soak: outcome=skipped", rendered)

    async def test_measure_covers_required_inventories_without_mutating_report(self):
        from scripts.release_performance import measure
        from scripts.toolkit import fingerprint_json, sha256_file

        with TemporaryDirectory() as raw:
            root = Path(raw)
            binaries = {}
            releases = {}
            for side in ("baseline", "candidate"):
                binary = root / side
                binary.write_bytes(side.encode())
                binaries[side] = {"core": binary, "stream_join_perf": binary}
                releases[side] = {"native_sha256": sha256_file(binary)}
            context = {
                "binaries": binaries,
                "releases": releases,
                "sites": {side: root / side for side in releases},
                "roots": {side: root for side in releases},
            }
            metadata = {
                key: value
                for name in ("machine", "dependency", "workload")
                for key, value in (
                    (f"{name}_identity", {"fixture": name}),
                    (f"{name}_fingerprint", fingerprint_json({"fixture": name})),
                )
            }

            async def python_sample(name, site, native, output):
                return {
                    "samples": [1.0] * 10,
                    "metadata": metadata,
                    "binary_sha256": native,
                    "worker": str(output),
                    "correctness": True,
                }

            async def rust_sample(name, binary, source, identity, output):
                return await python_sample(name, source, sha256_file(binary), output)

            original = {"cases": [], "errors": []}
            with (
                patch(
                    "scripts.release_performance.python_inventory",
                    new=AsyncMock(return_value=["case"]),
                ),
                patch(
                    "scripts.release_performance.rust_inventory",
                    new=AsyncMock(return_value=["case"]),
                ),
                patch(
                    "scripts.release_performance.rust_identities",
                    return_value={side: metadata for side in releases},
                ),
                patch(
                    "scripts.release_performance.python_observation",
                    side_effect=python_sample,
                ),
                patch(
                    "scripts.release_performance.rust_observation",
                    side_effect=rust_sample,
                ),
            ):
                result = await measure(context, root, original)
            self.assertEqual(original, {"cases": [], "errors": []})
            self.assertEqual(len(result["cases"]), 3)
            self.assertEqual(result["errors"], [])
            self.assertEqual(
                set(result["inventory"]), {"python", "core", "stream_join_perf"}
            )
            self.assertEqual(len(list(root.rglob("observation.json"))), 120)

    async def test_relative_output_is_resolved_before_builds(self):
        from scripts.release_performance import run_gate

        root = Path("target/release-relative-fixture")
        observed = []

        async def prepare(options):
            observed.append(options.output.is_absolute())
            raise RuntimeError("fixture stops before native build")

        options = argparse.Namespace(
            output=root, baseline_source=root / "base", allow_dependency_drift=False
        )
        with patch("scripts.release_performance.prepare", side_effect=prepare):
            self.assertEqual(await run_gate(options), 1)
        self.assertEqual(observed, [True])

    async def test_specialized_checks_keep_lifecycle_allocation_and_rolling_gates(self):
        from scripts.release_performance import ROLLING, specialized

        with TemporaryDirectory() as raw:
            root = Path(raw)
            for scenario in ROLLING:
                path = root / "cases" / scenario / "round-0/pair-0/candidate"
                path.mkdir(parents=True)
                (path / "pytest.json").write_text(
                    json.dumps({"benchmarks": [{"extra_info": {"scenario": scenario}}]})
                )
            context = _specialized_context(root)
            options = argparse.Namespace(output=root, allow_dependency_drift=False)
            original = {"errors": []}
            with (
                patch(
                    "scripts.release_performance.command", new=AsyncMock()
                ) as command,
                patch(
                    "scripts.release_performance.allocation",
                    new=AsyncMock(return_value={"valid": True}),
                ),
                patch(
                    "scripts.release_performance.load_stream_lifecycle",
                    return_value={
                        "checkpoint_bytes_p50": 100,
                        "checkpoint_bytes_p95": 100,
                        "checkpoint_duration_p50_seconds": 1.0,
                        "checkpoint_duration_p95_seconds": 1.0,
                        "recovery_duration_p50_seconds": 1.0,
                        "recovery_duration_p95_seconds": 1.0,
                        "machine_fingerprint": "a" * 64,
                        "dependency_fingerprint": "b" * 64,
                        "workload_fingerprint": "c" * 64,
                    },
                ),
            ):
                result = await specialized(context, options, original)
            self.assertEqual(original, {"errors": []})
            self.assertEqual(result["errors"], [])
            commands = [call.args[0] for call in command.call_args_list]
            self.assertEqual(
                sum(
                    "--minimum-rounds" in argv and argv[-1] == "20" for argv in commands
                ),
                2,
            )
            self.assertEqual(sum("--scenario" in argv for argv in commands), 2)
            self.assertEqual(sum("--compare" in argv for argv in commands), 1)

    async def test_failure_summary_runs_without_benchmark_dependencies(self):
        import os
        import sys

        from scripts.benchmark_suite.process import ROOT, command

        with TemporaryDirectory() as raw:
            root = Path(raw)
            await command(
                [
                    sys.executable,
                    "-S",
                    "-m",
                    "scripts.release_performance",
                    "--acceptance-summary",
                    "--output",
                    str(root),
                ],
                cwd=ROOT,
                log=root / "summary.log",
                env={
                    **os.environ,
                    "ACCEPTANCE_STEPS": "{}",
                    "GITHUB_STEP_SUMMARY": str(root / "github.md"),
                },
            )
            self.assertTrue((root / "acceptance.json").is_file())

    async def test_lifecycle_regression_blocks_release_without_calling_it_incomparable(
        self,
    ):
        from scripts.release_performance import run_gate

        with TemporaryDirectory() as raw:
            root = Path(raw)
            report = {
                "cases": [
                    {"id": "case", "result": {"verdict": "no-confirmed-regression"}}
                ],
                "errors": [],
                "lifecycle_regressions": [["checkpoint_bytes", 0.1]],
            }
            with (
                patch(
                    "scripts.release_performance.prepare",
                    new=AsyncMock(return_value={"releases": {}}),
                ),
                patch(
                    "scripts.release_performance.measure",
                    new=AsyncMock(return_value=report),
                ),
                patch(
                    "scripts.release_performance.specialized",
                    new=AsyncMock(return_value=report),
                ),
            ):
                result = await run_gate(argparse.Namespace(output=root))
            self.assertEqual(result, 1)
            summary = (root / "summary.md").read_text()
            self.assertIn("lifecycle: regression", summary)
            self.assertNotIn("evidence failure:", summary)

    async def test_prepare_rejects_rust_source_drift_from_the_sealed_release(self):
        import sys
        from types import ModuleType
        from unittest.mock import Mock

        from scripts.release_performance import prepare

        release_module = ModuleType("scripts.benchmark_suite.release")
        release_module.load_release = Mock(
            side_effect=[{"git_sha": "1" * 40}, {"git_sha": "2" * 40}]
        )
        with TemporaryDirectory() as raw:
            root = Path(raw)
            with (
                patch.dict(
                    sys.modules, {"scripts.benchmark_suite.release": release_module}
                ),
                patch("scripts.release_performance.command", new=AsyncMock()),
                patch(
                    "scripts.release_performance.install",
                    new=AsyncMock(return_value=root),
                ),
                patch(
                    "scripts.release_performance.build_binaries",
                    new=AsyncMock(return_value={}),
                ),
                patch(
                    "scripts.release_performance.build_provenance",
                    return_value={"git_sha": "3" * 40},
                ),
                patch(
                    "scripts.release_performance.with_compiled_dependencies",
                    side_effect=lambda identity, *args: identity,
                ),
                self.assertRaisesRegex(ValueError, "sealed release"),
            ):
                await prepare(argparse.Namespace(output=root, baseline_source=root))
