from __future__ import annotations

import argparse
import hashlib
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

    def test_acceptance_summary_reports_only_the_jobs_own_steps(self):
        from scripts.release_performance import acceptance_summary

        results = {
            "performance": {"outcome": "success", "conclusion": "success"},
            "security": {"outcome": "success", "conclusion": "success"},
            "soak": {"outcome": "failure", "conclusion": "failure"},
        }
        rendered = acceptance_summary(results, steps=("performance", "security"))
        self.assertIn("performance: outcome=success", rendered)
        self.assertIn("security: outcome=success", rendered)
        self.assertNotIn("soak:", rendered)

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

    async def test_measure_collects_only_selected_suite(self):
        from scripts.release_performance import measure

        with TemporaryDirectory() as raw:
            root = Path(raw)
            context = {
                "binaries": {
                    side: {"core": root / side} for side in ("baseline", "candidate")
                },
                "roots": dict.fromkeys(("baseline", "candidate"), root),
            }
            with (
                patch(
                    "scripts.release_performance.python_inventory", new=AsyncMock()
                ) as py,
                patch(
                    "scripts.release_performance.rust_inventory",
                    new=AsyncMock(return_value=["one"]),
                ),
                patch(
                    "scripts.release_performance.rust_identities",
                    return_value={"baseline": {}, "candidate": {}},
                ),
                patch("scripts.release_performance.sha256_file", return_value="seal"),
                patch(
                    "scripts.release_performance.collect_case",
                    new=AsyncMock(return_value={"id": "rust/core/one"}),
                ),
                patch(
                    "scripts.release_performance.evaluate_case",
                    return_value={"verdict": "no-confirmed-regression"},
                ),
            ):
                report = await measure(
                    context, root, {"cases": [], "errors": []}, "core"
                )
            py.assert_not_called()
            self.assertEqual(list(report["inventory"]), ["core"])
            self.assertEqual([row["id"] for row in report["cases"]], ["rust/core/one"])

    def test_merge_checks_suite_identity_inventory_and_raw_evidence(self):
        import zipfile

        from scripts.release_performance import harness_identity, merge_reports
        from scripts.toolkit import sha256_file

        with TemporaryDirectory() as raw:
            root = Path(raw)
            sources = root / "suites"
            expected = {"verdict": "no-confirmed-regression"}
            for suite in ("python", "core", "stream_join_perf"):
                case_id = "python/one" if suite == "python" else f"rust/{suite}/one"
                digest = hashlib.sha256(case_id.encode()).hexdigest()[:20]
                relative = Path("cases") / digest / "pairs.json"
                source = sources / f"release-performance-{suite}"
                (source / relative).parent.mkdir(parents=True)
                (source / relative).write_text(json.dumps({"id": case_id}))
                releases = {}
                for side, sha in (("baseline", "a" * 40), ("candidate", "b" * 40)):
                    destination = source / "builds" / side
                    destination.mkdir(parents=True)
                    wheel = destination / "fixture.whl"
                    with zipfile.ZipFile(wheel, "w") as archive:
                        archive.writestr("calc_flow/_native.so", side.encode())
                    release = {
                        "contract": "benchmark-release-v1",
                        "build_profile": "release",
                        "git_clean": True,
                        "git_sha": sha,
                        "wheel": wheel.name,
                        "wheel_sha256": sha256_file(wheel),
                        "native_sha256": hashlib.sha256(side.encode()).hexdigest(),
                    }
                    (destination / "release.json").write_text(json.dumps(release))
                    releases[side] = release
                    target = "allocation_regression" if suite == "python" else suite
                    log = source / "rust-builds" / side / f"build-{target}.jsonl"
                    log.parent.mkdir(parents=True)
                    log.write_text('{"reason": "build-finished", "success": true}\n')
                if suite != "python":
                    (source / "rust-provenance.json").write_text(
                        json.dumps(
                            {
                                side: {"git_sha": releases[side]["git_sha"]}
                                for side in releases
                            }
                        )
                    )
                (source / "results.json").write_text(
                    json.dumps(
                        {
                            "contract": "release-paired-v1",
                            "suite": suite,
                            "cases": [
                                {
                                    "id": case_id,
                                    "evidence": str(Path("/runner/output") / relative),
                                    "seals": {
                                        side: releases[side]["native_sha256"]
                                        for side in releases
                                    },
                                    "result": expected,
                                }
                            ],
                            "errors": [],
                            "allocation": {"baseline": {}, "candidate": {}},
                            "inventory": {
                                suite: {"baseline": ["one"], "candidate": ["one"]}
                            },
                            "harness": harness_identity(),
                            "dependency_lock_sha256": sha256_file(
                                Path(__file__).resolve().parents[1]
                                / "benchmarks/requirements.lock"
                            ),
                            "releases": releases,
                        }
                    )
                )
            with (
                patch(
                    "scripts.release_performance.evaluate_case", return_value=expected
                ) as evaluate,
                patch("scripts.release_performance.rust_identities"),
            ):
                merged = merge_reports(sources)
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(len(merged["cases"]), 3)
            core_provenance = sources / "release-performance-core/rust-provenance.json"
            provenance = core_provenance.read_text()
            core_provenance.unlink()
            with (
                patch(
                    "scripts.release_performance.evaluate_case", return_value=expected
                ),
                self.assertRaisesRegex(ValueError, "missing core Rust provenance"),
            ):
                merge_reports(sources)
            core_provenance.write_text(provenance)
            wrong_provenance = json.loads(provenance)
            wrong_provenance["baseline"]["git_sha"] = "c" * 40
            core_provenance.write_text(json.dumps(wrong_provenance))
            with (
                patch(
                    "scripts.release_performance.evaluate_case", return_value=expected
                ),
                self.assertRaisesRegex(ValueError, "core baseline Rust source differs"),
            ):
                merge_reports(sources)
            core_provenance.write_text(provenance)
            core_log = (
                sources
                / "release-performance-core/rust-builds/baseline/build-core.jsonl"
            )
            log_content = core_log.read_text()
            core_log.unlink()
            with (
                patch(
                    "scripts.release_performance.evaluate_case", return_value=expected
                ),
                self.assertRaisesRegex(
                    ValueError, "missing core baseline Rust build log"
                ),
            ):
                merge_reports(sources)
            core_log.write_text(log_content)
            allocation_log = (
                sources
                / "release-performance-python"
                / "rust-builds"
                / "baseline"
                / "build-allocation_regression.jsonl"
            )
            allocation_content = allocation_log.read_text()
            allocation_log.unlink()
            with (
                patch(
                    "scripts.release_performance.evaluate_case", return_value=expected
                ),
                patch("scripts.release_performance.rust_identities"),
                self.assertRaisesRegex(
                    ValueError, "missing python baseline Rust build log"
                ),
            ):
                merge_reports(sources)
            allocation_log.write_text(allocation_content)
            python_manifest = (
                sources / "release-performance-python/builds/baseline/release.json"
            )
            manifest_content = python_manifest.read_text()
            python_manifest.unlink()
            with self.assertRaisesRegex(
                ValueError, "missing python baseline sealed release manifest"
            ):
                merge_reports(sources)
            python_manifest.write_text(manifest_content)
            self.assertTrue(
                all(
                    (sources.parent / row["evidence"]).is_file()
                    for row in merged["cases"]
                )
            )
            core_result = sources / "release-performance-core" / "results.json"
            core = json.loads(core_result.read_text())
            core["releases"]["baseline"]["git_sha"] = "c" * 40
            core_result.write_text(json.dumps(core))
            with (
                patch(
                    "scripts.release_performance.evaluate_case", return_value=expected
                ),
                patch("scripts.release_performance.rust_identities"),
                self.assertRaisesRegex(ValueError, "release SHAs disagree"),
            ):
                merge_reports(sources)
            core["releases"]["baseline"]["git_sha"] = "a" * 40
            core_result.write_text(json.dumps(core))
            (sources / "release-performance-core" / "cases").rename(
                sources / "release-performance-core" / "lost-cases"
            )
            with (
                patch(
                    "scripts.release_performance.evaluate_case", return_value=expected
                ),
                patch("scripts.release_performance.rust_identities"),
                self.assertRaisesRegex(ValueError, "missing paired evidence"),
            ):
                merge_reports(sources)

    def test_merge_failure_writes_evidence_and_nonzero_verdict(self):
        from scripts.release_performance import run_merge

        with TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertEqual(run_merge(root / "missing", root / "output"), 1)
            report = json.loads((root / "output/results.json").read_text())
            self.assertIn("missing python suite results", report["errors"][0])
            self.assertIn("incomparable", (root / "output/summary.md").read_text())

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

    async def test_prepare_builds_only_the_selected_suite_binaries(self):
        import sys
        from types import ModuleType
        from unittest.mock import Mock

        from scripts.release_performance import prepare

        for suite, targets in (
            ("python", ("allocation_regression",)),
            ("core", ("core",)),
            ("stream_join_perf", ("stream_join_perf",)),
        ):
            with self.subTest(suite=suite), TemporaryDirectory() as raw:
                root = Path(raw)
                release_module = ModuleType("scripts.benchmark_suite.release")
                release_module.load_release = Mock(
                    side_effect=[{"git_sha": "a" * 40}, {"git_sha": "b" * 40}]
                )
                with (
                    patch.dict(
                        sys.modules,
                        {"scripts.benchmark_suite.release": release_module},
                    ),
                    patch("scripts.release_performance.command", new=AsyncMock()),
                    patch(
                        "scripts.release_performance.install",
                        new=AsyncMock(return_value=root),
                    ),
                    patch(
                        "scripts.release_performance.build_binaries",
                        new=AsyncMock(return_value={}),
                    ) as build,
                    patch(
                        "scripts.release_performance.build_provenance",
                        side_effect=[
                            {"git_sha": "a" * 40},
                            {"git_sha": "b" * 40},
                        ],
                    ),
                    patch(
                        "scripts.release_performance.with_compiled_dependencies",
                        side_effect=lambda identity, *args: identity,
                    ),
                ):
                    await prepare(
                        argparse.Namespace(
                            suite=suite, output=root, baseline_source=root
                        )
                    )
                self.assertEqual(
                    [call.kwargs["targets"] for call in build.call_args_list],
                    [targets] * 2,
                )
                self.assertEqual(
                    (root / "rust-provenance.json").is_file(), suite != "python"
                )
