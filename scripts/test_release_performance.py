from __future__ import annotations

import argparse
import hashlib
import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.toolkit import sha256_file


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


def _write_release_fixture(source: Path, side: str) -> dict:
    destination = source / "builds" / side
    destination.mkdir(parents=True)
    wheel = destination / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("calc_flow/_native.so", side.encode())
    release = {
        "contract": "benchmark-release-v1",
        "build_profile": "release",
        "git_clean": True,
        "git_sha": ("a" if side == "baseline" else "b") * 40,
        "wheel": wheel.name,
        "wheel_sha256": sha256_file(wheel),
        "native_sha256": hashlib.sha256(side.encode()).hexdigest(),
    }
    (destination / "release.json").write_text(json.dumps(release))
    return release


def _write_suite_builds(source: Path, suite: str) -> dict:
    releases = {
        side: _write_release_fixture(source, side) for side in ("baseline", "candidate")
    }
    target = "allocation_regression" if suite == "python" else suite
    for side in releases:
        log = source / "rust-builds" / side / f"build-{target}.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text('{"reason": "build-finished", "success": true}\n')
        (log.parent / "binary-sha256.json").write_text(
            json.dumps({target: hashlib.sha256(f"{suite}/{side}".encode()).hexdigest()})
        )
    if suite != "python":
        (source / "rust-provenance.json").write_text(
            json.dumps(
                {side: {"git_sha": releases[side]["git_sha"]} for side in releases}
            )
        )
    return releases


def _write_suite_fixture(sources: Path, suite: str) -> None:
    from scripts.release_performance import harness_identity

    case_id = "python/one" if suite == "python" else f"rust/{suite}/one"
    relative = (
        Path("cases") / hashlib.sha256(case_id.encode()).hexdigest()[:20] / "pairs.json"
    )
    source = sources / f"release-performance-{suite}"
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_text(json.dumps({"id": case_id}))
    releases = _write_suite_builds(source, suite)
    rust_seals = {
        side: hashlib.sha256(f"{suite}/{side}".encode()).hexdigest()
        for side in releases
    }
    seals = (
        {side: releases[side]["native_sha256"] for side in releases}
        if suite == "python"
        else rust_seals
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
                        "seals": seals,
                        "result": {"verdict": "no-confirmed-regression"},
                    }
                ],
                "errors": [],
                "allocation": {"baseline": {}, "candidate": {}},
                "inventory": {suite: {"baseline": ["one"], "candidate": ["one"]}},
                "harness": harness_identity(),
                "dependency_lock_sha256": sha256_file(
                    Path(__file__).resolve().parents[1] / "benchmarks/requirements.lock"
                ),
                "releases": releases,
                "rust_binary_sha256": {suite: rust_seals} if suite != "python" else {},
            }
        )
    )


def _merge_fixture(root: Path) -> Path:
    sources = root / "suites"
    for suite in ("python", "core", "stream_join_perf"):
        _write_suite_fixture(sources, suite)
    return sources


def _merged_fixture_report(sources: Path) -> dict:
    from scripts.release_performance import merge_reports

    with (
        patch(
            "scripts.release_performance.evaluate_case",
            return_value={"verdict": "no-confirmed-regression"},
        ),
        patch("scripts.release_performance.rust_identities"),
    ):
        return merge_reports(sources)


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
                "binary_sha256": {
                    side: {
                        target: sha256_file(binary)
                        for target, binary in binaries[side].items()
                    }
                    for side in binaries
                },
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
            self.assertEqual(
                result["rust_binary_sha256"],
                {
                    target: {
                        side: sha256_file(binaries[side][target]) for side in releases
                    }
                    for target in ("core", "stream_join_perf")
                },
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
                "binary_sha256": {
                    side: {"core": "seal"} for side in ("baseline", "candidate")
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
            self.assertEqual(
                report["rust_binary_sha256"],
                {"core": {"baseline": "seal", "candidate": "seal"}},
            )
            self.assertEqual([row["id"] for row in report["cases"]], ["rust/core/one"])

    async def test_measure_rejects_binary_changed_after_build(self):
        from scripts.release_performance import measure

        with TemporaryDirectory() as raw:
            root = Path(raw)
            binaries = {}
            for side in ("baseline", "candidate"):
                binary = root / side
                binary.write_bytes(side.encode())
                binaries[side] = {"core": binary}
            context = {
                "binaries": binaries,
                "binary_sha256": {side: {"core": "c" * 64} for side in binaries},
                "roots": dict.fromkeys(binaries, root),
            }
            with (
                patch(
                    "scripts.release_performance.rust_inventory",
                    new=AsyncMock(return_value=["one"]),
                ),
                patch(
                    "scripts.release_performance.rust_identities",
                    return_value={side: {} for side in binaries},
                ),
                self.assertRaisesRegex(ValueError, "binary changed after build"),
            ):
                await measure(context, root, {"cases": [], "errors": []}, "core")

    def test_merge_checks_suite_identity_inventory_and_raw_evidence(self):
        with TemporaryDirectory() as raw:
            sources = _merge_fixture(Path(raw))
            merged = _merged_fixture_report(sources)
            self.assertEqual(len(merged["cases"]), 3)
            self.assertTrue(
                all(
                    (sources.parent / row["evidence"]).is_file()
                    for row in merged["cases"]
                )
            )

    def test_merge_rejects_rust_binary_seal_drift(self):
        with TemporaryDirectory() as raw:
            sources = _merge_fixture(Path(raw))
            result = sources / "release-performance-core/results.json"
            core = json.loads(result.read_text())
            core["cases"][0]["seals"]["baseline"] = "c" * 64
            result.write_text(json.dumps(core))
            with self.assertRaisesRegex(ValueError, "core Rust binary seal differs"):
                _merged_fixture_report(sources)

    def test_merge_rejects_rust_binary_build_digest_drift(self):
        with TemporaryDirectory() as raw:
            sources = _merge_fixture(Path(raw))
            path = (
                sources
                / "release-performance-core/rust-builds/baseline/binary-sha256.json"
            )
            path.write_text(json.dumps({"core": "c" * 64}))
            with self.assertRaisesRegex(
                ValueError, "core baseline Rust build digest differs"
            ):
                _merged_fixture_report(sources)

    def test_merge_rejects_missing_build_and_pair_evidence(self):
        core_digest = hashlib.sha256(b"rust/core/one").hexdigest()[:20]
        missing = (
            (
                "release-performance-core/rust-provenance.json",
                "missing core Rust provenance",
            ),
            (
                "release-performance-core/rust-builds/baseline/build-core.jsonl",
                "missing core baseline Rust build log",
            ),
            (
                "release-performance-core/rust-builds/baseline/binary-sha256.json",
                "missing core baseline Rust build digest",
            ),
            (
                "release-performance-python/rust-builds/baseline/build-allocation_regression.jsonl",
                "missing python baseline Rust build log",
            ),
            (
                "release-performance-python/builds/baseline/release.json",
                "missing python baseline sealed release manifest",
            ),
            (
                f"release-performance-core/cases/{core_digest}/pairs.json",
                "missing paired evidence",
            ),
        )
        for relative, message in missing:
            with self.subTest(relative=relative), TemporaryDirectory() as raw:
                sources = _merge_fixture(Path(raw))
                (sources / relative).unlink()
                with self.assertRaisesRegex(ValueError, message):
                    _merged_fixture_report(sources)

    def test_merge_rejects_rust_source_sha_drift(self):
        with TemporaryDirectory() as raw:
            sources = _merge_fixture(Path(raw))
            path = sources / "release-performance-core/rust-provenance.json"
            provenance = json.loads(path.read_text())
            provenance["baseline"]["git_sha"] = "c" * 40
            path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(
                ValueError, "core baseline Rust source differs"
            ):
                _merged_fixture_report(sources)

    def test_merge_rejects_release_sha_drift(self):
        with TemporaryDirectory() as raw:
            sources = _merge_fixture(Path(raw))
            path = sources / "release-performance-core/results.json"
            core = json.loads(path.read_text())
            core["releases"]["baseline"]["git_sha"] = "c" * 40
            path.write_text(json.dumps(core))
            with self.assertRaisesRegex(ValueError, "release SHAs disagree"):
                _merged_fixture_report(sources)

    def test_merge_retains_partial_collection_errors(self):
        with TemporaryDirectory() as raw:
            sources = _merge_fixture(Path(raw))
            path = sources / "release-performance-core/results.json"
            core = json.loads(path.read_text())
            core["cases"] = []
            core["errors"] = ["ValueError: compiled identity failed"]
            del core["rust_binary_sha256"]
            path.write_text(json.dumps(core))
            merged = _merged_fixture_report(sources)
            self.assertIn(
                "core: ValueError: compiled identity failed", merged["errors"]
            )
            self.assertEqual(len(merged["cases"]), 2)

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
                for side in ("baseline", "candidate"):
                    digest = root / "rust-builds" / side / "binary-sha256.json"
                    self.assertEqual(json.loads(digest.read_text()), {})
