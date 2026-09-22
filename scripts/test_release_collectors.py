from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from scripts.toolkit import fingerprint_json, sha256_file


def identity() -> dict:
    values = {
        "machine": {"cpu": "fixture"},
        "dependency": {"python": "3.13"},
        "workload": {
            "scenario": "case",
            "scope": "pytest-native-boundary",
            "backend": "calc-flow",
            "scale": "overhead",
            "input_rows": 1000,
        },
    }
    return {
        key: value
        for name, raw in values.items()
        for key, value in (
            (f"{name}_identity", raw),
            (f"{name}_fingerprint", fingerprint_json(raw)),
        )
    }


def benchmark(extra: dict | None = None) -> dict:
    return {
        "name": "case",
        "fullname": "benchmarks/test_case.py::test_case",
        "extra_info": identity() if extra is None else extra,
        "stats": {
            "mean": 1.0,
            "stddev": 0.01,
            "rounds": 10,
            "median": 1.0,
            "data": [1.0] * 10,
        },
    }


class ReleaseCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_invocation_uses_current_harness_sealed_site_and_raw_samples(
        self,
    ):
        from scripts.benchmark_suite.release_collectors import python_observation

        with TemporaryDirectory() as raw:
            root = Path(raw)

            async def run(argv, *, cwd, log, env, **kwargs):
                self.assertIn("--benchmark-save-data", argv)
                self.assertIn("scripts.benchmark_suite.release_pytest", argv)
                self.assertTrue(env["PYTHONPATH"].startswith(str(root / "site")))
                (root / "pytest.json").write_text(
                    json.dumps({"benchmarks": [benchmark()]})
                )
                (root / "native.json").write_text(
                    json.dumps({"native_sha256": "a" * 64})
                )

            with patch(
                "scripts.benchmark_suite.release_collectors.command", side_effect=run
            ):
                result = await python_observation(
                    "benchmarks/test_case.py::test_case", root / "site", "a" * 64, root
                )
            self.assertEqual(result["samples"], [1.0] * 10)
            self.assertEqual(result["binary_sha256"], "a" * 64)

    async def test_rust_invocation_filters_one_case_and_keeps_compiled_identity(self):
        from scripts.benchmark_suite.release_collectors import rust_observation

        with TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "core"
            binary.write_bytes(b"sealed fixture")

            async def run(argv, *, cwd, log, env, **kwargs):
                self.assertEqual(argv[1:4], ["case/1000", "--exact", "--bench"])
                path = Path(env["CRITERION_HOME"]) / "case/1000/new"
                path.mkdir(parents=True)
                (path / "benchmark.json").write_text(
                    json.dumps({"full_id": "case/1000"})
                )
                (path / "sample.json").write_text(
                    json.dumps({"iters": [10, 20], "times": [100, 200]})
                )

            with patch(
                "scripts.benchmark_suite.release_collectors.command", side_effect=run
            ):
                result = await rust_observation(
                    "case/1000", binary, root, identity(), root / "run"
                )
            self.assertEqual(result["samples"], [1e-8, 1e-8])
            self.assertEqual(result["binary_sha256"], sha256_file(binary))


class ReleaseNativeTests(unittest.TestCase):
    def test_pytest_plugin_rejects_wrong_native_before_collection(self):
        from scripts.benchmark_suite.release_pytest import pytest_sessionstart

        with TemporaryDirectory() as raw:
            root = Path(raw)
            native = root / "native.so"
            native.write_bytes(b"native")
            package = ModuleType("calc_flow")
            package._native = ModuleType("calc_flow._native")
            package._native.__file__ = str(native)
            environment = {
                "CALC_FLOW_RELEASE_NATIVE": "a" * 64,
                "CALC_FLOW_RELEASE_OBSERVED": str(root / "native.json"),
            }
            with (
                patch.dict(sys.modules, {"calc_flow": package}),
                patch.dict("os.environ", environment),
            ):
                with self.assertRaisesRegex(ValueError, "native"):
                    pytest_sessionstart(None)
                self.assertEqual(
                    json.loads((root / "native.json").read_text())["native_sha256"],
                    sha256_file(native),
                )


class ReleaseRustBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_only_requested_release_targets(self):
        from scripts.benchmark_suite.rust import build_binaries

        with TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "built"
            binary.write_bytes(b"binary")
            called = []

            async def run(argv, *, cwd, log, **kwargs):
                target = argv[argv.index("--bench") + 1]
                called.append(target)
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(
                    json.dumps(
                        {
                            "reason": "compiler-artifact",
                            "target": {"name": target},
                            "executable": str(binary),
                        }
                    )
                )

            with patch("scripts.benchmark_suite.rust.command", side_effect=run):
                result = await build_binaries(
                    root,
                    root / "out",
                    root / "shared",
                    targets=("core", "stream_join_perf"),
                )
            self.assertEqual(called, ["core", "stream_join_perf"])
            self.assertEqual(set(result), set(called))


class ReleaseRawSampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_summary_without_saved_samples_is_not_pairing_evidence(self):
        from scripts.benchmark_suite.release_collectors import python_observation

        with TemporaryDirectory() as raw:
            root = Path(raw)
            row = benchmark()
            row["stats"].pop("data")

            async def run(*args, **kwargs):
                (root / "pytest.json").write_text(json.dumps({"benchmarks": [row]}))
                (root / "native.json").write_text(
                    json.dumps({"native_sha256": "a" * 64})
                )

            with (
                patch(
                    "scripts.benchmark_suite.release_collectors.command",
                    side_effect=run,
                ),
                self.assertRaisesRegex(ValueError, "raw samples"),
            ):
                await python_observation(row["fullname"], root / "site", "a" * 64, root)
