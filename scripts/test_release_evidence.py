"""Release evidence must be comparable before it can receive a timing verdict."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.toolkit import fingerprint_json
from scripts.verify_perf_gates import check_regression, load_baseline


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


class ReleaseIdentityTests(unittest.TestCase):
    def load(self, rows):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "result.json").write_text(json.dumps({"benchmarks": rows}))
            return load_baseline(root)

    def test_loader_preserves_raw_identity_and_rejects_corrupt_hashes(self):
        loaded = self.load([benchmark()])
        self.assertEqual(next(iter(loaded.values()))["metadata"], identity())
        for name in ("machine", "dependency", "workload"):
            for bad in (None, "", "wrong", "f" * 64):
                with (
                    self.subTest(name=name, bad=bad),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    self.load([benchmark({**identity(), f"{name}_fingerprint": bad})])

    def test_duplicate_cases_never_overwrite_evidence(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load([benchmark(), benchmark()])

    def test_identity_mismatch_is_incomparable_not_regression(self):
        baseline = self.load([benchmark()])
        raw = identity()
        raw["dependency_identity"] = {"python": "3.14"}
        raw["dependency_fingerprint"] = fingerprint_json(raw["dependency_identity"])
        candidate = self.load([benchmark(raw)])
        with self.assertRaisesRegex(ValueError, "incomparable.*dependency"):
            check_regression(baseline, candidate)

    def test_unpaired_summary_cannot_pass_release(self):
        values = self.load([benchmark()])
        before = copy.deepcopy(values)
        with self.assertRaisesRegex(ValueError, "paired.*required"):
            check_regression(values, values)
        self.assertEqual(values, before)


class BenchmarkRecordingTests(unittest.TestCase):
    def test_common_recording_attaches_identities_without_mutating_metadata(self):
        import importlib.util
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch

        stub = ModuleType("calc_flow")
        stub.Batch = object
        path = Path(__file__).resolve().parents[1] / "benchmarks/support.py"
        spec = importlib.util.spec_from_file_location("release_support_fixture", path)
        module = importlib.util.module_from_spec(spec)
        dependencies = {
            name: ModuleType(name) for name in ("numpy", "pyarrow", "psutil", "cpuinfo")
        }
        dependencies["psutil"].Process = lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=1000)
        )
        dependencies["cpuinfo"].get_cpu_info = lambda: {"brand_raw": "fixture"}
        with patch.dict(
            sys.modules, {"calc_flow": stub, spec.name: module, **dependencies}
        ):
            spec.loader.exec_module(module)
        original = {"existing": "value"}
        fixture = SimpleNamespace(extra_info=original)
        with (
            patch.object(module, "_machine_identity", return_value={"cpu": "fixture"}),
            patch.object(module, "version", return_value="1.0"),
        ):
            module.record_benchmark(
                fixture, scenario="case", input_rows=1000, output_rows=1000
            )
        self.assertEqual(original, {"existing": "value"})
        for name in ("machine", "dependency", "workload"):
            self.assertEqual(
                fixture.extra_info[f"{name}_fingerprint"],
                fingerprint_json(fixture.extra_info[f"{name}_identity"]),
            )
        self.assertEqual(
            fixture.extra_info["workload_identity"]["scope"], "pytest-native-boundary"
        )


class CriterionEvidenceTests(unittest.TestCase):
    def test_criterion_loader_requires_and_preserves_raw_identity(self):
        from scripts.verify_perf_gates import load_criterion

        with TemporaryDirectory() as raw:
            root = Path(raw)
            directory = root / "core/case/exact-base"
            directory.mkdir(parents=True)
            (directory / "estimates.json").write_text(
                json.dumps(
                    {
                        "mean": {
                            "point_estimate": 1e9,
                            "confidence_interval": {
                                "lower_bound": 0.99e9,
                                "upper_bound": 1.01e9,
                            },
                        }
                    }
                )
            )
            with self.assertRaises((ValueError, OSError)):
                load_criterion(root, "exact-base")
            (directory / "identity.json").write_text(json.dumps(identity()))
            loaded = load_criterion(root, "exact-base")
            self.assertEqual(loaded["core/case"]["metadata"], identity())

    def test_independent_criterion_estimates_cannot_receive_a_release_verdict(self):
        from scripts.verify_perf_gates import check_criterion_regression

        values = {
            "case": {
                "name": "case",
                "mean_seconds": 1.0,
                "lower_seconds": 0.99,
                "upper_seconds": 1.01,
                "metadata": identity(),
            }
        }
        with self.assertRaisesRegex(ValueError, "paired.*required"):
            check_criterion_regression(values, values)


class WorkloadDescriptorTests(unittest.TestCase):
    def test_flat_scope_scale_backend_or_rows_cannot_disagree_with_identity(self):
        from scripts.benchmark_suite.identity import validate_identity

        for key, bad in (
            ("scope", "other"),
            ("scale", "standard"),
            ("backend", "other"),
            ("input_rows", 2),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "workload"):
                validate_identity({**identity(), key: bad})
