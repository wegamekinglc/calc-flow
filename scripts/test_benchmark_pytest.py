from __future__ import annotations

import importlib.util
import json
import sys
import sysconfig
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite.legacy import _pytest_run, block_problem

_SMOKE_DEPENDENCIES = (
    "numpy",
    "pyarrow",
    "psutil",
    "cpuinfo",
    "pytest",
    "pytest_benchmark",
    "calc_flow",
)


class PytestHarnessTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in _SMOKE_DEPENDENCIES),
        "isolated recorder smoke requires the Python benchmark environment",
    )
    async def test_child_process_collects_old_checkout_with_current_support(self):
        from scripts.benchmark_suite.identity import validate_identity
        from scripts.benchmark_suite.process import ROOT

        with TemporaryDirectory() as raw:
            root = Path(raw)
            blocks = {}
            for side in ("baseline", "candidate"):
                source = root / side
                _write_old_checkout(source)
                if side == "candidate":
                    (source / "benchmarks/support.py").write_bytes(
                        (ROOT / "benchmarks/support.py").read_bytes()
                    )
                output = root / f"{side}-results"
                output.mkdir()
                rows = await _pytest_run(
                    {"family": "python", "scale": "overhead"},
                    source,
                    Path(sysconfig.get_paths()["purelib"]),
                    output,
                )
                self.assertEqual(len(rows), 1)
                for row in rows.values():
                    validate_identity(row["metadata"])
                blocks[side] = [rows] * 2
            name = next(iter(blocks["baseline"][0]))
            self.assertIsNone(block_problem(name, blocks))

    async def test_old_checkout_uses_current_driver_and_retains_workload_source(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "baseline"
            tests = source / "benchmarks"
            tests.mkdir(parents=True)
            test = tests / "test_case.py"
            test.write_text("# baseline workload\n")
            output = root / "output"
            output.mkdir()
            name = "benchmarks/test_case.py::test_case"

            async def run(argv, **kwargs):
                self.assertEqual(Path(argv[1]).name, "pytest_driver.py")
                self.assertNotIn("-p", argv)
                self.assertEqual(kwargs["cwd"], source)
                (output / "inventory.json").write_text(json.dumps([name]))
                (output / "pytest.json").write_text(
                    json.dumps(
                        {
                            "benchmarks": [
                                {
                                    "fullname": name,
                                    "stats": {"data": [1.0], "median": 1.0},
                                    "extra_info": {},
                                }
                            ]
                        }
                    )
                )

            with patch("scripts.benchmark_suite.legacy.command", side_effect=run):
                rows = await _pytest_run({"family": "python"}, source, root, output)
            self.assertIn("benchmark_source_sha256", rows[name]["metadata"])
            self.assertIn("benchmark_support_sha256", rows[name]["metadata"])

    def test_common_support_replaces_old_recorder_before_collection(self):
        from scripts.benchmark_suite.pytest_driver import load_support

        old = ModuleType("benchmarks.support")
        old.record_benchmark = lambda benchmark, **kwargs: setattr(
            benchmark, "extra_info", kwargs
        )
        fixture = SimpleNamespace(extra_info={})
        old.record_benchmark(fixture, scenario="case", input_rows=1000, output_rows=1)
        self.assertNotIn("machine_fingerprint", fixture.extra_info)
        modules = _dependencies()
        with patch.dict(sys.modules, {"benchmarks.support": old, **modules}):
            support = load_support()
            self.assertIs(sys.modules["benchmarks.support"], support)
            self.assertEqual(
                Path(support.__file__).resolve(),
                Path(__file__).resolve().parents[1] / "benchmarks/support.py",
            )
            with patch.object(support, "version", return_value="1.0"):
                support.record_benchmark(
                    fixture, scenario="case", input_rows=1000, output_rows=1
                )
        row = {
            "rows": 1000,
            "scope": "pytest-native-boundary",
            "metadata": fixture.extra_info,
        }
        blocks = {side: [{"case": row}] * 2 for side in ("baseline", "candidate")}
        self.assertIsNone(block_problem("case", blocks))
        for name in ("machine", "dependency", "workload"):
            with self.subTest(identity=name):
                corrupted = {**fixture.extra_info, f"{name}_fingerprint": "0" * 64}
                invalid = {**row, "metadata": corrupted}
                both = {side: [{"case": invalid}] * 2 for side in blocks}
                self.assertIsNotNone(block_problem("case", both))
                self.assertIn("corrupt", block_problem("case", both))
        del fixture.extra_info["machine_fingerprint"]
        self.assertIn("missing", block_problem("case", blocks))


def _dependencies() -> dict:
    modules = {
        name: ModuleType(name)
        for name in ("numpy", "pyarrow", "psutil", "cpuinfo", "calc_flow")
    }
    modules["calc_flow"].Batch = object
    modules["psutil"].Process = lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=1000)
    )
    modules["cpuinfo"].get_cpu_info = lambda: {"brand_raw": "fixture CPU"}
    return modules


def _write_old_checkout(source: Path) -> None:
    for package in ("benchmarks", "scripts", "scripts/benchmark_suite"):
        path = source / package
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").touch()
    (source / "scripts/benchmark_suite/pytest_plugin.py").write_text(
        "raise AssertionError('loaded baseline collection plugin')\n"
    )
    (source / "benchmarks/support.py").write_text(
        "def record_benchmark(benchmark, **metadata):\n"
        "    benchmark.extra_info = dict(metadata)\n"
    )
    (source / "benchmarks/test_record.py").write_text(
        "import pytest\n"
        "from benchmarks.support import record_benchmark\n"
        "@pytest.mark.benchmark(min_rounds=1, max_time=0.001)\n"
        "def test_record(benchmark):\n"
        "    record_benchmark(benchmark, scenario='case', "
        "input_rows=1, output_rows=1)\n"
        "    assert benchmark(lambda: 1) == 1\n"
    )
