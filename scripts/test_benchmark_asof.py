from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite.asof import CASES, asof_rows
from scripts.benchmark_suite.rust import (
    bench_targets,
    build_binaries,
    measure_rust,
    run_binary,
)


def report() -> dict:
    cases = []
    for name, config in CASES.items():
        sample = {
            "config": config,
            "seconds": 0.1,
            "output_rows": config["pending"],
            "chunks": [[128, 0.1]] * (config["pending"] // 128),
            "max_chunk_bytes": 8192,
            "rss_available": True,
            "rss_before_bytes": 10,
            "rss_peak_bytes": 20,
            "allocation_peak_bytes": 30,
            "allocation_total_bytes": 1000,
            "allocation_count": 100,
            "checkpoint_before_bytes": 200,
            "checkpoint_after_bytes": 100,
            "admission_seconds_untimed": 0.1,
            "restore_seconds_untimed": 0.1 if config["restored"] else None,
            "capture_seconds_untimed": 0.01,
            "after_status": {
                "pending_left_rows": 0,
                "matched_rows": config["pending"],
                "retained_right_rows": config["retained"],
            },
            "validated_all_rows": False,
        }
        cases.append(
            {
                "name": name,
                "config": config,
                "oracle": {**sample, "validated_all_rows": True},
                "samples": [sample] * 20,
            }
        )
    return {
        "schema": "calc-flow.asof-finalization.v1",
        "scope": "operator-watermark-settlement",
        "cases": cases,
    }


class AsofEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_maintained_native_target_keeps_eight_cases_and_raw_diagnostics(self):
        self.assertIn("stream_asof_perf", bench_targets(Path.cwd()))
        evidence = report()
        with TemporaryDirectory() as raw:
            root = Path(raw)

            async def command(argv, **_kwargs):
                self.assertIn("--output", argv)
                Path(argv[argv.index("--output") + 1]).write_text(json.dumps(evidence))

            with patch("scripts.benchmark_suite.rust.command", side_effect=command):
                rows = await run_binary(
                    "stream_asof_perf", root / "native", root, root, "candidate"
                )
            self.assertEqual(len(rows), 8)
            for case in evidence["cases"]:
                row = rows[f"stream_asof_perf/{case['name']}"]
                self.assertEqual(row["samples"], [0.1] * 20)
                self.assertEqual(row["metadata"]["observations"], case["samples"])
                self.assertEqual(row["metadata"]["oracle"], case["oracle"])

    def test_incomplete_or_corrupt_asof_evidence_is_rejected(self):
        invalid = []
        for field, value in (("schema", "bad"), ("scope", "bad"), ("cases", [])):
            invalid.append({**report(), field: value})
        item = report()
        item["cases"].pop()
        invalid.append(item)
        item = report()
        item["cases"].append(item["cases"][0])
        invalid.append(item)
        for field, value in (
            ("seconds", float("nan")),
            ("seconds", -1),
            ("output_rows", 1),
            ("allocation_total_bytes", None),
            ("allocation_peak_bytes", -1),
            ("rss_available", "yes"),
            ("chunks", []),
            ("after_status", {}),
        ):
            item = copy.deepcopy(report())
            item["cases"][0]["samples"][0][field] = value
            invalid.append(item)
        item = report()
        item["cases"][0]["samples"] = item["cases"][0]["samples"][:19]
        invalid.append(item)
        item = report()
        item["cases"][0]["oracle"]["validated_all_rows"] = False
        invalid.append(item)
        with TemporaryDirectory() as raw:
            path = Path(raw) / "result.json"
            for index, item in enumerate(invalid):
                with self.subTest(index=index):
                    path.write_text(json.dumps(item))
                    with self.assertRaises(ValueError):
                        asof_rows(path)


class AsofInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_source_rebuilds_its_product_library_before_linking(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            shared = root / "target"
            deps = shared / "release/deps"
            deps.mkdir(parents=True)
            library = deps / "libcalc_flow-samehash.rlib"
            metadata = deps / "libcalc_flow-samehash.rmeta"
            dependency = deps / "libdatafusion-dependency.rlib"
            library.write_bytes(b"baseline implementation")
            metadata.write_bytes(b"baseline metadata")
            dependency.write_bytes(b"reusable dependency")
            compiled = []

            async def command(argv, **kwargs):
                if not library.exists():
                    compiled.append(kwargs["cwd"])
                    self.assertFalse(metadata.exists())
                    library.write_bytes(b"candidate implementation")
                target = argv[argv.index("--bench") + 1]
                executable = deps / target
                executable.write_bytes(library.read_bytes())
                kwargs["log"].write_text(
                    json.dumps(
                        {
                            "reason": "compiler-artifact",
                            "target": {"name": target},
                            "executable": str(executable),
                        }
                    )
                )

            with (
                patch(
                    "scripts.benchmark_suite.rust.bench_targets",
                    return_value=["core", "stream_asof_perf"],
                ),
                patch("scripts.benchmark_suite.rust.command", side_effect=command),
            ):
                binaries = await build_binaries(root, root, shared)
            self.assertEqual(compiled, [root])
            self.assertEqual(dependency.read_bytes(), b"reusable dependency")
            for binary in binaries.values():
                self.assertEqual(binary.read_bytes(), b"candidate implementation")

    async def test_removed_target_still_requires_an_explicit_migration(self):
        binaries = [
            {"core": Path("core"), "previous": Path("previous")},
            {"core": Path("core")},
        ]
        with (
            patch(
                "scripts.benchmark_suite.rust.build_binaries",
                AsyncMock(side_effect=binaries),
            ),
            patch("scripts.benchmark_suite.rust._rust_provenance", return_value={}),
            self.assertRaisesRegex(ValueError, "targets removed"),
        ):
            await measure_rust(
                {"id": "rust", "family": "rust"},
                {},
                {"baseline": Path("base"), "candidate": Path("head")},
                Path("target/test"),
            )

    async def test_added_target_is_new_coverage_without_fabricating_baseline(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "binary"
            binary.write_bytes(b"fixture")
            binaries = {
                "baseline": {"core": binary},
                "candidate": {"core": binary, "stream_asof_perf": binary},
            }
            row = {
                "rows": 100_000,
                "scope": "operator-watermark-settlement",
                "metadata": {},
                "samples": [0.1],
            }

            async def block(_binaries, _source, _output, _stamps, side):
                return (
                    {"stream_asof_perf/wide_f100_slow": row}
                    if side == "candidate"
                    else {},
                    [],
                )

            with (
                patch(
                    "scripts.benchmark_suite.rust.build_binaries",
                    AsyncMock(side_effect=list(binaries.values())),
                ),
                patch("scripts.benchmark_suite.rust._rust_provenance", return_value={}),
                patch(
                    "scripts.benchmark_suite.rust.declared_migrations", return_value={}
                ),
                patch(
                    "scripts.benchmark_suite.rust._stamp_fingerprints",
                    return_value={"baseline": {}, "candidate": {}},
                ),
                patch("scripts.benchmark_suite.rust._rust_block", side_effect=block),
                patch(
                    "scripts.benchmark_suite.rust._allocation_reports",
                    AsyncMock(return_value={}),
                ),
                patch("scripts.benchmark_suite.rust.allocation_rows", return_value=[]),
            ):
                result = await measure_rust(
                    {"id": "rust", "family": "rust"},
                    {},
                    {"baseline": root, "candidate": root},
                    root,
                )
            self.assertEqual(result["errors"], [])
            case = result["cases"][0]
            self.assertEqual(case["comparison"], "new")
            self.assertEqual(case["result"]["verdict"], "new-coverage")
            self.assertEqual(case["baseline"], [])
            self.assertEqual(case["candidate"], [[0.1], [0.1]])
