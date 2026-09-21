from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite.join_materialization import materialization_rows


def observation() -> dict:
    return {
        **dict.fromkeys(
            (
                "allocation_peak_bytes",
                "allocation_total_bytes",
                "allocation_count",
                "rss_before_bytes",
                "rss_peak_bytes",
                "rss_first_emit_bytes",
                "chunks",
                "max_chunk_bytes",
                "queue_high_water_bytes",
                "blocked_sends",
            ),
            1,
        ),
        "rss_available": True,
        "blocked_seconds": 0.0,
        "seconds": 0.1,
        "output_rows": 100_000,
    }


def report() -> dict:
    return {
        "schema": "calc-flow.join-materialization.v1",
        "scope": "operator-bounded-edge",
        "cases": [
            {
                "name": "wide_f100_slow",
                "config": {"incoming": 1000, "fan": 100},
                "oracle": {"validated_all_rows": True, "output_rows": 100_000},
                "samples": [observation() for _ in range(20)],
            }
        ],
    }


class MaterializationEvidenceTests(unittest.TestCase):
    def test_missing_or_corrupt_memory_backpressure_evidence_is_rejected(self):
        invalid = []
        for field in (
            "allocation_peak_bytes",
            "rss_peak_bytes",
            "blocked_sends",
            "rss_available",
        ):
            item = report()
            del item["cases"][0]["samples"][0][field]
            invalid.append(item)
        for field, value in (
            ("allocation_total_bytes", -1),
            ("blocked_seconds", -1),
            ("rss_available", "yes"),
        ):
            item = report()
            item["cases"][0]["samples"][0][field] = value
            invalid.append(item)
        with TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            for item in invalid:
                path.write_text(json.dumps(item))
                with self.assertRaises(ValueError):
                    materialization_rows(path)

    def test_invalid_or_incomplete_native_evidence_is_rejected(self):
        invalid = []
        for field, value in (
            ("schema", "unknown"),
            ("scope", "unknown"),
            ("cases", []),
        ):
            invalid.append({**report(), field: value})
        for field, value in (
            ("samples", []),
            ("samples", [{"seconds": 0.1, "output_rows": 100_000}] * 19),
            ("oracle", {"validated_all_rows": False, "output_rows": 100_000}),
        ):
            item = report()
            item["cases"][0][field] = value
            invalid.append(item)
        for field, value in (
            ("seconds", -1),
            ("seconds", float("nan")),
            ("seconds", True),
            ("output_rows", 99),
        ):
            item = report()
            item["cases"][0]["samples"][0][field] = value
            invalid.append(item)
        duplicate = report()
        duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
        invalid.append(duplicate)
        with TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            for index, item in enumerate(invalid):
                with self.subTest(index=index):
                    path.write_text(json.dumps(item))
                    with self.assertRaises(ValueError):
                        materialization_rows(path)
