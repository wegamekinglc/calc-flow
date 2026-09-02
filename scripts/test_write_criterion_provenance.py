"""Tests for structured Criterion benchmark provenance."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.write_criterion_provenance import build_provenance


class CriterionProvenanceTests(unittest.TestCase):
    def test_records_exact_source_and_comparable_fingerprints(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("Cargo.lock").write_text("locked\n", encoding="utf-8")
            bench = root / "crates/calc-flow/benches/core.rs"
            bench.parent.mkdir(parents=True)
            bench.write_text("fn main() {}\n", encoding="utf-8")
            with (
                patch(
                    "scripts.write_criterion_provenance._command",
                    side_effect=("1" * 40, "", "rustc 1.88.0", "cargo 1.88.0"),
                ),
                patch(
                    "scripts.write_criterion_provenance._machine_identity",
                    return_value={"cpu_model": "stable", "logical_cpu_count": 8},
                ),
            ):
                document = build_provenance(
                    root, [Path("crates/calc-flow/benches/core.rs")]
                )

        self.assertEqual(document["git_sha"], "1" * 40)
        self.assertIs(document["tracked_worktree_clean"], True)
        self.assertEqual(document["benchmarks"], ["core"])
        self.assertRegex(document["machine_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(document["dependency_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(document["workload_fingerprint"], r"^[0-9a-f]{64}$")

    def test_rejects_missing_or_escaping_benchmark_sources(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("Cargo.lock").write_text("locked\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "benchmark source"):
                build_provenance(root, [Path("missing.rs")])
            with self.assertRaisesRegex(ValueError, "escapes repository"):
                build_provenance(root, [Path("../outside.rs")])

    def test_rejects_dirty_tracked_worktree(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("Cargo.lock").write_text("locked\n", encoding="utf-8")
            bench = root / "core.rs"
            bench.write_text("fn main() {}\n", encoding="utf-8")
            with (
                patch(
                    "scripts.write_criterion_provenance._command",
                    side_effect=("1" * 40, " M core.rs"),
                ),
                self.assertRaisesRegex(ValueError, "clean tracked worktree"),
            ):
                build_provenance(root, [Path("core.rs")])


if __name__ == "__main__":
    unittest.main()
