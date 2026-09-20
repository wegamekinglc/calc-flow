"""Tests for structured Criterion benchmark provenance."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import scripts.write_criterion_provenance as provenance
from scripts.write_criterion_provenance import build_provenance


@contextmanager
def stable_identity():
    with (
        patch(
            "scripts.write_criterion_provenance._git_head",
            return_value="1" * 40,
        ),
        patch(
            "scripts.write_criterion_provenance._git_status",
            return_value="",
        ),
        patch(
            "scripts.write_criterion_provenance._rustc_version",
            return_value="rustc 1.88.0",
        ),
        patch(
            "scripts.write_criterion_provenance._cargo_version",
            return_value="cargo 1.88.0",
        ),
        patch(
            "scripts.write_criterion_provenance._machine_identity",
            return_value={"cpu_model": "stable", "logical_cpu_count": 8},
        ),
    ):
        yield


class CriterionProvenanceTests(unittest.TestCase):
    def test_git_head_uses_fixed_argv_without_shell(self) -> None:
        with patch(
            "scripts.write_criterion_provenance.command_output",
            return_value="1" * 40,
        ) as run:
            result = provenance._git_head(Path("/repository"))

        self.assertEqual(result, "1" * 40)
        run.assert_called_once_with(
            ["git", "rev-parse", "HEAD^{commit}"], cwd=Path("/repository")
        )

    def test_records_exact_source_and_comparable_fingerprints(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("Cargo.lock").write_text("locked\n", encoding="utf-8")
            bench = root / "crates/calc-flow/benches/core.rs"
            bench.parent.mkdir(parents=True)
            bench.write_text("fn main() {}\n", encoding="utf-8")
            with (
                patch(
                    "scripts.write_criterion_provenance._git_head",
                    return_value="1" * 40,
                ),
                patch(
                    "scripts.write_criterion_provenance._git_status",
                    return_value="",
                ),
                patch(
                    "scripts.write_criterion_provenance._rustc_version",
                    return_value="rustc 1.88.0",
                ),
                patch(
                    "scripts.write_criterion_provenance._cargo_version",
                    return_value="cargo 1.88.0",
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
                    "scripts.write_criterion_provenance._git_head",
                    return_value="1" * 40,
                ),
                patch(
                    "scripts.write_criterion_provenance._git_status",
                    return_value=" M core.rs",
                ),
                self.assertRaisesRegex(ValueError, "clean tracked worktree"),
            ):
                build_provenance(root, [Path("core.rs")])

    def test_scoped_workload_fingerprints_isolate_each_benchmark_source(
        self,
    ) -> None:
        sources = [
            Path("crates/calc-flow/benches/core.rs"),
            Path("crates/calc-flow/benches/join.rs"),
        ]
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("Cargo.lock").write_text("locked\n", encoding="utf-8")
            benches = root / "crates/calc-flow/benches"
            benches.mkdir(parents=True)
            for source in sources:
                benches.joinpath(source.name).write_text(
                    "fn main() {}\n", encoding="utf-8"
                )
            with stable_identity():
                before = build_provenance(root, sources)
                benches.joinpath("join.rs").write_text(
                    "fn main() {} // changed\n", encoding="utf-8"
                )
                after = build_provenance(root, sources)
                alone = build_provenance(root, [sources[0]])

        self.assertEqual(
            after["scoped_workload_fingerprints"]["core"],
            before["scoped_workload_fingerprints"]["core"],
        )
        self.assertNotEqual(
            after["scoped_workload_fingerprints"]["join"],
            before["scoped_workload_fingerprints"]["join"],
        )
        self.assertNotEqual(
            after["workload_fingerprint"], before["workload_fingerprint"]
        )
        self.assertEqual(
            alone["workload_fingerprint"],
            after["scoped_workload_fingerprints"]["core"],
        )

    def test_duplicate_benchmark_stems_fail_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            root.joinpath("Cargo.lock").write_text("locked\n", encoding="utf-8")
            for relative in ("benches/core.rs", "other/core.rs"):
                bench = root / relative
                bench.parent.mkdir(parents=True)
                bench.write_text("fn main() {}\n", encoding="utf-8")
            with (
                stable_identity(),
                self.assertRaisesRegex(ValueError, "duplicate benchmark"),
            ):
                build_provenance(root, [Path("benches/core.rs"), Path("other/core.rs")])


if __name__ == "__main__":
    unittest.main()
