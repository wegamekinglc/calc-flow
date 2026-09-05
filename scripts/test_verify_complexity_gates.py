"""Unit tests for the complexity ratchet gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import verify_complexity_gates as gates


class TestMarkerCounting(unittest.TestCase):
    def test_counts_markers_case_insensitively_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.rs").write_text(
                "// #lizard forgives\n// #Lizard Forgives\n", encoding="utf-8"
            )
            (root / "b.py").write_text("x = 1\n", encoding="utf-8")
            counts = gates.count_markers(
                [root / "a.rs", root / "b.py"], gates.RUFF_MARKER, root=root
            )
        self.assertEqual(counts, {"a.rs": 2})

    def test_counts_clippy_allows_only_in_rust_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k.rs").write_text(
                "#[allow(clippy::too_many_lines)]\nfn f() {}\n", encoding="utf-8"
            )
            counts = gates.count_markers(
                [root / "k.rs"], gates.CLIPPY_MARKER, root=root
            )
        self.assertEqual(counts, {"k.rs": 1})


class TestFlattenCounts(unittest.TestCase):
    def test_flattens_plain_marker_sections(self) -> None:
        flat = gates.flatten_counts("lizard_forgives", {"a.rs": 2, "b.py": 1})
        self.assertEqual(flat, {"lizard_forgives:a.rs": 2, "lizard_forgives:b.py": 1})

    def test_flattens_per_rule_ruff_sections(self) -> None:
        flat = gates.flatten_counts(
            "ruff_complexity", {"lower.py": {"C901": 2, "PLR0913": 1}}
        )
        self.assertEqual(
            flat,
            {
                "ruff_complexity:lower.py:C901": 2,
                "ruff_complexity:lower.py:PLR0913": 1,
            },
        )


class TestCompareStates(unittest.TestCase):
    def test_passes_when_counts_hold_or_shrink(self) -> None:
        baseline = {
            "lizard_forgives": {"a.rs": 3},
            "ruff_complexity": {"b.py": {"C901": 2}},
        }
        current = {"lizard_forgives": {"a.rs": 3, "gone.rs": 0}, "ruff_complexity": {}}
        self.assertEqual(gates.compare_states(baseline, current), [])

    def test_fails_when_a_marker_count_grows(self) -> None:
        baseline = {"lizard_forgives": {"a.rs": 2}}
        current = {"lizard_forgives": {"a.rs": 3}}
        self.assertEqual(
            gates.compare_states(baseline, current),
            ["lizard_forgives:a.rs grew from 2 to 3"],
        )

    def test_fails_when_a_location_is_new(self) -> None:
        baseline = {"lizard_forgives": {}}
        current = {"lizard_forgives": {"new.rs": 1}}
        self.assertEqual(
            gates.compare_states(baseline, current),
            ["lizard_forgives:new.rs grew from 0 to 1"],
        )

    def test_fails_when_one_rule_grows_while_another_shrinks(self) -> None:
        baseline = {"ruff_complexity": {"a.py": {"C901": 3, "PLR0913": 1}}}
        current = {"ruff_complexity": {"a.py": {"C901": 2, "PLR0913": 2}}}
        self.assertEqual(
            gates.compare_states(baseline, current),
            ["ruff_complexity:a.py:PLR0913 grew from 1 to 2"],
        )

    def test_passes_when_a_baseline_section_vanishes_from_current(self) -> None:
        baseline = {"lizard_forgives": {"a.rs": 1}}
        current = {}
        self.assertEqual(gates.compare_states(baseline, current), [])


class TestBaselineIo(unittest.TestCase):
    def test_missing_baseline_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "baseline.json"
            with self.assertRaises(FileNotFoundError):
                gates.load_baseline(missing)

    def test_write_baseline_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            state = {"lizard_forgives": {"a.rs": 1}}
            gates.write_baseline(path, state)
            gates.write_baseline(path, state)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), state)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))


class TestRuffParsing(unittest.TestCase):
    def test_run_ruff_complexity_groups_findings_per_rule(self) -> None:
        payload = json.dumps(
            [
                {"filename": "python/calc_flow/array.py", "code": "C901"},
                {
                    "filename": str(
                        gates.REPO_ROOT / "python" / "calc_flow" / "array.py"
                    ),
                    "code": "C901",
                },
                {"filename": "python/calc_flow/pipeline.py", "code": "PLR0913"},
                {
                    "filename": str(
                        Path("/definitely-not-the-repo/outside.py").resolve()
                    ),
                    "code": "C901",
                },
            ]
        )
        completed = type(
            "Completed", (), {"returncode": 1, "stdout": payload, "stderr": ""}
        )()

        def fake_run(*_args: object, **_kwargs: object) -> object:
            return completed

        with patch.object(gates.subprocess, "run", fake_run):
            counts = gates.run_ruff_complexity()
        self.assertEqual(
            counts,
            {
                "python/calc_flow/array.py": {"C901": 2},
                "python/calc_flow/pipeline.py": {"PLR0913": 1},
            },
        )

    def test_run_ruff_complexity_rejects_tool_failure(self) -> None:
        completed = type(
            "Completed", (), {"returncode": 2, "stdout": "", "stderr": "boom"}
        )()

        def fake_run(*_args: object, **_kwargs: object) -> object:
            return completed

        with (
            patch.object(gates.subprocess, "run", fake_run),
            self.assertRaises(RuntimeError),
        ):
            gates.run_ruff_complexity()


class TestMain(unittest.TestCase):
    def test_update_baseline_writes_current_state(self) -> None:
        state = {
            "lizard_forgives": {},
            "clippy_complexity_allows": {},
            "ruff_complexity": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            with (
                patch.object(gates, "collect_current_state", lambda: state),
                patch.object(gates, "BASELINE_PATH", path),
            ):
                self.assertEqual(gates.main(["--update-baseline"]), 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), state)

    def test_holding_ratchet_exits_zero(self) -> None:
        state = {"lizard_forgives": {"a.rs": 1}, "ruff_complexity": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            gates.write_baseline(path, state)
            with (
                patch.object(gates, "collect_current_state", lambda: state),
                patch.object(gates, "BASELINE_PATH", path),
            ):
                self.assertEqual(gates.main([]), 0)

    def test_regression_exits_nonzero(self) -> None:
        baseline_state = {"lizard_forgives": {"a.rs": 1}, "ruff_complexity": {}}
        current_state = {"lizard_forgives": {"a.rs": 2}, "ruff_complexity": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            gates.write_baseline(path, baseline_state)
            with (
                patch.object(gates, "collect_current_state", lambda: current_state),
                patch.object(gates, "BASELINE_PATH", path),
            ):
                self.assertEqual(gates.main([]), 1)


if __name__ == "__main__":
    unittest.main()
