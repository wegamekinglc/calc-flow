"""Unit tests for the M7-02 security gate runner."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from scripts.verify_security_gates import (
    AUDIT_COMMANDS,
    THREAT_MODEL,
    main,
    print_checklist,
)


class TestThreatModel(unittest.TestCase):
    def test_threat_model_is_non_empty(self) -> None:
        self.assertGreater(len(THREAT_MODEL), 10)

    def test_all_entries_have_evidence(self) -> None:
        for entry in THREAT_MODEL:
            with self.subTest(threat=entry.threat):
                self.assertTrue(entry.evidence.strip())
                self.assertTrue(entry.boundary.strip())

    def test_covers_required_threats(self) -> None:
        threats = {entry.threat for entry in THREAT_MODEL}
        required = {
            "secret-value-in-config",
            "credential-leak-in-error",
            "path-traversal",
            "symlink-traversal",
            "decompression-bomb",
            "oversized-message",
            "malicious-schema",
            "sql-identifier-injection",
            "sql-query-injection",
            "clickhouse-dedup-token-forgery",
            "tls-disabled-by-default",
        }
        missing = required - threats
        self.assertEqual(missing, set(), f"missing threats: {missing}")

    def test_audit_commands_present(self) -> None:
        names = {name for name, _ in AUDIT_COMMANDS}
        self.assertIn("cargo audit", names)
        self.assertIn("cargo deny", names)
        self.assertIn("npm audit", names)


class TestChecklistOutput(unittest.TestCase):
    def test_print_produces_output(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured):
            print_checklist()
        output = captured.getvalue()
        self.assertIn("Threat-Model Coverage Checklist", output)
        self.assertIn("secret-value-in-config", output)
        self.assertIn("Audit Commands", output)

    def test_main_checklist_only_exits_zero(self) -> None:
        import sys

        old_argv = sys.argv
        sys.argv = ["verify_security_gates", "--checklist-only"]
        try:
            with redirect_stdout(io.StringIO()):
                result = main()
        finally:
            sys.argv = old_argv
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
