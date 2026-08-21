"""Unit tests for the M7-02 security gate runner."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts.verify_security_gates import (
    AUDIT_COMMANDS,
    THREAT_EVIDENCE,
    THREAT_MODEL,
    EvidenceRef,
    main,
    print_checklist,
    run_audits,
    validate_evidence,
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
        names = {command.name for command in AUDIT_COMMANDS}
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


class TestEvidenceValidation(unittest.TestCase):
    def test_every_threat_has_existing_named_evidence(self) -> None:
        self.assertEqual(validate_evidence(), ())

    def test_stale_evidence_symbol_fails_closed(self) -> None:
        threat = next(iter(THREAT_EVIDENCE))
        stale = EvidenceRef(
            THREAT_EVIDENCE[threat][0].path,
            "definitely_missing_security_evidence_symbol",
        )
        with patch.dict(THREAT_EVIDENCE, {threat: (stale,)}):
            failures = validate_evidence()

        self.assertTrue(any("missing symbol" in failure for failure in failures))


class TestAuditExecution(unittest.TestCase):
    @patch("scripts.verify_security_gates.subprocess.run")
    def test_runs_every_declared_audit(self, run) -> None:
        run.return_value = SimpleNamespace(returncode=0)
        with redirect_stdout(io.StringIO()):
            result = run_audits()
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, len(AUDIT_COMMANDS))
        for call, command in zip(run.call_args_list, AUDIT_COMMANDS, strict=True):
            self.assertEqual(call.args[0], list(command.argv))
            self.assertEqual(call.kwargs["cwd"], command.cwd)
            self.assertTrue(call.kwargs["check"])

    @patch("scripts.verify_security_gates.subprocess.run")
    def test_audit_failure_stops_the_gate(self, run) -> None:
        # This test constructs the exception; it never launches a process.
        from subprocess import CalledProcessError  # nosec B404

        run.side_effect = CalledProcessError(7, ["cargo", "audit"])
        with redirect_stdout(io.StringIO()):
            result = run_audits()
        self.assertEqual(result, 7)
        self.assertEqual(run.call_count, 1)

    @patch("scripts.verify_security_gates.run_audits", return_value=9)
    def test_main_default_executes_the_audits(self, run_audits_mock) -> None:
        import sys

        old_argv = sys.argv
        sys.argv = ["verify_security_gates"]
        try:
            with redirect_stdout(io.StringIO()):
                result = main()
        finally:
            sys.argv = old_argv
        self.assertEqual(result, 9)
        run_audits_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
