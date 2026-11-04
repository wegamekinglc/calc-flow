"""Require the measured executable and product library to be rebuilt."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.benchmark_suite.sum_state_build import compiled_binary


class SumStateBuildTests(unittest.TestCase):
    def messages(self):
        return [
            {
                "reason": "compiler-artifact",
                "target": {"name": "calc_flow", "kind": ["lib"]},
                "fresh": False,
            },
            {
                "reason": "compiler-artifact",
                "target": {"name": "grouped_sum_state_benchmark", "kind": ["example"]},
                "fresh": False,
                "executable": "/target/example",
            },
        ]

    def test_selects_rebuilt_diagnostic_executable(self):
        self.assertEqual(compiled_binary(self.messages()), Path("/target/example"))

    def test_rejects_cached_product_from_another_worktree(self):
        messages = self.messages()
        messages[0]["fresh"] = True
        with self.assertRaisesRegex(ValueError, "rebuild"):
            compiled_binary(messages)

    def test_rejects_missing_or_duplicate_diagnostic(self):
        messages = self.messages()
        for invalid in [messages[:1], messages + [messages[1]]]:
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "rebuild"),
            ):
                compiled_binary(invalid)
