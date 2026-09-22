from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_release_collects_pairs_and_always_uploads_failure_evidence(self):
        text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn("  acceptance-gates:\n", text)
        self.assertIn("  crate:\n", text)
        job = text.split("  acceptance-gates:\n", 1)[1].split("  crate:\n", 1)[0]
        self.assertIn("python -m scripts.release_performance", job)
        self.assertIn("--require-hashes -r benchmarks/requirements.lock", job)
        for step in ("performance", "security", "soak"):
            self.assertIn(f"id: {step}", job)
        self.assertIn("--acceptance-summary", job)
        self.assertIn("ACCEPTANCE_STEPS: ${{ toJSON(steps) }}", job)
        for heading in (
            "Summarize acceptance outcomes",
            "Upload release performance evidence",
        ):
            self.assertIn(f"- name: {heading}\n", job)
            block = job.split(f"- name: {heading}\n", 1)[1].split("      - ", 1)[0]
            self.assertIn("if: always()", block)
        self.assertIn("retention-days: 30", job)
        self.assertNotIn("continue-on-error", job)
        self.assertIn("needs: acceptance-gates", text)
