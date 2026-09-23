from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_release_collects_pairs_and_always_uploads_failure_evidence(self):
        text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn("  acceptance-gates:\n", text)
        self.assertIn("  crate:\n", text)
        job = text.split("  acceptance-gates:\n", 1)[1].split("  soak-gates:\n", 1)[0]
        self.assertIn("python -m scripts.release_performance", job)
        self.assertIn("--require-hashes -r benchmarks/requirements.lock", job)
        for step in ("performance", "security"):
            self.assertIn(f"id: {step}", job)
        self.assertIn("--steps performance,security", job)
        self.assertNotIn("id: soak", job)
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

    def test_release_soaks_run_in_a_parallel_job_gating_all_packaging(self):
        text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn("  soak-gates:\n", text)
        job = text.split("  soak-gates:\n", 1)[1].split("  crate:\n", 1)[0]
        self.assertIn("needs: prepare-python-release", job)
        self.assertIn("id: soak", job)
        self.assertIn('CALC_FLOW_STREAM_SOAK: "1"', job)
        self.assertIn('CALC_FLOW_M5_CHECKPOINT_SOAK: "1"', job)
        self.assertIn('CALC_FLOW_M7_WINDOW_STATE_SOAK: "1"', job)
        self.assertIn("twenty_minute_two_source_slow_sink", job)
        self.assertIn("twenty_minute_epoch_checkpoint_restart", job)
        self.assertIn("high_cardinality_window_state_exceeds_legacy_json_limit", job)
        self.assertNotIn("continue-on-error", job)
        for downstream in (
            "  crate:\n",
            "  wheels:\n",
            "  sdist:\n",
            "  studio-wheel:\n",
        ):
            block = text.split(downstream, 1)[1].split("  audits:\n", 1)[0]
            self.assertIn("needs: [acceptance-gates, soak-gates]", block[:200])
