from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_wheel_inspection_keeps_platform_checks_and_scopes_helper_tests(
        self,
    ) -> None:
        text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        wheels = text.split("  wheels:\n", 1)[1].split("  sdist:\n", 1)[0]
        self.assertEqual(wheels.count("python scripts/inspect_wheel.py core-wheel"), 1)
        self.assertEqual(wheels.count("python -m unittest discover -s scripts"), 1)
        helper = wheels.split("      - name: Verify release helpers\n", 1)[1].split(
            "      - name: Inspect wheel contents\n", 1
        )[0]
        self.assertIn("matrix.os == 'ubuntu-latest'", helper)
        self.assertIn("matrix.target == 'x86_64'", helper)
        self.assertIn("matrix.python_tag == 'cp313'", helper)
        inspector = wheels.split("      - name: Inspect wheel contents\n", 1)[1].split(
            "      - name: Smoke native wheel\n", 1
        )[0]
        self.assertNotIn("if:", inspector)
        self.assertNotIn("unittest", inspector)

    def test_release_collects_three_suites_in_parallel_before_merging_verdict(self):
        text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        collection = text.split("  performance-collection:\n", 1)[1].split(
            "  acceptance-gates:\n", 1
        )[0]
        self.assertIn("suite: [python, core, stream_join_perf]", collection)
        self.assertIn('--suite "${{ matrix.suite }}"', collection)
        self.assertIn("--require-hashes -r benchmarks/requirements.lock", collection)
        self.assertIn("if: always()", collection)
        self.assertIn("release-performance-${{ matrix.suite }}", collection)
        acceptance = text.split("  acceptance-gates:\n", 1)[1].split(
            "  soak-gates:\n", 1
        )[0]
        self.assertIn("performance-collection", acceptance)
        self.assertIn("actions/download-artifact@", acceptance)
        self.assertIn("--merge-suites", acceptance)
        self.assertIn(
            "COLLECTION_RESULT: ${{ needs.performance-collection.result }}", acceptance
        )
        self.assertIn('test "${COLLECTION_RESULT}" = success', acceptance)

    def test_release_collects_pairs_and_always_uploads_failure_evidence(self):
        text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn("  acceptance-gates:\n", text)
        self.assertIn("  crate:\n", text)
        job = text.split("  acceptance-gates:\n", 1)[1].split("  soak-gates:\n", 1)[0]
        self.assertIn("python -m scripts.release_performance", job)
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
