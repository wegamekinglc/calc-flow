from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseWorkflowTests(unittest.TestCase):
    def test_release_jobs_only_prepare_build_verify_test_and_publish_core(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        jobs = re.findall(
            r"^  ([a-z-]+):$", workflow.split("jobs:\n", 1)[1], re.MULTILINE
        )

        self.assertEqual(
            jobs,
            [
                "prepare-python-release",
                "wheels",
                "sdist",
                "wheel-python-versions",
                "verify-core-artifacts",
                "publish-python-core",
            ],
        )
        self.assertNotIn("release_performance", workflow)
        self.assertNotIn("cargo test", workflow)
        self.assertNotIn("npm audit", workflow)
        self.assertNotIn("studio-wheel", workflow)
        self.assertNotIn("smoke_wheel.py", workflow)

    def test_installed_wheel_unit_tests_gate_tagged_publication(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        unit_tests = workflow.split("  wheel-python-versions:\n", 1)[1].split(
            "  verify-core-artifacts:\n", 1
        )[0]
        verification = workflow.split("  verify-core-artifacts:\n", 1)[1].split(
            "  publish-python-core:\n", 1
        )[0]
        publication = workflow.split("  publish-python-core:\n", 1)[1]

        self.assertIn("needs: [wheels, sdist]", unit_tests)
        self.assertIn(
            "name: wheel-ubuntu-latest-x86_64-${{ matrix.python_tag }}", unit_tests
        )
        self.assertIn("python -m pytest -q", unit_tests)
        self.assertIn("needs: [wheels, sdist]", verification)
        self.assertIn("--dist-dir dist --core-only", verification)
        self.assertIn(
            "needs: [verify-core-artifacts, wheel-python-versions]", publication
        )
        self.assertIn("sha256sum --check ../release-manifest.txt", publication)
        self.assertIn(
            "if: github.event_name == 'push' && github.ref_type == 'tag'", publication
        )
        self.assertIn("id-token: write", publication)
        self.assertIn("packages-dir: dist", publication)


if __name__ == "__main__":
    unittest.main()
