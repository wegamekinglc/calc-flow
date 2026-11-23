from __future__ import annotations

import unittest
from pathlib import Path

from scripts import sparse_changelog_wheels as wheels


class GroupHashRegistrationTests(unittest.TestCase):
    def test_only_registered_artifact_source_and_tree_are_accepted(self):
        workflow = "a" * 40
        commit = "b" * 40
        tree = "c" * 40
        registry = {
            "baseline": {
                "requested_sha": workflow,
                "artifact": 7,
                "checkout": commit,
                "tree": tree,
            }
        }
        self.assertEqual(wheels.product_tree(registry, workflow, 7, commit, tree), tree)
        for observed in (
            ("d" * 40, 7, commit, tree),
            (workflow, 8, commit, tree),
            (workflow, 7, "e" * 40, tree),
            (workflow, 7, commit, "f" * 40),
        ):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                wheels.product_tree(registry, *observed)

    def test_dedicated_job_registers_its_480_second_bound(self):
        from scripts import sparse_changelog_ci as ci

        self.assertEqual(ci.GROUP_HASH_JOB_SECONDS, 480)
        job = (Path(ci.ROOT) / ".github/workflows/benchmarks.yml").read_text()
        definition = job.split("  sparse-changelog:\n", 1)[1].split(
            "  suite-recovery:\n", 1
        )[0]
        self.assertIn("timeout-minutes: 8", definition)
