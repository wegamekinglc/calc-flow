from __future__ import annotations

import copy
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.prune_actions_caches import cache_family, stale_caches

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 31, 9, 30, tzinfo=UTC)
GRACE = timedelta(hours=24)


def _cache(
    identifier: int,
    key: str,
    *,
    created: str,
    accessed: str | None = None,
    ref: str = "refs/heads/main",
) -> dict:
    return {
        "id": identifier,
        "key": key,
        "ref": ref,
        "size_in_bytes": 1024,
        "created_at": created,
        "last_accessed_at": accessed or created,
    }


class CacheFamilyTests(unittest.TestCase):
    def test_family_strips_trailing_environment_and_lockfile_hashes(self):
        self.assertEqual(
            cache_family("v0-rust-rust-core-Linux-x64-29c9df49-6ac3f920"),
            "v0-rust-rust-core-Linux-x64",
        )
        self.assertEqual(
            cache_family(
                "setup-uv-2-x86_64-unknown-linux-gnu-ubuntu-26.04-3.13.17-pruned-"
                "7de170440d01583754a2b8ff6af8143c0806824ebd1241722f1a33161c08c5a2"
            ),
            "setup-uv-2-x86_64-unknown-linux-gnu-ubuntu-26.04-3.13.17-pruned",
        )

    def test_family_keeps_short_or_inner_segments(self):
        self.assertEqual(cache_family("pip-linux-x64-3.13"), "pip-linux-x64-3.13")
        self.assertEqual(cache_family("tool-deadbeef01-linux"), "tool-deadbeef01-linux")


class StaleCacheTests(unittest.TestCase):
    def test_closed_pull_request_caches_are_stale(self):
        caches = [
            _cache(
                1,
                "a-1234abcd",
                created="2026-10-31T08:00:00Z",
                ref="refs/pull/518/merge",
            ),
            _cache(
                2,
                "a-1234abcd",
                created="2026-10-31T08:00:00Z",
                ref="refs/pull/515/merge",
            ),
        ]

        stale = stale_caches(caches, {515}, now=NOW, grace=GRACE)

        self.assertEqual(
            [(cache["id"], reason) for cache, reason in stale],
            [(1, "pull request #518 is closed")],
        )

    def test_replaced_caches_go_once_the_replacement_matures(self):
        caches = [
            _cache(
                1,
                "v0-rust-core-Linux-x64-00000000-11111111",
                created="2026-10-29T03:00:00Z",
            ),
            _cache(
                2,
                "v0-rust-core-Linux-x64-22222222-11111111",
                created="2026-10-30T03:00:00Z",
            ),
            _cache(
                3,
                "v0-rust-core-Linux-x64-33333333-11111111",
                created="2026-10-31T03:00:00Z",
            ),
        ]

        stale = stale_caches(caches, set(), now=NOW, grace=GRACE)

        # Entry 2's only replacement is younger than the grace period.
        self.assertEqual(
            [(cache["id"], reason) for cache, reason in stale],
            [(1, "replaced by cache 2 at 2026-10-30T03:00:00Z")],
        )

    def test_caches_restored_after_a_replacement_stay(self):
        # A scheduled run keeps restoring its release-baseline entry even after
        # a manual run created a sibling for another baseline.
        caches = [
            _cache(
                1,
                "v0-rust-benchmark-baseline-build-Linux-x64-7a6d296c-90301d77",
                created="2026-10-28T10:34:00Z",
                accessed="2026-10-30T22:02:00Z",
            ),
            _cache(
                2,
                "v0-rust-benchmark-baseline-build-Linux-x64-7a6d296c-6f8c2572",
                created="2026-10-29T15:34:00Z",
            ),
        ]

        self.assertEqual(stale_caches(caches, set(), now=NOW, grace=GRACE), [])

    def test_families_are_scoped_to_their_ref(self):
        caches = [
            _cache(
                1,
                "v0-rust-core-Linux-x64-00000000-11111111",
                created="2026-10-28T03:00:00Z",
            ),
            _cache(
                2,
                "v0-rust-core-Linux-x64-22222222-11111111",
                created="2026-10-29T03:00:00Z",
                ref="refs/pull/515/merge",
            ),
        ]

        self.assertEqual(stale_caches(caches, {515}, now=NOW, grace=GRACE), [])

    def test_inputs_are_not_mutated(self):
        caches = [
            _cache(1, "a-00000000", created="2026-10-28T03:00:00Z"),
            _cache(2, "a-11111111", created="2026-10-29T03:00:00Z"),
            _cache(
                3, "b-22222222", created="2026-10-29T03:00:00Z", ref="refs/pull/9/merge"
            ),
        ]
        snapshot = copy.deepcopy(caches)
        open_pulls = {515}

        stale_caches(caches, open_pulls, now=NOW, grace=GRACE)

        self.assertEqual(caches, snapshot)
        self.assertEqual(open_pulls, {515})


class CacheCleanupWorkflowTests(unittest.TestCase):
    def test_cleanup_runs_before_each_benchmark_and_on_closed_pulls(self):
        workflow = (ROOT / ".github/workflows/cache-cleanup.yml").read_text(
            encoding="utf-8"
        )
        # The complete benchmark suite starts at 22:00 and 10:00 UTC.
        self.assertIn('cron: "30 21 * * *"', workflow)
        self.assertIn('cron: "30 9 * * *"', workflow)
        self.assertIn("types: [closed]", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("actions: write", workflow)
        # Run the reviewed default-branch script, not the closed pull request.
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", workflow)
        self.assertIn(
            'python -m scripts.prune_actions_caches --repository "$GITHUB_REPOSITORY"',
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
