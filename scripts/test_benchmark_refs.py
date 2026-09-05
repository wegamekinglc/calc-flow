from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite.refs import git_revision, resolve_refs


class BenchmarkRefsTests(unittest.TestCase):
    def test_revision_boundary_rejects_options_paths_and_shell_text(self):
        for revision in ("--help", "HEAD", "../main", "HEAD; false"):
            with (
                self.subTest(revision=revision),
                patch(
                    "scripts.benchmark_suite.refs.subprocess.check_output"
                ) as command,
                self.assertRaisesRegex(ValueError, "unsupported benchmark revision"),
            ):
                git_revision(revision)
            command.assert_not_called()

    def test_event_shas_are_resolved_without_shell_expansion(self):
        base, head = "a" * 40, "b" * 40
        with (
            patch.dict(
                "os.environ", {"BENCHMARK_BASE_SHA": base, "BENCHMARK_HEAD_SHA": head}
            ),
            patch(
                "scripts.benchmark_suite.refs.subprocess.check_output",
                side_effect=[base, head],
            ) as command,
        ):
            self.assertEqual(resolve_refs(), (base, head))
        self.assertTrue(
            all(call.kwargs["shell"] is False for call in command.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["cwd"].is_dir() for call in command.call_args_list)
        )

    def test_non_sha_event_input_is_rejected_before_git(self):
        with (
            patch.dict("os.environ", {"BENCHMARK_HEAD_SHA": "HEAD; false"}),
            patch("scripts.benchmark_suite.refs.subprocess.check_output") as command,
            self.assertRaises(ValueError),
        ):
            resolve_refs()
        command.assert_not_called()

    def test_absent_push_base_uses_first_parent(self):
        base, head = "a" * 40, "b" * 40
        with (
            patch.dict(
                "os.environ",
                {"BENCHMARK_BASE_SHA": "0" * 40, "BENCHMARK_HEAD_SHA": head},
            ),
            patch(
                "scripts.benchmark_suite.refs.subprocess.check_output",
                side_effect=[base, base, head],
            ) as command,
        ):
            self.assertEqual(resolve_refs(), (base, head))
        argv = command.call_args_list[0].args[0]
        self.assertTrue(Path(argv[0]).is_absolute())
        self.assertEqual(argv[1:], ["rev-parse", "--verify", "HEAD^"])


if __name__ == "__main__":
    unittest.main()
