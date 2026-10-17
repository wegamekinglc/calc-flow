from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite import refs

BASE, HEAD, PARENT, RELEASE = "a" * 40, "b" * 40, "c" * 40, "d" * 40
TAG = "calc-flow-python-v2026.10.13"
LATEST_RELEASE = [
    "for-each-ref",
    "--merged=HEAD",
    "--sort=-version:refname",
    "--count=1",
    "--format=%(refname:lstrip=2)",
    "refs/tags/calc-flow-python-v*",
]


def environment(*, head: str = HEAD, base: str = "", event: str = "") -> dict[str, str]:
    return {
        "BENCHMARK_HEAD_SHA": head,
        "BENCHMARK_BASE_SHA": base,
        "BENCHMARK_EVENT": event,
    }


def fields(
    resolved: refs.BenchmarkRefs,
) -> tuple[str, str, str, str, str]:
    return (
        resolved.base,
        resolved.head,
        resolved.baseline_kind,
        resolved.baseline_ref,
        resolved.rust_base,
    )


class BenchmarkRefsTests(unittest.TestCase):
    def resolve(self, variables: dict[str, str], outputs: list[str]):
        with (
            patch.dict("os.environ", variables),
            patch(
                "scripts.benchmark_suite.refs.subprocess.check_output",
                side_effect=outputs,
            ) as command,
        ):
            resolved = refs.resolve_refs()
        for call in command.call_args_list:
            self.assertIs(call.kwargs["shell"], False)
            self.assertTrue(call.kwargs["cwd"].is_dir())
            self.assertTrue(Path(call.args[0][0]).is_absolute())
        return resolved, [call.args[0][1:] for call in command.call_args_list]

    def test_revision_boundary_rejects_options_paths_and_shell_text(self):
        for revision in (
            "--help",
            "HEAD",
            "../main",
            "HEAD; false",
            TAG,
            f"{TAG}^{{commit}}",
            f"refs/tags/{TAG}",
            f"refs/tags/{TAG}^{{commit}}\n",
            f"refs/tags/{TAG}; false^{{commit}}",
            "refs/tags/--help^{commit}",
            "refs/tags/calc-flow-python-v2026.010.13^{commit}",
            "refs/tags/v5.0.0^{commit}",
            "refs/heads/main^{commit}",
        ):
            with (
                self.subTest(revision=revision),
                patch(
                    "scripts.benchmark_suite.refs.subprocess.check_output"
                ) as command,
                self.assertRaisesRegex(ValueError, "unsupported benchmark revision"),
            ):
                refs.git_revision(revision)
            command.assert_not_called()

    def test_release_tag_revision_reaches_git_unchanged(self):
        with patch(
            "scripts.benchmark_suite.refs.subprocess.check_output",
            return_value=RELEASE + "\n",
        ) as command:
            self.assertEqual(refs.git_revision(f"refs/tags/{TAG}^{{commit}}"), RELEASE)
        self.assertEqual(
            command.call_args.args[0][1:],
            ["rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}"],
        )

    def test_event_shas_are_resolved_without_shell_expansion(self):
        resolved, argv = self.resolve(environment(base=BASE), [BASE, HEAD])
        self.assertEqual(fields(resolved), (BASE, HEAD, "explicit", BASE, BASE))
        self.assertEqual(
            argv,
            [
                ["rev-parse", "--verify", f"{BASE}^{{commit}}"],
                ["rev-parse", "--verify", f"{HEAD}^{{commit}}"],
            ],
        )

    def test_non_sha_event_input_is_rejected_before_git(self):
        with (
            patch.dict("os.environ", environment(head="HEAD; false")),
            patch("scripts.benchmark_suite.refs.subprocess.check_output") as command,
            self.assertRaises(ValueError),
        ):
            refs.resolve_refs()
        command.assert_not_called()

    def test_absent_push_base_uses_first_parent(self):
        resolved, argv = self.resolve(
            environment(base="0" * 40, event="push"), [PARENT, PARENT, HEAD]
        )
        self.assertEqual(
            fields(resolved), (PARENT, HEAD, "first-parent", "HEAD^", PARENT)
        )
        self.assertEqual(argv[0], ["rev-parse", "--verify", "HEAD^"])

    def test_schedule_compares_against_the_latest_release_tag_commit(self):
        resolved, argv = self.resolve(
            environment(event="schedule"),
            [TAG + "\n", RELEASE, PARENT, RELEASE, HEAD, PARENT],
        )
        self.assertEqual(fields(resolved), (RELEASE, HEAD, "release", TAG, PARENT))
        self.assertEqual(
            argv,
            [
                LATEST_RELEASE,
                ["rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}"],
                ["rev-parse", "--verify", "HEAD^"],
                ["rev-parse", "--verify", f"{RELEASE}^{{commit}}"],
                ["rev-parse", "--verify", f"{HEAD}^{{commit}}"],
                ["rev-parse", "--verify", f"{PARENT}^{{commit}}"],
            ],
        )

    def test_schedule_on_the_release_commit_uses_the_first_parent(self):
        resolved, argv = self.resolve(
            environment(event="schedule"), [TAG, HEAD, PARENT, PARENT, HEAD]
        )
        self.assertEqual(
            fields(resolved), (PARENT, HEAD, "first-parent", "HEAD^", PARENT)
        )
        self.assertEqual(argv[2], ["rev-parse", "--verify", "HEAD^"])

    def test_schedule_without_a_release_tag_uses_the_first_parent(self):
        resolved, argv = self.resolve(
            environment(event="schedule", base="0" * 40), ["\n", PARENT, PARENT, HEAD]
        )
        self.assertEqual(
            fields(resolved), (PARENT, HEAD, "first-parent", "HEAD^", PARENT)
        )
        self.assertEqual(argv[:2], [LATEST_RELEASE, ["rev-parse", "--verify", "HEAD^"]])

    def test_non_sha_rust_base_is_rejected_without_a_further_git_call(self):
        with (
            patch.dict("os.environ", environment(event="schedule")),
            patch(
                "scripts.benchmark_suite.refs.subprocess.check_output",
                side_effect=[TAG + "\n", RELEASE, "not-a-sha", RELEASE, HEAD],
            ) as command,
            self.assertRaisesRegex(
                ValueError, "Rust baseline must be a full commit SHA"
            ),
        ):
            refs.resolve_refs()
        self.assertEqual(len(command.call_args_list), 5)
        self.assertEqual(
            command.call_args_list[-1].args[0][1:],
            ["rev-parse", "--verify", f"{HEAD}^{{commit}}"],
        )

    def test_rewritten_rust_base_commit_is_rejected_by_reverification(self):
        with (
            patch.dict("os.environ", environment(event="schedule")),
            patch(
                "scripts.benchmark_suite.refs.subprocess.check_output",
                side_effect=[TAG + "\n", RELEASE, PARENT, RELEASE, HEAD, "f" * 40],
            ) as command,
            self.assertRaisesRegex(
                ValueError, "resolved commit does not match the requested SHA"
            ),
        ):
            refs.resolve_refs()
        self.assertEqual(len(command.call_args_list), 6)
        self.assertEqual(
            command.call_args_list[-1].args[0][1:],
            ["rev-parse", "--verify", f"{PARENT}^{{commit}}"],
        )

    def test_malformed_release_tag_is_rejected_before_further_git(self):
        for output in (
            "--upload-pack=touch pwned",
            f"{TAG}; false",
            f"{TAG}\ncalc-flow-python-v2026.10.12",
            f"refs/tags/{TAG}",
            f"../{TAG}",
            "calc-flow-python-v2026.010.13",
            "calc-flow-python-v2026.10.13-rc1",
            "calc-flow-python-v2026.10",
            "v5.0.0",
        ):
            with (
                self.subTest(output=output),
                patch.dict("os.environ", environment(event="schedule")),
                patch(
                    "scripts.benchmark_suite.refs.subprocess.check_output",
                    side_effect=[output],
                ) as command,
                self.assertRaisesRegex(ValueError, "unsupported benchmark release tag"),
            ):
                refs.resolve_refs()
            self.assertEqual(command.call_count, 1)

    def test_explicit_base_wins_over_the_schedule_release(self):
        resolved, argv = self.resolve(
            environment(event="schedule", base=BASE), [BASE, HEAD]
        )
        self.assertEqual(fields(resolved), (BASE, HEAD, "explicit", BASE, BASE))
        self.assertNotIn(LATEST_RELEASE, argv)

    def test_non_schedule_events_keep_the_first_parent(self):
        for event in ("workflow_dispatch", "pull_request", "push", "", "Schedule"):
            with self.subTest(event=event):
                resolved, argv = self.resolve(
                    environment(event=event), [PARENT, PARENT, HEAD]
                )
                self.assertEqual(
                    fields(resolved), (PARENT, HEAD, "first-parent", "HEAD^", PARENT)
                )
                self.assertNotIn(LATEST_RELEASE, argv)

    def test_github_output_records_baseline_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "github-output"
            destination.write_text("matrix={}\n", encoding="utf-8")
            with (
                patch.dict("os.environ", environment(event="schedule")),
                patch(
                    "scripts.benchmark_suite.refs.subprocess.check_output",
                    side_effect=[TAG, RELEASE, PARENT, RELEASE, HEAD, PARENT],
                ),
            ):
                refs.write_refs(destination)
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                f"matrix={{}}\nbase={RELEASE}\nhead={HEAD}\n"
                f"baseline_kind=release\nbaseline_ref={TAG}\n"
                f"rust_base={PARENT}\n",
            )

    def test_baseline_description_names_kind_ref_and_exact_commit(self):
        self.assertEqual(
            refs.describe_baseline("release", TAG, RELEASE),
            f"release {TAG} (`{RELEASE}`)",
        )
        self.assertEqual(
            refs.describe_baseline("first-parent", "HEAD^", PARENT),
            f"first parent (`{PARENT}`)",
        )
        self.assertEqual(
            refs.describe_baseline("explicit", BASE, BASE), f"explicit (`{BASE}`)"
        )

    def test_baseline_description_rejects_unresolved_provenance(self):
        for kind, ref, base in (
            ("", "", BASE),
            ("tag", TAG, BASE),
            ("release", "", RELEASE),
            ("release", "v5.0.0", RELEASE),
            ("release", f"{TAG}`\n# injected", RELEASE),
            ("release", TAG, ""),
            ("first-parent", "HEAD^", "HEAD^"),
        ):
            with (
                self.subTest(kind=kind, ref=ref, base=base),
                self.assertRaisesRegex(ValueError, "baseline"),
            ):
                refs.describe_baseline(kind, ref, base)


if __name__ == "__main__":
    unittest.main()
