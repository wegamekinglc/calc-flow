from __future__ import annotations

import unittest
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

from scripts.classify_ci_changes import changed_paths, docs_only


class ClassifyCiChangesTests(unittest.TestCase):
    def test_markdown_and_document_assets_are_docs_only(self) -> None:
        self.assertTrue(
            docs_only(
                (
                    "README.md",
                    "docs/introduction.md",
                    "docs/images/runtime.svg",
                    "examples/README.md",
                )
            )
        )

    def test_source_or_configuration_change_requires_full_ci(self) -> None:
        self.assertFalse(
            docs_only(
                (
                    "docs/introduction.md",
                    "crates/calc-flow/src/lib.rs",
                )
            )
        )
        self.assertFalse(docs_only((".github/workflows/ci-linux.yml",)))

    def test_empty_change_set_requires_full_ci(self) -> None:
        self.assertFalse(docs_only(()))

    @patch("scripts.classify_ci_changes.subprocess.run")
    def test_changed_paths_disables_rename_detection(self, run: MagicMock) -> None:
        run.return_value = CompletedProcess(
            args=(), returncode=0, stdout=b"src/lib.rs\0docs/lib.md\0"
        )

        self.assertEqual(
            changed_paths("base", "head"),
            ("src/lib.rs", "docs/lib.md"),
        )
        command = run.call_args.args[0]
        self.assertIn("--no-renames", command)


if __name__ == "__main__":
    unittest.main()
