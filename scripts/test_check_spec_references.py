"""Unit tests for the frozen spec-reference tag validator."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.check_spec_references import (
    REPOSITORY_ROOT,
    find_citations,
    main,
    parse_artifact_sections,
    validate_references,
)

ARTIFACT_TEXT = """# Symbolic Computation Contract Freeze

## 3. D1 — Public boundary and namespace

## 5. D3 — Types, promotion, nulls, NaNs, and failures

### 5.1 Table and row-local values

### 5.2 Stateful numeric values

## 7. D5 — Rolling temporal frames
"""


def _write_tree(root: Path, artifact: str | None, sources: dict[str, str]) -> None:
    if artifact is not None:
        artifact_dir = root / ".codex" / "artifacts" / "specs"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "symbolic-computation-contract.md").write_text(
            artifact, encoding="utf-8"
        )
    crates = root / "crates"
    crates.mkdir(exist_ok=True)
    for name, text in sources.items():
        (crates / name).write_text(text, encoding="utf-8")


class TestParseArtifactSections(unittest.TestCase):
    def test_collects_decision_and_numeric_headings(self) -> None:
        sections = parse_artifact_sections(ARTIFACT_TEXT)
        self.assertEqual(sections.decision_ids, frozenset({"D1", "D3", "D5"}))
        self.assertEqual(sections.numeric_ids, frozenset({"3", "5", "5.1", "5.2", "7"}))


class TestFindCitations(unittest.TestCase):
    def test_extracts_decisions_and_contract_sections(self) -> None:
        text = "/// frozen (SCE-00 D3, contract section 5.2; D5)\n"
        citations = find_citations(text, "SCE-00")
        self.assertEqual(
            [(citation.decision, citation.identifier) for citation in citations],
            [(True, "3"), (True, "5"), (False, "5.2")],
        )

    def test_follows_parenthesized_tags_across_comment_lines(self) -> None:
        text = "/// frame (SCE-00 D3,\n/// contract section 5.2).\n"
        citations = find_citations(text, "SCE-00")
        self.assertEqual(
            [(citation.decision, citation.identifier) for citation in citations],
            [(True, "3"), (False, "5.2")],
        )

    def test_unparenthesized_tag_ends_at_the_line(self) -> None:
        text = "/// SCE-00 D7 metrics.\n/// D12 unrelated\n"
        citations = find_citations(text, "SCE-00")
        self.assertEqual(
            [(citation.decision, citation.identifier) for citation in citations],
            [(True, "7")],
        )

    def test_bare_namespace_has_no_citations(self) -> None:
        self.assertEqual(find_citations("/// frozen (SCE-00).\n", "SCE-00"), ())


class TestValidateReferences(unittest.TestCase):
    def test_valid_tags_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tree(
                root,
                ARTIFACT_TEXT,
                {"ok.rs": "/// (SCE-00 D3, contract section 5.2; D5)\n"},
            )
            self.assertEqual(validate_references(root), ())

    def test_dangling_decision_subsection_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tree(root, ARTIFACT_TEXT, {"bad.rs": "/// (SCE-00 D3.2)\n"})
            failures = validate_references(root)
        self.assertEqual(len(failures), 1)
        self.assertIn("crates/bad.rs:1: SCE-00 D3.2", failures[0])

    def test_unknown_decision_and_section_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tree(
                root,
                ARTIFACT_TEXT,
                {"bad.rs": "/// (SCE-00 D14, contract section 9.9)\n"},
            )
            failures = validate_references(root)
        self.assertEqual(len(failures), 2)
        self.assertTrue(any("D14" in failure for failure in failures))
        self.assertTrue(any("contract section 9.9" in failure for failure in failures))

    def test_missing_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tree(root, None, {"ok.rs": "/// (SCE-00 D3)\n"})
            failures = validate_references(root)
        self.assertEqual(len(failures), 1)
        self.assertIn("missing spec artifact", failures[0])

    def test_unmapped_namespaces_are_out_of_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_tree(root, ARTIFACT_TEXT, {"ok.rs": "/// (ZZZ-99 D40)\n"})
            self.assertEqual(validate_references(root), ())

    def test_repository_tree_has_no_dangling_references(self) -> None:
        self.assertEqual(validate_references(REPOSITORY_ROOT), ())


class TestMain(unittest.TestCase):
    def test_main_exits_zero_on_the_repository_tree(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
