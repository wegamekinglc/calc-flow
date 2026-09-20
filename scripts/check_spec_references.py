"""Frozen spec-reference tag validator.

Rust comments cite frozen contract sections with tags such as
``(SCE-00 D3, contract section 5.2)``. Each validated tag namespace maps to
one artifact under ``.codex/artifacts/specs/``; this check resolves every
cited section against that artifact's headings and fails on dangling
references like ``D3.2``, which the symbolic computation contract never
defined.

Usage:
    uv run python scripts/check_spec_references.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = "crates"
SPEC_ARTIFACT_DIR = Path(".codex") / "artifacts" / "specs"

# Only namespaces mapped here are validated. Other tags cite issue tickets
# or notes outside the frozen spec set and stay out of scope by design.
NAMESPACE_ARTIFACTS: dict[str, str] = {
    "SCE-00": "symbolic-computation-contract.md",
}

HEADING_RE = re.compile(r"^#{1,6}\s+(\d+(?:\.\d+)*)(?:\.(?=\s)|\s)(.*)$", re.MULTILINE)
DECISION_HEADING_RE = re.compile(r"\b(D\d+)\b")
DECISION_CITATION_RE = re.compile(r"\bD(\d+(?:\.\d+)*)\b")
CONTRACT_SECTION_RE = re.compile(r"\bcontract section (\d+(?:\.\d+)*)\b")


@dataclass(frozen=True, slots=True)
class ArtifactSections:
    """Section identifiers defined by one spec artifact's headings."""

    decision_ids: frozenset[str]
    numeric_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class Citation:
    """One section citation parsed from a spec tag in a source file."""

    line: int
    decision: bool
    identifier: str

    @property
    def display(self) -> str:
        if self.decision:
            return f"D{self.identifier}"
        return f"contract section {self.identifier}"


def parse_artifact_sections(text: str) -> ArtifactSections:
    """Collects decision and numeric section ids from artifact headings."""
    decision_ids: set[str] = set()
    numeric_ids: set[str] = set()
    for match in HEADING_RE.finditer(text):
        numeric_ids.add(match.group(1))
        decision_ids.update(DECISION_HEADING_RE.findall(match.group(2)))
    return ArtifactSections(
        decision_ids=frozenset(decision_ids), numeric_ids=frozenset(numeric_ids)
    )


def _tag_tail(text: str, start: int, end: int) -> str:
    """Returns the citation text following one namespace occurrence.

    A parenthesized tag may span comment lines, so its tail runs to the
    closing paren; an unparenthesized tag ends at the end of its line.
    """
    line_start = text.rfind("\n", 0, start) + 1
    if "(" in text[line_start:start]:
        close = text.find(")", end)
        if close != -1:
            return text[end:close]
    line_end = text.find("\n", end)
    return text[end : len(text) if line_end == -1 else line_end]


def find_citations(text: str, namespace: str) -> tuple[Citation, ...]:
    """Extracts every section citation attached to one namespace tag."""
    citations: list[Citation] = []
    for match in re.finditer(re.escape(namespace), text):
        line = text.count("\n", 0, match.start()) + 1
        tail = _tag_tail(text, match.start(), match.end())
        citations.extend(
            Citation(line=line, decision=True, identifier=identifier)
            for identifier in DECISION_CITATION_RE.findall(tail)
        )
        citations.extend(
            Citation(line=line, decision=False, identifier=identifier)
            for identifier in CONTRACT_SECTION_RE.findall(tail)
        )
    return tuple(citations)


def _resolves(citation: Citation, sections: ArtifactSections) -> bool:
    if citation.decision:
        return f"D{citation.identifier}" in sections.decision_ids
    return citation.identifier in sections.numeric_ids


def validate_references(root: Path) -> tuple[str, ...]:
    """Returns every dangling spec reference under the repository root."""
    failures: list[str] = []
    sections_by_namespace: dict[str, ArtifactSections] = {}
    for namespace, filename in sorted(NAMESPACE_ARTIFACTS.items()):
        artifact = SPEC_ARTIFACT_DIR / filename
        try:
            artifact_text = (root / artifact).read_text(encoding="utf-8")
        except OSError:
            failures.append(f"{namespace}: missing spec artifact {artifact.as_posix()}")
            continue
        sections_by_namespace[namespace] = parse_artifact_sections(artifact_text)
    for path in sorted((root / SCAN_ROOT).rglob("*.rs")):
        text = path.read_bytes().decode("utf-8", errors="replace")
        relative = path.relative_to(root).as_posix()
        for namespace, sections in sections_by_namespace.items():
            for citation in find_citations(text, namespace):
                if not _resolves(citation, sections):
                    artifact = SPEC_ARTIFACT_DIR / NAMESPACE_ARTIFACTS[namespace]
                    failures.append(
                        f"{relative}:{citation.line}: {namespace} "
                        f"{citation.display} is not a section of {artifact.as_posix()}"
                    )
    return tuple(sorted(failures))


def main() -> int:
    failures = validate_references(REPOSITORY_ROOT)
    if failures:
        print(f"spec reference check found {len(failures)} dangling reference(s):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("spec reference check passed: every spec tag section resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
