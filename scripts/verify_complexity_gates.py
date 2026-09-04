"""Codacy-style complexity ratchet gate.

Complexity findings are ratcheted, not absolute: the committed baseline in
``scripts/complexity_baseline.json`` records the current ``# lizard
forgives`` markers, clippy ``too_many_*`` allows, and ruff complexity
findings (C901, PLR0912, PLR0913, PLR0915). The gate fails only when a
count grows or a new location carries one, so refactors shrink the
baseline until the ruff rules can graduate into ``pyproject.toml``'s
``[tool.ruff.lint] select``.

Usage:
    uv run python scripts/verify_complexity_gates.py
    uv run python scripts/verify_complexity_gates.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 -- fixed, module-owned ruff invocation only
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "scripts" / "complexity_baseline.json"
SCAN_ROOTS: tuple[str, ...] = ("benchmarks", "crates", "python", "scripts", "web-ui")
SKIPPED_PARTS = frozenset({"node_modules", "__pycache__", ".venv", "target"})
SOURCE_SUFFIXES = frozenset({".rs", ".py"})
RUFF_COMPLEXITY_SELECT = ("C901", "PLR0912", "PLR0913", "PLR0915")
RUFF_MARKER = "lizard forgives"
CLIPPY_MARKER = "clippy::too_many"


def source_files() -> list[Path]:
    """Return every Rust and Python source file under the scan roots."""

    files: list[Path] = []
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if SKIPPED_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return files


def relative_key(path: Path, root: Path) -> str:
    """Key a scanned file relative to its scan root in forward slashes."""

    if path.is_absolute():
        return path.relative_to(root).as_posix()
    return path.as_posix()


def count_markers(
    files: Sequence[Path], marker: str, root: Path = REPO_ROOT
) -> dict[str, int]:
    """Count case-insensitive marker occurrences per scanned file."""

    counts: dict[str, int] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        found = text.lower().count(marker.lower())
        if found:
            counts[relative_key(path, root)] = found
    return dict(sorted(counts.items()))


def run_ruff_complexity() -> dict[str, dict[str, int]]:
    """Return per-file per-rule ruff complexity finding counts."""

    completed = subprocess.run(  # nosec B603  # nosemgrep
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            ".",
            "--select",
            ",".join(RUFF_COMPLEXITY_SELECT),
            "--output-format",
            "json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"ruff complexity scan failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    findings = json.loads(completed.stdout or "[]")
    counts: dict[str, dict[str, int]] = {}
    for finding in findings:
        file_path = Path(str(finding.get("filename", "")))
        if file_path.is_absolute():
            # Ruff reports absolute paths on some platforms; baseline keys are
            # always repository-relative.
            try:
                file_path = file_path.relative_to(REPO_ROOT)
            except ValueError:
                continue
        file_name = file_path.as_posix()
        code = str(finding.get("code") or "")
        if not code or not file_name:
            continue
        per_file = counts.setdefault(file_name, {})
        per_file[code] = per_file.get(code, 0) + 1
    return {name: dict(sorted(rules.items())) for name, rules in sorted(counts.items())}


def collect_current_state() -> dict[str, Any]:
    """Capture every ratcheted complexity section for the baseline."""

    files = source_files()
    rust_files = [path for path in files if path.suffix == ".rs"]
    return {
        "clippy_complexity_allows": count_markers(rust_files, CLIPPY_MARKER),
        "lizard_forgives": count_markers(files, RUFF_MARKER),
        "ruff_complexity": run_ruff_complexity(),
    }


def flatten_counts(section: str, counts: Mapping[str, Any]) -> dict[str, int]:
    """Flatten a section to one leaf key per counted location.

    Marker sections map file -> count while the ruff section maps
    file -> rule -> count; flattening both makes the comparison airtight
    so a drop in one rule can never mask growth in another.
    """

    flat: dict[str, int] = {}
    for key, value in sorted(counts.items()):
        if isinstance(value, Mapping):
            for rule, count in sorted(value.items()):
                flat[f"{section}:{key}:{rule}"] = int(count)
        else:
            flat[f"{section}:{key}"] = int(value)
    return flat


def compare_states(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Return failure lines for every grown or new complexity location."""

    baseline_flat: dict[str, int] = {}
    current_flat: dict[str, int] = {}
    for section in sorted(set(baseline) | set(current)):
        if section not in baseline:
            current_flat.update(flatten_counts(section, current.get(section, {})))
        elif section not in current:
            baseline_flat.update(flatten_counts(section, baseline[section]))
        else:
            baseline_flat.update(flatten_counts(section, baseline[section]))
            current_flat.update(flatten_counts(section, current[section]))

    failures = [
        f"{key} grew from {baseline_flat.get(key, 0)} to {count}"
        for key, count in sorted(current_flat.items())
        if count > baseline_flat.get(key, 0)
    ]
    return failures


def load_baseline(path: Path) -> dict[str, Any]:
    """Load the committed baseline, failing closed when it is absent."""

    if not path.is_file():
        raise FileNotFoundError(
            f"complexity baseline {path} is missing; regenerate it with "
            "--update-baseline and commit the result"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(path: Path, state: Mapping[str, Any]) -> None:
    """Write the ratchet state deterministically."""

    payload = json.dumps(state, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the committed baseline from the current tree",
    )
    arguments = parser.parse_args(argv)

    current = collect_current_state()
    if arguments.update_baseline:
        write_baseline(BASELINE_PATH, current)
        print(f"complexity baseline updated: {BASELINE_PATH}")
        return 0

    baseline = load_baseline(BASELINE_PATH)
    failures = compare_states(baseline, current)
    if failures:
        print("complexity ratchet failed (fix, or consciously update the baseline):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("complexity ratchet holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
