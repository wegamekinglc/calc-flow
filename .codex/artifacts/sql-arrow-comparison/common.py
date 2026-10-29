"""Reuse the reviewed fixtures and process ownership without changing old files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / ".codex/artifacts/arrow-ipc-audit"
sys.path.insert(0, str(PRIOR))

from collect_adjacent import identity  # noqa: E402
from extension_transport import native_sample  # noqa: E402
from fixtures import (  # noqa: E402
    CASES,
    NATIVE,
    ROWS,
    digest,
    encode,
    fixture,
    oracle,
    save,
)

ARROYO = "40588f3d036faf05520103a149a1b5e46680ace8e4e98b68c85082a1cdf68cb0"
ARMS = ("sql", "arroyo", "sql_plan")


def read(path):
    return json.loads(Path(path).read_text())


def milliseconds(span):
    return (span[1] - span[0]) / 1e6


def check_rows(case, table):
    expected = oracle(case)
    if table.column_names != ["id", "y"] or any(
        not table[name].equals(expected[name]) for name in expected.column_names
    ):
        raise ValueError("complete Arrow columns, values or original sequence differ")


def inventory(preflight=False):
    return [
        (case.name, group, arm)
        for case in CASES[:2]
        for group in range(1 if preflight else 5)
        for arm in (ARMS if group % 2 == 0 else ARMS[::-1])
    ]


__all__ = [
    "ARMS",
    "ARROYO",
    "CASES",
    "NATIVE",
    "PRIOR",
    "ROOT",
    "ROWS",
    "check_rows",
    "digest",
    "encode",
    "fixture",
    "inventory",
    "identity",
    "milliseconds",
    "native_sample",
    "read",
    "save",
]
