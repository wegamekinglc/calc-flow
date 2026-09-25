"""Guard the published Python package's oldest supported syntax."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PythonCompatibilityTests(unittest.TestCase):
    def test_published_sources_parse_as_python_39(self) -> None:
        sources = sorted((ROOT / "python/calc_flow").rglob("*.py"))
        self.assertTrue(sources)
        for source in sources:
            with self.subTest(source=source.relative_to(ROOT)):
                ast.parse(
                    source.read_text(encoding="utf-8"),
                    filename=str(source),
                    feature_version=(3, 9),
                )
