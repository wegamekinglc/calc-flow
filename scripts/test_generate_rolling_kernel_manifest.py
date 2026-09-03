"""Tests for fail-closed rolling-kernel manifest generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_rolling_kernel_manifest import SOURCE, render


class GenerateRollingKernelManifestTests(unittest.TestCase):
    def test_checked_in_manifest_renders(self) -> None:
        generated = render()

        self.assertIn('primitive: "mean"', generated)
        self.assertIn("GeneratedTransition::Numeric", generated)

    def test_missing_primitive_fails_closed(self) -> None:
        document = json.loads(SOURCE.read_text(encoding="utf-8"))
        document["kernels"].pop()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "primitives and order"):
                render(path)

    def test_typed_transition_requires_both_lifecycles(self) -> None:
        document = json.loads(SOURCE.read_text(encoding="utf-8"))
        document["kernels"][2]["stream"] = False

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernels.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires batch and stream"):
                render(path)


if __name__ == "__main__":
    unittest.main()
