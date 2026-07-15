from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseConfigTests(unittest.TestCase):
    def test_python_projects_ship_license_files(self) -> None:
        for project in (ROOT, ROOT / "web-ui/backend"):
            with self.subTest(project=project):
                config = tomllib.loads((project / "pyproject.toml").read_text())
                self.assertEqual(config["project"]["license-files"], ["LICENSE"])
                license_text = (project / "LICENSE").read_text()
                self.assertIn("Apache License", license_text)
                self.assertIn("Version 2.0, January 2004", license_text)

    def test_crate_excludes_integration_tests_and_ships_license(self) -> None:
        crate = ROOT / "crates/calc-flow"
        config = tomllib.loads((crate / "Cargo.toml").read_text())
        self.assertIn("tests/**", config["package"]["exclude"])
        license_text = (crate / "LICENSE").read_text()
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0, January 2004", license_text)

    def test_release_maturin_actions_pin_tool_and_rust_versions(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        action_count = workflow.count("uses: PyO3/maturin-action@")
        self.assertEqual(action_count, 3)
        self.assertEqual(workflow.count("maturin-version: v1.14.1"), action_count)
        self.assertEqual(workflow.count('rust-toolchain: "1.88.0"'), action_count)


if __name__ == "__main__":
    unittest.main()
