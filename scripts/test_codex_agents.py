from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_AGENTS = {
    "cf-api-designer",
    "cf-critic",
    "cf-doc-writer",
    "cf-implementer",
    "cf-orchestrator",
    "cf-performancer",
    "cf-reviewer",
    "cf-simplifier",
    "cf-spec-writer",
    "cf-tester",
}

MIRRORS = (
    (
        ".claude/api-notes/docs-examples.md",
        ".codex/artifacts/api-notes/docs-examples.md",
    ),
    (
        ".claude/specs/head-operator.md",
        ".codex/artifacts/specs/head-operator.md",
    ),
    (
        ".claude/rules/code-style.md",
        ".codex/guidance/code-style.md",
    ),
)

LEGACY_AGENT_MARKERS = (
    ".claude/",
    "EnterWorktree",
    "`Agent`",
    "`SendMessage`",
    "`TaskCreate`",
    "`TaskUpdate`",
    "`TaskList`",
    "`TaskGet`",
    "`Bash`",
    "`Read`",
    "`Write`",
    "`Edit`",
    "`NotebookEdit`",
    "`CronCreate`",
    "`ScheduleWakeup`",
)


class CodexAgentConfigTests(unittest.TestCase):
    def _agent_configs(self) -> dict[str, dict[str, object]]:
        agent_dir = ROOT / ".codex/agents"
        return {
            path.stem: tomllib.loads(path.read_text())
            for path in sorted(agent_dir.glob("*.toml"))
        }

    def test_project_config_enables_agents_without_changing_permissions(
        self,
    ) -> None:
        config = tomllib.loads((ROOT / ".codex/config.toml").read_text())

        self.assertEqual(config["approval_policy"], "on-request")
        self.assertEqual(config["sandbox_mode"], "workspace-write")
        self.assertIs(config["agents"]["enabled"], True)
        self.assertEqual(set(config["agents"]), {"enabled"})

    def test_expected_custom_agent_roster_is_complete(self) -> None:
        configs = self._agent_configs()

        self.assertEqual(set(configs), EXPECTED_AGENTS)
        declared_names = {config["name"] for config in configs.values()}
        self.assertEqual(declared_names, EXPECTED_AGENTS)
        for filename, config in configs.items():
            with self.subTest(agent=filename):
                self.assertEqual(config["name"], filename)
                self.assertIsInstance(config["description"], str)
                self.assertTrue(config["description"].strip())
                self.assertIsInstance(config["developer_instructions"], str)
                self.assertTrue(config["developer_instructions"].strip())
                self.assertNotIn("model", config)
                self.assertNotIn("model_reasoning_effort", config)

    def test_custom_agents_use_codex_paths_and_terminology(self) -> None:
        for filename, config in self._agent_configs().items():
            text = "\n".join(
                (
                    str(config["description"]),
                    str(config["developer_instructions"]),
                )
            )
            with self.subTest(agent=filename):
                for marker in LEGACY_AGENT_MARKERS:
                    self.assertNotIn(marker, text)

    def test_custom_agent_instruction_tables_are_aligned(self) -> None:
        for filename, config in self._agent_configs().items():
            blocks: list[list[str]] = []
            current: list[str] = []
            for line in str(config["developer_instructions"]).splitlines():
                if line.startswith("|"):
                    current.append(line)
                elif current:
                    blocks.append(current)
                    current = []
            if current:
                blocks.append(current)

            for block in blocks:
                with self.subTest(agent=filename, header=block[0]):
                    self.assertEqual(
                        len({len(line) for line in block}),
                        1,
                        "\n".join(block),
                    )

    def test_team_readme_lists_every_agent_and_codex_artifact_path(self) -> None:
        readme = (ROOT / ".codex/agents/README.md").read_text()

        for agent in EXPECTED_AGENTS:
            self.assertIn(f"`{agent}`", readme)
        for path in (
            ".codex/artifacts/specs/",
            ".codex/artifacts/api-notes/",
            ".codex/artifacts/critiques/",
            ".codex/guidance/code-style.md",
        ):
            self.assertIn(path, readme)
        self.assertNotIn("EnterWorktree", readme)
        self.assertNotIn(".claude/", readme)

    def test_generic_guidance_and_artifacts_are_exact_mirrors(self) -> None:
        for source, destination in MIRRORS:
            with self.subTest(destination=destination):
                self.assertEqual(
                    (ROOT / destination).read_bytes(),
                    (ROOT / source).read_bytes(),
                )

    def test_agents_guide_points_to_codex_team(self) -> None:
        guide = (ROOT / "AGENTS.md").read_text()

        self.assertIn(".codex/agents/README.md", guide)
        self.assertIn(".codex/artifacts/", guide)
        self.assertIn("preserved compatibility", guide)


if __name__ == "__main__":
    unittest.main()
