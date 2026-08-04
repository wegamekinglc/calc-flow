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

CODE_STYLE_SKILL = ".agents/skills/code-style/SKILL.md"

MIRRORS = (
    (
        ".claude/api-notes/docs-examples.md",
        ".codex/artifacts/api-notes/docs-examples.md",
    ),
    (
        ".claude/specs/head-operator.md",
        ".codex/artifacts/specs/head-operator.md",
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
            path.stem: tomllib.loads(path.read_text(encoding="utf-8"))
            for path in sorted(agent_dir.glob("*.toml"))
        }

    def test_project_config_enables_agents_without_changing_permissions(
        self,
    ) -> None:
        config = tomllib.loads(
            (ROOT / ".codex/config.toml").read_text(encoding="utf-8")
        )

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

    def test_every_custom_agent_enables_code_style_skill(self) -> None:
        expected_skill_config = {
            "config": [
                {
                    "name": "code-style",
                    "enabled": True,
                }
            ]
        }

        for filename, config in self._agent_configs().items():
            with self.subTest(agent=filename):
                self.assertEqual(config.get("skills"), expected_skill_config)

    def test_code_style_guidance_is_a_valid_skill(self) -> None:
        skill = (ROOT / CODE_STYLE_SKILL).read_text(encoding="utf-8")

        self.assertFalse((ROOT / ".codex/guidance/code-style.md").exists())
        self.assertTrue(skill.startswith("---\nname: code-style\n"))
        self.assertIn("\ndescription: ", skill.split("---", 2)[1])
        for heading in (
            "## Python",
            "## Web UI and backend",
            "## Tests",
            "## Markdown",
            "## Verification",
        ):
            self.assertIn(heading, skill)

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
        readme = (ROOT / ".codex/agents/README.md").read_text(encoding="utf-8")

        for agent in EXPECTED_AGENTS:
            self.assertIn(f"`{agent}`", readme)
        for path in (
            ".codex/artifacts/specs/",
            ".codex/artifacts/api-notes/",
            ".codex/artifacts/critiques/",
            CODE_STYLE_SKILL,
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

    def test_claude_agent_guidance_omits_obsolete_artifact_namespaces(
        self,
    ) -> None:
        obsolete_namespaces = (
            ".claude/specs",
            ".claude/api-notes",
            ".claude/critiques",
        )

        for path in sorted((ROOT / ".claude/agents").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                for namespace in obsolete_namespaces:
                    self.assertNotIn(namespace, text)

    def test_agents_guide_points_to_codex_team(self) -> None:
        guide = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn(".codex/agents/README.md", guide)
        self.assertIn(".codex/artifacts/", guide)
        self.assertIn("preserved compatibility", guide)


if __name__ == "__main__":
    unittest.main()
