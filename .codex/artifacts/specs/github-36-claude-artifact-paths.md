# GitHub #36 — Claude Compatibility Artifact Paths Specification

## Status

Ready for critic review. This specification resolves the ambiguous disposition
of the two pre-migration artifact copies; no public API design is required.

## Source

- GitHub issue:
  [#36](https://github.com/wegamekinglc/calc-flow/issues/36),
  “docs: .claude/agents mirror still points artifacts at
  .claude/specs|api-notes|critiques”
- Multica mirror: DAL-42 / issue 42
- Canonical guidance: `AGENTS.md` and `.codex/agents/`
- Historical migration design:
  `docs/superpowers/specs/2026-07-24-claude-to-codex-agent-team-design.md`
- Existing structural checks: `scripts/test_codex_agents.py`

## Problem Statement

`AGENTS.md` declares `.codex/agents/` canonical and `.claude/agents/` a
compatibility mirror. It also declares `.codex/artifacts/` the only active
namespace for team-produced specifications, API notes, and critiques.

The compatibility mirror does not reflect that contract. Eight active
`.claude/agents/` documents still tell agents to read or write artifacts under
`.claude/specs/`, `.claude/api-notes/`, or `.claude/critiques/`. A Claude user
following those instructions can fork current team artifacts across two trees.

Two pre-migration files also remain under the legacy artifact directories:

- `.claude/specs/head-operator.md`
- `.claude/api-notes/docs-examples.md`

Those files are not active write targets. The current test suite intentionally
preserves them as byte-for-byte compatibility copies of their canonical
`.codex/artifacts/` counterparts.

## Goals

- Make every active `.claude/agents/` instruction use the canonical
  `.codex/artifacts/` paths for new and upstream team artifacts.
- Preserve the role boundaries, Claude-specific frontmatter, tools, and
  terminology of the compatibility definitions.
- Add a focused regression check so stale legacy artifact paths cannot return
  to active `.claude/agents/` guidance.
- Keep the canonical `.codex/agents/` definitions and artifact locations
  unchanged.

## Non-Goals

- No Rust, Python, Studio REST/OpenAPI, schema, runtime, or packaging changes.
- No public API additions, removals, or compatibility shims.
- No rewrite of historical `docs/superpowers/` designs or plans that describe
  the repository state at the time they were written.
- No general migration of `.claude/rules/`, `CLAUDE.md`, or other Claude
  compatibility surfaces.
- No semantic redesign of the specialist workflow or role responsibilities.

## Selected Approach

Synchronize the artifact-path semantics of the active Claude compatibility
documents from their canonical Codex counterparts. Preserve the two
pre-migration artifact copies as frozen compatibility inputs, but remove all
active guidance that presents their parent directories as current artifact
locations.

This approach is selected because `scripts/test_codex_agents.py` currently
defines the two files as exact mirrors. Deleting or moving them would broaden
this documentation fix into a separate preservation-policy change and would
invalidate an executable repository contract. A future issue may retire those
copies by deliberately changing that contract.

Rejected alternatives:

1. Delete the two legacy artifact copies and update the mirror test. This
   removes ambiguity at the filesystem level but changes the active
   preservation contract beyond the path-guidance defect.
2. Leave the `.claude/agents/` write targets intact and add redirects or
   explanatory notes. This still permits new artifacts to fork across two
   namespaces and contradicts `AGENTS.md`.

## Required Changes

### Active Compatibility Guidance

Replace stale artifact path references in these files:

| File                                   | Required canonical paths                                      |
| -------------------------------------- | ------------------------------------------------------------- |
| `.claude/agents/README.md`             | specs, API notes, and critiques under `.codex/artifacts/`     |
| `.claude/agents/cf-spec-writer.md`     | `.codex/artifacts/specs/<feature-slug>.md`                    |
| `.claude/agents/cf-api-designer.md`    | `.codex/artifacts/api-notes/<feature-slug>.md`                |
| `.claude/agents/cf-critic.md`          | canonical inputs and `.codex/artifacts/critiques/` output     |
| `.claude/agents/cf-orchestrator.md`    | canonical spec and critique paths in examples and reports     |
| `.claude/agents/cf-implementer.md`     | canonical upstream artifact directories                      |
| `.claude/agents/cf-reviewer.md`        | canonical upstream artifact directories                      |
| `.claude/agents/cf-doc-writer.md`      | canonical upstream directories and API-note example source   |

The canonical path mapping is exact:

| Legacy active guidance  | Canonical active guidance          |
| ----------------------- | ---------------------------------- |
| `.claude/specs/`        | `.codex/artifacts/specs/`           |
| `.claude/api-notes/`    | `.codex/artifacts/api-notes/`       |
| `.claude/critiques/`    | `.codex/artifacts/critiques/`       |

The compatibility files must remain valid Claude agent definitions. Do not
copy TOML syntax or Codex-only coordination language into the Markdown files.
Where a canonical Codex instruction differs only because of platform-specific
tools, preserve the Claude-specific wording and synchronize only the artifact
location and shared semantics.

### Preserved Compatibility Artifacts

Retain:

- `.claude/specs/head-operator.md`
- `.claude/api-notes/docs-examples.md`

They must remain byte-for-byte equal to:

- `.codex/artifacts/specs/head-operator.md`
- `.codex/artifacts/api-notes/docs-examples.md`

No active agent instruction may identify the legacy directories as a place to
create new artifacts or as the authoritative place to read upstream work.

### Regression Coverage

Extend `scripts/test_codex_agents.py` with a focused check over
`.claude/agents/*.md` that rejects these exact prefixes:

- `.claude/specs/`
- `.claude/api-notes/`
- `.claude/critiques/`

The check must not reject `.claude/` generally because the compatibility files
may legitimately reference Claude-specific guidance, tools, or runtime
locations outside the three obsolete artifact namespaces.

Keep the existing exact-mirror check for the two preserved compatibility
artifacts.

## Compatibility and Documentation

- The change affects internal team guidance only.
- Existing Codex-created artifacts remain authoritative and unchanged.
- Existing historical Claude artifact copies remain readable but are frozen.
- `CHANGELOG.md` does not require an entry because there is no product or
  developer-facing calc-flow API behavior change.
- Historical design and plan documents retain their original paths as
  provenance; they are not active operational guidance.

## Acceptance Criteria

- [ ] `rg -n '\.claude/(specs|api-notes|critiques)' .claude/agents` returns
      no matches.
- [ ] Every output and upstream-artifact reference in the eight affected
      `.claude/agents/` files uses the corresponding `.codex/artifacts/`
      directory.
- [ ] `.codex/agents/` and all pre-existing canonical files under
      `.codex/artifacts/` are unchanged by the guidance reconciliation.
- [ ] The two preserved `.claude` artifact files remain byte-for-byte equal to
      their `.codex/artifacts/` counterparts.
- [ ] `python -m unittest scripts.test_codex_agents` passes with a regression
      check covering the active Claude compatibility guidance.
- [ ] `git diff --check` passes, including aligned Markdown tables and final
      newlines.
- [ ] No Rust, Python package, Studio, OpenAPI, generated schema, release, or
      historical `docs/superpowers/` files change.

## Affected Surfaces

- Active compatibility guidance: `.claude/agents/`
- Structural regression test: `scripts/test_codex_agents.py`
- Preserved compatibility evidence: `.claude/specs/head-operator.md` and
  `.claude/api-notes/docs-examples.md`

No public API surface is involved. `cf-api-designer` is not required.

## Remaining Risk

The GitHub issue suggests archiving or deleting the two legacy artifact files.
This specification instead preserves them because current main explicitly
tests them as exact compatibility mirrors. The critic should treat any desire
to retire those files as a separate policy decision and blocking scope change,
not as an implicit part of this narrow guidance fix.
