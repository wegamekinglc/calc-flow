# GitHub #36 Claude Compatibility Artifact Paths - Critic Critique

## Target

- Spec:
  `.codex/artifacts/specs/github-36-claude-artifact-paths.md`
  (commit `fc96d69`)
- GitHub issue:
  [#36](https://github.com/wegamekinglc/calc-flow/issues/36),
  “docs: .claude/agents mirror still points artifacts at
  .claude/specs|api-notes|critiques”
- Canonical repository contract: `AGENTS.md`
- Structural compatibility contract: `scripts/test_codex_agents.py`
- Historical migration design:
  `docs/superpowers/specs/2026-07-24-claude-to-codex-agent-team-design.md`

No API note applies because this change has no public Rust, Python, REST,
OpenAPI, or generated-schema surface.

## Verdict

**Proceed with caveats**

There are no blocking design findings. The spec resolves the apparent conflict
between GitHub #36's suggested deletion/archive cleanup and the repository's
executable compatibility contract in the least disruptive way: repair active
guidance now, retain the two byte-identical historical copies, and leave any
retirement of those copies to a deliberate preservation-policy change.

The implementation should strengthen the proposed regression marker slightly
so the unit test rejects both a bare legacy directory and paths below it. This
is a narrow test detail and does not require another spec-writing cycle.

## Findings

### Blocking Issues

None.

### Significant Concerns

None.

### Minor / Style Notes

- **Slash-terminated markers leave a small regression gap** - The regression
  section requires exact markers `.claude/specs/`, `.claude/api-notes/`, and
  `.claude/critiques/`, but the acceptance command searches the namespace names
  without requiring a trailing slash. A future instruction such as “read
  artifacts under `.claude/specs`” would pass a slash-terminated string check
  while violating the acceptance criterion and the canonical namespace rule.
  - **Suggested fix:** In `scripts/test_codex_agents.py`, scan all top-level
    `.claude/agents/*.md` files and reject `.claude/specs`,
    `.claude/api-notes`, and `.claude/critiques` without including the trailing
    slash in the forbidden markers. This still permits unrelated `.claude/`
    compatibility references and covers both directory labels and descendant
    paths.

- **“Frozen” should mean preserved, not independently maintained** - The spec
  calls the two legacy copies “frozen compatibility inputs” while also
  requiring them to remain byte-for-byte equal to their canonical
  `.codex/artifacts/` counterparts. Those requirements are consistent for this
  issue, but “frozen” could be misread later as allowing the canonical copy to
  evolve without updating the compatibility copy.
  - **Suggested fix:** Do not add new prose to active agent instructions about
    these copies. In implementation and review reports, describe them as
    “preserved exact mirrors under the existing test contract.” If future work
    edits either canonical artifact, it must deliberately preserve equality or
    change the compatibility contract.

## Conflict Resolution Review

GitHub #36 says to “archive or delete” the two legacy files as part of a
suggested fix; it does not state deletion as a separate acceptance criterion.
The current repository carries stronger contrary evidence:

- `scripts/test_codex_agents.py` lists
  `.claude/specs/head-operator.md` and
  `.claude/api-notes/docs-examples.md` in `MIRRORS` and compares their bytes
  with the canonical Codex copies.
- Both comparisons pass at `fc96d69`.
- The implemented migration design says generic Markdown artifacts were copied
  byte-for-byte and defines preservation as not deleting, renaming, or editing
  tracked `.claude/` files.
- Current `AGENTS.md` makes `.codex/artifacts/` authoritative for new team
  artifacts while retaining `.claude/agents/` as a compatibility surface whose
  shared semantics must be synchronized from the Codex definitions.

Accordingly, deleting or moving the two copies would not be a routine cleanup:
it would intentionally reverse a documented and tested preservation decision.
The spec is correct to keep that policy change out of this guidance fix. The
copies cease to be operationally ambiguous once no active Claude agent
instruction sends readers or writers to their parent directories.

## Active Scope Verification

Exactly eight active `.claude/agents/` Markdown files currently contain one or
more of the three obsolete namespaces:

1. `.claude/agents/README.md`
2. `.claude/agents/cf-api-designer.md`
3. `.claude/agents/cf-critic.md`
4. `.claude/agents/cf-doc-writer.md`
5. `.claude/agents/cf-implementer.md`
6. `.claude/agents/cf-orchestrator.md`
7. `.claude/agents/cf-reviewer.md`
8. `.claude/agents/cf-spec-writer.md`

The remaining active documents,
`.claude/agents/cf-performancer.md`,
`.claude/agents/cf-simplifier.md`, and
`.claude/agents/cf-tester.md`, contain none of those namespaces and should not
receive speculative edits.

The regression check should nevertheless scan every top-level
`.claude/agents/*.md` file, not hard-code the current list of eight. That makes
the check cover the README, all ten roles, and any future active role added at
the same level.

## Historical Documentation Review

Historical `docs/superpowers/` specifications and plans intentionally record
the Claude-native layout that existed when those documents were authored. In
particular:

- `docs/superpowers/specs/2026-07-20-agent-team-design.md` describes the
  original `.claude/specs/`, `.claude/api-notes/`, and `.claude/critiques/`
  workflow.
- `docs/superpowers/specs/2026-07-24-claude-to-codex-agent-team-design.md`
  records the later copy-and-preserve migration and explicitly excludes
  historical `docs/superpowers/` rewrites.
- `docs/superpowers/plans/2026-07-24-claude-to-codex-agent-team.md` contains
  commands and mappings tied to that completed migration.

Changing those records would erase provenance and is not necessary to correct
active instructions. The spec is correct to leave all historical designs and
plans untouched and to scope the new regression check to `.claude/agents/`.

## Axis Check

| Axis                    | Result                                                                                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness             | **OK:** new and upstream team artifacts converge on the one canonical `.codex/artifacts/` namespace.                                                             |
| Hidden assumptions      | **OK with caveat:** preservation is backed by current tests and history; report it as an exact-mirror contract rather than an independent legacy source.         |
| Missing edge cases      | **Caveat:** reject bare legacy directory names as well as slash-qualified descendant paths.                                                                      |
| Backwards compatibility | **OK:** Claude agent frontmatter and platform-specific wording remain; the two tested generic artifact mirrors remain byte-identical.                            |
| Performance             | **OK:** static Markdown and one small structural scan add no runtime or hot-path cost.                                                                           |
| Surface and ergonomics  | **OK:** Claude and Codex users receive one active artifact namespace without conflating unrelated `.claude/` compatibility paths.                                |
| Test plan               | **OK with caveat:** scan all eleven active Markdown definitions and keep the existing exact-mirror test; use markers that also catch bare directory references.  |
| Risk and scope          | **OK:** eight guidance files plus one focused structural test are sufficient; artifact deletion and historical-document rewrites are correctly excluded.         |

## Counter-Proposals

Do not delete, move, or separately archive the two legacy artifacts in this
change. If maintainers later want a single physical copy, open a preservation
policy issue that explicitly updates `MIRRORS`, states whether Git history
alone is sufficient archival evidence, and revises any current compatibility
promise. Mixing that decision into GitHub #36 would make a narrow active-path
repair depend on a broader repository-history policy.

## Questions for the Author

No blocking questions.

For implementation clarity only: should the regression constants omit the
trailing slash so the automated test exactly matches the broader namespace
coverage of the acceptance `rg`? The critic recommends yes.

## Exact Guidance for `cf-implementer`

1. Modify only the eight active files listed above and
   `scripts/test_codex_agents.py`.
2. In those eight files, replace only the three artifact namespace semantics:
   `.claude/specs/` to `.codex/artifacts/specs/`,
   `.claude/api-notes/` to `.codex/artifacts/api-notes/`, and
   `.claude/critiques/` to `.codex/artifacts/critiques/`.
3. Preserve YAML frontmatter, Claude tool names, role boundaries, examples, and
   all unrelated `.claude/` references.
4. Add a focused unit test that scans every top-level
   `.claude/agents/*.md` file and rejects the three obsolete namespace strings
   without requiring a trailing slash.
5. Keep the existing `MIRRORS` entries and exact-byte test unchanged. Do not
   delete, rename, edit, or archive
   `.claude/specs/head-operator.md` or
   `.claude/api-notes/docs-examples.md`.
6. Do not edit `.codex/agents/`, pre-existing canonical artifacts,
   `docs/superpowers/`, `CLAUDE.md`, product documentation, `CHANGELOG.md`, or
   runtime/package/generated surfaces.
7. Verify the focused unit test, the no-obsolete-namespace search, both byte
   comparisons, the intended diff scope, and `git diff --check`.

## Actual Checks and Limits

- Read the target spec at commit `fc96d69`, `AGENTS.md`,
  `docs/introduction.md`, the historical migration design, the current
  structural test, and the active Claude path references.
- Read GitHub #36 and confirmed that archive/delete appears under “Suggested
  fix,” not as an independently stated acceptance requirement.
- `rg` found the obsolete namespaces in exactly the eight active files listed
  above.
- `git ls-files` confirmed eleven top-level `.claude/agents/*.md` documents in
  total.
- `cmp` confirmed both preserved Claude artifacts are currently byte-identical
  to their canonical Codex counterparts.
- `git blame` traced the `MIRRORS` entries and exact-byte assertion to the
  Codex-native agent-team migration.
- No build, implementation test, code generation, benchmark, artifact
  deletion, product-source edit, PR operation, or remote mutation was
  performed. Those are outside this pre-implementation critique.

## Remaining Risks

- Duplicate physical files remain discoverable to a user browsing `.claude/`
  without reading active guidance. This is accepted preservation behavior, not
  an active write/read contract.
- A future change to either generic canonical artifact must preserve the
  byte-equality contract or retire it deliberately.
- A plain substring test is intentionally narrow; it prevents the three
  obsolete artifact namespaces but does not and should not ban `.claude/`
  generally.
