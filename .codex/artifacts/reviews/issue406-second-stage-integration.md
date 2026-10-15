## Branch Review: issue 406 second-stage integration

**Author:** Cheng Li | **Branch:** `feature/issue406-second-stage`

**Reviewed head:** `d1902a72f317d26827acdea615ae4c19c3d820a0`

**Reviewed issue source:** `33e9a184c39a05b14f07770f51de337c5ac313e6`

**Pinned published main:** `fb14646b713eee5cc64e0625175c7d65f60166e7`

**Merge:** `d4d3b80a3dd15914fc020a9c617cce684a9e9ea2`

**Scope:** bounded integration and review carry-forward. The combined branch
changes 114 files against pinned main, including 84 Rust files; this report is
not a fresh review of every implementation file. Existing increment reports
retain their exact source, test and measurement scopes.

### Summary

The merge combines the reviewed ASOF and Join increments with the published
main window changes, then carries the final compiled-expression report and
supporting documentation. No Rust behavior was added by the merge: each issue
Rust file matches its reviewed parent and each main Rust file matches pinned
main. The combined changelog and guides preserve both parents' contracts.

### Build and Test Results

- Rust integration compilation/lint: **Passed**, from owner evidence in
  `target/issue406-integration-20261014/core-clippy-integrated.log`.
  `cargo clippy -p calc-flow --lib --tests -- -D warnings` completes with
  default features in 2m42s tool time, 162.151s runner time, exit zero.
  This is core lib/test compilation and lint, not a runtime or all-feature
  verification matrix.
- Formatting, whitespace and generated project-v3 schema no-drift against
  pinned main: owner records **Passed**. The reviewer also verified the schema
  bytes equal pinned main while checking source provenance.
- Runtime tests: **Not rerun**. Existing passing increment evidence remains
  attributed to its original source. No fresh integration runtime result is
  inferred from Clippy or identical file contents.
- Python integration: **Not run** at this head. The earlier two capability
  inventory tests used their recorded compatible old native import; they do
  not establish current-native Python execution or recovery.
- New unresolved integration failures: **None found**. Required CI,
  cross-platform and combined 90% Rust line coverage remain unverified.

The reviewer independently checked the owner's
`target/issue406-integration-20261014/merge-provenance.json` against Git objects
and current disk contents:

- The actual common base is
  `ac5d7dffabe098d33ee9c07c912d3c6c63898552`.
- All 84 issue-branch Rust paths changed from that base equal the issue parent's
  bytes at `33e9a184`; all eight main Rust paths equal pinned main's bytes.
  The independently derived path sets match the inventory and have zero Rust
  overlap. Committed and working-tree bytes match for all these files.
- All 50 Rust files in the approved phase 2.2 manifest still match their recorded
  hashes. Its independently recomputed source digest remains:

  ```text
  706257eb1a3e504deffe4b6dff91515a544aadae83330235bc254f21d8f00302
  ```

- All six Rust files in the approved phase 2.3 manifest still match their recorded
  hashes. Its independently recomputed source digest remains:

  ```text
  94c6c1919367193c933556cc0830a5d58901ac62e964239913290873c15d713e
  ```

- The phase 2.2 review equals its issue-parent version. The phase 2.3 review
  equals the final approved report carried from its implementation worktree.
  Commit `d1902a72` changes four documentation/report files and no Rust files.

These are source-preservation results, not new correctness, timing, allocation,
RSS, release or coverage measurements. Published-main identity is pinned to the
owner's recorded pre-integration remote check; the reviewer performed no remote
query.

### Blocking Issues

None for the bounded integration/carry scope. No reviewed source was replaced
by an independently edited merge variant, and no conflicting merged contract
was found in the inspected documentation sections.

### Style Issues

None identified in the integration resolution. The prepared local PR title uses
the `perf:` prefix and remains under 70 characters; its body has Summary and
Test plan sections and references issue 406 without closing it. Those files
remain primary-owned and were read without editing or publishing them.

### Test Coverage

The existing ASOF routing/capture/workspace/copy, Join output/state/recovery and
compiled-expression reports continue to describe their focused controls. No
extra passing claim is created by carrying them into this branch. In particular,
the existing ASOF bulk-copy timing attempt remains stopped before warmups or
measured samples; earlier paired evidence retains its original revisions and
uncertainty. No stopped acquisition was resumed or retried during integration.

Disjoint Rust edits and the combined Clippy pass support a proportional
source-only integration handoff. They do not replace runtime regression CI or
certify all interactions between independently changed operators. No concrete
additional local runtime control was required by the inspected merge changes.

### Documentation Consistency

The reviewer compared the resolved changelog and both combined guides against
each parent. Both conflicting changelog entries are preserved verbatim. The
Join layout-3, carried-charge, legacy migration, terminal recovery and emitted
counter contracts remain intact. The published-main window sections preserve
open-window budget semantics and net checkpoint-delta/tombstone behavior.
Neither operator's checkpoint description was overwritten by the other.

The compiled-expression section keeps exact-schema/template eligibility,
per-call filter ownership, SQL fallback metrics and conservative UDF/plan
fallbacks. Its separate-driver statement for arithmetic/filter projections
remains compatible with compiled evaluation. The merged text does not extend
ASOF's output-projection qualification to arithmetic consumers. No material
ASOF documentation mismatch was exposed by these combined sections.

The integration analysis correctly retains all eight issue targets and later
phases, identifies scoped evidence, and makes no new performance claim. The
prepared local PR body likewise distinguishes source approvals and old-native
inventory checks from incomplete CI, coverage, new-native integration and
performance acceptance. Main's historical window timings are retained as its
own changelog evidence, not revalidated here.

### Verdict

**Approve — integration consistency only, at
`d1902a72f317d26827acdea615ae4c19c3d820a0`.**

Prior increment approvals retain their exact scopes. This report does not
certify a fresh 114-file implementation review, new performance results, required
CI/cross-platform checks, combined 90% Rust coverage, current-native Python or
release integration, merge readiness, or completion of issue 406. All eight
original performance targets and remaining phase 0/1/3/4/5 requirements remain
active. No build, runtime test, benchmark, CI query, remote mutation or commit
was performed by this reviewer; only this report was written.
