# Issue 406: reviewed increment integration

The complete issue406 objective and all eight performance targets remain active.
This local integration combines already reviewed ASOF routing/capture/workspace/
copy and Join projection/state/compiled-expression increments with published
main fb14646b713eee5cc64e0625175c7d65f60166e7. No remote mutation or performance
acquisition belongs to this step.

## Exact sources

- Reviewed phase2.3 source:33e9a184c39a05b14f07770f51de337c5ac313e6.
- Published main:fb14646b713eee5cc64e0625175c7d65f60166e7, independently
  confirmed with git ls-remote origin refs/heads/main before integration.
- Local merge:d4d3b80a, with only CHANGELOG requiring manual resolution.
- Final phase2.3 review/documentation was carried unchanged by d1902a72.

The source-preservation proof in root
target/issue406-integration-20261014/merge-provenance.json verifies all84Rust
files changed by the issue branch remain byte-identical to33e9a184 and all8Rust
files changed by main remain byte-identical to published main. The sets have
no overlap. Both conflicting changelog entries are preserved verbatim; the
guides merge automatically. Current Join state and compiled-expression source
seals still match every file:

```text
Join state:706257eb1a3e504deffe4b6dff91515a544aadae83330235bc254f21d8f00302
Expressions:94c6c1919367193c933556cc0830a5d58901ac62e964239913290873c15d713e
```

The unchanged per-increment
[Join state review](../reviews/issue406-join-state-projection.md) and
[compiled-expression review](../reviews/issue406-compiled-expressions.md)
approve their exact source scopes. Other ASOF/Join output reports preserve
their original source commits, measured builds and limitations. These local
source approvals do not prove the original eight performance goals.

## Focused integration verification

Core default-feature lib/test Clippy passes after the main integration:
root target/issue406-integration-20261014/core-clippy-integrated.log,
2m42s tool time,162.151s runner time. The purpose was compilation and static
lint of the merged core, not a runtime regression matrix:

```bash
cargo clippy -p calc-flow --lib --tests -- -D warnings
```

Formatting, whitespace and generated project-v3 schema no-drift against the
pinned main pass. No unchanged passing runtime test was repeated. Source-only
merge consistency and combined-document review are approved by the
[bounded integration review](../reviews/issue406-second-stage-integration.md)
at d1902a72f317d26827acdea615ae4c19c3d820a0. All owned build/formatter handles are settled.

## Delivery and remaining acceptance

Branch:feature/issue406-second-stage. The prepared PR title/body are in root
target/issue406-integration-20261014/pr-title.txt and pr-body.txt. Its scope
explicitly references406 without closing it. The branch is local and unpushed;
no new PR or current-head remote CI exists. The previous user-authorized PR421
is already merged and is included through current main.

Required CI/cross-platform checks, combined90%Rust coverage, release/new-native
Python integration and performance acceptance remain unverified. The two Python
capability inventory tests used their recorded compatible baseline native
import; they are not current-native execution evidence. Historical stopped
performance attempts remain sealed and are not resumed or retried here.
Remaining phase0/1/3/4/5 requirements and all eight original acceptance rows
remain in the development plan.

## Final local integration review

cf-reviewer approves integration consistency at
d1902a72f317d26827acdea615ae4c19c3d820a0. It independently derives the exact
84/8 Rust path sets, verifies every committed/worktree byte against the
respective parents, confirms zero overlap and recomputes both stage seals.
The resolved changelog and combined guides preserve the reviewed source
scopes. No fresh114-file source review, runtime or performance test is claimed.
The subsequent artifact commit changes documentation only. No local blocker
remains for publishing this reviewed increment; remote checks and all original
issue-wide requirements remain pending.
