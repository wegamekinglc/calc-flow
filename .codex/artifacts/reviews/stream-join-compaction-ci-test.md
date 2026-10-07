## PR #371 Integration Test Correction Review

**Verdict: Approve.** Reviewed local HEAD
`2f5912e7df017b77ef257cd45c961549e10e0021`, tree
`d498ec8038561a1f617d205e2dcc9822169be9b7`, against parent
`583d749ad4dcee0caf6edd7748f3d576e3dbe127`.

Only `crates/calc-flow/tests/stream_join_state.rs` changes, with five added and
three removed lines. The old test bytes equal the actual failing remote
`f8f35523938a809eacfe13dd09f0c866f3f6400a` test bytes. Production source is
unchanged.

### Correctness

The correction preserves the compaction and recovery assertions. Four dirty
synchronous checkpoint captures now explicitly assert absence of `left-base`.
The test then awaits `prepare_checkpoint_async` before inserting the fifth
row. The real method delegates to `prepare_compaction`, which rebuilds a due
base through the existing async worker submission, finish and installation.
It is not a test-only force-compaction helper.

Epoch 5 still requires both the canonical `left-base` and new `left-delta-5`.
The test still restores that snapshot, carries the base through another
checkpoint, restores again and checkpoints again. Its final restored state
must match exactly four rows for the continuation right input; the fifth row
remains outside that interval. These assertions were not removed or relaxed.
The new absence checks also detect premature synchronous compaction.

No blocker or style issue was found. The updated comment matches the production
async checkpoint-preparation lifecycle. No public or normative runtime behavior
changed, so no unrelated documentation update is required.

### Verification Evidence

The saved historical CI evidence contains the same failed compaction assertion
in three jobs: each reported nine passed and one failed test, with five deltas
and no base. The author correction receipt records all ten integration tests
passing, scoped `--test stream_join_state` Clippy passing with `-D warnings`,
and fmt, whitespace and generated-contract checks clean. These passing checks
were not repeated by this reviewer; no native build or timing was performed.

Actual SHA256 bindings verified in this review:

- Current test: `fcc4d56db4b3813f44fe528b144d586654ff388facd9c866daf01d2e901db4bd`.
- Original test: `2b027119df008b507c3781fc6450bd61b24f6b1748587bca3c0a4e459ee0f9f3`.
- Exact parent-to-head diff: `09868ef61fc90585453a49a6d58c4e62675290d7b6382729860af4323bddfd5a`.
- Correction receipt: `84ac9ddb6509f35afa10ee13c195344cf7ce7f5f3e84385469b7989412347800`.
- Handoff receipt: `44d1b1a9b00e66f4d29443eafc34f555f21f1365e902558984599a5a5fc3f9ef`.
- Historical CI RED: `54cb362828743386a12f7654ac16ede636c956f618a7f7a15ed0ada3bbd833eb`.

Receipts are under repository `target/issue363-integration/`. The recorded
handoff diff exactly equals the actual Git diff. No new CI snapshot was taken;
required checks still need to be green at the final published head before merge.
This test-only change does not turn the retained performance binaries into
fresh builds of this new Git head; their original identities remain in force.

The reviewer changed only this owned report. Untracked performance reviews and
other contributors' files were preserved. No push, merge, J2 review or unrelated
performance assessment was performed. Reviewer-owned processes: **0**.
