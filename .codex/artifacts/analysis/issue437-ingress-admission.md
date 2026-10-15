# Per-ingress admission preparation

DAL-344 / GitHub #437; part of #406 section 4.1.
Starting and freshly verified `origin/main`:
`718ab62ff5ad562994a694a20102b89991fc6b2b`.
Dependencies #430 and #435 were already merged. The task has its own branch,
`feature/dal344-ingress-admission`; no sibling worktree or PR is reused.

## Ownership and bounds

Join/ASOF operator entry creates one owned Tokio task per live ingress, after
entry/restore and launch gates. Other operators retain their existing path.
Each task dequeues at most one envelope, publishes readiness before doing CPU
preparation, and waits for the owner to acknowledge dispatch before reading
again. The existing round-robin ready-input selection remains with the owner;
preparation completion never determines which ready ingress it selects.

Join binds the declared key/time columns and streams pure validity and time
normalization facts in 64-row blocks. Its bounded fact channel holds 64 blocks
(4,096 rows); the producer additionally holds one current block. It yields at
least every 64 blocks or when the channel fills. There is no full-batch
selection/identity/charge cache or per-row allocation. Conversion errors remain
row bits so the owner preserves physical-row-ID and row-error precedence.
The owner computes lateness and retention against its current input/opposite
progress, accounts the original logical quantum, and performs existing
admission, charges, funding, dictionary resolution, emission and commit.

ASOF prepares null checks and a late-row count against the ingress's last
acknowledged watermark. For Utf8/UInt64 identity columns it also prepares the
existing pure within-batch order proof, including cross-record boundaries.
The owner rejects stale watermark summaries/proofs, rechecks schema/protocol,
applies metrics and late policy, and compares the first identity with resident
state. General/wide/composite key codec, canonical-byte ranks, dictionary
publication, duplicate checks and workspace funding retain their existing
owner paths. No Join dense IDs are interpreted as ASOF ranks.

Controls use the same FIFO envelope queue. Barrier alignment still excludes
blocked ingresses; at most one post-barrier envelope may be staged without
being installed or included in state. Restore supplies initial progress before
workers start. On success, cancellation, failed dispatch or panic, the owner
cancels independent worker waits and drains every handle before returning.
Worker panics surface as TaskPanicked; operator panics retain their outer
containment after child cleanup. No new thread pool or background reaper exists.

Caller inputs, retained logical V1 fees, state/checkpoint versions and bytes,
lineage, fingerprints and public interfaces are unchanged. No binding or
schema regeneration is required.

## Verification

Validation uses pinned Rust 1.88.0, locked dependencies and repository-local
`target/`, with `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0` and
`CARGO_PROFILE_TEST_DEBUG=0`. Dependency artifacts were copied into this
worktree; the changed core and tests are rebuilt here.

The focused regression cases cover a blocked output while the opposite
preparation counter advances without publication; one-envelope backpressure;
input-wait cancellation; failed emission ahead of a speculative schema error;
worker panic and complete join; fresh opposite-watermark retention; sliced,
nullable, multi-record Join output/status/fee parity; and ASOF serial parity,
checkpoint bytes/recovery, stale watermark, resident duplicates and schema/protocol error
priority.

These are correctness/structural overlap checks. No throughput, latency, CPU or
final performance acceptance claim is made. Full regression, platform matrices
and the existing Rust 90% coverage gate remain with CI and #443.

Static complexity comparison uses Lizard 1.23.0 at the unchanged limit of 8.
Across the changed admission/ASOF/task modules and the two new production
modules, the baseline and candidate both have the same 11 existing functions
above the threshold; there are no new or increased violations. No check
configuration, exemptions or thresholds were changed.

The first run exposed an invalid test fixture that tried to declare nullable
ASOF identity fields. Construction correctly rejected that schema. The corrected
regression supplies a mismatched nullable batch to a valid declared schema,
then separately verifies terminal protocol priority with a valid batch.
The Join output-parity fixture also explicitly releases collected output and
operator ownership before draining its gather home, with bounded cleanup waits.

Final changed-source checks:

```bash
CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test --locked -p calc-flow --lib ingress_admission -- --test-threads=1
```

11/11 new regressions pass (2,115 filtered), in 0.32 s after the final test
rebuild. The last rebuild took 1 minute 59 seconds; this is build time, not a
performance measurement. The following existing affected controls pass 54/54
(2,072 filtered) in 15.16 s using that same rebuilt binary:

```bash
target/debug/deps/calc_flow-71dd66c0d944c938 --test-threads=1 \
  operator::asof::admission::identity_tests::columnar::direct:: \
  operator::asof::admission::identity_tests::columnar::general:: \
  operator::join::tests::batch_timestamp \
  operator::join::tests::columnar_state_tests::test_batch_admission \
  operator::join::tests::optimization_tests:: columnar_charge \
  runtime::streaming::operator_task::tests::asof_tests:: \
  runtime::streaming::operator_task::tests::join_checkpoint_owner_tests:: \
  runtime::streaming::runner::tests::terminal_join_launch_cancel_tests:: \
  runtime::streaming::runner::operator_fusion::asof_chain_entry_tests::

target/debug/deps/calc_flow-71dd66c0d944c938 --test-threads=1 \
  runtime::streaming::runner::operator_fusion::asof_chain_tests::emission_controls::
```

The second command passes 3/3 fused ASOF emission/metrics/progress failure
controls (2,123 filtered) in 0.05 s. Total focused tests: 68 passed, zero failed.
The failed fixture and interrupted cleanup diagnosis described above are not
counted as passing evidence; all 11 new tests were rebuilt and completed.

Format, whitespace and unchanged project-schema checks pass:

```bash
cargo fmt --all --check
git diff --check
git diff --exit-code -- schemas/project-v3.schema.json
```

The only source edit after the successful tests moves the unchanged
`acknowledge` helper above the test module to satisfy Clippy's item-order rule.
The existing test results remain valid for those unchanged bodies; final
Clippy and format checks compile/check the final declaration order.

Final Clippy passes with the unchanged lint settings:

```bash
CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
```

The two new regression modules also have no complexity violations at limit 8.
Remote CI, Codacy and review status are recorded with the exact pushed head in
the PR/issue handoff; local checks do not establish merge readiness.
