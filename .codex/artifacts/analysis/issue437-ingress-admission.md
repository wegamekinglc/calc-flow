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

## Replay capacity fallback repair

Repair starts at PR #451 head
`e16e5db88da88d7c033963eee034b0f61af19d7b`. The independent review supplied
`replay-pressure-regression.patch` and `replay-pressure-checks.txt` as Multica
attachments. Its unchanged-head reproduction failed at max_state_bytes 65,536
and 64 rows: serial admission accepted both sides, while prepared admission
consumed its facts again after stopping replay and failed with a closed-channel
Internal error. That reviewed failure is reused without repeating an unchanged
failing build.

Cache the successfully received ASOF facts in the envelope's existing OnceLock,
replacing the order-only cache. The facts are a fixed-size Copy value, so retries
reuse the same immutable classification and proof without another channel read
or allocation. Owner validation still checks current schema, protocol, watermark,
resident identity boundaries and counters on every attempt. The replay/status
rollback, publication and cleanup paths are unchanged.

The supplied combination regression is retained under its original test name.
It reuses the existing pressure-pair helper for all 15 supplied limit/row
combinations. Both prepared and serial controls compare acceptance, full status,
finalization and state fees; only explicitly measured replay-log bytes are
subtracted for comparison. The 65,536-byte / 64-row case asserts that replay was
actually dropped. Current inventory accounting is checked and every pool must
return to zero after reset/drop and awaited gather drain.

The repaired combination regression passes 1/1 (2,126 filtered), covering all
15 supplied combinations in 0.61 s after a 3 minute 55 second rebuild. Eight
additional affected replay/admission/cleanup tests pass (2,119 filtered) in
0.87 s using the same rebuilt binary:

```bash
CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  cargo test --locked -p calc-flow --lib \
  test_review_ingress_facts_preserve_replay_pressure_fallback \
  -- --test-threads=1 --nocapture

target/debug/deps/calc_flow-71dd66c0d944c938 --test-threads=1 \
  operator::asof::tests::source_replay::replay_ \
  operator::asof::admission::identity_tests::columnar::ingress:: \
  runtime::streaming::operator_task::ingress_admission::tests::test_ingress_admission_panic_cleanup_joins_every_worker \
  runtime::streaming::operator_task::tests::ingress_admission_tests::test_ingress_admission_joins_workers_waiting_for_input
```

The unaffected parts of the original 68-check evidence remain valid; no full
regression or performance collection is repeated. Lizard at complexity limit
8 reports no warnings in either changed ingress or replay-test module. Format,
whitespace and unchanged project-schema checks pass.

The repair also passes `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`
with the same build settings above. No public interface, checkpoint layout,
limit, lint threshold or exemption changed. The repaired head's remote
CI/Codacy/review snapshot is recorded in the follow-up handoff.
