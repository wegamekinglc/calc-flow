# Issue 406: batched snapshot segment staging specification

## Source and selected scope

- [Issue #406](https://github.com/wegamekinglc/calc-flow/issues/406), superseded
  where applicable by the user's separate checkpoint-off/on requirements and
  authorization to continue all development in PR #407 without a PR #405 dependency.
- [Domain vocabulary](../../../docs/introduction.md#the-basic-vocabulary),
  [managed durability](../../../docs/runtime-envelope.md#manifest-publication-and-recovery),
  and the preceding [publication specification](issue406-segment-publication.md)
  and [critique](../critiques/issue406-segment-publication.md).
- Current implementation: `state/backend.rs`, `state/local.rs`,
  `state/local/publication.rs`, and `state/transaction.rs` under
  `crates/calc-flow/src/`.
- Audited evidence in the original checkout's
  `target/issue406-checkpoint-diagnostic/`: the 100 ms checkpoint steady window
  contains 6,506 fsync intervals with a 32.3446 s union over 56.698 s. The
  classified staging work includes 1,808 temporary-file syncs totaling 8.8603 s
  and 1,808 staging-owner-directory syncs totaling 8.4365 s across 452
  checkpoints, each with four snapshot segments. Directory-creation syncs are
  separate. These are instrumented syscall observations, not job-blocked time
  or a predicted wall-time saving.

The selected change reduces repeated staging-directory confirmation within one
owner snapshot. It preserves each file's synchronization and the existing
validation, publication, checkpoint acknowledgement, and recovery boundaries.
Scope is the Rust state backend and the managed operator/sink staging paths.
Duplicate manifest validation, Join kernels, checkpoint cadence, file-sync
parallelism, new formats, and Python/Studio changes are outside this increment.

## Completion and compatibility requirements

- **FR1 — Additive backend surface.** Add a provided async method on the public
  `StateLineageBackend` trait with signature
  `stage_segments(&self, segments: &[(&StateHandle, &[u8])]) -> Result<()>`.
  Both handles and payloads are borrowed and immutable; no new public payload
  container is required. The default delegates to existing `stage_segment`
  operations sequentially after input metadata preflight, preserving each
  backend's existing single-operation guarantees. Existing implementations
  require no new method. Local overrides the batch operation; its existing
  single entry point retains a complete per-call staging durability boundary.
- **FR2 — Deterministic preflight.** Before staging mutation, check the whole
  collection for conflicting identities at the same path. Local also validates
  every managed path and rejects conflicting handles resolving to the same
  staging target, including `.arrow`/`.segment` aliases. Exact duplicate pairs
  are processed once in first-appearance order; a repeated handle with
  different bytes must fail, never discard an unchecked payload. Compare
  payload equality only for duplicate handles before deduplication. This does
  not require an additional whole-payload copy or hash pass: ordinary
  length/checksum validation retains its ordered per-file worker boundary.
  Metadata preflight errors in a later entry can precede earlier entries'
  I/O errors; otherwise operation/error order follows first appearance.
- **FR3 — Local durability.** Hold the local publication serialization lock
  for the complete finite staging batch. Preserve every new file's write,
  flush, file sync, no-clobber persistence, and integrity checks. After all
  requested new/reused staging files complete, sync each affected staging
  owner directory exactly once, in deterministic first-appearance order.
  Success means all requested staged bytes and those directory boundaries
  have completed. No committed-file rename/publication belongs to this batch.
  Keep ancestor/epoch/owner directory-creation syncs separate. The existing
  Unix durability and non-Unix file-content/visibility limits remain explicit.
- **FR4 — Retry and partial failure.** Existing matching staged bytes are
  insufficient proof of a prior staging-directory sync. Both single and batch
  retry must validate visible bytes and confirm the affected staging directory;
  a persistent injected sync failure must fail every retry. A directory left
  visible after failed creation sync likewise needs its ancestor boundary
  completed on retry, including a fresh lineage session. Invalid metadata,
  wrong length/checksum, symlinks, unexpected entries, and committed conflicts
  retain their existing fail-closed checks and error families. A failure after
  some files persist may leave complete staged orphans; no rollback of those
  files is promised. Preserve existing cleanup/reachability rules.
- **FR5 — One managed snapshot.** `collect_staged_state_segments` first
  classifies the complete snapshot into established durable carries, unknown
  matching committed files, and new segments. Stage the new set with one
  backend batch call, then retain staged validation and existing batched
  publication. Carries retain original handles and avoid staging, validation,
  publication, and payload reloads. Unknown visible committed files require
  publication confirmation, without inventing a missing staging file.
  Operator and sink paths obey the same boundary. Returned handles retain
  canonical ordering. A staging/classification failure adds no session
  carried/verified entries, working pins, publication, manifest, or ACK;
  previously established session entries remain intact.
- **FR6 — Owned cancellation.** Admit the staging batch through one
  `owner_settled` operation. An already-cancelled token starts no batch work.
  Once admitted, lock waits and every remaining segment belong to the finite
  settlement unit. Cancellation waits for settlement while preserving lineage
  ownership and serialization, then returns cancellation without proceeding
  to validation/publication or advancing managed caches/pins/ACKs. No admitted
  worker may still mutate files after that return. The trait has no token:
  cancellation during a lock wait does not promise to prevent later worker
  admission. Arbitrarily dropping a public trait future gains no new managed
  cancellation guarantee.
- **FR7 — Resource and lifecycle bounds.** Bookkeeping may retain handle/path
  metadata proportional to the input; it must not copy all payloads into an
  aggregate batch. Additional owned payload copies remain bounded by the
  current file, and payload staging remains sequential. Keep filesystem I/O
  and payload validation in appropriate owned workers rather than blocking
  Tokio executor threads. Empty input succeeds without staging I/O. Existing
  segment limits, validation prerequisites, manifest format/version, source
  cursors, delivery guarantees, retention, and recovery remain unchanged.
  Checkpoint-off continues to create no state files or epochs.

Public rustdoc must include a compiling borrowed-slice example, default-backend
compatibility, duplicate/preflight behavior, partial visibility, managed
cancellation scope, and platform durability limits. Preserve path-bearing I/O
errors and their source; metadata conflicts use `InvalidArgument`, while
payload-integrity and existing committed-file failures preserve the existing
error families. Update the normative durability paragraph in
`docs/runtime-envelope.md` and `CHANGELOG.md`; generated project, Python, and
Studio contracts must not drift.

## Focused acceptance criteria

Extend existing test entries rather than grow the current large libtest table.
Record behavioral RED before implementation; a missing-method compile failure
alone does not establish the intended durability regression. The existing
`publication_tests::Operation` probe may count/inject operations while retaining
real filesystem sync calls in successful tests. Directory durability assertions
apply on Unix; non-Unix checks assert the existing visibility/integrity contract.

- [ ] **AC1 — Work reduction and integrity.** For N > 1 fresh segments sharing
  one staging owner directory, observe N actual file syncs and one
  `StagingSync`, after all file persistence. Reopen and validate exact bytes.
  Two owner directories produce one staging sync each; directory creation and
  subsequent publication syncs remain separately observable. Cover both
  managed operator and sink snapshots, plus unchanged carry-only/mixed cases.
- [ ] **AC2 — Failure and retry.** Fail staging-directory sync after files
  persist: repeated single and batch retries keep failing under injection,
  then succeed after removal. Fail on a later file and on a later directory:
  earlier visible files are valid, no managed success escapes, and retry
  succeeds. Extend failed directory-creation tests to the staging ancestors
  and fresh-session retry. No failed epoch displaces a previous recoverable
  manifest or advances session caches/pins.
- [ ] **AC3 — Inputs and legacy fallback.** Cover empty, singleton, exact
  duplicates, conflicting handles, conflicting duplicate payloads, derived
  staging-path aliases, incorrect length/checksum, and invalid managed paths.
  A final metadata conflict causes zero writes for preceding entries. A strict
  legacy backend implementing only previous required methods receives ordered
  single-stage calls, returns its original errors, and stops on the first
  operation failure. Caller handles and payloads remain unchanged.
- [ ] **AC4 — Cancellation and ACK.** Block a real staging worker, cancel,
  and prove the managed future stays pending until the entire admitted batch
  settles. A second lineage open and competing publication cannot pass the
  held ownership/serialization boundary. Cover cancellation before admission
  and during backend-lock wait. After settlement, no worker remains and no
  new publication, cache entry, pins, operator ACK, or forwarded barrier was
  produced. Extend the existing operator-task checkpoint boundary test.
- [ ] **AC5 — Existing contracts and verification.** Focused local publication,
  managed transaction publication, and operator-task segment-publication tests
  pass, alongside their directly affected recovery/retention controls. Run
  `cargo fmt --all --check`, core `--lib --tests --all-features` Clippy with
  warnings denied, the new method's rustdoc example, generated-contract drift,
  and `git diff --check`. Confirm the implementation retains one-file payload
  ownership and no new blocking I/O on Tokio. Full regression, coverage, and
  cross-platform gates remain CI work; pending CI is not merge readiness.
- [ ] **AC6 — Evidence and PR handoff.** Follow the user's replacement
  eight-workload acceptance table, preserved in the
  [development plan](../analysis/issue406-development-plan.md#required-performance-coverage).
  The PR #407 update includes every required metric, pinned main/current
  comparisons and separately labeled checkpoint-off, low-frequency and enabled
  results where applicable. Report incomplete evidence and uncertainty rather
  than omitting a row. Final specialist review is required.

## Performance target and evidence limits

The deterministic acceptance target for this increment is N staging-directory
syncs reduced to one for N > 1 same-directory snapshot segments, retaining N
new-file syncs and all distinct ancestor, committed-directory, manifest, and
retention boundaries. Wall-time benefit requires measurement.

The performancer freezes cases, artifacts/commits, warmups, fixed sample counts,
timing boundaries and the total budget before execution. The already executed
initial and diagnostic stages, including bounded trace analysis, consumed
232.870 s. Their proposed final seven-workload stage was never executed and is
superseded by the user's explicit complete eight-row acceptance request. Its
240 s allocation is not a limit or authorization for the replacement task.
The complete task needs a newly reviewed budget including setup, oracles,
measurement and cleanup; build time is reported separately. Missing harness
metrics must be implemented and validated before execution. A missed target
does not authorize favorable retries or unrecorded inventory changes.

Use the maintained two-round paired rule where paired verdicts are supported:
both lower confidence bounds above +5% establish regression; both upper bounds
below -5% establish improvement. One sustained run per mode reports descriptive
throughput, not a paired improvement verdict. A sustained-checkpoint coverage
claim needs at least 20 consecutive nonterminal, nonempty epochs with advancing
source cursors and completed retirement wholly inside the steady interval.
Report complete output checks, checkpoint coverage, validator cost, and owned
cleanup; do not subtract concurrent validation or fsync durations from job time.

The new eight-row user table supplies the final workload targets. The earlier
prepared-input Polars ratio and sustained 1 s/100 ms observations remain
historical supplementary evidence, not substitutes for those eight rows.
Checkpoint-disabled execution and checkpoint-enabled overhead retain separate
measurements and goals; checkpoint-specific capture/restore and compaction
metrics retain their necessary lifecycle.
Main `588726c8bce0f1c57b90077064834c4a472ee5f2` has no true-off API; label that
cell unsupported. Its known success-EOF checkpoint failure makes checkpoint-on
timings diagnostic and lifecycle-invalid; do not patch or relabel that baseline
to manufacture a valid speedup. Preserve independent valid incremental evidence.

## Handoff

No user decision remains for this increment. Route this public trait addition
and durability contract through API/critic review before implementation;
implementation must satisfy the focused failure and ownership criteria above.
