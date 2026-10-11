# Issue 406: batched state-segment publication specification

## Source and scope

- [Issue #406](https://github.com/wegamekinglc/calc-flow/issues/406), with the
  user's checkpoint-off/on corrections and authorization to continue in PR #407.
- Baseline: `53f44fad`; [measured follow-up](../analysis/issue406-join-eviction.md#checkpoint-on-follow-up-assessment).
- [Vocabulary](../../../docs/introduction.md#the-basic-vocabulary),
  [durability and recovery](../../../docs/runtime-envelope.md#manifest-publication-and-recovery),
  [verification policy](../../../AGENTS.md#verification).
- Relevant implementation: `state/backend.rs`, `state/local.rs`,
  `state/transaction.rs`, `state/transaction/working.rs`, and the streaming
  operator-task/checkpoint-task call sites under `crates/calc-flow/src/`.

The fresh checkpoint-on trace attributes 81 directory fsync calls to state
directories, but includes distinct staging, creation and publication operations.
Only repeated committed-directory synchronization within one publication batch
is selected for this increment. The managed path publishes operator segments in
`stage_operator_state_locked` before its checkpoint ACK; the final manifest's
`staged_segments` is empty. Changing only the manifest publication loop would
miss the measured path.

This is a Rust state-backend and managed-transaction change. It introduces no
new checkpoint format, project/runtime option, Python binding or Studio contract.
Join kernels, state compaction, checkpoint cadence, sink guarantees and manifest
retention policy are outside scope.

## Required completion contract

- **FR1 — Batch boundary.** One managed call publishes the new or not-yet-proven
  durable handles needed by one owner snapshot in one epoch. It must finish
  before working state is returned and before the operator checkpoint ACK.
  The existing shared sink-segment staging path must retain the same guarantee
  if it uses this batch operation. No batching across operator ACKs or epochs.
- **FR2 — Local completion means durable publication.** For the local backend
  on Unix, success requires every requested segment's contents to have been
  synced and validated, every
  requested committed path to contain the matching immutable bytes, and all
  affected committed directories to have been synced after the relevant
  renames. A successful batch performs one publication sync per distinct
  committed target directory, including retry-visible targets requiring durable
  confirmation. Directory-creation syncs are separate and are not removed.
  Other backends retain their existing committed-read/publication contracts;
  the provided batch default does not strengthen them into local fsync proofs.
- **FR3 — Preserve earlier boundaries.** Keep each segment's file flush/sync,
  staging-directory sync, length/checksum checks and managed-path validation.
  Keep ancestor syncs when creating managed directories. Do not introduce a
  path traversal, symlink, overwrite or unchecked existing-file fast path.
  Source-directory deletion durability must not be weakened from the current
  staging/publication contract.
- **FR4 — Failures are not a multi-file rollback promise.** A failed rename or
  directory sync returns the existing path-bearing I/O error; mismatched bytes,
  invalid paths and missing validation retain their established error families.
  Earlier renames may remain externally visible. No failed batch supplies an
  ACK or new durable manifest. Complete but unreferenced segment files are
  permitted orphans, reclaimed only by existing reachability-based cleanup.
- **FR5 — Local retry proves durability.** For the local backend, matching
  committed bytes alone never prove that a previous publication's directory
  sync succeeded. A retry after partial
  rename or failed directory sync must verify visible files and complete the
  missing durability boundary, even when no staging file remains. If the sync
  still fails, the retry must fail again. If publication creates a managed
  directory and its creation sync fails, a retry must not infer success merely
  from its existence. Preserve and test the ancestor boundary on paths touched
  by this change; redesigning unrelated root initialization is outside scope.
- **FR6 — Session state advances only after success.** Newly observed or
  published handles may enter `session.carried` and `session.verified` only
  after the complete requested publication succeeds and cancellation has been
  resolved. The batch must not increment working pins before that point.
  Failure/cancellation leaves the prior carried/verified entries and pin counts
  intact; already-valid earlier carries remain valid. Preserve the existing
  checked, all-or-nothing working-pin acquisition.
- **FR7 — No retry bypass in transaction staging.** The current
  `stage_state_segment` branch that finds matching committed bytes must not
  immediately mark an unknown handle carried/verified and skip publication.
  Distinguish three observable states: known-durable carry (neither validation
  nor publication), unknown matching committed file (publication confirmation
  without reading a missing staging file), and newly staged file (staged
  validation followed by publication). Thus validation and publication inputs
  need not be the same collection. Merely changing the existing
  `needs_publication` flag is insufficient. Handles already established by a
  successful session publication, or by existing validated recovery, retain
  their established reuse rules. This classification also applies to shared
  sink-segment staging.
- **FR8 — Owned cancellation.** A token already cancelled when `owner_settled`
  admits the publication operation starts no new operation. Once admitted,
  the entire finite batch is the settlement unit, including backend-lock waits
  and later segments. The trait has no cancellation token: cancellation while
  waiting for its lock does not promise to prevent a later filesystem worker.
  Managed cancellation waits for the admitted batch to settle while preserving
  lineage ownership and publication serialization, then returns cancellation
  with no new pins/ACK. No worker may continue renaming or syncing after that
  return. A successful worker after cancellation may leave durable orphan
  files, which later retries still handle correctly. This does not introduce
  a new guarantee for arbitrary dropping of a public trait future.
- **FR9 — Determinism and bounded resources.** Preserve canonical snapshot
  handle order and deterministic publication/error order. Directory deduplication
  must not rely on unordered iteration. Batch bookkeeping may retain bounded
  handle/path metadata proportional to the input; it must not retain a new
  aggregate of segment payloads or perform parallel payload reads. Existing
  state, snapshot and manifest limits and allocation-failure behavior remain.
- **FR10 — Manifest and retention ordering.** Preserve segment durability →
  operator ACK → manifest file sync → manifest installation/directory sync →
  sink commit → source checkpoint ACK. Preserve installed/indeterminate
  manifest outcomes and old-manifest deletion durability before orphan
  collection. This increment must not batch or remove these separate syncs.

## Minimal compatible surface for review

The current public factory is `StateBackend`; its operation trait is
`StateLineageBackend`, not a separate `StateSession` trait. The proposed minimal
extension is a provided asynchronous `StateLineageBackend::publish_segments`
method accepting a borrowed slice of `StateHandle` and returning `Result<()>`.
Its provided default first preflights the whole input for conflicting identities
at the same committed path and deduplicates exact handles in first-appearance
order, before any publication mutation. It then processes the unique handles
sequentially: `verify_committed_segment` success accepts the already-committed
handle under the old managed committed-read semantics; only `NotFound` calls
the existing `publish_segment`; every other error propagates unchanged.
Verification itself is not a directory-sync proof. Existing third-party
backends need no new required method, and their single-segment publication may
continue to require validated staging and reject repeat publication.

The local override must implement FR2/FR5 for both matching visible files and
new staged files before succeeding. The existing single-segment entry point
retains its completion guarantee; the local implementation must also repair
its sync-failure retry without imposing a new cross-session, staging-free
republication precondition on third-party implementations. Public rustdoc for
the provided method must cover the default's committed-read compatibility,
local durability, partial visibility, validation prerequisites, cancellation
scope, errors and the existing platform durability limit.

This is a compatibility proposal for the critic/API review, not an instruction
to add a new global batching abstraction. An internal-only alternative is
acceptable only if it preserves backend polymorphism without downcasts or
bypassing other backends' existing publication semantics.

- Empty batch: succeeds without filesystem publication work.
- One handle: the default preserves the backend's existing committed-read or
  staged-publication semantics; the local override performs any required
  directory sync even when the file already matches.
- Carry-only snapshot: already-established session carries retain their
  original epochs and avoid new stage, validation, publication and payload
  reads; the backend need not receive an empty publication call.
- Mixed carries and new handles: publish only those needing confirmation;
  the returned snapshot remains canonical and includes both sets.
- Exact repeated handles: idempotent; they must not require a second rename
  of an already-moved staging file. Default and local implementations process
  the handle once, in first-appearance order. Local publication still syncs
  each affected directory once.
- Conflicting handles for the same committed path: reject as invalid input
  after whole-input preflight and before any publication mutation, even if
  earlier entries are otherwise valid; do not choose one payload silently.
  Invalid managed paths likewise fail before local publication mutation.
- Different target directories: a valid backend-level batch may contain
  several owners/epochs within its leased lineage; each affected directory
  independently satisfies FR2. The managed optimization remains one snapshot.

## Focused red/green acceptance

Add or extend focused tests before implementation, recording the expected
behavioral failure on the baseline. A compile-only missing-method failure is
not sufficient durability evidence. A test-only filesystem-operation probe may
observe ordering and inject failures; it must not substitute successful mock
syncs for production syncs in the green path. Unix tests assert actual sync
attempts; non-Unix tests cover visibility/integrity without claiming power-loss
durability.

- [x] **AC1 — Same-directory work reduction.** Given at least two newly staged
  and validated segments, publication performs all file/staging syncs, then the
  renames, then exactly one committed-directory publication sync, and only then
  succeeds. Reopen and verify all bytes. Baseline shows repeated directory sync.
- [x] **AC2 — Partial rename.** Fail after the first rename of a two-segment
  batch. No pins/ACK/manifest are added; prior session cache remains intact.
  Retrying succeeds with exact bytes and a completed directory sync. The first
  visible file must not create a carried-cache shortcut.
- [x] **AC3 — Sync failure and repeated retry.** Fail the directory sync after
  all renames. An unchanged retry reaches a sync and fails while injection
  remains active; after removal it succeeds. Cover both batch and single entry
  points, and the managed staging path's visible-file branch.
- [x] **AC4 — Multiple directories and creation.** Publish into two owner
  directories and prove one publication sync per directory. A failure on the
  second directory produces no managed success; retry confirms both targets.
  First-use paths touched by publication preserve creation/ancestor sync
  ordering, including retry after a failed sync for directories that this
  operation creates. This test does not expand the change into unrelated root
  initialization. Symlink/unexpected-entry controls remain green.
- [x] **AC5 — Cancellation settlement.** Block a real publication worker at a
  controlled boundary, cancel, and prove the managed future stays pending until
  the complete admitted batch settles. Attempt a second lineage open and a
  conflicting publication to prove that the lineage lease and publication
  serialization remain held until settlement. Cover already-cancelled admission
  starting no operation and cancellation during a backend-lock wait without
  assuming it prevents later worker admission. On return, no worker remains,
  no new cache/pins/ACK exists, and a fresh session or retry sees only valid
  bytes or recoverable orphans.
- [x] **AC6 — Compatibility and edge cases.** Exercise empty, single, repeated
  identical, conflicting same-path, carry-only and mixed snapshots using a
  strict old backend that implements only the previous required methods,
  requires validated staging for publication, and rejects repeat publication.
  The default accepts unknown matching committed handles through verification
  without calling that publication method, deduplicates exact handles before
  operations, and publishes only on `NotFound`. Other verification/publication
  errors propagate unchanged and stop the sequence. A conflicting final input
  causes zero publication calls for all earlier valid entries. Caller slices
  and handles remain unchanged.
- [x] **AC7 — Managed recovery boundary.** A multi-segment operator snapshot
  cannot ACK while publication is incomplete. A successful epoch restores with
  exact bytes/state; a failed unpublished epoch cannot displace the previous
  manifest. Existing manifest installed/unknown outcome and retention ordering
  controls pass. Checkpoint-off still creates no state files or epochs.
- [x] **AC8 — Verification and docs.** Run the selected local/transaction/
  operator-ACK tests, directly affected existing recovery/retention controls,
  core `--lib --tests` Clippy and formatting checks. Check generated-contract
  drift and whitespace. Update normative durability wording and changelog only
  where behavior/surface changed. Full regression, coverage and platform gates
  remain CI responsibilities; a pending snapshot is not merge readiness.

## Performance evidence and decision

The deterministic target is one committed-directory publication sync for N > 1
segments in the same target directory, with all preceding/later durability
operations preserved. Wall-time improvement is a measured outcome, not presumed
from syscall counts. Use sealed baseline `53f44fad` and candidate release builds
with equal optimization/dependency settings, reporting build time separately.

Before any execution, freeze the maintained harness, at most five representative
cases, counts and a total measurement allowance of at most 600 seconds including
fixtures, startup, warmup, correctness, restore and cleanup. The required paired
comparison separates true checkpoint-off and 100 ms checkpoint-on for the
existing 200,000-rows-per-side interval Join, 8,192-row batches. Use two rounds of
ten alternating AB/BA pairs per selected mode, with declared warmups, and the
maintained paired confidence/regression rule. A lower confidence bound above +5%
in both rounds is the maintained regression criterion; inconclusive timing does
not establish speedup or equivalence. Trace/probe diagnostics run separately
from uninstrumented timing.

A sustained-checkpoint acceptance claim additionally requires at least 20
consecutive nonterminal epochs with nonempty state in one continuing job, plus
completed-epoch evidence, output oracle, phase/latency quantiles, checkpoint
bytes, RSS, successful restore and owned cleanup. Finite job repetitions or
retained terminal manifests do not meet this condition. Include a fixed sustained
case only within the predeclared budget; if it cannot complete, report the
unverified acceptance explicitly and settle all owned processes.

The larger issue's goals remain distinct and unverified by this increment:
checkpoint-off elapsed time ≤ 1.25× matching Polars single-thread time;
checkpoint-on throughput at a 1-second cadence ≥ 90% of off and at 100 ms ≥ 75%
of off. Low-frequency checkpoint runs still require separate labeling because
terminal publication remains enabled. Do not claim these broader targets from
the narrow baseline/candidate comparison.

## Handoff

The [critic review](../critiques/issue406-segment-publication.md) approved the
revised contract before implementation. AC1–AC8 now have focused validation
evidence; the [implementation and measurement report](../analysis/issue406-segment-publication.md)
records the tests, paired comparison and continuous-checkpoint recovery result.
Required CI and cross-platform gates remain separate from this local acceptance.
