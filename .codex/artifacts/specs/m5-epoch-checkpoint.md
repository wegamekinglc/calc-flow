# Continuous Streaming 3.0 — M5 Epoch Checkpoint Specification

| Field             | Value                                                                       |
| ----------------- | --------------------------------------------------------------------------- |
| Status            | **Proposed — blocks M5 implementation until review is complete**            |
| Priority          | **P0 — crash recovery and exactly-once depend on this protocol**            |
| Baseline          | `main@a5fc2c395e347041f8d16384be99af7e23d2ebff`                             |
| Milestone         | M5 — epoch checkpoints, complete-job recovery, and exactly-once sinks       |
| Artifact slug     | `m5-epoch-checkpoint`                                                       |
| Intended audience | runtime, state, connector, compiler, benchmark, test, and review owners     |

## 1. Authority and precedence

This document is the controlling M5 delta wherever the following inputs are
silent, stale, or inconsistent with the current `main` implementation:

1. `docs/research/2026-08-02-arroyo-risingwave-streaming-research.md`;
2. `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`;
3. `.codex/artifacts/specs/continuous-streaming-runtime.md`;
4. `.codex/artifacts/api-notes/continuous-streaming-runtime.md`;
5. the M3 and M4 delta packages;
6. baseline commit `a5fc2c395e347041f8d16384be99af7e23d2ebff`.

Precedence for an M5 conflict is:

```text
this M5 delta
> compatible M4 and M3 delta requirements
> compatible continuous-streaming runtime specification and API note
> compatible main-plan requirements
> research recommendations and historical milestone notes
```

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD
NOT**, and **MAY** are normative.

## 2. Current-main reconciliation

| Topic                         | Current evidence                                                                   | M5 interpretation                                                                                                    |
| ----------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Manifest model                | M4 ships the strict canonical `CheckpointManifest` v3 under `state/manifest.rs`    | M5 reuses that exact information model and productionizes publication; it does not create a second v3 model          |
| Legacy checkpoint surface     | public v2 runners still use `checkpoint.rs` and `FileCheckpointStore`              | M5 keeps the v2 public API intact; public A6 replacement remains a separate post-M5 atomic integration               |
| Manifest publication          | segment and manifest crash ordering exists only in an M4 `#[cfg(test)]` harness    | M5 promotes the proven transaction into production code with selection, retention, recovery, and fault injection     |
| Runtime configuration hash    | current `ManifestExpectation` rejects an exact hash mismatch                       | the hash remains canonical diagnostic metadata, but a changed operational configuration does not invalidate state    |
| M3 progress snapshot          | exact in-process snapshots include execution trace and monotonic timer coordinates | M5 persists a bounded semantic projection; trace history and process-local `Instant` coordinates are never durable   |
| Source edge ownership         | `LiveProgressCoordinator` owns all source-output senders                           | source cuts and barrier fan-out must be serialized through that owner; the checkpoint coordinator never writes edges |
| Operator control              | operator and sink tasks fail closed on every barrier                               | M5 implements alignment, operator snapshot, sink pre-commit, and barrier acknowledgement before enabling checkpoints |
| Terminal processing           | window operators emit final results from `on_end` and no message may follow end    | final recovery uses an out-of-band terminal epoch after the data-plane end cut; it never sends a post-end barrier    |
| Exactly-once                  | whole-job preflight rejects every exactly-once requirement                         | M5 derives a per-output capability proof and enables only outputs whose complete reachable path satisfies it         |
| Production connectors         | ordinary test/runtime source and sink bindings are the existing boundary           | M5 uses deterministic replayable and transactional test bindings; production connectors remain M6                    |

## 3. Scope

M5 includes:

1. a production manifest transaction and deterministic latest-completed
   selection over the M4 v3 model;
2. durable semantic restore for source position, watermark policy, operator
   progress, operator state handles, and sink pre-commit metadata;
3. one-in-flight, strictly increasing checkpoint epoch coordination;
4. global source pause and exact progress cut before source-owned barrier
   injection;
5. blocked-ingress multi-input barrier alignment;
6. asynchronous state staging and operator acknowledgement;
7. transactional and approved epoch-idempotent sink lifecycle coordination;
8. manifest-before-external-commit ordering and restart completion;
9. an explicit terminal checkpoint after all final outputs are pre-committed;
10. whole-job capability validation before connector lifecycle work;
11. deterministic metrics, bounded status, fault injection, paired benchmarks,
    and an exact-head 20-minute recovery soak.

## 4. Non-goals and milestone firewall

M5 does not include:

- removal or semantic replacement of public v2 `Checkpoint`,
  `CheckpointStore`, `FileCheckpointStore`, `MicroBatchRunner`, or
  `StreamingRunner`;
- public A6 source-driven runner, job status, checkpoint trigger, cancel, wait,
  or recovery APIs;
- Python, Studio, REST, OpenAPI, project-v3, or generated schema changes;
- production Kafka, PostgreSQL CDC, ClickHouse, or object-store connectors;
- unaligned checkpoints, concurrent checkpoints, partial graph checkpoints,
  savepoints, rescaling, schema evolution, or state migration;
- session windows, early triggers, changelog output, or retractions;
- treating ordinary at-least-once sinks as exactly-once;
- persisting M3's complete execution trace, pending receipts, Tokio timer
  deadlines, or process-local monotonic clock coordinates;
- weakening bounded queues, cancellation precedence, manifest parsing, state
  checksums, path safety, lineage leases, or payload redaction.

M5 implementation types remain crate-private unless they were already public
in M4. Test coverage MUST exercise crate-private production paths through
inline runtime tests and focused public state tests; it MUST NOT expose
provisional public APIs merely to make integration tests convenient.

## 5. Controlling decisions

### D1 — M4 manifest v3 is reused behind a legacy-v2 firewall

M5 MUST use the existing M4 `CheckpointManifest`, source/operator/sink entries,
state checksum, and strict JSON parser. A second v3 structure or a lossy
translation is forbidden. The historical instruction to replace
`checkpoint.rs` is interpreted as productionizing manifest operations, not as
removing the public v2 types before A6.

Production manifest storage MAY live under a new private checkpoint module or
under `state`, but public v2 exports MUST remain source- and behavior-compatible.
The M5 coordinator receives the production manifest transaction explicitly;
legacy v2 runners continue to receive their legacy `CheckpointStore`.

The semantic pipeline fingerprint, pipeline name, exact source/operator/sink
ID sets, manifest version, epoch, checksums, and handle ownership are recovery
identity. `runtime_config_hash` is diagnostics only. A mismatch MUST be exposed
in private recovery evidence and metrics and MUST NOT reject or mutate the
selected manifest.

### D2 — The durable manifest is the sole checkpoint truth

One lineage-exclusive production transaction MUST provide:

1. state-segment stage, durable write, validation, and publication;
2. strict manifest construction and validation against the prepared job;
3. atomic manifest publication after every referenced segment is committed;
4. deterministic valid-manifest listing and latest-completed selection;
5. reachability-based retention and orphan collection serialized with publish;
6. recovery reads that revalidate manifest bytes, handles, segment length, and
   SHA-256 before returning state.

Directory order, timestamps, temporary files, highest state-segment epoch, or
sink-visible output MUST NOT select recovery state. A failed epoch never
becomes recoverable. Epoch allocation after restart begins at
`selected_manifest.epoch + 1` with checked arithmetic.

### D3 — Durable progress is a bounded semantic projection

The source manifest entry persists only the durable cursor, next source
sequence, source identity hash, ended flag, and watermark-policy state already
defined by M4. The operator entry persists only ingress active/idle/ended
state, last accepted watermark, bounded inline metadata, and immutable state
handles.

Recovery MUST reconstruct a new M3 progress driver from prepared source
configuration plus those semantic values. It starts a new bounded trace and
receipt namespace. Generated-watermark and idle timers re-arm from the new
driver origin using their complete configured interval or timeout. No restored
timer may emit a watermark below the persisted watermark. Pending submissions,
admission gates, trace records, and monotonic nanoseconds are not serialized.

Restore MUST reject a cursor that the prepared source cannot seek exactly, an
identity mismatch, regressed sequence/watermark state, an invalid operator
ingress set, or a state segment mismatch before source open or sink recovery.

### D4 — Checkpoints are single-flight FIFO epochs

The job has at most one in-flight checkpoint. Periodic and internal test
triggers enter one bounded FIFO. An epoch is allocated only after the prior
epoch is completed or the job has failed. Epoch zero remains invalid.

For each epoch the coordinator records a monotonic phase:

```text
requested
-> sources_cut
-> operators_snapshotted
-> sinks_precommitted
-> manifest_durable
-> sinks_committed
-> completed
```

Every participant ID is known from preflight. Duplicate acknowledgement is
accepted only when it is byte-for-byte identical; a missing, foreign, future,
regressed, or conflicting acknowledgement fails the job. Timeout, manifest
failure, state failure, sink failure, task panic, or cancellation ends the
epoch deterministically; no later epoch begins.

### D5 — A source barrier is injected at a global progress cut

The checkpoint coordinator sends one request to every source binding, but it
never owns or writes a graph edge. Each source stops polling, drains any
already accepted raw event through its `LiveProgressCoordinator`, and reports
its fully admitted observed position. The progress owner waits until all
source submissions before the cut are settled and no timer/control emission
can overtake the cut.

Only then does the progress owner fan out `Barrier(epoch)` through each
source-owned route, in deterministic binding and edge order, and advance that
source's durable position to the reported cursor. Source polling resumes only
after successful barrier fan-out and acknowledgement. A data or watermark
accepted after the cut cannot appear before the barrier.

Idle sources participate. Ended sources report their final durable cursor and
ended policy state but do not reopen and do not receive a data-plane barrier
after their terminal message. The coordinator accounts for their terminal cut
out of band.

### D6 — Operators align by blocking only arrived ingresses

An operator tracks the expected epoch and per-required-ingress barrier state.
After barrier `E` arrives on ingress A, A is excluded from receive selection;
its channel remains bounded and is not drained. Other unblocked ingresses
continue until the same epoch arrives. Idle does not exempt an ingress.
Ended ingresses contribute a terminal cut without receiving another message.

When all required ingress cuts are present, the task:

1. freezes the operator after all pre-barrier data/control is applied;
2. calls `checkpoint(E)` and validates the returned operator/epoch identity;
3. stages any dirty segments through the production state transaction;
4. sends a deterministic operator acknowledgement to the coordinator;
5. immediately forwards `Barrier(E)` after the local snapshot succeeds;
6. unblocks all live ingresses.

M5 selects immediate forwarding after successful local snapshot rather than
waiting for global manifest publication. This bounds alignment latency. A
later global failure cancels the job and leaves downstream pre-commit state to
abort or recovery according to D7. Snapshot failure forwards no barrier.

### D7 — Sink pre-commit precedes manifest; external commit follows it

For an exactly-once output, every bound sink MUST be transactional or an
approved epoch-idempotent mechanism with sufficient retention. An ordinary
sink remains valid only for at-least-once output.

On barrier `E`, a transactional sink has already received every pre-barrier
batch. It pre-commits `E`, returns bounded canonical metadata, and sends an
acknowledgement. The coordinator publishes the manifest only after all
source, operator, and sink acknowledgements for `E` are complete. External
commit starts only after the manifest is durable.

Commit and recovery MUST be idempotent by `(pipeline_fingerprint, sink_id,
epoch)`. If failure occurs before manifest durability, prepared sink output is
aborted. If failure occurs after manifest durability, prepared output MUST NOT
be aborted; restart calls `recover` and completes the manifest-described
commit before any source resumes. A multi-sink partial commit is therefore
completed forward, never rolled back.

The manifest is the only durable completion intent. M5 MUST NOT add a second
mutable completion document. A sink that cannot determine idempotently whether
the manifest-described epoch is already committed cannot satisfy exactly-once.

### D8 — Terminal recovery uses an out-of-band final epoch

No `Barrier` may follow `EndOfInput` on an edge. After every source reaches its
terminal cut, operators process `on_end`, emit all final-only window outputs,
and forward end. Transactional sinks consume those outputs but do not close
irreversibly. They pre-commit a coordinator-owned terminal epoch after their
end acknowledgement.

Operators then capture their post-`on_end` state for that same terminal epoch.
The coordinator publishes a manifest containing final source cursors, ended
progress, post-end operator state, and final sink pre-commit metadata. It then
commits all sinks and marks the job complete in memory. Recovery from that
manifest completes any sink commit and returns terminal success without
reopening sources or re-emitting final windows.

M5 does not add a second terminal flag to the M4 model. Terminal classification
is derived only when every exact source entry has `ended = true` and every
operator ingress entry is `Ended`; any mixed terminal shape is invalid. The
existing `RecoveryStatus::Final` continues to mean that the manifest is a
complete recoverable epoch, not that the stream necessarily ended.

A terminal epoch is ordered after the last periodic epoch and follows the same
single-flight transaction. If a periodic checkpoint is active when the final
cut begins, terminal processing waits for it or inherits its deterministic
failure; epochs never overlap.

### D9 — Exactly-once is proven per output before lifecycle work

Whole-job preflight MUST derive each requested output guarantee from every
source, operator, edge policy, and sink on the reachable subgraph.
Exactly-once requires:

- every reachable source can replay from the exact persisted cursor;
- every reachable progress policy has the M5 semantic projection;
- every reachable stateful operator snapshots and restores deterministically;
- every edge is bounded and lossless;
- every bound sink is transactional or approved epoch-idempotent;
- every idempotency retention horizon covers checkpoint retention and maximum
  recovery delay;
- the effective plan contains no volatile UDF or nondeterministic operator.

Validation identifies the exact output and first incompatible component. It
runs before source open, operator reset/restore, sink begin/recover, lease
acquisition with side effects, or task spawn. At-least-once outputs may share a
job with exactly-once outputs only when their sink lifecycle cannot block or
weaken the exactly-once path.

### D10 — Recovery order is fail-closed and side-effect aware

Startup follows this order:

1. compile and preflight the complete immutable job;
2. acquire the lineage-exclusive state session;
3. list, parse, and select the latest valid completed manifest;
4. validate semantic identity, exact participant sets, handles, and segments;
5. record any runtime-configuration diagnostic mismatch;
6. construct operators and restore progress plus operator state;
7. open sources paused and seek every non-ended source exactly;
8. open sinks and complete manifest-described recovery idempotently;
9. allocate the next epoch and spawn gated tasks;
10. release gates only after all recovery steps succeed.

Any error before step 10 closes already opened resources in reverse ownership
order. No source poll, sink write, edge send, or operator data callback occurs
before the recovery gate opens.

### D11 — Coordination remains bounded, observable, and payload-free

Checkpoint command and acknowledgement channels are bounded. Status stores
only the current epoch, phase, participant counts, elapsed duration, last
completed epoch, failure category, and configuration-hash mismatch flag. It
never stores batch payloads, cursors, connector metadata, state bytes,
pre-commit metadata, filesystem paths, or secrets.

Metrics use deterministic participant IDs and checked counters. Required
metrics cover requested/completed/failed checkpoints, phase latency,
alignment duration, state and manifest bytes, restore duration, sink commit
retries, orphan cleanup, and terminal checkpoint outcome. High-cardinality
epoch values are status fields, not metric labels.

Cancellation remains biased over timer, source, edge, state, and sink work.
Every participant must settle or be reaped before the private job reports a
terminal result. No checkpoint task, blocking worker, open lease, temporary
file, or sink transaction may outlive its owner.

### D12 — Fault evidence, performance, and soak are exact-head gates

Fault tests inject cancellation, I/O error, panic, and restart after every
externally observable transaction boundary: source admission, source cut,
partial alignment, state stage, sink pre-commit, manifest write/rename,
partial multi-sink commit, completed commit, retention, and compaction.

Each fault case asserts selected epoch, restored cursor/watermark/idle/end,
restored window state, visible sink records, duplicate and missing counts,
temporary artifact reachability, and deterministic terminal error.

Paired benchmarks measure barrier data-path overhead, alignment, state
checkpoint, manifest publication, restore, and transactional commit. The
steady-state data path MUST NOT regress by more than 5% without an approved
exception and noise analysis.

The final implementation head MUST pass a 20-minute soak with two replayable
sources, union, final-only window, bounded slow transactional test sink,
periodic checkpoints, retention/compaction, and deterministic process restart.
Evidence contains the raw commit SHA, 120 one-minute samples, zero missing or
duplicate records, bounded tasks/queues/state/manifests, and zero resources
after terminal completion. Any push invalidates earlier soak and CI evidence.

## 6. Implementation work packages

### WP0 — Freeze the current-main M5 contract

- publish this specification, the companion API note, and adversarial critique;
- reconcile the historical M5 task list and runtime boundary;
- approve the legacy-v2 firewall, runtime-hash diagnostic rule, durable
  progress projection, terminal checkpoint, and sink recovery protocol.

### WP1 — Production manifest transaction and durable restore primitives

- promote M4 test publication/selection logic into production-owned modules;
- split manifest validation into semantic identity and diagnostic runtime
  configuration comparison;
- implement strict listing, latest-valid selection, retention, and recovery
  segment reads;
- add crash-boundary, corrupt-manifest, mismatch, and cleanup tests.

### WP2 — Coordinator, source cut, and epoch lifecycle

- add bounded command/ack types and the single-flight coordinator;
- route source pause, settlement, and barrier injection through the progress
  owner;
- implement epoch timeout/failure, status, metrics, and cancellation cleanup;
- add deterministic one- and two-source cut tests including idle and ended.

### WP3 — Operator alignment, snapshot, and restore

- add per-ingress alignment state and blocked receive selection;
- stage operator state and acknowledge before immediate barrier forwarding;
- restore operator progress/state before gates open;
- test every two-input arrival ordering, bounded backpressure, failure, and
  post-restore final result.

### WP4 — Sink transactions and capability proof

- introduce private transactional/epoch-idempotent sink bindings;
- derive per-output delivery capability during whole-job preflight;
- implement begin, pre-commit, manifest-durable notification, commit, abort,
  and recovery;
- add a deterministic filesystem-backed transactional test sink and partial
  multi-sink commit tests.

### WP5 — Terminal epoch, fault matrix, and merge evidence

- implement post-end terminal checkpoint without a post-end barrier;
- complete the full fault matrix and handoff report;
- run full repository verification, paired benchmarks, and 20-minute soak;
- resolve all review, Codacy, and CI findings on the exact final head.

## 7. Acceptance matrix

### Manifest and restore

- **AC-01:** M5 uses one M4 `CheckpointManifest` v3 model; no duplicate model or
  lossy translation exists.
- **AC-02:** public v2 checkpoint and runner APIs remain unchanged through M5.
- **AC-03:** v2, unknown-field, duplicate-key, missing-nullable, deep, oversize,
  corrupt-checksum, and wrong-handle documents fail closed.
- **AC-04:** pipeline fingerprint and participant-set mismatch fail before any
  connector or operator side effect.
- **AC-05:** runtime-configuration hash mismatch restores successfully and is
  observable as a payload-free diagnostic.
- **AC-06:** only a fully published valid manifest selects recovery; orphan and
  temporary artifacts never do.
- **AC-07:** latest-valid selection is deterministic with corrupt, missing, and
  partially published higher epochs.
- **AC-08:** recovery validates every referenced segment length and checksum
  before installing operator state.
- **AC-09:** epoch allocation resumes at selected epoch plus one and rejects
  overflow.
- **AC-10:** retention preserves every handle reachable from retained or
  in-flight manifests and follows no links.

### Progress and source cuts

- **AC-11:** a barrier follows exactly the data/control covered by its durable
  source cursor on every output edge.
- **AC-12:** no post-cut data or timer emission overtakes the barrier.
- **AC-13:** idle sources participate and ended sources contribute a terminal
  cut without reopen or a post-end barrier.
- **AC-14:** periodic and test triggers allocate strictly increasing,
  single-flight FIFO epochs.
- **AC-15:** checkpoint timeout deterministically fails and cancels the job;
  it never skips an epoch.
- **AC-16:** restored progress cannot regress persisted source or operator
  watermarks.
- **AC-17:** restore starts a new bounded trace/receipt namespace and persists
  no process-local timer coordinate.
- **AC-18:** restored generated-watermark and idle timers re-arm from the new
  origin with the full configured delay.

### Alignment and state

- **AC-19:** after one ingress receives barrier E, that ingress is not drained
  while other live ingresses continue.
- **AC-20:** an operator snapshots only after every required ingress cut for E
  is present.
- **AC-21:** pre-barrier data is represented in E and post-barrier data is not.
- **AC-22:** local snapshot success immediately forwards the barrier and
  unblocks ingress; global failure later cancels the job.
- **AC-23:** snapshot/stage failure forwards no barrier and leaves the previous
  manifest authoritative.
- **AC-24:** duplicate identical barrier state is idempotent; foreign,
  conflicting, future, or regressed epochs fail closed.
- **AC-25:** restored window state and progress produce the same final result
  under every batch boundary and two-input barrier arrival ordering.
- **AC-26:** alignment retains bounded edge backpressure without an internal
  unbounded buffer.

### Sink transaction and exactly-once

- **AC-27:** ordinary sinks fail exactly-once preflight before lifecycle work.
- **AC-28:** every reachable source/operator/edge/sink capability participates
  in each requested output's proof.
- **AC-29:** a sink pre-commits only after all data before barrier E is written.
- **AC-30:** manifest publication begins only after every required ack and
  external commit begins only after manifest durability.
- **AC-31:** pre-manifest failure aborts prepared output; post-manifest failure
  preserves it for recovery.
- **AC-32:** commit/recover is idempotent by pipeline, sink, and epoch.
- **AC-33:** restart completes a partially committed multi-sink epoch forward
  without rewriting or losing data.
- **AC-34:** no second durable completion marker competes with the manifest.
- **AC-35:** epoch-idempotent retention shorter than the required recovery
  horizon fails preflight.
- **AC-36:** at-least-once output continues to expose its allowed duplicate
  boundary and is never reported as exactly-once.

### Terminal, lifecycle, and evidence

- **AC-37:** no edge receives a barrier after `EndOfInput`.
- **AC-38:** the terminal manifest contains final cursors, ended progress,
  post-`on_end` operator state, and final sink pre-commit metadata.
- **AC-39:** recovery from a terminal manifest commits sinks if necessary and
  returns success without reopening sources or re-emitting windows.
- **AC-40:** terminal checkpoint and periodic checkpoint never overlap.
- **AC-41:** cancellation wins over waits and leaves no task, lease, worker,
  temporary file, or transaction owned by the job.
- **AC-42:** status and metrics contain no rows, cursors, state bytes, connector
  payloads, pre-commit data, paths, attributes, or secrets.
- **AC-43:** the complete fault matrix proves selected epoch, restored state,
  visible output, zero missing/duplicates for the transactional sink, cleanup,
  and deterministic terminal error.
- **AC-44:** paired benchmarks show no unexplained steady-state regression
  above 5%.
- **AC-45:** full Rust, Python, Studio, frontend, supply-chain, generated-file,
  and diff checks pass on the final implementation head.
- **AC-46:** the exact-head 20-minute soak records 120 samples, deterministic
  restart, bounded resources, zero missing/duplicates, and terminal zero.
- **AC-47:** all actionable Copilot, Codacy, and CI findings are resolved on the
  same head used for final evidence.
- **AC-48:** M5 adds no Python, Studio, project-v3, production connector, or
  public A6 surface.

## 8. Required implementation order

Implementation follows `WP0 -> WP1 -> WP2 -> WP3 -> WP4 -> WP5`.

WP1 establishes the only durable truth before runtime participants can depend
on it. WP2 establishes the exact source cut and coordinator lifecycle before
operator alignment. WP3 establishes barrier flow before sink transactions.
WP4 completes exactly-once capability and commit ordering. WP5 adds the
terminal path and runs evidence only on the integrated implementation.

Each work package starts with a focused failing test. A stacked review PR may
target the milestone integration branch, but no partial work package merges to
`main`. The final M5 implementation reaches `main` atomically only after all
acceptance criteria pass.

## 9. Pull-request and merge strategy

The planning package uses branch `feature/continuous-streaming-m5-plan` and a
standalone documentation PR.

After approval, implementation uses integration branch
`feature/continuous-streaming-m5-epoch-checkpoint`. WP1–WP5 MAY use stacked
review PRs targeting that integration branch. The integration branch receives
one final PR to `main`; reviewers must be able to verify that no partial
checkpoint or exactly-once claim is present on `main` before that merge.

Any implementation push invalidates prior exact-head CI, review-resolution,
Codacy, benchmark, and soak evidence. Merge requires the final remote head to
be mergeable and clean, with every required check and review thread resolved.
Post-M5 public A6 integration remains a separate milestone and PR.

## 10. Completion definition

M5 is complete only when:

1. WP1–WP5 are implemented test-first and reviewed against this delta;
2. AC-01–AC-48 are satisfied on one exact remote head;
3. the full verification matrix and paired benchmarks pass;
4. the 20-minute soak passes on that same head;
5. fault-matrix and soak handoffs contain independently checkable raw evidence;
6. no unresolved Copilot, Codacy, CI, or review-thread issue remains;
7. the final M5 integration PR merges atomically into `main`.
