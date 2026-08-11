# Public Continuous Streaming A6 - Specification

## Source

- Issue: [GitHub #94](https://github.com/wegamekinglc/calc-flow/issues/94)
- Multica issue: DAL-103
- Request date: 2026-08-11
- Revision date: 2026-08-12
- Baseline: `main@aa3bbf0b40aef74898a59b6d0d0028c59a2d6993`
- Baseline delta: merged PR #108 (`d4a85bcecbac0eefa8ff1c5bf75a5b54fe23a447`)
  adds manual-checkpoint FIFO coordination; merged PR #110
  (`aa3bbf0b40aef74898a59b6d0d0028c59a2d6993`) freezes source capabilities,
  binds cursors to source identities, and extends durable identity validation.
- Review input: initial `cf-critic` verdict **Block**, reviewed spec SHA-256
  `f33e8c629650c5cc7df2ea9aa0ef68f7ac332800fd46bf3f14b6cd5c64231489`
- Review input: revised API note SHA-256
  `a4d7441f760cbea33a1657923bc0378adb610fb7c452ccefe3e287e60b2fcec3`
  and second `cf-critic` verdict **Block**, reviewed spec SHA-256
  `e28da9ec0b5596b505c3c443b136c3bd46abb3ba8d8f6c337e5830189a9fc361`
  and critique SHA-256
  `afa5738142f93ff314b2e59438585f6956970fbf41c1675d337d9fa42b14737b`
- Review input: the prior zero-blocker critique and spec SHA-256
  `bf60b2f401df35ef500c27b27db7861fd2fc1df3040495428c31c985f6180b72`
  are superseded by the independent PR #113 **Request Changes** review on
  exact head `cb479fc3c2060e68f98959cac988b583534513e4`; that review requires this
  specification, its API note, and its critique to incorporate PRs #108/#110.
- Related docs: [`docs/introduction.md`](../../../docs/introduction.md),
  [`docs/runtime-envelope.md`](../../../docs/runtime-envelope.md),
  [`continuous-streaming-runtime.md`](continuous-streaming-runtime.md), and
  [`m5-epoch-checkpoint.md`](m5-epoch-checkpoint.md)

This specification is the controlling Public A6 delta where the earlier
continuous-streaming specification or API note conflicts with the verified
post-M5 current-main baseline.
The companion API note must define exact signatures without weakening the
behavioral requirements below.

## Problem Statement

M0-M5 and the merged #108/#110 follow-ups now provide a crate-private
source-driven continuous runtime with durable manifest recovery,
transactional sink completion, manual checkpoint FIFO coordination, frozen
source capabilities, stable cursor ownership, bounded supervision, and
per-output delivery proof. The public Rust, PyO3, and Python surfaces still
export the v2 micro-batch and push-based runners, while the older A6 design
depends on a v3 `CheckpointStore` that does not exist on current `main` and
assumes a reusable runner rather than a one-shot owner transfer.

Public A6 must expose the implemented runtime as one coherent breaking cut. It
must make ownership, recovery, manual-checkpoint completion, capability proof,
terminal observation, redaction, and legacy removal precise enough that the API
designer, implementer, and reviewer do not need to infer semantics from private
types.

## Goals

- Freeze the public continuous-runtime contract against the exact verified
  post-#108/#110 current-main baseline.
- Expose one source-driven Rust runner whose consuming `start(self)` transfers
  all runtime ownership to one non-cloneable `StreamingJob`.
- Project the same engine and lifecycle through PyO3 and functional Python
  adapters; Python must not become a second runtime.
- Use one leased checkpoint-runtime owner around the current v3
  `CheckpointManifest`, engine-created local state backend, and manifest
  transaction as the only durable checkpoint truth, with no new or repurposed
  `CheckpointStore` and no erased local-backend constructor.
- Define new-run, resume, manual checkpoint, graceful shutdown, cancellation,
  terminal wait, handle drop, and awaiter drop semantics.
- Model manifest publication as durable, installed with unknown durability, or
  absent; never infer absence from a failed parent-directory synchronization.
- Freeze stable participant and cursor identities, the once-sampled source
  capability descriptor, explicit exact-replay positioning, lossless delivery,
  and the per-output exactly-once capability proof.
- Restrict Python connector lifecycle I/O to cooperative async callbacks and
  give each sink only its own recovery evidence.
- Publish explicit status, error, and log allowlists and sensitive-data
  prohibitions, with one safe Rust wrapper for every fallible A6 lifecycle
  operation and one source-free Python projection.
- Remove the v2 runner/checkpoint symbols atomically across Rust exports, PyO3
  registration, Python exports, `_native.pyi`, and every in-repository
  consumer, including Studio startup, examples, benchmarks, docs, and release
  smoke tests.
- Name every M5 fault-catalog case and map A6 requirements to existing and
  missing tests, cross-surface E2E evidence, and the 20-minute soak.

## Non-Goals

- This specification issue does not change runtime behavior, public exports,
  package versions, generated files, or tests.
- Public A6 does not add production Kafka, PostgreSQL, ClickHouse, file,
  object-store, HTTP, or WebSocket connectors.
- Public A6 does not add or change Studio routes, SSE, frontend behavior,
  project v3, `web-ui/openapi.json`, or TypeScript contracts. The atomic
  cutover does include the Studio backend's private persistence wiring and
  package smoke tests so existing v2 startup and checkpoint routes do not
  break when public legacy symbols disappear.
- Public A6 does not add savepoints, unaligned or concurrent checkpoints,
  rescaling, state migration, schema evolution, or in-place job restart.
- Public A6 does not accept caller-supplied or non-local `StateBackend`
  implementations in the managed continuous runner. Such a future extension
  requires a separately reviewed namespace-identity capability; the existing
  public state APIs remain available outside this A6 owner.
- Public A6 does not add pause/resume of a live job. “Resume” means starting a
  new job from a durable manifest after a previous job ended or failed.
- Public A6 does not preserve aliases, shims, overloads, or deprecation periods
  for the removed v2 runner and checkpoint surfaces.
- Public A6 does not expose private coordinator, edge-control, reaper,
  diagnostic-launch, or fault-injector types.
- Public A6 does not change the v3 manifest wire format or create a second
  durable completion marker.

## Current-Main Delta

| Topic                 | Current `main` evidence                                                                                                                                           | Public A6 decision                                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stream plan           | `StreamExecutionPlan`, `StreamRequirements`, `DeliveryGuarantee`, `EdgeBudget`, and `StreamRuntimeConfig` are public                                              | Reuse these types; do not create an A6-only plan model                                                                                                      |
| Public runner         | `runtime/streaming/mod.rs` exports the v2 `StreamingRunner` over `BatchExecutionPlan::step`                                                                       | Replace it atomically with the source-driven runner; do not overload `step` and `start` on one type                                                         |
| Private runner        | `ContinuousRunner::start(&self)` allows a runner-scoped registry and later starts                                                                                 | Public `StreamingRunner::start(self)` is one-shot; a returned `StreamingJob` owns the running lifecycle                                                     |
| Private job           | `ContinuousJob` provides status, manual checkpoint, wait, shutdown, cancel, drop cancellation, deterministic outcome, and reaper transfer                         | Project those semantics through public data-only types, subject to the allowlists below                                                                     |
| Checkpoint storage    | M5 uses independently supplied `state_backend` and `manifest_root`; only the state lineage is leased                                                              | Public A6 captures one local managed root, derives state/manifest children internally, and leases the complete canonical namespace; no erased backend input |
| Manifest              | Public `CheckpointManifest` v3 is strict, bounded, canonical, manifest-last, and the sole recovery truth                                                          | Reuse it byte-for-byte; no public mutable manifest save API is added                                                                                        |
| Legacy checkpoint     | `Checkpoint`, `CheckpointStore`, and `FileCheckpointStore` remain public v2 types and have Studio/example/benchmark consumers                                     | Remove the public types only in the same cut that migrates every in-repository consumer                                                                     |
| Recovery              | M5 validates every canonical candidate in ascending order; any invalid candidate fails selection                                                                  | Preserve that fail-closed rule; if all candidates validate, resume from the highest, without in-place restart                                               |
| Manual checkpoint     | #108 adds `CheckpointRequest::Manual`, `ContinuousJob::trigger_checkpoint`, one bounded FIFO, distinct epochs, and waiter-drop persistence                        | Expose the existing manual operation without creating a second request path or changing its ordering                                                        |
| Checkpoint completion | #108 resolves a manual waiter at `Completed`, after manifest durability and every sink commit acknowledgement                                                     | Public manual success uses the same completed point; installed-unknown and failed/partial commit paths return safe typed failures                           |
| Source cursor         | #110 binds configured, restored, and emitted cursors to the stable source binding ID and rejects foreign ownership                                                | Expose the stable binding identity on `Cursor`; never accept or persist a cursor under another source                                                       |
| Source capabilities   | #110 samples schema, watermark, replay, delivery, and row/byte bounds once before open and fingerprints them into durable identity                                | Expose an explicit frozen descriptor; recovery rejects drift and exactly-once requires lossless exact replay                                                |
| Capability proof      | Current preflight rejects lossy or non-replayable reachable sources before open, but still permits a `replayable`-boolean fallback                                | Remove the fallback: require explicit exact pause/report/seek capability plus connector conformance evidence per reachable source                           |
| Sink recovery         | Each private transactional sink receives the complete cross-participant `CheckpointManifest`                                                                      | Pass a sink-scoped recovery value containing only epoch/terminal state and that sink's delivery/pre-commit evidence                                         |
| Error boundary        | Existing `calc_flow::Result<T>` can retain paths, raw I/O sources, panic payloads, and connector text                                                             | Every fallible A6 owner/runner/job method returns only `CalcFlowError::Streaming(StreamingError)` on failure; raw causes are discarded at one seam          |
| Status                | Private status contains payload-free counters but also internal task/cursor details; a drop counter is unusable after owner drop                                  | Publish only the A6 allowlist, include frozen source capability and completed checkpoint fields, and observe owner drop by warning                          |
| PyO3/Python           | Python exposes only batch `ExecutionPlan`, `_MicroBatchRunner`, and the push `_StreamingRunner`; legacy adapters invoke sync callbacks before awaitability checks | Replace registration/wrappers/typing atomically; require async connector lifecycle callbacks and keep `status()` synchronous in event loops                 |
| Plan status           | The v3 plan header on the PR branch says `M0-M5 internal complete; Public A6 pending`                                                                             | Preserve that status until the public implementation and its gates complete                                                                                 |

## Controlling Decisions

### D1 - One atomic public boundary

Rust exports, PyO3 registration, Python wrappers, `calc_flow.__all__`,
`python/calc_flow/_native.pyi`, Studio backend wiring, shipped examples,
benchmarks, public docs, and release smoke checks must change in one
implementation commit or one atomic PR merge. No intermediate revision may
export the new runner with the legacy checkpoint model, remove a legacy symbol
before every repository consumer has migrated, or let runtime behavior,
typing, documentation, and packaged Studio startup disagree.

The cut is intentionally breaking. Removed names must fail import or compile;
they must not resolve to compatibility adapters.

### D2 - One-shot runner and owning job

`StreamingRunner` is an unstarted owner of exactly one plan, exact source and
sink bindings, runtime configuration, and one managed checkpoint-runtime
owner. Its public start operation consumes the Rust runner (`start(self)`) and
can produce at most one `StreamingJob`. It has no public `step`, `reset`,
`plan_snapshot`, restart, or runner-level `shutdown` operation.

`StreamingJob` is the only public owner after successful start. It is not
`Clone`; observer futures may share internal state without creating another
owning handle. The job owns or transitively owns every runtime task, connector,
queue endpoint, cancellation token, state-lineage lease, manifest transaction,
manifest-namespace lease, and cleanup/reaper responsibility.

`start(self)` returns a job only after whole-job preflight, complete checkpoint
namespace leasing, lineage selection, operator reset/restore, connector
open/recovery, task registration, and the live-data gate have succeeded.
Before that point, no source poll, sink write, or operator data callback may
occur. Dropping or cancelling a pending start must request cancellation, settle
every begun task and conforming connector, release every state and manifest
lease, and make no job handle observable.

Python cannot express a move in its type system, so `start_async()` and
blocking `start()` must enforce the same one-shot state at runtime. The Python
runner becomes consumed whether start returns a job or raises after native
launch has begun; another start call raises the frozen consumed-runner error
selected by the API note. A validation failure detected before native launch
may leave only ordinary Python argument objects reusable; it must not expose a
partially initialized native runner.

### D3 - Job lifecycle, ownership, and observer cancellation

The public states are exactly `running`, `draining`, `completed`, `cancelled`,
`failed`, and `recovery_required`. The last four are terminal and immutable.
Terminal selection is serialized with this precedence when triggers are
observed at the same decision point: task failure or panic, explicit cancel,
deadline, then graceful or natural completion.

The public job operations are:

- `id`: stable job ID, synchronous and infallible;
- `status`: synchronous, cheap, deterministic snapshot using the allowlist;
- `trigger_checkpoint`: manual FIFO request returning its completed epoch;
- `shutdown`: graceful drain and terminal checkpoint;
- `cancel`: explicit no-drain request;
- `wait`: observation of the immutable terminal outcome.

`wait`, `shutdown`, and `cancel` are idempotent observers of one terminal
outcome. Repeated calls after terminal completion return the same semantic
outcome. Dropping a `wait` observer has no job effect. Dropping a `shutdown` or
`cancel` observer does not revoke the already-recorded transition; another
call reattaches to the same convergence. Dropping a manual-checkpoint observer
does not dequeue the request.

These convergence guarantees apply to connectors that satisfy their public
cooperative async lifecycle contract. Python adapter validation must reject a
synchronous lifecycle method before invoking it. An accepted async callback
must not perform blocking Python or native work on the event-loop/runtime
thread and must yield cancellation at await points; the engine does not claim
to preempt connector code that violates that contract.

Graceful shutdown stops new source admission, drains the accepted prefix,
processes end, publishes the terminal manifest, completes any required sink
commit, joins all owned work, closes connectors, and releases the lineage
before returning `completed`. Natural finite-source completion follows the
same terminal-checkpoint and resource-release path.

Explicit cancellation makes no drain promise. A manifest-absent transaction is
aborted; durable and installed-with-unknown-durability intent is preserved and
must settle through the D5 rules or be recoverable by the next job. `cancel`
returns only after owned tasks and cooperative async cleanup have settled.
Failure and deadline paths meet the same join and release guarantee.

Dropping the owning `StreamingJob` is non-blocking: it records an explicit
abandonment, requests cancellation, and transfers cleanup ownership to the
self-contained runtime reaper. It must never synchronously join on a Tokio
worker. The reaper must eventually close resources and release the complete
checkpoint namespace; the public drop path is observable through exactly one
allowlisted warning event. An internal/process-level counter may exist, but it
is not a `JobStatus` field because no public job handle remains after the sole
owner is dropped. Because the caller abandoned the owner, drop provides no
returned outcome; callers requiring proof of release must await `cancel`,
`shutdown`, or `wait` as appropriate.

### D4 - Checkpoint storage, new lineages, and resume

A6 reuses `CheckpointManifest` v3 and the M5 local state/manifest machinery
behind one owned managed checkpoint-runtime boundary; the API note owns its
exact spelling. The only A6 checkpoint-owner constructor captures one
caller-supplied local managed root. It must not accept `Arc<dyn StateBackend>`,
`LocalStateBackend`, a separate state root, or a separate manifest root. The
owner constructs its `LocalStateBackend` internally and derives fixed,
non-overlapping state and manifest children under the managed root. No public
constructor, builder, conversion, or escape hatch may reintroduce an erased
local backend beside an independently supplied manifest path.

Construction performs only lexical validation and immutable path capture.
During asynchronous start, the owner creates and canonicalizes the managed
root, resolves symlink aliases, and acquires one cross-process exclusive lease
covering that complete canonical root before manifest scan, connector open,
state mutation, or task spawn. It then opens the engine-created state lineage
inside the derived state child. Failure at either acquisition releases every
acquired lease before returning a conflict. Two jobs using the same canonical
root, including different lexical or symlink aliases, conflict before
lifecycle work. Fixed engine-owned child names prevent equal, ancestor, and
descendant state/manifest relationships by construction; callers cannot name
either child independently. Both the complete-root lease and state-lineage
lease remain owned until terminal cleanup settles or the reaper completes.

The managed namespace identity contains the canonical managed root, pipeline
name, and semantic fingerprint. Runtime tuning may set
`runtime_config_changed` but never changes that identity or rejects recovery.
A6 does not define, rename, or repurpose a manifest-oriented
`CheckpointStore`, and it does not give callers public state-backend access or
a method that publishes arbitrary manifests. Existing standalone public
`StateBackend` and `LocalStateBackend` APIs are not accepted by this owner.

For a namespace whose candidates all validate, automatic start selection has
exactly two modes:

- if the lineage has no completed manifest, start is a new run: operators are
  reset, sources open at their configured beginning, sequences start at zero,
  and the first checkpoint epoch is one;
- if the lineage has a completed manifest, start is a resume: the highest
  selected completed epoch is validated, state and progress are restored, sink
  recovery completes, non-ended sources seek to exact cursors, and the next
  epoch is selected epoch plus one.

There is no “ignore existing state” flag. A caller that wants a new lineage
must use a fresh managed checkpoint namespace. Resume is always a new job; a
terminal or failed `StreamingJob` is never restarted in place.

Temporary and orphan files do not select recovery. Every canonical manifest
candidate is validated in deterministic ascending epoch order, matching the
baseline. If any lower or higher candidate is corrupt, unreadable, or
semantically mismatched, the whole lineage fails closed; A6 does not skip that
candidate or fall back to an older epoch or empty state. If every candidate is
valid, sparse epochs select the highest. A terminal manifest completes only
sink-scoped recovery and returns a completed job outcome without polling
sources or re-emitting operator end output.

### D5 - Manual checkpoint durable completion

Manual, periodic, and terminal checkpoints share the same bounded,
single-flight coordinator. Requests are FIFO, never coalesced, and each
accepted request owns one strictly increasing epoch. Manual requests wait
behind an active periodic request; terminal completion waits behind the active
epoch and is ordered after all earlier accepted manual requests. This is the
existing #108 coordinator path: Public A6 exposes it and must not add a second
manual queue, allocator, waiter registry, or completion seam.

Manifest publication has exactly three semantic outcomes:

- **absent** - the expected canonical bytes for `E` were not installed; the
  prepared transaction may be aborted and recovery must not select `E`;
- **installed with unknown durability** - atomic rename installed the expected
  canonical bytes, but parent-directory synchronization did not complete; the
  current process can observe `E`, while a crash may preserve or lose it;
- **durable** - the expected canonical bytes are installed and the parent
  directory has been synchronized.

Publication durability and manual completion are distinct. Every referenced
segment is durable and every sink has pre-committed before publication, but
`trigger_checkpoint` succeeds with epoch `E` only when the coordinator reaches
`Completed`: the manifest is durable **and every expected sink commit has been
acknowledged**. The method does not wait for later retention, orphan
collection, or compaction. Public A6 therefore preserves the #108 completion
point and must not return at `ManifestDurable` or merely after parent sync.

Cancellation after manifest durability must finish the in-flight sink commit
attempt before reporting success or converge the job to `recovery_required`.
A partial or failed sink commit before `Completed` fails the manual observer;
recovery uses the durable manifest to finish the intent forward on the next
start. Once `trigger_checkpoint` has returned `E`, all sink commits for `E`
were acknowledged, an immediate crash can select `E`, and a later maintenance
failure cannot retroactively revoke the returned result. Status and `wait`
surface any later terminal failure without changing the completed epoch.

An installed-with-unknown-durability outcome completes the manual observer
with a distinct typed indeterminate-publication error that safely identifies
`E`. Its stable category is `checkpoint_publication_unknown`, distinct from
generic I/O, timeout, cancellation, and manifest mismatch. It does not advance
`last_completed_epoch`, does not commit or abort the prepared sink intent, and
converges the job to `recovery_required`. The next start may select `E` if the
canonical file persisted and validates; if the file was lost, it selects the
highest remaining wholly valid candidate (or a new lineage when none exists)
and reconciles connector intent accordingly. The observer must never be told
that `E` is absent.

Periodic and terminal checkpoints use the same classification. An
installed-unknown outcome preserves intent and converges to
`recovery_required`; the typed failure is surfaced through status/outcome even
when no manual observer exists.

Failure, cancellation, deadline, or job convergence with an absent publication
completes the request with a typed error and no selectable `E`. A dropped Rust
future or cancelled Python awaiter loses only observation: the queued request
still executes. The public status field `last_completed_epoch` and
`JobOutcome.completed_epoch` advance only at `Completed`, after durability and
all expected sink commit acknowledgements; status may also name an installed
epoch whose durability is unknown without treating it as completed.

### D6 - Stable identities and capability proof

Stable identities are data, not caller-selected filesystem paths:

- source ID: the exact external graph input/binding ID;
- operator ID: the compiled graph node ID;
- edge ID: the compiler-produced stable edge ID;
- output ID: the external graph output ID;
- sink ID: an explicit portable identifier, globally unique within the job,
  even when sinks are attached to different outputs.

Runner preflight requires the source map to equal the plan source-ID set and
the sink routes to cover the plan output-ID set. Duplicate, missing, unknown,
or cross-output-colliding IDs fail before operator entry, connector open,
lineage mutation, or task spawn. Manifest participant sets must equal the
preflight sets on resume.

Every configured, restored, and emitted source cursor is owned by the exact
source binding ID. Preflight binds an unbound configured cursor or rejects a
cursor already owned by another source; recovery reconstructs the cursor under
the manifest participant ID; live admission rejects a foreign cursor before
progress or data can advance. The manifest cursor entry need not duplicate the
source ID because its enclosing source participant is authoritative, but the
public `Cursor` value must retain and validate that ownership in memory.

Capabilities are sampled exactly once during whole-job preflight, before
connector open, and frozen for the job. The descriptor includes declared
schema, native-watermark capability, normalized watermark policy, explicit
replay positioning, lossless/lossy delivery, maximum batch rows, and maximum
batch bytes. Its canonical capability fingerprint participates in both the
prepared-job fingerprint and durable per-source identity; restart with drift
in any of these declared values fails identity validation instead of silently
changing the job.

Source capability data contains an explicit replay-positioning value with at
least `exact_pause_report_and_seek` and `unsupported`; a bare `replayable`
boolean or the current private fallback from that boolean is not accepted as
Public A6 proof. The explicit positioning value is sampled once, included in
the durable source identity, and checked again on recovery.

The default watermark policy is exactly source-provided watermarks; Python
`watermark_policy=None` has the same meaning. The resolved policy, including
that default, is part of durable source identity and capability validation.

Exactly-once for output `O` requires every component reachable from `O` to
prove all of the following:

- every source declares lossless delivery and
  `exact_pause_report_and_seek`, passes connector conformance tests proving
  that an emitted persisted cursor is honored on reopen without gaps or
  duplicates, and declares a positive maximum batch row and byte bound
  compatible with first-hop edges;
- progress and watermark policy have the M5 durable semantic projection;
- every stateful operator has deterministic checkpoint/restore behavior and
  every selected UDF/operator is deterministic;
- every edge is bounded and lossless (`Block`); and
- every sink is transactional or has an approved epoch-idempotent mechanism
  with unbounded/sufficient retention.

Transactional capability is established by implementing the complete
transactional lifecycle, never by a boolean claim. Epoch-idempotent evidence
records mechanism and retention class. Ordinary sinks and bounded-retention
idempotent sinks remain at-least-once. Capability failure names the requested
output, component kind, and stable component ID. An incompatible component on
a disjoint output does not invalidate `O`; a job may expose exactly-once and
at-least-once outputs together without upgrading the latter.

The connector conformance suite must include an adversarial source that claims
exact positioning but ignores the restore cursor; that source must fail the
exact-replay contract and cannot qualify an exactly-once output. Type spelling
alone is evidence of the declared contract, not evidence that an arbitrary
third-party implementation obeys it.

Status exposes the frozen requested and effective guarantee per output. The
manifest keeps per-sink delivery evidence. Neither surface claims an
end-to-end guarantee stronger than the derived proof.

### D7 - Status allowlist

Public status is a cloned, deterministic, data-only snapshot. Map keys use
stable IDs and `BTreeMap` order in Rust; Python returns fresh dictionaries in
that order. Durations cross Rust as `Duration` and Python/status JSON as
non-negative integer microseconds. The allowlist is exhaustive:

- job: `job_id`, `state`, optional `terminal_cause`, requested/effective
  guarantee per output, `task_count`, `task_errors`, and
  `metrics_overflowed`;
- edge: current/high-water envelope, row, and byte charges; blocked-send count
  and duration; the configured envelope/row and byte limits;
- source: frozen replay-positioning and delivery capabilities, declared
  row/byte bounds, next sequence, ended flag, poll/data/fan-out counts,
  row/byte counts, and error count;
- operator: input/fan-out counts, row/byte counts, processing duration, error
  count, ended flag, late/null event-time counters, maximum lateness, and
  whether a DataFusion runtime was created;
- sink: output ID, effective delivery kind, delivered batch/row/byte counts,
  write duration, error count, and ended flag;
- checkpoint: optional current epoch/phase, terminal flag, source/operator/
  sink pre-commit/sink commit acknowledgement and expected counts, elapsed
  duration, optional `last_completed_epoch`, optional failure category, and
  `runtime_config_changed`; an optional installed-but-durability-unknown epoch
  is permitted only after the D5 outcome and never aliases
  `last_completed_epoch`.

Unknown or future internal fields must not appear automatically. Adding a
public status field requires a separately reviewed contract and tests.

### D8 - Error and log allowlists; sensitive-data prohibition

Every fallible public method introduced or replaced by A6 on the managed
checkpoint owner, `StreamingRunner`, or `StreamingJob` returns the repository
standard `calc_flow::Result<T>`. At this boundary its only failure value is a
new `CalcFlowError::Streaming(StreamingError)` wrapper. The companion API note
must freeze the exact variant spelling and method signatures; it must not
return an unwrapped `CalcFlowError::Io`, `Operator`, `ExternalProvider`,
`TaskPanicked`, or other raw-bearing variant from an A6 method.

`StreamingError` is an owned, data-only safe value containing exactly the
allowlisted fields below. One conversion seam maps internal failures before
they cross into the public runner/job channel, outcome storage, diagnostics,
PyO3, or logging:

| Internal failure context                               | Public streaming category        |
| ------------------------------------------------------ | -------------------------------- |
| A6 argument, route, or capability validation           | `validation`                     |
| Stream-plan compile or frozen-plan incompatibility     | `compile`                        |
| Complete managed-root or lineage lease conflict        | `conflict`                       |
| Explicit or cooperative lifecycle cancellation         | `cancelled`                      |
| Checkpoint deadline before the D5 completed point      | `checkpoint_timeout`             |
| Candidate, identity, checksum, or manifest mismatch    | `checkpoint_mismatch`            |
| Installed manifest with unknown parent-sync durability | `checkpoint_publication_unknown` |
| Engine-owned filesystem or state/manifest I/O          | `io`                             |
| Operator callback or execution failure                 | `operator`                       |
| Source or sink open/lifecycle callback failure         | `connector`                      |
| Contained runtime task panic                           | `task_panicked`                  |
| Unclassified engine invariant failure                  | `internal`                       |

The conversion retains no raw source object, filesystem path, panic payload,
connector value, Python exception, traceback, or original message. Rust
`Display` and every `Debug` form render only the safe fields;
`StreamingError::source()` returns `None`. The outer
`CalcFlowError::Streaming` may expose only that safe value as its source and
must add no raw source or hidden formatting. Thus `start().await` after a
credential-bearing connector-open error returns
`CalcFlowError::Streaming(StreamingError { category: connector, ... })`, and a
pre-install checkpoint filesystem failure returns the same wrapper with
category `io`.

PyO3 converts only the safe `StreamingError` value into the frozen Python
exception hierarchy/category selected by the API note. Its text and data use
the same allowlist; `__cause__` and `__context__` must not retain the native or
Python callback exception. `JobOutcome.errors` stores the same safe value, not
a second conversion of the raw error.

Public streaming errors and `JobOutcome` entries may contain only:

- stable error code/category (`validation`, `compile`, `conflict`, `cancelled`,
  `checkpoint_timeout`, `checkpoint_mismatch`,
  `checkpoint_publication_unknown`, `io`, `operator`, `connector`,
  `task_panicked`, or `internal`);
- safe engine-generated message;
- optional job ID, epoch, checkpoint phase, component kind, stable component
  ID, and bounded diagnostic ID;
- a stable ordering position for attached secondary errors.

Logs may contain only event name, severity, job ID, terminal state/cause,
stable component kind/ID, epoch/phase, counts, durations, error code, and
bounded diagnostic ID. The one drop warning event is
`abandoned_streaming_job`.

The following are forbidden from public status, error display/source chains,
`Debug`, logs, metrics labels, Python exception text, and diagnostic records:

- batches, row values, Arrow/array contents, or `BatchMetadata.attributes`;
- cursor order bytes, cursor payload, latest/durable cursor values, source
  offsets, or connector pre-commit metadata;
- state bytes, state handles, segment IDs, checksums, manifest bytes, canonical
  manifest JSON, staging/committed paths, or managed roots;
- connector options, credentials, secret values/references, authorization
  headers, connection strings, URLs, query strings, client-library raw errors,
  or Python callback representations;
- UDF source/callables, panic payload text, private task names/IDs, private
  launch IDs, or runtime/state identity hashes.

Connector and Python callback failures must pass through the single safe
conversion seam before any observer can see them. The managed checkpoint is
not a public diagnostic surface: it may contain only the cursor, pre-commit,
state-handle, checksum, identity, and other bounded fields required by the v3
recovery schema. It must not contain raw batches, connector options or
credentials, client-library errors, Python callback representations, UDF
source/callables, or panic payload text.

The full manifest is engine-owned recovery data and is never passed to a
connector. Each transactional sink receives a fresh sink-scoped recovery value
containing only the selected epoch, terminal flag, that sink's frozen delivery
evidence, and that sink's own bounded pre-commit mapping. It contains no source
cursor or progress, operator state/handle/checksum, manifest path/bytes, or
other sink ID/evidence/pre-commit mapping. Rust and Python expose the same
projection, and connector mutation cannot change the engine-owned manifest.

Canary-secret tests must exercise constructor/preflight, start-time local I/O,
connector open, operator failure, contained panic, checkpoint timeout,
pre-install checkpoint I/O, and installed-unknown publication. They search
`Display`, normal and alternate `Debug`, complete `source()` chains, outcome,
status, logs, metrics, diagnostics, serialized safe fields, Python exception
text, `__cause__`, and `__context__` for every forbidden value. A separate
manifest census must prove that the managed manifest contains only its
schema-approved recovery fields and none of the categories prohibited from
checkpoints in the preceding sentence. Stable participant IDs are permitted
only after portable-ID validation; callers must not encode secrets into IDs.

### D9 - PyO3 and Python parity

PyO3 owns native runner/job lifecycles and invokes the Rust engine. Python
adapters validate/copy mappings, bridge async source/sink methods, reject a
running event loop from lifecycle methods that block the calling thread, and
never implement graph, checkpoint, recovery, or delivery semantics
independently.

All potentially blocking connector lifecycle methods are declared `async def`:
source `open`/`next`/`close`; ordinary sink `open`/`write`/`close`; and
transactional sink `open`/`begin_epoch`/`write`/`pre_commit`/`commit`/`abort`/
`recover`/`close`. Adapter construction or pure preflight validates the method
shape without calling it and rejects every synchronous implementation before
native launch. Only pure, CPU-local capability accessors remain synchronous.
Accepted lifecycle coroutines must obey the cooperative rule in D3.

Python exposes async methods `start_async`, `trigger_checkpoint_async`,
`shutdown_async`, `cancel_async`, and `wait_async`, with blocking twins named
without `_async`. Blocking methods check for a running loop before any other
work and raise builtin `RuntimeError` with:

```text
<method>() cannot run inside an event loop; use <method>_async()
```

`status` is the only status operation. It is synchronous, CPU-local, permitted
inside a running event loop, and returns a fresh defensive copy. There is no
`status_async`; the event-loop rejection above applies to `start`,
`trigger_checkpoint`, `shutdown`, `cancel`, and `wait`, not `id` or `status`.

The Python managed-checkpoint constructor accepts exactly one local managed
directory and performs only lexical validation and immutable path capture. It
does not expose separate state/manifest directories or accept a native backend
object. Directory creation, canonicalization, derived-child construction,
locking, scanning, and all other filesystem I/O occur during async native
start (or after the blocking `start` event-loop check), never synchronously in
an active event loop.

Async awaiter cancellation follows this exhaustive table:

| Awaited operation                  | Required native effect when the Python awaiter is cancelled                      |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| `start_async`                      | Cancel and reap the provisional launch; never expose a half-started job          |
| `wait_async`                       | Detach only the observer; do not affect the job                                  |
| `trigger_checkpoint_async`         | Keep the accepted request queued; detach only the observer                       |
| `shutdown_async` or `cancel_async` | Preserve the recorded transition and native convergence; a later call reattaches |

If the native operation was terminal before Python cancellation linearized,
the native result/error wins. Otherwise native cleanup or observer detachment
must be established before `asyncio.CancelledError` propagates. For a
conforming connector, observed terminal completion leaves no pending Python
task, native await lease, live callback root, connector, state/manifest lease,
or runtime worker.

### D10 - Atomic legacy-symbol cutover

The implementation must apply this inventory exactly:

| Surface                   | Remove                                                                                                               | Add or replace                                                                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Rust crate exports        | `Checkpoint`, `CHECKPOINT_FORMAT_VERSION`, `MAX_CHECKPOINT_DOCUMENT_BYTES`, `CheckpointStore`, `FileCheckpointStore` | Keep `CheckpointManifest`, `MANIFEST_FORMAT_VERSION`, state backends; expose a local single-root managed checkpoint owner                |
| Rust I/O exports          | `Source`, `SourceItem`, `BatchingSource`, `Sink`, `SinkRouter`                                                       | Public identity-bound cursor/events, frozen source capability descriptor, bindings, async sink contracts, and sink-scoped recovery value |
| Rust runners              | `MicroBatchRunner`; push `StreamingRunner::{step,reset,plan_snapshot}`                                               | One-shot source-driven `StreamingRunner::start(self)` and owning `StreamingJob`                                                          |
| PyO3 registration         | `_MicroBatchRunner`, legacy `_StreamingRunner`, `_FileCheckpointStore`                                               | Native stream plan, one-shot runner/job, managed checkpoint owner, status/outcome, and async connector projections                       |
| Python runtime/store      | `Source` v2 protocol, `MicroBatchRunner`, push-runner methods, public v2 `FileCheckpointStore`                       | Async continuous source/sink protocols/bindings, single-directory managed checkpoint owner, one-shot runner/job, status/outcome/config   |
| Python exports and typing | Legacy names above in `calc_flow.__all__` and `_native.pyi`                                                          | Every supported A6 public type/error/enum and exact registered native signatures                                                         |
| Studio backend            | Direct top-level import/construction of public `FileCheckpointStore`                                                 | Studio-private persistence wiring that preserves existing `/api/v2` checkpoint load/delete behavior and existing v2 checkpoint documents |
| Examples and benchmarks   | v2 micro-batch/push runner and public checkpoint-store call sites                                                    | Public A6 examples/benchmarks or explicit removal where the scenario no longer exists                                                    |
| Docs and release checks   | Current-surface claims and smoke assertions for removed symbols                                                      | A6 docs plus core/Studio import, `create_app()`, wheel, example, benchmark-discovery, and removed-symbol smoke checks                    |

The batch `BatchExecutionPlan` and its Python execution surface remain
available. A6 must expose `StreamExecutionPlan` through PyO3/Python without
renaming it back to ambiguous `ExecutionPlan`. Removal tests cover Rust use
sites, native module attributes, Python import/export, the stub, repository
search, Studio import/startup/routes, example discovery, benchmark collection,
and package smoke tests; checking only `__all__` is insufficient. The
Studio-private persistence migration is not a public alias or compatibility
shim in `calc_flow`.

## Functional Requirements

- **FR1** - All public A6 behavior is implemented by the M5 Rust runtime; PyO3
  and Python contain adapters only.
- **FR2** - Runner construction consumes one stream plan, exact source/sink
  bindings, and one single-root local managed checkpoint owner and performs
  pure validation without opening connectors or performing filesystem I/O.
- **FR3** - Rust start consumes the runner and returns at most one owning job.
- **FR4** - Start publishes no job until complete namespace leasing,
  recovery/reset, resource open, task registration, and gate readiness.
- **FR5** - Python rejects synchronous connector lifecycle methods before
  invocation; dropped/cancelled start settles all provisional ownership for a
  conforming async connector and releases both state and manifest leases.
- **FR6** - One managed checkpoint owner derives non-overlapping state and
  manifest children, rejects any caller-supplied backend or child root, and
  enforces complete-root conflicts across processes and symlink aliases.
- **FR7** - New lineages and resumes follow D4 with no ignore/reset flag.
- **FR8** - Resume validates every canonical candidate plus manifest identity,
  participant sets, state handles/checksums, source cursor ownership, frozen
  capability identity, and sink delivery evidence before data work; any
  invalid candidate poisons the lineage.
- **FR9** - Terminal-manifest resume completes with sink-scoped recovery and no
  source polling or duplicate final operator output.
- **FR10** - Exactly one checkpoint is active; accepted periodic/manual/
  terminal requests execute through the existing #108 FIFO without
  coalescing or a second public-side queue.
- **FR11** - Manifest publication is classified as absent,
  installed-with-unknown-durability, or durable exactly as D5.
- **FR12** - Manual checkpoint success and public completed-epoch fields use
  only the D5 `Completed` point after parent-synchronized manifest durability
  and every expected sink commit acknowledgement.
- **FR13** - An absent publication may abort and exposes no selectable epoch;
  an installed-unknown publication preserves intent, returns a typed
  indeterminate error, and transitions to `recovery_required`.
- **FR14** - Post-durable, pre-completed sink failure preserves
  forward-recoverable intent, fails the manual observer, and may transition the
  job to `recovery_required`; failure after `Completed` cannot revoke success.
- **FR15** - Graceful and natural completion publish one terminal epoch after
  end processing and never send a barrier after end.
- **FR16** - Explicit cancel has no drain promise but joins and releases owned
  resources before its awaited result for conforming connectors.
- **FR17** - Wait/shutdown/cancel are idempotent observers of one immutable
  terminal outcome.
- **FR18** - Job-handle drop is non-blocking, emits one safe warning, and
  transfers cleanup to an owner that eventually releases resources; it adds no
  unusable per-job status field.
- **FR19** - Stable participant IDs, cursor ownership, and exact route coverage
  are validated before lifecycle work and again as live cursors are admitted.
- **FR20** - Exact replay uses an explicit positioning capability, durable
  identity, and conformance evidence; a `replayable` boolean, fallback inferred
  from it, foreign cursor, or ignored restore cursor is insufficient.
- **FR21** - The complete source descriptor in D6 is sampled once before open,
  frozen, and fingerprinted into durable identity; per-output proof derives
  only from that value, and lossy sources, ordinary sinks, or
  insufficient-retention sinks can never be reported exactly-once.
- **FR22** - Every sink receives only its own D8 recovery projection, never the
  complete manifest or another participant's data.
- **FR23** - Every fallible A6 checkpoint-owner, runner, and job operation maps
  internal failures exactly once to `CalcFlowError::Streaming(StreamingError)`;
  status, outcomes, logs, metrics labels, debug/source formatting, Python
  exceptions, and managed manifests comply with D7-D8.
- **FR24** - Python async-only connector, blocking/async lifecycle,
  cancellation, GC, status-in-event-loop, and defensive-copy semantics comply
  with D9.
- **FR25** - The complete in-repository legacy inventory in D10 is migrated and
  removed in one atomic cut without breaking Studio v2 startup/routes,
  examples, benchmark collection, documentation, or package smoke tests.
- **FR26** - Public Rust examples compile, Python examples undergo signature
  and runtime checks, and removed examples fail with the expected missing
  symbol/method.
- **FR27** - The v3 plan status is updated to “M0-M5 internal complete; Public
  A6 pending” before A6 implementation is claimed ready.

## Non-Functional Requirements

- **Performance** - A6 adds no extra data-path task, queue, batch copy, Arrow
  materialization, or per-batch serialization relative to the M5 private path.
  The public implementation head is compared with baseline
  `aa3bbf0b40aef74898a59b6d0d0028c59a2d6993` using the existing common-edge
  harness; a greater-than-5% regression is accepted only with statistically
  conclusive same-host evidence and explicit approval. Private checkpoint
  paths remain absolute-only where the baseline has no corresponding public
  path.
- **Boundedness** - Existing edge budgets, source prefetch bound, checkpoint
  channel bounds, status cardinality, diagnostic retention bounds, and
  manifest size/depth limits remain unchanged. A manual request queue is
  bounded and backpressures request submission rather than allocating without
  limit.
- **State and checkpoints** - Snapshot/restore/reset and manifest semantics are
  exactly D4-D6. Manifest format remains v3 and is not rewritten by this
  public projection. The public A6 owner accepts one local root and derives
  its internal namespaces; lease acquisition/release and all local filesystem
  work remain asynchronous and owner-settled.
- **Compatibility** - A6 is deliberately incompatible with the v2 runner and
  checkpoint public API, but preserves batch execution, project v2 behavior,
  manifest v3 bytes, existing `tests/fixtures/v1/`, and existing Studio v2
  route/startup/checkpoint-document behavior. Rust/PyO3/Python, the Studio
  package, examples, benchmarks, docs, and release smoke checks must agree at
  every merged revision.
- **Determinism** - Status map ordering, terminal error selection, checkpoint
  request ordering, stable IDs, manifest bytes, and output capability reports
  are repeatable for equal inputs.
- **Security and privacy** - D7-D8 are fail-closed. Raw internal causes are
  discarded at the single streaming-error conversion seam, every outward
  formatting/source/Python path uses only the safe value, and no sink receives
  cross-participant recovery data.
- **Documentation-only issue** - This A6-01 issue changes specifications,
  API notes, critique, and plan status only; it does not alter runtime behavior.

## Inputs and Outputs

| Name                     | Type                                                          | Units                             | Range / Constraints                                                                                            |
| ------------------------ | ------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Stream plan              | Rust/Python `StreamExecutionPlan`                             | n/a                               | Compiled, immutable, semantic fingerprint fixed                                                                |
| Source bindings          | Map from source ID to owned source binding                    | n/a                               | Exact plan input-ID set; frozen descriptor; lossless exact replay for exactly-once                             |
| Sink bindings            | Map from output ID to ordered owned sink bindings             | n/a                               | Exact output coverage; sink IDs globally unique; async lifecycle                                               |
| Checkpoint runtime       | One local managed directory captured by an owned A6 boundary  | n/a                               | Internal fixed state/manifest children; complete canonical-root lease; no backend input or public manifest I/O |
| Runtime config           | `StreamRuntimeConfig`                                         | microseconds, rows, bytes, epochs | Exact-microsecond durations; positive limits; retained epochs positive                                         |
| Job ID                   | `u64` / Python `int`                                          | n/a                               | Stable for one job; non-secret                                                                                 |
| Epoch                    | `Epoch` / Python non-negative `int`                           | checkpoint ordinal                | First checkpoint epoch 1; strictly increasing; checked `u64`                                                   |
| Status                   | Public immutable snapshot / fresh Python dictionary           | counts and microseconds           | D7 allowlist only; synchronous in Python event loops; stable-ID ordering                                       |
| Job outcome              | `JobOutcome` / frozen Python value                            | n/a                               | Terminal state, cause, optional completed epoch, ordered sanitized errors                                      |
| Manual checkpoint result | Completed epoch or typed safe checkpoint error                | checkpoint ordinal                | Success after manifest durability and all sink commit acks; indeterminate error identifies installed `E`       |
| Fallible A6 result       | `calc_flow::Result<T>` with safe streaming wrapper on failure | n/a                               | Only `CalcFlowError::Streaming(StreamingError)`; no raw cause, path, payload, or callback value                |
| Source cursor            | Public `Cursor` passed only to source lifecycle and manifest  | connector-defined                 | Stable source binding ID, non-empty bounded order key, strict bounded JSON; never in status/errors/logs        |
| Sink pre-commit          | Strict bounded `JsonMap`                                      | connector-defined                 | Durable manifest data; never in status/errors/logs                                                             |
| Sink recovery value      | Engine-created sink-scoped immutable value                    | n/a                               | Selected epoch/terminal flag plus only that sink's evidence and pre-commit                                     |

## Named M5 Fault Catalog

The public A6 validation catalog contains exactly these 48 stable case IDs.
The four modes mean injected I/O error, panic, cancellation, and process
restart at the named durable boundary.

| Boundary             | I/O case                           | Panic case                            | Cancel case                            | Restart case                            |
| -------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | --------------------------------------- |
| Source admission     | `m5_fault/source_admission/io`     | `m5_fault/source_admission/panic`     | `m5_fault/source_admission/cancel`     | `m5_fault/source_admission/restart`     |
| Source cut           | `m5_fault/source_cut/io`           | `m5_fault/source_cut/panic`           | `m5_fault/source_cut/cancel`           | `m5_fault/source_cut/restart`           |
| Partial alignment    | `m5_fault/partial_alignment/io`    | `m5_fault/partial_alignment/panic`    | `m5_fault/partial_alignment/cancel`    | `m5_fault/partial_alignment/restart`    |
| State stage          | `m5_fault/state_stage/io`          | `m5_fault/state_stage/panic`          | `m5_fault/state_stage/cancel`          | `m5_fault/state_stage/restart`          |
| Sink pre-commit      | `m5_fault/sink_pre_commit/io`      | `m5_fault/sink_pre_commit/panic`      | `m5_fault/sink_pre_commit/cancel`      | `m5_fault/sink_pre_commit/restart`      |
| Manifest write       | `m5_fault/manifest_write/io`       | `m5_fault/manifest_write/panic`       | `m5_fault/manifest_write/cancel`       | `m5_fault/manifest_write/restart`       |
| Manifest rename      | `m5_fault/manifest_rename/io`      | `m5_fault/manifest_rename/panic`      | `m5_fault/manifest_rename/cancel`      | `m5_fault/manifest_rename/restart`      |
| Manifest parent sync | `m5_fault/manifest_parent_sync/io` | `m5_fault/manifest_parent_sync/panic` | `m5_fault/manifest_parent_sync/cancel` | `m5_fault/manifest_parent_sync/restart` |
| Partial sink commit  | `m5_fault/partial_sink_commit/io`  | `m5_fault/partial_sink_commit/panic`  | `m5_fault/partial_sink_commit/cancel`  | `m5_fault/partial_sink_commit/restart`  |
| Completed commit     | `m5_fault/completed_commit/io`     | `m5_fault/completed_commit/panic`     | `m5_fault/completed_commit/cancel`     | `m5_fault/completed_commit/restart`     |
| Retention            | `m5_fault/retention/io`            | `m5_fault/retention/panic`            | `m5_fault/retention/cancel`            | `m5_fault/retention/restart`            |
| Compaction           | `m5_fault/compaction/io`           | `m5_fault/compaction/panic`           | `m5_fault/compaction/cancel`           | `m5_fault/compaction/restart`           |

Every case records the selected epoch; restored source cursor, watermark,
idle/end and sequence state; restored operator/window state; visible ordinary
and transactional sink output; missing and duplicate counts; reachable and
temporary artifacts; terminal state/error; task/connector/lease cleanup; and
whether a manual observer received completed success, an absent-publication
error, or an installed-unknown error. “Success” means `Completed` after every
sink commit acknowledgement; partial-sink-commit cases must fail the observer
and preserve forward recovery. Manifest-rename and parent-sync cases must
exercise both post-crash persistence outcomes permitted by D5. A case is not
covered merely because the enum cross-product is enumerated.

## Verification and Test Mapping

| Contract                          | Existing evidence on `aa3bbf0b…`                                                                                                                  | Missing A6/public evidence                                                                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| One-shot start and launch cleanup | Private launch/drop/reaper tests in `runtime/streaming/runner.rs`                                                                                 | Rust consuming signature; Python one-shot; gated launch cancellation with cooperative callbacks; sync callback rejected before invocation          |
| Owning job lifecycle              | Idempotent terminal observers plus graceful/cancel/deadline/drop tests                                                                            | Public Rust/Python parity; one drop warning; no unusable `abandoned_job_drops`; synchronous `status()` works in an event loop                      |
| Complete namespace lease          | State backend has same-lineage in-process and file locks; manifest transaction canonicalizes but does not lease its root                          | Single-root owner, derived-child isolation, same-root/cross-process/symlink conflicts, and compile-fail rejection of backend/two-root constructors |
| New lineage/resume                | Empty-lineage, distinct-cursor reopen, restore, and terminal tests; selector validates every candidate                                            | Empty/`R + 1`/terminal public tests plus corrupt-lower/valid-higher and valid-lower/corrupt-higher fail-closed cases                               |
| Manual checkpoint                 | #108 covers bounded FIFO allocation, periodic/manual/terminal ordering, dropped observers, cancellation, shutdown draining, and terminal failures | Public Rust/Python operation, safe error mapping, Python observer cancellation, and exact signature checks                                         |
| Checkpoint completion             | #108 resolves manual waiters after durable manifest plus all sink commit acks and covers partial-commit recovery-required restart                 | Public completed-epoch projection, installed-unknown safe error, post-completion maintenance behavior, and cross-surface restart evidence          |
| Source identity                   | #110 binds configured/restored/emitted cursors to binding IDs and rejects foreign cursors before progress                                         | Public cursor identity signatures plus Rust/Python configured, restored, emitted, and cross-source rejection tests                                 |
| Source capability identity        | #110 samples the full descriptor once before open and fingerprints schema/watermark/replay/delivery/bounds into durable identity                  | Public frozen descriptor, no boolean fallback, restart drift matrix, and Python parity                                                             |
| Exact replay capability           | Private progress models `ExactPauseReportAndSeek`; #110 freezes it but current preflight still accepts the `replayable` fallback                  | Public enum/identity projection and connector conformance test whose claimed-exact source ignores the restore cursor                               |
| Per-output capability             | #110 rejects lossy and non-replayable reachable sources before open; existing tests cover disjoint output proofs                                  | Public requested/effective report and Python projection; reachable-only volatile UDF plus exact-replay cases                                       |
| Sink-scoped recovery/redaction    | Private recovery passes the full manifest to every transactional sink                                                                             | Two-sink isolation plus safe-wrapper mapping and constructor/start/checkpoint/outcome/format/source/Python canary censuses                         |
| Atomic repository cutover         | Repository search finds Studio, examples, benchmarks, tests, docs, and release assertions using legacy symbols                                    | Removed-symbol census plus Studio import/`create_app()`/route/package smoke, example compile/run, benchmark collection, docs/release checks        |
| Cross-surface durability          | M5/#108/#110 Rust recovery tests                                                                                                                  | Rust-to-Python and Python-to-Rust restart E2E using one managed checkpoint owner and deterministic conforming connectors                           |
| Fault catalog                     | Fault catalog enumeration plus focused transaction/task/manual/source-identity tests                                                              | Executed named 48-case matrix with absent/installed-unknown/completed assertions; enumeration alone is insufficient                                |
| 20-minute soak                    | Private ignored `twenty_minute_epoch_checkpoint_restart` contract                                                                                 | Same 3-generation/120-sample public-runner contract on exact final head; no shorter substitute                                                     |

The cross-surface E2E uses one shared deterministic connector protocol and
strictly identical plan/participant identities. Direction A checkpoints in
Rust and resumes in a fresh Python process; direction B checkpoints in Python
and resumes in a fresh Rust process. Both directions verify manifest bytes,
selected/next epoch, source seek, final window output, sink commit idempotency,
and zero live resources. Only filesystem manifests, state segments, and sink
evidence may cross the process boundary.

The public 20-minute soak keeps the M5 contract: one parent launches exactly
three child OS generations over global ranges `0..40`, `40..80`, and
`80..120`; it records exactly 120 ten-second samples over 1,200 measured
seconds and verifies bounded queues/tasks/state/manifests/RSS, restart
continuity, no missing/duplicate transactional output, no temporary artifacts,
and terminal zero ownership. Evidence includes the exact commit and executable
hash; any implementation push invalidates the result.

## Acceptance Criteria

- [ ] The specification, API note, and critique all use the slug
  `a6-public-continuous-runtime` and baseline
  `aa3bbf0b40aef74898a59b6d0d0028c59a2d6993` or explicitly rebase together to
  a newer reviewed `main`.
- [ ] The independent critique reports zero blockers separately for ownership,
  checkpoint completion, recovery, capability proof, redaction, and cutover.
- [ ] Rust public examples compile and signature tests prove the runner is
  consumed by `start(self)` and the job is the sole non-cloneable owner.
- [ ] Python rejects every synchronous connector lifecycle method before
  invocation; a cooperative async callback blocked on an await point can be
  cancelled at every launch/runtime gate with no task, callback root,
  connector, temporary artifact, state lease, or manifest lease left live.
- [ ] The public A6 owner accepts exactly one local managed root; compile-fail,
  native-registration, stub, and Python-signature tests prove that no
  `Arc<dyn StateBackend>`, backend object, separate state root, separate
  manifest root, or two-root constructor is accepted.
- [ ] Start derives fixed non-overlapping state and manifest children; two jobs
  using the same canonical root conflict before lifecycle work in-process and
  cross-process, including lexical and symlink aliases, and every acquired
  root/lineage lease is released after failed start or terminal cleanup.
- [ ] Given an empty lineage, start resets state and produces epoch one; given
  a valid manifest at `R`, start restores before data work and produces next
  epoch `R + 1`.
- [ ] Both corrupt-lower/valid-higher and valid-lower/corrupt-higher candidate
  sets fail closed; an all-valid sparse set selects its highest epoch.
- [ ] A terminal-manifest resume gives each sink only its own scoped recovery
  value and completes without polling a source or re-emitting final operator
  output.
- [ ] Manual, periodic, and terminal requests are bounded, FIFO,
  single-flight, non-coalescing, and allocate distinct increasing epochs.
- [ ] A manual request returns only at `Completed`, after manifest install plus
  parent sync and every expected sink commit acknowledgement, but before later
  retention/orphan/compaction work; an immediate crash can recover the returned
  epoch and no pre-completed commit failure is reported as success.
- [ ] Rename-installed/parent-sync-failed returns the D5 typed indeterminate
  error for `E`, leaves `last_completed_epoch` unchanged, performs neither sink
  commit nor abort, and reaches `recovery_required`; real restart tests cover
  both persistence and loss of `E` without claiming absence, using a real
  directory-sync failure rather than only a post-sync test hook.
- [ ] Dropping a manual observer keeps its request; an absent publication
  returns error with no selectable epoch; post-durable/pre-completed sink
  failure fails the observer and preserves recovery intent, while a failure
  after `Completed` does not revoke success and remains visible through
  terminal status/outcome.
- [ ] Graceful, natural, cancel, deadline, failure, start-drop, job-drop, and
  Python-GC paths satisfy D2-D3 for conforming connectors; job drop emits one
  safe warning and `JobStatus` contains no `abandoned_job_drops` field.
- [ ] Python exposes async-only connector lifecycle I/O, only synchronous
  `status()`, allows `status()` in an active event loop, rejects other blocking
  lifecycle twins there before argument validation, and passes every D9
  cancellation row with no pending native/Python ownership.
- [ ] Configured, restored, and emitted cursors retain the exact source binding
  ID; a foreign cursor fails before connector open or live progress admission,
  and Rust/Python recovery cannot attach one source's cursor to another source.
- [ ] The complete source descriptor is sampled once before open and its
  schema, native-watermark capability, normalized policy, replay positioning,
  delivery capability, and row/byte bounds are stable across status, durable
  identity, manifest validation, Rust, and Python; any restart drift fails
  closed.
- [ ] A lossy source or a source claiming exact positioning while ignoring the
  restore cursor fails connector conformance and cannot qualify an
  exactly-once output; neither `replayable: true` nor its fallback is accepted
  as Public A6 proof.
- [ ] Every exactly-once request validates the complete reachable path before
  lifecycle work and reports the same requested/effective guarantee in Rust,
  Python, status, and manifest evidence.
- [ ] Duplicate, missing, unknown, and cross-output participant IDs fail before
  lifecycle work with the safe output/component/ID diagnostic.
- [ ] The D7 field census passes exactly and no unapproved internal field is
  serialized or exposed.
- [ ] Signature and runtime tests prove every fallible public A6 owner/runner/
  job method returns `calc_flow::Result<T>` and every failure is exactly
  `CalcFlowError::Streaming(StreamingError)`, never a raw-bearing variant.
- [ ] Start-time local I/O, connector open, operator failure, panic, checkpoint
  timeout, pre-install checkpoint I/O, and installed-unknown canaries retain
  the required category and coordinates while every D8 forbidden value is
  absent from `Display`, normal/alternate `Debug`, full `source()` chains,
  outcome, status, logs, metrics, diagnostics, safe serialization, Python
  exception text, `__cause__`, and `__context__`.
- [ ] A separate manifest census permits only schema-approved recovery fields
  and rejects every checkpoint-prohibited value.
- [ ] A two-sink Rust/Python canary proves neither sink can observe source
  cursor/progress, operator state, manifest storage, or the other sink's ID,
  delivery evidence, or pre-commit mapping.
- [ ] Rust exports, PyO3 registration, Python exports, `_native.pyi`, Studio
  backend, examples, benchmarks, docs, and release checks complete every D10
  replacement in one atomic change; Studio import, default `create_app()`, v2
  checkpoint routes/documents, package smoke, examples, and benchmark
  collection remain green.
- [ ] All 48 named M5 fault cases execute and record the full required outcome;
  rename/parent-sync cases cover absent, installed-unknown, durable, and both
  installed-unknown crash outcomes; enumeration alone is insufficient.
- [ ] Rust-to-Python and Python-to-Rust restart E2E tests pass using only
  one managed checkpoint namespace and durable filesystem evidence across
  process boundaries.
- [ ] The exact-head public-runner 20-minute, three-generation, 120-sample soak
  passes and records terminal zero ownership.
- [ ] The v3 plan header says “M0-M5 internal complete; Public A6 pending” and
  `docs/introduction.md`/`docs/runtime-envelope.md` are updated when the public
  implementation lands.
- [ ] Every touched surface passes the complete verification matrix in
  `AGENTS.md`, including Rust/PyO3/Python tests, rustdoc, formatting, linting,
  coverage, generated-file checks where applicable, and `git diff --check`.
- [ ] A6-01 itself changes no runtime behavior.

## Open Questions

None in the controlling specification. The companion API note and prior
critique remain stale at `main@70f7e3d1e9306c419a0b2358527ec888c2ed9934` and
are not approval for this revision. The API designer must rebase the exact
signatures to `main@aa3bbf0b40aef74898a59b6d0d0028c59a2d6993`, preserve
the single-root local owner and safe streaming-error wrapper, expose the
#108 `Completed` manual-checkpoint seam, and add the #110 identity-bound cursor
and frozen source descriptor without a `replayable` fallback. An independent
critic must then re-run ownership, checkpoint completion, recovery, capability
proof, redaction, and cutover against the refreshed spec/API-note hashes.
