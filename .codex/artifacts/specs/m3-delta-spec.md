# Continuous Streaming 3.0 — M3 Delta Specification

| Field | Value |
|---|---|
| Status | **Proposed** |
| Priority | **P0 — blocks M3 implementation and merge** |
| Baseline | `main@88243e9565795cd0bce01a155f75e735f4b47728` |
| Milestone | M3 — event-time progress and watermark coordination |
| Intended audience | runtime, connector, compiler, test, and review owners |

## 1. Authority and references

This document is the controlling delta for M3 wherever an earlier document is silent, ambiguous, or inconsistent with it. It does not replace requirements that are compatible with it.

Normative inputs:

1. `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md` — Continuous Streaming 3.0 development plan (the “main plan”).
2. The repository's controlling continuous-streaming runtime specification (the “runtime spec”).
3. The M2 completion artifacts attached to PR #83, including the final implementation, acceptance-criteria evidence, review resolution, CI/Codacy results, and the successful 20-minute soak result (the “M2 completion artifacts”).
4. Baseline commit `88243e9565795cd0bce01a155f75e735f4b47728`.

Precedence for an M3 conflict is:

```text
this M3 delta
> compatible controlling runtime-spec requirements
> compatible main-plan requirements
> historical M2 notes and implementation details
```

The M2 completion artifacts remain authoritative evidence for M2 lifecycle, cancellation, backpressure, and soak behavior. M3 MUST preserve those properties unless this delta explicitly changes a behavior.

## 2. Problem statement

The pre-delta M3 plan contains four implementation-blocking boundary conflicts:

1. M3 refers to late-data behavior in a window operator that does not exist until M4.
2. M3 uses “recoverable” language although durable checkpoint manifests and restart recovery belong to M5.
3. watermark policy is configured at a source binding, while `compile_stream()` intentionally has no binding or connector context.
4. multi-input progress excludes idle inputs from the watermark minimum but does not fully specify downstream `Idle`, reactivation, or terminal ordering.

Resolving these conflicts also requires freezing timer ownership, deterministic arbitration, capability negotiation, observable status, and milestone boundaries. This specification makes those decisions.

## 3. Normative language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**, and **MAY** are normative as described by RFC 2119-style usage.

An implementation satisfies M3 only when every applicable acceptance criterion in §14 is backed by named automated test evidence and all merge gates in §16 pass.

## 4. Terms

- **Binding**: the prepared association between one logical graph source and one source descriptor/configuration.
- **Prepared job**: an immutable, validated, crate-private job representation produced before any runtime side effect.
- **Preflight**: whole-job validation of graph, source descriptors, schemas, capabilities, and watermark policies.
- **Ingress**: one input edge as observed by the multi-input progress aggregator.
- **Live ingress**: an ingress whose state is `Active` or `Idle`; an `Ended` ingress is not live.
- **Effective ingress watermark**: the monotonic watermark retained for an ingress after validating source-provided or generated progress.
- **Aggregate watermark**: the last watermark emitted downstream by a multi-input progress aggregator.
- **Idle epoch**: a maximal interval during which the live-ingress set is non-empty and every live ingress is `Idle`.
- **Reactivation**: transition from `Idle` to `Active` caused by `Data` or a watermark that is legal for the binding's normalized mode.
- **Logical instant**: a progress-driver clock position used for deterministic timer comparison; it is not a wall-clock timestamp contract.
- **Raw input/control**: an adapter-submitted binding identity plus payload/control kind and, when supported by that source, an opaque upstream replay cursor. It contains no runtime ordering key, logical time, local sequence, timer key, or timer generation.
- **Accepted raw envelope**: raw input/control for which the binding admission gate was open and the driver/inbox atomically allocated a receipt identity and binding-local inbox sequence, took ownership of the one-shot completion sender, and enqueued the envelope. A rejected submission is not accepted.
- **Admission attempt**: every call to submit raw input/control, whether accepted or immediately rejected. Before reading or changing a gate, the driver checked-allocates a stable global admission-attempt ordinal; an ordinal-exhaustion attempt is instead identified by a non-item `DriverPhaseError` coordinate.
- **Completion receipt**: the one-shot `Result` settlement for one accepted raw envelope. Success means that envelope's containing drain committed; failure means it did not commit or was rejected by terminal/cancel/fatal cleanup. A send attempt to a dropped receiver still completes settlement.
- **Drain epoch**: the driver-owned, checked, monotonic identity of one invocation of semantic drain arbitration, including a required empty invocation recorded by the replay contract.
- **Inbox fence**: a driver-owned, checked freeze record for one binding inbox in one drain epoch, identified by a fence sequence and an inclusive upper inbox sequence or `Empty`. The fence fixes which already-accepted envelopes are eligible for that epoch.
- **DrainFenceTrace**: the drain-only projection of `ProgressExecutionTrace`. It records drain epochs, inbox fences, selected item identities/key ranges, due timers, and drain outcomes, but by itself is not sufficient to replay admission or receipt behavior.
- **ProgressExecutionTrace**: the ordered, lossless, crate-private in-memory trace that controls M3 replay. It contains drain-fence records plus every admission decision, terminal/gate-close linearization, accepted-envelope settlement disposition, and driver-phase failure. Its record position and all semantic ordinals are driver-owned and checked.
- **Admission gate generation**: the checked per-binding generation observed by each admission attempt and advanced by a successful End, cancel, or fatal close. A closed gate retains its close cause, close ordinal, and next-inbox-sequence cut.
- **Settlement disposition**: the stable semantic result assigned exactly once to an accepted envelope: `CommitSuccess`, `TransactionError`, `PostEndTailReject`, `Cancelled`, or `Fatal`. Cross-receipt send/wakeup order is not an M3 semantic guarantee; trace comparison canonically orders settlement records by envelope identity.
- **DriverPhaseError**: a deterministic progress failure before selected-item evaluation has a complete `ReadyKey`, identified by phase and driver/admission/fence/counter/trace coordinate and never by a fabricated `ReadyKey`.
- **Upstream delivery/replay cursor**: a source-defined, equality-comparable position that can be reported while paused and sought exactly for same-process replay; row counts, wall time, and “latest” are not cursors.
- **Control frontier**: an equality-comparable source/driver position proving which upstream control records have been delivered and which remain replayable.
- **ReadyKey**: the driver-created total-order key `(logical_instant, class_rank, binding_ordinal, local_sequence)` assigned to a ready item. It is never caller supplied.
- **Quiescent event boundary**: a point after the preceding finite ready snapshot has either committed atomically or failed without commit, with every selected completion receipt settled, all successful emissions returned as one batch, and before capture of the next finite ready snapshot. No scratch transition, buffered timer mutation, or accepted-but-unsettled envelope may be pending; restore-capable capture MUST reject otherwise.
- **Captured logical coordinate**: the complete equality-checked restore position comprising the driver logical instant; each binding's upstream delivery/replay cursor and control frontier; the completed `ProgressExecutionTrace` prefix and consumed trace position; next trace-record, admission-attempt, gate-close, drain-epoch, receipt, global, binding-local, inbox, and fence sequence coordinates; each binding's exact gate generation/state/close cut and last completed fence; and the scheduled timer coordinates retained by the snapshot.
- **Transient snapshot**: a crate-private in-memory representation of progress state for same-process deterministic replay; it is not a checkpoint.
- **Mode-legal connector watermark**: a connector-emitted watermark accepted only when the normalized policy permits native watermark emission.
- **Concrete window assignment**: one specific row-to-window association produced by M4 window assignment, including a concrete `window_end`.

## 5. Scope

M3 includes:

1. crate-private watermark policy representation and normalization;
2. side-effect-free source descriptors sufficient for preflight schema/capability validation;
3. whole-job prepare/preflight before connector opening or task/timer creation;
4. generated bounded-out-of-orderness watermarks over supported Arrow timestamp units;
5. source-provided watermark validation and monotonic forwarding;
6. idle-timeout scheduling and idle/reactivation state transitions;
7. deterministic multi-input progress aggregation;
8. a job-scoped, single-writer progress driver, complete admission/fence/terminal/settlement execution trace, completion-receipt protocol, and timer arbiter;
9. transient in-memory progress snapshot/restore at receipt-quiescent boundaries;
10. progress status and metrics that do not classify rows as late;
11. deterministic tests, stress tests, and a 20-minute soak.

## 6. Non-goals

M3 does not include:

- window assignment, triggering, accumulation, or final-only window execution;
- row/assignment late-data classification, counting, tagging, splitting, or dropping;
- `LateDataPolicy::Drop` execution;
- allowed lateness, late side outputs, or retractions; all remain Continuous Streaming 3.0 non-goals;
- durable snapshot encoding, serialization, checkpoint barriers, manifests, process restart, deadline rebasing, exactly-once recovery, or cross-version compatibility;
- crash-durable receipt recovery or a claim that an M3 in-process receipt provides M5 exactly-once recovery;
- public source-driven runner A6, which remains post-M5;
- public connector-trait, `open()`, Python, REST, or OpenAPI changes;
- connector capability discovery that requires opening the connector or performing I/O;
- silent combination of connector-provided and runtime-generated watermarks.

## 7. Controlling decisions

### D1 — Late-data execution moves to M4

M3 **MUST** preserve the data path regardless of the relationship between a row's event time and any current watermark. M3 **MUST NOT** use `event_time < watermark` or `event_time <= watermark` to classify, count, tag, split, or drop rows or assignments. M3 **MUST NOT** expose a late-row metric, including `late_rows_observed`, `late_rows_dropped`, or aliases with equivalent meaning.

The actual `LateDataPolicy::Drop` behavior belongs to M4's final-only window execution. M4 **MUST** first create each concrete window assignment and **MUST** drop only an assignment for which:

```text
assignment.window_end <= WM_in
```

This is assignment-level behavior. For hopping windows, one row MAY retain open assignments while its already-closed assignments are dropped. M4 MUST NOT replace this predicate with a row-level comparison. Allowed lateness, side output, and retraction remain out of scope for 3.0.

### D2 — M3 restore is transient, exact-coordinate, and in-memory only

M3 **MUST** implement a crate-private transient snapshot/restore seam owned by the job-scoped progress driver. Capture **MUST** occur only at a quiescent event boundary. A restore-capable capture **MUST** include a complete captured logical coordinate: driver logical instant; for every binding, its exact upstream delivery/replay cursor and control frontier; the lossless completed `ProgressExecutionTrace` prefix and consumed trace position; next trace-record, admission-attempt, gate-close, drain-epoch, receipt, global, binding-local, inbox, and fence sequence coordinates; each binding's gate generation/state/close cut and exact last completed fence; and exact pending timer deadlines, generations, and sequences. Any inbox capacity or fence-selection configuration capable of changing grouping **MUST** be included in the prepared/config fingerprint and compared exactly on restore; restore **MUST NOT** accept a free replacement value.

Before restore, every upstream **MUST** be paused and positioned at a cursor/control frontier exactly equal, field by field, to the captured coordinate. Restore **MUST** also require exact equality of prepared-job fingerprint, ordered binding table (identity and driver-assigned ordinal), normalized-config fingerprints, driver logical instant, completed execution-trace prefix/consumed position, admission gate generations/states/close cuts, last-completed fences, and all next-allocation coordinates. “Compatible,” “equivalent,” “at least,” or best-effort positions **MUST NOT** be accepted. Any missing or unequal field **MUST** fail before state mutation, timer registration, connector resume/open, task spawn, admission decision, receipt settlement, or emission.

A source that cannot be paused and deterministically repositioned, or cannot report a stable delivery/replay cursor and control frontier, **MUST NOT** participate in M3 restore. Creation of a restore-capable `StreamProgressSnapshot` **MUST** fail deterministically when any binding lacks that capability; implementations **MUST NOT** claim replay for such a job.

For the exact captured coordinate and the same ordered raw input/control **attempts**, driver logical-clock trace, and post-capture `ProgressExecutionTrace`, admission decisions, emissions, completion-receipt dispositions/results, transaction and driver-phase errors, gate transitions, and terminal outcomes **MUST** be item-for-item deterministic. The full admission/fence/terminal/settlement trace is observable logical execution; `DrainFenceTrace` alone is insufficient. The same raw attempts and logical-clock trace with a different `ProgressExecutionTrace` is a different logical execution, and M3 **MUST NOT** promise identical admission decisions, transaction/settlement grouping, receipts, emissions, or terminal outcomes for it.

An M3 snapshot **MUST NOT** be serialized, written to a manifest, persisted, treated as process-restart recovery, accepted across versions, or used to promise durable recovery. M3 **MUST NOT** define wall-clock deadline rebasing. Durable encoding, checkpoint coordination, manifests, restart, deadline rebasing, and exactly-once recovery are M5 responsibilities.

Documentation **MUST** describe the M3 contract as:

> snapshot-ready and deterministically replayable progress state

It **MUST NOT** call the feature durable recovery or crash recovery.

### D3 — Policy validation belongs to whole-job preflight

`compile_stream()` **MUST** remain a pure graph/operator compiler and **MUST NOT** require a binding catalog, connector instance, connector schema lookup, or connector open.

A crate-private whole-job preparation operation **MUST**:

1. compile the logical graph;
2. collect side-effect-free source descriptors;
3. validate every declared schema, source capability, and watermark policy;
4. normalize each binding's watermark mode;
5. produce an immutable prepared job and fingerprints.

Every preflight failure **MUST** occur before connector `open()`, runtime task spawn, timer registration, data pull, or other connector side effect. Generated watermark mode **MUST** fail preflight when the declared schema is dynamic or unknown, the event-time column is missing, or its Arrow type is unsupported.

M3 **MUST NOT** change the public v2 connector trait, public `open()` contract, or public runner surface to accomplish preflight.

### D4 — Multi-input progress and Idle semantics are complete

Each ingress **MUST** be in exactly one of:

```text
Active { watermark: Option<EventTime> }
Idle { watermark: Option<EventTime> }
Ended { final_watermark: Option<EventTime> }
```

`Active(None)` **MUST** participate in the active set and **MUST** prevent an aggregate minimum from being known. `Active(Some(w))` participates with `w`. `Idle` and `Ended` **MUST** be excluded from the active minimum.

The aggregator **MUST** emit a watermark only when the active minimum is known and strictly greater than its last emitted aggregate watermark. It **MUST NOT** regress or duplicate aggregate watermarks.

When the live-ingress set is non-empty and every live ingress is `Idle`, the aggregator **MUST** emit exactly one plain `Idle` per idle epoch. Repeated ingress `Idle` events **MUST NOT** create another epoch or another downstream `Idle`.

`Data` or a mode-legal connector watermark received for an idle ingress **MUST** reactivate the ingress before applying that event, and **MUST** clear the aggregate idle latch. A subsequent transition back to all-live-idle begins a new idle epoch and MAY emit one new downstream `Idle`.

When an ingress ends, it is permanently excluded. If ending it advances the aggregate minimum, the aggregator **MUST** emit `Watermark` before any terminal emission. When the final ingress ends, it **MUST** emit exactly one `EndOfInput`; it **MUST NOT** first emit `Idle`, synthesize `Watermark::MAX`, or emit another terminal item.

Any connector/source `Data`, `Watermark`, `Idle`, end, or other input/control received after that ingress reached `Ended` **MUST** be a protocol error and **MUST NOT** be silently suppressed. The only post-end arrival that MAY be ignored is a driver-internal timer callback already unregistered by end and proven stale by its driver-owned generation; ignoring it MUST produce no state, timer, or output change.

Processing one finite ready snapshot **MUST** be transactional. The driver **MUST** validate and evaluate selected items in `ReadyKey` order until the first deterministic error and **MUST** identify the transaction error by that first failing key. It MUST buffer all emissions, timer operations, cursor acknowledgements, and owned completion senders. On error, it MUST stop semantic evaluation, discard the complete scratch transaction, return no emission, mutate no timer or cursor, and settle every selected raw envelope exactly once with the same stable failure identifying the first error; unvisited suffix items are not semantically evaluated. Only an error-free snapshot MAY atomically commit scratch state/timer operations/cursor acknowledgements and then settle each selected raw envelope successfully. A receiver dropped before settlement only discards the send result and MUST NOT roll back or change the committed semantic outcome.

This first-key rule begins only after the frozen transaction has complete `ReadyKey` values. A failure before that boundary is a D9 `DriverPhaseError`, has no item key, and follows the same zero-partial-effect and complete fatal-settlement guarantees without inventing a first item.

### D5 — One deterministic progress driver owns ordering, clock, timers, and progress

One job-scoped, single-writer progress driver **MUST** own:

- all binding progress states;
- generated and accepted watermarks;
- the logical timer heap;
- multi-input aggregation;
- status and transient snapshot capture/restore;
- ready-set arbitration.

It is also the **only** owner and allocator of the semantic logical clock, finite-ready-snapshot capture, binding ordinal, binding-local sequence, global sequence, `ReadyKey`, timer deadline, timer generation, timer sequence, and idle epoch. Preparation supplies an immutable ordered binding table; the driver assigns each ordinal exactly once from that table before accepting work. A caller, connector, source adapter, or operator **MUST** submit only raw input/control and **MUST NOT** provide, override, or forge `logical_time`, binding ordinal, local/global sequence, `ReadyKey`, timer key/deadline/generation/sequence, or idle epoch.

All accepted input/control **MUST** execute at the driver's current logical instant. An input/control API **MUST NOT** accept a caller-provided `now`. The driver MAY read or advance a fake/paused clock through its own clock facility, but that facility MUST NOT let an adapter attach time to an individual raw item.

Timer cadence is frozen as follows:

- a generated-watermark binding's first deadline **MUST** equal the driver logical instant at which that binding enters running state plus `emit_interval`;
- after an expiry whose prior scheduled deadline is `D`, future cadence **MUST** remain phase-anchored to `D + n * emit_interval`, never to callback execution time;
- when the driver current logical instant has crossed multiple ticks, M3 **MUST** coalesce them into exactly one semantic watermark expiry, evaluate one candidate after same-instant input/control, and select the smallest checked `D + n * emit_interval` strictly greater than the current driver logical instant as the next deadline; it **MUST NOT** emit or evaluate once per skipped tick;
- an idle deadline **MUST** equal the driver logical instant of the last qualifying activity plus `idle_timeout`; qualifying activity is accepted `Data`, a mode-legal connector watermark, or the event that reactivates the binding; rearm **MUST** use that same rule and MUST NOT use callback/wall-clock time;
- only the driver MAY checked-increment generations/sequences and register, replace, or unregister a timer.

Every addition/multiplication used to derive a timer deadline **MUST** be checked. An unrepresentable deadline is a deterministic fatal progress error under the same atomicity rule as protocol errors.

M3 **MUST NOT** create a separate sleeper task per binding or use independently racing wall-clock timers to decide semantic order.

At the start of each drain, the driver **MUST** checked-allocate the next drain epoch and each inbox fence, freeze the complete ready set currently visible to it, assign driver-owned globally comparable `ReadyKey` values, and process that finite snapshot without admitting newly ready work. It MUST append the drain records to `ProgressExecutionTrace`, including every per-inbox upper fence, selected key range and membership, and due timer identity. Required empty drain epochs—an invoked epoch that observes a wake/fence/terminal-settlement coordinate but selects no semantic item—MUST be recorded as `Empty`; an external no-op poll that never enters driver drain arbitration is not an epoch. Items in that snapshot with the same logical instant **MUST** be ordered by:

```text
InputOrControl < WatermarkTimer < IdleTimer
then stable binding ordinal
then binding-local sequence
```

Newly readied work **MUST NOT** be inserted into the currently sorted finite snapshot; it is normally eligible at the next drain. Replay MUST reproduce the complete `ProgressExecutionTrace`, including the recorded drain epoch, inbox upper fences, selected item range/membership, due timer identities, required empty epochs, interleaved admission decisions, gate closes, and settlements exactly; adapters still MUST NOT provide semantic keys or fences. Timer generation **MUST** only reject stale driver-internal timer entries and **MUST NOT** affect semantic ordering.

Every counter that can affect ordering, liveness, receipt identity, fencing, admission/gate linearization, trace position, or snapshot equality—including trace-record ordinal, consumed-trace cursor, admission-attempt ordinal, gate generation, gate-close ordinal, receipt sequence, inbox sequence, fence sequence, drain epoch, local/global sequence, timer generation, timer sequence, binding-ordinal allocation, and idle epoch—**MUST** use checked increment. Exhaustion **MUST** be a deterministic fatal runtime/progress error using the existing public error container. It **MUST NOT** wrap, saturate, reuse a value, partially commit, emit, mutate timers, or acknowledge a cursor. Before a raw envelope becomes accepted, all admission-side allocations MUST be checked as one atomic operation; failure rejects that submission without accepting it or consuming any allocation. Exhaustion found after envelopes were accepted MUST close admission and deterministically failure-settle every accepted envelope exactly once before the driver exits.

Cancellation and terminal completion **MUST** unregister all progress timers and central wakes, close admission, and deterministically settle every accepted raw envelope before exit; they **MUST NOT** allow any subsequent progress emission. Tests **MUST** use a fake or paused logical clock and **MUST NOT** rely on real sleeps for ordering assertions.

### D6 — Capability negotiation is explicit; no silent merge

Preflight **MUST** apply this matrix:

| Descriptor capability | `SourceProvided` | `Generated` | `Disabled` |
|---|---|---|---|
| `NeverEmits` | reject | allow | allow |
| `EmitsNative` | allow | reject | reject |
| `RuntimeToggleable` | allow only with existing private hook enabling native emission | allow only with existing private hook disabling native emission | allow only with existing private hook disabling native emission |
| `Unknown` | reject | reject | reject |

`RuntimeToggleable` support **MUST** reuse an existing crate-private connector control hook. M3 **MUST NOT** add a public hook. If that hook does not exist for a connector, its capability **MUST NOT** be advertised as `RuntimeToggleable`.

The runtime **MUST NOT** silently merge connector-provided and generated watermarks. In `Generated` or `Disabled` mode, a connector watermark **MUST** be a protocol error. In `SourceProvided` mode, runtime generation **MUST NOT** run.

### D7 — All M3 surfaces are crate-private; public A6 remains post-M5

All new M3 policy, descriptor, normalized-mode, prepared-job, driver, aggregator, snapshot, status, and internal error surfaces **MUST** remain crate-private and **MUST NOT** be re-exported.

The public API baseline **MUST** remain unchanged. M3 **MUST NOT** expose source-driven runner A6, change Python bindings, or alter REST/OpenAPI schemas. Public A6 remains gated on completed M4 window/state semantics and M5 durable checkpoint/exactly-once semantics.

### D8 — Admission, receipts, finite fences, and terminal tails are one driver-owned protocol

Each accepted raw envelope **MUST** remain owned by the driver/inbox together with its one-shot completion sender until exactly one settlement attempt. A successful receipt MUST be sent only after the containing drain's live state, timers, frontier/cursor acknowledgement, and fence coordinate commit. A transaction failure MUST failure-settle every selected raw envelope with the stable first-key-ordered error while committing none of those semantic effects. Cancellation and fatal cleanup MUST failure-settle every still-accepted envelope. Every admission decision and settlement disposition MUST be recorded losslessly in `ProgressExecutionTrace`. A dropped receiver changes only whether the send is observed; it MUST NOT change progress state, cursor acknowledgement, trace disposition, or driver outcome.

Successful commit of an `End` MUST linearize closure of that binding's admission gate at a checked gate-close ordinal and checked next generation. In the same serialized commit operation, the driver MUST record the prior/new gate generation, close cut (`next_inbox_sequence` at linearization), and exact identities of every same-binding envelope accepted after the frozen upper fence but not selected; it MUST atomically take/mark and settle those envelopes exactly once with the stable post-End protocol error without data/progress emission or cursor acknowledgement. Future submissions MUST be rejected immediately at admission with that same stable category and their admission-attempt records MUST identify the closed generation/cause. This terminal-tail rejection is part of the successful End commit, is the exception to normal next-drain eligibility, and MUST NOT depend on another drain.

When the committed End makes all ingresses ended, the driver MUST NOT complete terminal exit until every accepted envelope has been settlement-attempted exactly once, every binding admission gate is closed, the logical timer heap and central wake are empty, and no detached progress work remains. `EndOfInput` ordering remains governed by D4: any end-induced watermark precedes it, and neither `Idle` nor a sentinel/MAX watermark is synthesized.

These repairs MUST NOT weaken D1–D7. In particular, they MUST preserve the four original conflict resolutions; the first-round driver-ownership, timer-cadence, exact-coordinate restore, post-End transaction blockers and checked-exhaustion minor repair; the prohibition on M3 late-row metrics or classification; and the M4 window, M5 durable-recovery, and post-M5 A6 boundaries.

### D9 — Full execution trace controls admission and settlement replay; error identity is phase-correct

`ProgressExecutionTrace` **MUST** be the only trace named as sufficient for deterministic receipt replay. In lossless linearization order it MUST record:

1. each submit attempt's trace position, checked admission-attempt ordinal, binding, observed gate generation/state, and exactly one decision: `Accepted` with receipt identity and inbox sequence, or `ImmediateRejected` with stable reason;
2. each drain freeze and outcome already required by D5, including selected keys/timers and required empty epochs;
3. each End, cancellation, or fatal terminal transition's cause, driver logical instant, checked gate-close ordinal, and for each affected binding its old/new generation, close state, close cut/next inbox sequence, and exact canonically ordered extracted-tail envelope identities;
4. every accepted envelope's exactly-one settlement disposition and stable outcome, linked to the owning drain or terminal transition and canonically ordered by `(binding_ordinal, inbox_sequence, receipt_identity)` within a settlement batch; and
5. every replay/allocator/freeze/trace failure identified as either a selected-item error or a `DriverPhaseError`.

Cross-receipt completion wake/send order is deliberately not an M3 observable guarantee and is not replayed; only the per-envelope disposition/result and owning coordinate are guaranteed. A dropped receiver is not a distinct disposition. The canonical order exists for equality and trace consumption, not as a promise about task wake order, so no settlement ordinal or snapshot counter is required.

Replay MUST first validate the complete replay plan—exact raw attempt identities/order, logical-clock records, trace structural integrity, one admission record per attempt, one settlement record per accepted envelope and none for an immediate rejection, terminal ownership, and a complete expected suffix—before runtime mutation. It MUST then consume each record in linearization order and validate the expected admission decision, fence, terminal close cut/tail membership, and settlement batch while holding the relevant driver/gate ownership and before the governed state mutation, emission, receipt send, or admission result becomes visible. A missing, extra, reordered, too-short, too-long, or mismatched record MUST produce a deterministic `DriverPhaseError`; effects from earlier fully validated and committed trace records remain committed, but the mismatched record and its epoch/transition commit none of their governed effects.

The first-`ReadyKey` rule applies only after every item in a frozen transaction has a complete key and selected-item validation/evaluation begins. Admission, drain/fence allocation, inbox freeze, key construction, replay-structure/record validation, gate-close planning, or other pre-key failure MUST use `DriverPhaseError { phase, stable_coordinate, reason }` with no `ReadyKey` field or sentinel. It MUST commit zero partial state/emission/timer/cursor effect for its phase and MUST failure-settle every already-accepted envelope during deterministic fatal cleanup. When a drain epoch exists, the trace records a distinct phase-failure outcome; if the epoch or trace-record ordinal itself could not be allocated, the error is recorded by the last exact coordinate outside an invented epoch/key, and replay beyond that terminal exhaustion is unsupported.

## 8. Functional requirements

### 8.1 Preparation and policy

- **FR-001** `compile_stream()` MUST compile a graph without source bindings or connector metadata.
- **FR-002** whole-job preparation MUST be side-effect-free and MUST complete before runtime creation.
- **FR-003** each source descriptor MUST expose a declared schema state (`Known` or `DynamicOrUnknown`) and a watermark capability without opening the source.
- **FR-004** preparation MUST resolve a stable binding identity and immutable binding-table order independent of runtime readiness; the driver MUST be the sole allocator/owner of the corresponding binding ordinals.
- **FR-005** preparation MUST validate policy durations: applicable timeouts and intervals MUST be positive, finite, and representable by the logical clock.
- **FR-006** `Generated` MUST validate a known event-time column and one of the supported Arrow timestamp units.
- **FR-007** policy/capability resolution MUST follow D6 and MUST produce exactly one normalized mode.
- **FR-008** prepared-job and normalized-binding fingerprints MUST cover all configuration that can change progress behavior or finite-fence grouping, including inbox capacity and fence-selection configuration; restore MUST require exact equality and MUST NOT accept a free replacement.
- **FR-009** any preflight error MUST contain a stable binding-qualified configuration path.
- **FR-010** no prepared job may contain a connector instance that was opened during preparation.

### 8.2 Generated and source-provided watermarks

- **FR-011** supported event-time input MUST include Arrow timestamps in seconds, milliseconds, microseconds, and nanoseconds.
- **FR-012** generated progress MUST compute the maximum representable non-null event time observed for the configured column.
- **FR-013** a null-only or empty batch MUST NOT advance maximum observed event time or watermark.
- **FR-014** generated watermark candidates MUST account for configured bounded out-of-orderness without unchecked overflow or underflow.
- **FR-015** a generated watermark MUST be emitted only on an eligible watermark-timer event and only if it strictly advances the last generated watermark.
- **FR-016** source-provided watermarks MUST be validated for monotonicity; regression MUST be a protocol error and equality MUST be suppressed.
- **FR-017** effective ingress watermark MUST never regress across data, timer, idle, reactivation, or end transitions.
- **FR-018** `Generated` and `Disabled` MUST reject connector watermarks; `SourceProvided` MUST NOT schedule generated-watermark timers.

### 8.3 Idle and multi-input progress

- **FR-019** applicable idle timeout MUST be re-armed from the driver logical instant of accepted `Data`, a mode-legal connector watermark, or reactivation; it MUST NOT use caller, callback, or wall-clock time.
- **FR-020** a stale idle or watermark timer generation MUST be ignored without emission or state regression.
- **FR-021** idle transition MUST retain the ingress's last effective watermark for status and snapshot purposes while excluding it from minimum computation.
- **FR-022** reactivation MUST occur before processing the reactivating event.
- **FR-023** an active ingress without a watermark MUST block an aggregate minimum even if every other active ingress has one.
- **FR-024** aggregate watermark MUST be the minimum of all active ingresses only when every active ingress has a known watermark.
- **FR-025** an all-idle live set MUST produce one `Idle` per epoch, and only one.
- **FR-026** an all-ended set MUST produce one `EndOfInput` and no synthetic idle or sentinel watermark.
- **FR-027** end-induced aggregate advancement MUST precede `EndOfInput`.
- **FR-028** any selected connector/source input/control after scratch or committed end MUST be the first-key-ordered stable protocol error when it is the first failing selected item and MUST cause its entire finite ready snapshot to commit no state, emission, timer operation, or upstream-cursor acknowledgement; every selected raw envelope MUST receive the deterministic drain failure. Only a driver-internal generation-proven stale timer MAY be ignored without output. Already-accepted post-fence terminal tails follow FR-062 rather than waiting for another drain.

### 8.4 Driver, snapshot, and replay

- **FR-029** all semantic progress/order state and allocation MUST have exactly one writer: the job-scoped progress driver.
- **FR-030** each driver drain MUST checked-allocate a drain epoch and per-inbox fences, freeze the currently visible finite ready set, assign its `ReadyKey` values, use the D5 total order, and record the exact finite grouping as drain records within `ProgressExecutionTrace`; an adapter may submit only unkeyed raw input/control and may not supply fences or trace records.
- **FR-031** timer generation MUST be driver-allocated and validation-only; driver-owned ordinal and local sequence MUST provide stable tie-breaking.
- **FR-032** snapshot capture MUST be rejected outside a quiescent event boundary.
- **FR-033** `StreamProgressSnapshot` MUST contain the prepared-job fingerprint and the complete captured logical coordinate defined by D2, including completed `ProgressExecutionTrace` prefix/consumed position; next trace-record, admission-attempt, gate-close, drain, receipt, inbox, fence, local, and global coordinates; exact gate generations/states/close cuts; and last completed inbox fences. An undefined “compatible position” MUST NOT be used.
- **FR-034** each binding snapshot MUST contain binding identity and driver-assigned ordinal, normalized-config fingerprint, exact upstream delivery/replay cursor, exact control frontier, `Active`/`Idle`/`Ended` state, admission-gate state/generation/close cause/close ordinal/next-inbox close cut, last accepted source watermark, generated maximum observed event time, last generated watermark, effective ingress watermark, watermark timer deadline/generation/sequence, idle timer deadline/generation/sequence, next binding-local sequence, next inbox sequence, next fence sequence, and exact last completed inbox fence.
- **FR-035** aggregate/driver snapshot MUST contain every ingress progress entry, last emitted aggregate watermark, idle latch state, idle epoch, terminal state, next global/receipt/drain/trace-record/admission-attempt/gate-close coordinates, consumed execution-trace position, and the lossless completed `ProgressExecutionTrace` prefix; restore-capable capture MUST prove there is no accepted-but-unsettled envelope.
- **FR-036** restore MUST validate exact equality of prepared-job fingerprint, binding identities/ordinals/order, config and fence-grouping fingerprints, state cardinality, driver logical instant, completed execution-trace prefix/consumed position, gate generations/states/close cuts, last fences, every upstream delivery/replay cursor and control frontier, every next allocation coordinate, and timer coordinates before any state change, admission decision, receipt settlement, or runtime side effect.
- **FR-037** accepted restore MUST reproduce pending timer state and sequence positions exactly.
- **FR-038** identical post-restore ordered raw input/control attempts, logical-clock trace, and full `ProgressExecutionTrace` MUST produce item-for-item identical admission decisions, outputs, completion-receipt dispositions/results, transaction/driver-phase errors, gate transitions, and terminal outcomes. `DrainFenceTrace` alone is insufficient; a different full execution trace is explicitly a different logical execution even if raw attempts and clock match.
- **FR-039** M3 snapshot values MUST remain in memory and MUST NOT implement a stable serialized format.
- **FR-040** cancel, fatal cleanup, and terminal end MUST record their exact cause, logical instant, gate transitions/close cuts, extracted accepted identities, and settlement dispositions in `ProgressExecutionTrace`; they MUST close admission, empty the logical timer heap and central wake, exactly-once settle every accepted envelope, and prevent detached work and post-terminal emissions.

### 8.5 Data-path neutrality and scope control

- **FR-041** M3 MUST forward all accepted data without watermark-based late classification or mutation.
- **FR-042** M3 metrics, status, and errors MUST NOT expose late-row observations or drops.
- **FR-043** M3 MUST have no implementation dependency on `operator/window.rs` or any M4-only window executor.
- **FR-044** no public Rust, Python, REST, or OpenAPI surface may change.

### 8.6 Exact ownership, cadence, transactionality, and exhaustion

- **FR-045** caller/adapter submission MUST be limited to raw input/control and MUST reject caller-supplied logical instant, `ReadyKey`, ordinal, sequence, or timer coordinates.
- **FR-046** all accepted input/control MUST execute at the driver current logical instant; no input/control method may accept `now`.
- **FR-047** initial generated-watermark deadline MUST equal binding-running logical instant plus `emit_interval`, using checked arithmetic.
- **FR-048** generated-watermark cadence MUST remain anchored to its prior scheduled deadline; multiple missed ticks MUST coalesce into one semantic expiry and the next phase-aligned deadline strictly after current driver time.
- **FR-049** idle deadline MUST equal the driver logical instant of the last qualifying activity plus `idle_timeout`, using checked arithmetic.
- **FR-050** only the driver may allocate/increment timer generation/sequence or register, replace, and unregister timer entries.
- **FR-051** after all selected items have complete `ReadyKey` values, each finite ready snapshot MUST be evaluated against scratch state in key order only until the first deterministic selected-item error, with emissions, timer operations, cursor acknowledgements, and owned completion senders buffered; it MAY commit atomically only if every selected item succeeds.
- **FR-052** a selected-item failing finite ready snapshot MUST return exactly the first key-ordered deterministic error, MUST commit zero state, emissions, timer operations, or delivery-cursor acknowledgements, and MUST failure-settle every selected raw envelope exactly once with that error even when its item lies in the unevaluated suffix. Pre-key failures follow FR-075 and MUST NOT fabricate a `ReadyKey`.
- **FR-053** restore-capable capture MUST reject a binding that cannot pause, report an exact stable replay cursor/control frontier, and later reposition to them.
- **FR-054** restore MUST pause and exactly position all upstreams before validation/commit and MUST reject any missing or unequal coordinate before state/timer mutation.
- **FR-055** trace-record, consumed-trace-cursor, admission-attempt, gate-generation, gate-close, receipt, inbox, fence, drain-epoch, local/global, timer generation/sequence, binding-ordinal, idle-epoch, and any equivalent ordering/liveness/settlement counter MUST use checked increment.
- **FR-056** counter exhaustion MUST be a deterministic fatal progress error through the existing public error container and MUST obey the zero-state/output/timer/cursor-commit rule; wrap, saturation, value reuse, and a stranded accepted envelope are forbidden. Admission-side allocation failure MUST accept nothing; exhaustion after acceptance MUST failure-settle every accepted envelope before exit.
- **FR-057** the driver MUST record a lossless `ProgressExecutionTrace`. Its drain projection MUST retain every drain epoch, driver logical instant, inbox fence identity/upper sequence/`Empty`, ordered selected identity/key range/`Empty`, due timer identity, required empty epoch, and selected-item or driver-phase outcome; the full trace MUST also retain FR-068–FR-072 lifecycle records.
- **FR-058** deterministic replay and recorded stress reproduction MUST consume the exact ordered raw input/control attempts, logical-clock trace, and full `ProgressExecutionTrace`; natural arrival timing or output backpressure MAY differ only when the supplied full trace remains identical, and a different trace carries no promise of identical admission, transaction, or settlement grouping.
- **FR-059** admission MUST checked-allocate and record a stable attempt ordinal before gate observation, then atomically decide and record exactly one of: accepted with receipt/inbox identity and sender transfer/enqueue, or immediate rejection with observed gate generation/state and stable reason. Every accepted envelope MUST receive exactly one settlement attempt on success, transaction failure, terminal-tail rejection, cancellation, or fatal cleanup.
- **FR-060** a success receipt MUST be attempted only after live state, timers, cursor/frontier, fence coordinates, and matching trace records commit; a selected transaction failure receipt MUST identify the first key-ordered error and MUST NOT acknowledge the delivery cursor; each disposition/outcome and owning coordinate MUST be recorded before its receipt send.
- **FR-061** dropping a receipt receiver MUST affect only observation of the send result; it MUST NOT roll back commit, retry settlement, alter the replay trace, or leave driver-owned work pending.
- **FR-062** successful End commit MUST atomically close that binding's admission gate and record its cause/logical instant/close ordinal, old/new generation, next-inbox close cut, and exact extracted same-binding envelopes accepted after the frozen fence but not selected; those terminal-tail envelopes MUST receive exactly one recorded stable post-End failure without semantic emission or cursor acknowledgement and MUST NOT require a later drain.
- **FR-063** every future submission to a binding whose End committed MUST be rejected immediately with the stable post-End admission/protocol error, MUST NOT allocate receipt/inbox identity or enqueue an accepted envelope, and MUST record its attempt ordinal plus observed closed gate generation/cause and rejection.
- **FR-064** all-ended terminal exit MUST occur only after the full terminal transition and all settlement dispositions match/commit in the execution trace, all accepted envelopes have been settlement-attempted, all admission gates are closed, timers and central wake are cleared, and detached progress work is absent.
- **FR-065** cancellation MUST atomically record cause/logical instant and per-binding gate generation/close ordinal/close cut/extracted identities, close every admission gate, take every queued/unselected accepted envelope, record and settle each once with the stable cancellation error, clear timers/wake, and then exit.
- **FR-066** the identity of a multi-error **selected-item** transaction MUST always be the first failing item in the D5 total order; suffix validation/evaluation hooks MUST NOT run after that error, although all selected raw completion senders MUST still be failure-settled. It MUST NOT be applied to a pre-key driver-phase failure.
- **FR-067** the replay/receipt/fence repairs MUST preserve D1–D7, the four original conflict resolutions, the first-round B-01–B-04 and N-01 repairs, M3's no-late-metric rule, and the M4/M5/post-M5 A6 scope firewall.
- **FR-068** `ProgressExecutionTrace` MUST contain exactly one admission record for every raw submit attempt, losslessly identifying trace position, attempt ordinal, binding, observed gate generation/state, and accepted receipt/inbox identity or immediate stable rejection.
- **FR-069** every End, cancellation, and fatal terminal transition MUST have one lossless record containing cause, driver logical instant, gate-close ordinal, and per-binding old/new generation, close state, next-inbox close cut, and exact canonically ordered extracted-tail identities/ranges.
- **FR-070** every accepted envelope MUST have exactly one settlement record linked by accepted identity to its owning drain/terminal transition with one stable disposition: commit success, selected transaction error, post-End tail rejection, cancellation, or fatal cleanup. Immediate rejections MUST have no settlement record.
- **FR-071** cross-receipt send/wakeup order MUST NOT be an M3 semantic guarantee. Settlement records MUST use canonical envelope-identity order for equality/replay; no settlement ordinal/counter is required, and receiver drop MUST NOT change the recorded disposition.
- **FR-072** replay MUST prevalidate the complete raw-attempt/clock/execution-trace plan, then consume and validate each admission, drain, terminal, and settlement record before exposing its governed admission result, state/timer/cursor commit, emission, or receipt. Missing, extra, reordered, too-short, too-long, or mismatched records MUST fail deterministically.
- **FR-073** the captured coordinate MUST round-trip the completed execution-trace prefix, consumed trace position, next trace-record/admission-attempt/gate-close coordinates, and every gate's generation/state/close cause/close ordinal/next-inbox close cut in addition to all prior coordinates.
- **FR-074** trace-record ordinal, consumed-trace cursor, admission-attempt ordinal, gate generation, and gate-close ordinal MUST use checked increment. Exhaustion before acceptance/commit MUST produce no partial governed effect; exhaustion after earlier acceptances MUST deterministically fatal-settle them without wrap or stranded sender.
- **FR-075** admission, drain/fence allocation, inbox freeze, key construction, trace/replay validation, gate-close planning, or other failure before complete selected keys MUST be a `DriverPhaseError` with stable phase/coordinate/reason and no `ReadyKey`; it MUST commit no partial governed semantic effect and MUST trigger complete fatal settlement/cleanup.
- **FR-076** a replay plan MUST prove before execution that every raw attempt has exactly one admission record, every accepted identity has exactly one settlement and every rejection none, every settlement owner exists, all terminal cuts/tails are self-consistent, and the expected trace suffix is complete.
- **FR-077** while holding admission/driver ownership, replay MUST wait for every tail/cleanup identity recorded as accepted, reject any unexpected accepted identity, and validate the exact gate transition/close cut and settlement batch before the corresponding live commit or receipt/rejection becomes visible.
- **FR-078** fatal cleanup MUST be recorded with the selected-item error or `DriverPhaseError` cause, logical instant, all gate transitions/cuts, exact selected/unselected accepted membership, and fatal settlement outcomes; cancellation MUST use the equivalent explicit lifecycle record even when no drain epoch exists.
- **FR-079** these full-trace and phase-error refinements MUST preserve every prior D1–D8 decision and AC-01–AC-75 without weakening M2 safety, M3 no-late behavior, or M4/M5/post-M5 A6 boundaries.

## 9. Required state and transition semantics

### 9.1 Binding progress

The normalized binding state MUST retain enough information to satisfy FR-034. Its externally relevant transitions are:

| Current | Event | Next | Required effect |
|---|---|---|---|
| `Active` | accepted `Data` | `Active` | at driver current instant, update generated maximum if enabled; set idle deadline to current instant + timeout |
| `Active` | legal connector WM | `Active` | at driver current instant, validate/update source/effective WM; set idle deadline to current instant + timeout |
| `Active` | eligible WM timer | `Active` | emit only a strictly advancing generated WM |
| `Active` | current idle timer | `Idle` | retain effective WM; participate in idle-epoch calculation |
| `Idle` | accepted `Data` | `Active` | at driver current instant, reactivate and clear aggregate idle latch before data processing; set new idle deadline |
| `Idle` | legal connector WM | `Active` | at driver current instant, reactivate before watermark validation/aggregation; set new idle deadline |
| `Active` or `Idle` | end | `Ended` | in scratch, cancel binding timers and preserve final effective WM; on successful commit, linearize admission-gate close and terminal-tail extraction/settlement |
| `Ended` | any selected connector/source input/control | error | if first in total order, become the stable first error; fail the complete finite snapshot atomically and failure-settle every selected raw envelope, with no state, emission, timer, or cursor-ack commit |

A driver-internal callback whose generation proves it is stale produces no state transition or output even if its binding has ended. This is the only permitted post-end suppression. An illegal connector watermark produces a protocol error, not reactivation. After a successful End commit, a new submit is rejected synchronously at the closed admission gate; an envelope already accepted after the committed End's frozen fence is atomically removed from next-drain eligibility and receives the same stable post-End failure exactly once as part of that commit. Each admission decision, gate generation/close cut, extracted identity, and settlement disposition MUST appear at its exact linearization position in `ProgressExecutionTrace`.

### 9.2 Aggregate idle epoch

The aggregate idle latch is set only after downstream `Idle` has been emitted for the current all-live-idle interval. It is cleared when:

- an idle ingress reactivates;
- a membership change leaves at least one live ingress active; or
- the live set becomes empty at terminal end.

Repeated `Idle` events while the latch is set do not change the epoch. A new epoch begins only after the latch was cleared by reactivation/membership transition and the job later becomes all-live-idle again. Epoch numbering MUST be monotonic, driver-owned, checked on increment, and included in the transient snapshot. Exhaustion fails the finite snapshot atomically.

### 9.3 Finite ready-set arbitration

At the start of a drain at driver-owned logical instant `t`, the driver checked-allocates one drain epoch and one fence record for every binding inbox, then freezes all raw input/control at or below each recorded upper inbox sequence plus all due internal timers. The driver, and only the driver, allocates each binding ordinal/local sequence and constructs each `ReadyKey`. Raw submitters cannot influence those fields or fence membership. The driver sorts each item using:

```text
(t, class_rank, binding_ordinal, local_sequence)

class_rank:
  InputOrControl  = 0
  WatermarkTimer  = 1
  IdleTimer       = 2
```

Work made ready above a frozen upper fence waits for the next drain, including work at the same logical instant, except that a successful End commit atomically extracts and rejects its same-binding terminal tail under D8/FR-062. The whole sorted snapshot is evaluated on scratch state before any commit. This rule prevents an unbounded input producer from mutating the already-selected ordering and preserves snapshot-level error atomicity.

The driver MUST append the drain-fence and outcome records to `ProgressExecutionTrace` before exposing the governed epoch effects. They MUST losslessly identify the epoch, logical instant, each inbox's fence sequence and inclusive upper inbox sequence or `Empty`, ordered selected raw identities/cursors and timer identities, inclusive first/last `ReadyKey` or `Empty`, and exactly one outcome: committed, selected-item failure with first error key, or driver-phase failure with no key. A driver drain invocation that selects nothing but is needed to observe a scheduled wake, frozen empty fence, or terminal-settlement coordinate is a required empty epoch and MUST be recorded. A poll that does not enter semantic drain arbitration is outside the trace.

Replay MUST drive each epoch from the full execution trace, not re-discover fence or admission/settlement membership from thread timing, queue occupancy, gate races, or output-sink delay. The same raw attempts and clock trace with a different execution trace are intentionally distinguishable. Inbox capacity and any fence-selection limit or policy are semantic configuration covered by the exact fingerprint. Output backpressure is not replayed as wall-clock timing; any semantic effect it had is represented by the resulting trace records.

### 9.4 Logical clock and timer cadence

Adapters submit raw work without `now`; every item frozen by a drain executes at that drain's driver current logical instant. The first generated-watermark deadline is:

```text
running_instant + emit_interval
```

For an expired generated-watermark timer with scheduled deadline `D` and driver current instant `t >= D`, the driver evaluates exactly one semantic expiry after all same-instant `InputOrControl`. It then computes, using checked arithmetic, the unique smallest phase-aligned deadline:

```text
D + n * emit_interval > t, for the smallest integer n >= 1
```

Thus missed ticks are coalesced into one expiry and do not cause burst emissions. The phase remains anchored to `D`, not callback execution time. An idle deadline is always `qualifying_activity_instant + idle_timeout`; it is not periodic and is replaced on each qualifying activity. Only the driver may create, rearm, replace, or unregister either timer.

### 9.5 Transactional finite-snapshot processing

For each frozen finite snapshot, the driver MUST:

1. validate the expected drain-fence record before extracting or mutating inbox state; a pre-key mismatch or allocation/freeze/key-construction error follows §9.8;
2. clone or otherwise stage the relevant progress/timer/counter state as scratch state and retain ownership of every selected raw envelope's one-shot completion sender;
3. after every selected item has a key, validate and evaluate it against scratch state until the first error in `ReadyKey` order, and run no semantic validation/evaluation hook for the suffix after that error;
4. buffer emissions, timer operations, delivery-cursor acknowledgements, terminal admission operations, and receipt dispositions without making them visible;
5. if a selected item fails, discard all scratch semantic changes, validate expected replay records or atomically append the selected-failure outcome and complete selected settlement batch, failure-settle every selected raw envelope exactly once with the stable first-key-ordered error, and return that one error; no selected cursor is acknowledged;
6. otherwise, before live commit and while holding each affected admission gate, compute the exact End transition—old/new generation, gate-close ordinal, close cut, and extracted terminal-tail identities—and validate expected replay records or stage new transition and selected-success/tail-failure settlement records;
7. atomically commit the staged state, timer operations, cursor/frontier acknowledgements, completed fence coordinates, End gate closures/terminal-tail extraction, staged trace records, and committed drain outcome;
8. only after that commit, settle selected envelopes with commit success, settle extracted terminal tails with the stable post-End failure, and return the complete buffered emission batch; send/wakeup order between distinct receipts is unspecified;
9. if the commit made all ingresses ended, clear the central wake and prove the trace suffix for this transition is consumed, all accepted envelopes settled, every gate closed, and no queued/detached progress work before terminal exit.

No item prefix is independently committed. Consequently, a valid item ordered before a post-end input in the same finite snapshot does not leak an emission before the snapshot fails. Every selected raw envelope receives the same deterministic failure even when its item lies after the first error and was not evaluated. Failure to send because a receipt receiver was dropped counts as the envelope's one settlement attempt and has no semantic effect.

Submission and End commit MUST serialize on the binding admission gate. Admission checked-allocates an attempt ordinal, observes the gate generation/state, computes the accepted receipt/inbox identity or immediate stable rejection, and validates/appends that admission record before enqueue or return becomes visible. For acceptance it checks receipt/inbox successors without consuming either on failure, transfers sender ownership, enqueues, and only then reports acceptance. End commit validates/appends its transition record, atomically changes the gate generation/state to closed at the recorded close cut, and takes every exact recorded same-binding envelope above the frozen fence. Therefore an attempt is exactly one of accepted work or an immediate rejection, and an accepted envelope is exactly one of selected work, an extracted terminal tail, cancellation/fatal cleanup work, or future selected work; it cannot fall between outcomes.

### 9.6 Checked counters

Every allocation/increment of trace-record ordinal, consumed-trace cursor, admission-attempt ordinal, gate generation, gate-close ordinal, receipt sequence, inbox sequence, fence sequence, drain epoch, binding ordinal, local/global sequence, timer generation, timer sequence, idle epoch, or equivalent ordering/liveness/settlement counter MUST be checked before governed semantic mutation or work acceptance. Admission checks all required accepted-envelope values atomically before enqueue and consumes none if any next value is unrepresentable. A drain-side exhaustion closes admission during recorded fatal cleanup and failure-settles all accepted selected and unselected envelopes before exit. Wraparound, saturation, reuse, best-effort continuation, partial output, cursor acknowledgement, stranded completion senders, and fabricated item keys are forbidden.

### 9.7 Full execution-trace state machine

The normative trace is logically equivalent to this ordered vocabulary; concrete type spelling is non-normative:

```rust
enum ProgressTraceRecord {
    Admission(AdmissionTraceRecord),
    DrainFence(DrainFenceRecord),
    DrainOutcome(DrainOutcomeRecord),
    TerminalTransition(TerminalTransitionRecord),
    SettlementBatch(SettlementBatchRecord),
    DriverPhaseFailure(DriverPhaseFailureRecord),
}
```

Every record has a checked trace-record ordinal. An accepted-envelope identity MUST include binding identity/ordinal, admission-attempt ordinal, receipt identity, inbox sequence, and replay cursor/control identity sufficient for exact equality. An admission record MUST contain the observed gate generation/state and accepted or immediate-rejected decision. A terminal record MUST contain cause (`EndCommit`, `Cancellation`, or `Fatal`), logical instant, owning drain when one exists, and for every affected gate the checked close ordinal, old/new generation, closed state/cause, `next_inbox_sequence` close cut, and exact extracted envelope identities/range. A settlement batch MUST link each accepted identity to its owner and one disposition/outcome. Canonical batch order is `(binding_ordinal, inbox_sequence, receipt_identity)` and has no cross-receipt wake-order meaning.

Live recording and replay use the same state machine. Record mode appends each record at its linearization point. Replay mode receives a complete prevalidated expected trace and consumes the next exact record before the governed effect. End replay holds the gate and waits until every recorded accepted tail identity is present at/below the recorded close cut, rejects an unexpected identity or cut, validates the terminal/settlement records, and only then commits gate closure. Cancellation and fatal cleanup do the same across gates in binding-ordinal order even when no drain exists. An admission attempt after a recorded close consumes an immediate-rejection record; it MUST NOT be converted into an accepted tail, and a recorded accepted tail MUST NOT be rediscovered as an immediate rejection.

Before execution, replay MUST cross-check the complete raw-attempt list and trace: admission records are one-for-one and ordered; accepted identities are unique and each owns exactly one settlement; rejected attempts own none; selected, tail, cancel, and fatal membership is disjoint and complete; every settlement owner/terminal cut exists; trace ordinals and gate generations/closes are contiguous; and the expected suffix ends exactly at the declared replay completion. A too-short/too-long trace, missing/extra settlement, reordered admission, cut mismatch, extra actual tail, or unconsumed expected suffix is invalid before replay begins when structurally detectable and otherwise before the mismatched record's governed effect. Earlier fully matched epochs are not rolled back.

### 9.8 Failure identity domains

Failures are disjoint:

```text
SelectedItemError { first_error_key, stable_reason }
DriverPhaseError { phase, stable_coordinate, stable_reason }
```

`SelectedItemError` is legal only after a frozen set has complete keys and evaluation has begun; it chooses the first failing `ReadyKey`, stops suffix hooks, and uses §9.5 atomic rollback. `DriverPhaseError` covers admission/trace ordinal allocation, gate observation or close planning, drain-epoch/fence allocation, inbox freeze, local/timer key allocation, replay-plan/record/suffix mismatch, or any other failure without a complete item key. Its coordinate MUST identify the exact phase plus the available attempt/gate/fence/counter/trace position. It MUST NOT contain a sentinel or borrowed `ReadyKey`.

Before returning `DriverPhaseError`, the driver MUST expose none of the mismatched phase's state, timer, cursor, emission, admission, or receipt effect. It MUST validate/record fatal gate transitions and settlement membership when trace capacity permits, close admission, and exactly-once fatal-settle every accepted envelope. If trace-record or drain-epoch exhaustion prevents allocation of that record/epoch, the last exact counter coordinate identifies the terminal error outside an invented trace position or epoch; earlier trace remains valid, but deterministic replay beyond that terminal exhaustion is unsupported.

## 10. Preparation and capability contract

The descriptor read used in preflight MUST be side-effect-free. It MAY read immutable configuration already available in memory, but MUST NOT open sockets/files, consume messages, spawn tasks, register timers, or call connector `open()`.

Preparation MUST reject:

- missing or duplicate binding identity;
- duplicate binding identity or an immutable binding-table order that cannot be assigned unique driver-owned ordinals;
- dynamic/unknown schema with `Generated`;
- missing, ambiguous, or unsupported generated event-time column;
- non-positive or unrepresentable emit interval/idle timeout/out-of-orderness;
- every forbidden policy/capability pair in D6;
- `RuntimeToggleable` without an existing private toggle hook;
- a policy that would allow connector and runtime watermark emission simultaneously.

There is no fallback from a rejected policy to another mode. The error MUST require the caller to correct configuration or descriptor capability.

## 11. Snapshot contract

The minimum complete snapshot is logically equivalent to:

```rust
pub(crate) struct StreamProgressSnapshot {
    prepared_job_fingerprint: ConfigFingerprint,
    coordinate: CapturedLogicalCoordinate,
    bindings: Vec<BindingProgressSnapshot>,
    aggregate: MultiInputProgressSnapshot,
}

pub(crate) struct CapturedLogicalCoordinate {
    driver_logical_instant: LogicalInstant,
    binding_frontiers: Vec<BindingReplayFrontier>,
    next_global_sequence: u64,
    next_receipt_sequence: u64,
    next_drain_epoch: u64,
    next_trace_record_ordinal: u64,
    next_admission_attempt_ordinal: u64,
    next_gate_close_ordinal: u64,
    completed_progress_execution_trace: ProgressExecutionTrace,
    consumed_trace_position: TracePosition,
    unsettled_accepted_envelopes: Zero,
}

pub(crate) struct BindingReplayFrontier {
    binding_identity: BindingIdentity,
    driver_assigned_ordinal: BindingOrdinal,
    upstream_delivery_replay_cursor: ReplayCursor,
    control_frontier: ControlFrontier,
    next_local_sequence: u64,
    next_inbox_sequence: u64,
    next_fence_sequence: u64,
    last_completed_fence: Option<InboxFenceCoordinate>,
    admission_gate: AdmissionGateState,
    admission_gate_generation: u64,
    admission_gate_close: Option<AdmissionGateCloseCoordinate>,
}
```

The type spelling is non-normative; the information content is normative. The captured coordinate MUST be complete and compared field by field; there is no “compatible position” predicate. `ProgressExecutionTrace` MUST be a lossless ordered prefix, not merely a drain projection or digest, and `consumed_trace_position` MUST identify its exact next replay record. Each gate close coordinate MUST retain close cause, checked close ordinal, next generation, and `next_inbox_sequence` cut. Each last-completed fence MUST retain its fence sequence, inclusive upper inbox sequence or `Empty`, drain epoch, and outcome coordinate. In addition to FR-034/FR-035, implementations MUST distinguish absent timers from scheduled timers and MUST retain their exact logical deadline, generation, and sequence.

Snapshot capture MUST NOT race a driver state transition and MUST occur only at the §4 quiescent boundary. A restore-capable capture MUST reject if any accepted envelope or owned completion sender remains unsettled; the coordinate MUST therefore prove zero such envelopes. Before restore validation, all upstreams MUST be paused and MUST prove exact positioning at the stored delivery/replay cursor and control frontier. Exact position, fingerprints including fence/queue configuration, ordered binding table, driver instant, full execution-trace prefix/consumed position, next trace/admission/gate-close coordinates, last-completed fences, admission gate state/generation/close cut, every other next sequence coordinate, and timer coordinates MUST all match before any state or timer is installed. Restore MUST be all-or-nothing: validation failure MUST leave no state change, opened/resumed connector, task, registered timer, accepted envelope/admission result, cursor acknowledgement, receipt settlement, or emission.

After restore, replay MUST begin at `consumed_trace_position`, `next_admission_attempt_ordinal`, `next_gate_close_ordinal`, `next_drain_epoch`, and every binding's exact gate generation/state/close cut and `next_inbox_sequence`/`next_fence_sequence`; it MUST allocate `next_trace_record_ordinal`, `next_receipt_sequence`, and every other counter exactly. It MUST reproduce the supplied post-capture `ProgressExecutionTrace` entry by entry. A mismatch—including missing/extra/reordered admission or settlement; too-short/too-long/unconsumed suffix; gate generation/close cut/tail inequality; missing/extra required empty epoch; different fence/selected membership/due timer; or wrong terminal cause—MUST produce `DriverPhaseError` before the mismatched record's governed effect. The same raw attempts and clock trace under another execution trace is outside the identical-replay promise and MUST be reported as trace inequality rather than implementation nondeterminism.

If any source cannot provide and later seek to a stable replay cursor/control frontier, creation of this restore-capable snapshot MUST fail deterministically and restore MUST be reported unsupported for the job. The runtime MUST NOT approximate a coordinate using row counts, wall time, or “latest” offsets.

Snapshot data is in-memory only. It MUST NOT be serialized, persisted, passed to another process, accepted across versions, or used after process restart. It is an internal testability and future-checkpoint seam. M5 MAY replace its concrete Rust representation while preserving a migration/versioning contract defined in M5.

## 12. Errors

M3 MUST reuse the existing public error family. It MUST NOT add a public error variant solely for this work. Internal preparation and protocol errors MUST preserve a stable machine/test-visible path and an actionable reason.

Canonical path forms:

```text
sources.<binding>.watermark_policy.event_time_column
sources.<binding>.watermark_policy.emit_interval
sources.<binding>.watermark_policy.idle_timeout
sources.<binding>.capabilities.native_watermarks
runtime.nodes.<node>.ingress.<ingress>.progress
runtime.progress.snapshot.prepared_job_fingerprint
runtime.progress.snapshot.coordinate.driver_logical_instant
runtime.progress.snapshot.coordinate.next_drain_epoch
runtime.progress.snapshot.coordinate.next_receipt_sequence
runtime.progress.snapshot.coordinate.next_trace_record_ordinal
runtime.progress.snapshot.coordinate.next_admission_attempt_ordinal
runtime.progress.snapshot.coordinate.next_gate_close_ordinal
runtime.progress.snapshot.coordinate.progress_execution_trace
runtime.progress.snapshot.coordinate.consumed_trace_position
runtime.progress.snapshot.coordinate.bindings.<binding>.upstream_delivery_replay_cursor
runtime.progress.snapshot.coordinate.bindings.<binding>.control_frontier
runtime.progress.snapshot.coordinate.bindings.<binding>.next_inbox_sequence
runtime.progress.snapshot.coordinate.bindings.<binding>.next_fence_sequence
runtime.progress.snapshot.coordinate.bindings.<binding>.last_completed_fence
runtime.progress.snapshot.coordinate.bindings.<binding>.admission_gate_generation
runtime.progress.snapshot.coordinate.bindings.<binding>.admission_gate_close_cut
runtime.progress.snapshot.bindings.<binding>.normalized_config_fingerprint
runtime.progress.admission.<binding>.post_end
runtime.progress.drains.<epoch>.first_error
runtime.progress.driver_phase.<phase>.<coordinate>
runtime.progress.counters.<counter>
```

Required error categories include:

- invalid policy value;
- missing/unsupported/unknown event-time schema;
- policy/capability conflict;
- connector watermark illegal for normalized mode;
- source watermark regression;
- event after ingress end;
- immediate submission after committed End and terminal-tail rejection after a successful End fence;
- execution-trace replay mismatch, including missing/extra/reordered admission, terminal, settlement, or empty-drain record; too-short/too-long/unconsumed suffix; gate generation/close-cut/tail inequality; or fence/selected/timer inequality;
- snapshot captured at non-quiescent boundary;
- snapshot capture with an accepted-but-unsettled envelope/receipt;
- prepared-job, binding, config/fence grouping, exact cursor/frontier, completed execution trace/consumed position, last fence, gate generation/state/close cut, next allocation, timer-coordinate, or driver-logical-instant mismatch on restore;
- restore requested for a source without exact pause/replay positioning;
- timestamp conversion or candidate arithmetic overflow;
- timer-deadline arithmetic overflow;
- trace-record/consumed-trace/admission-attempt/gate-generation/gate-close/receipt/inbox/fence/drain-epoch/local/global sequence, timer generation/sequence, binding ordinal, idle epoch, or equivalent counter exhaustion.

Errors MUST NOT include secrets or arbitrary connector payload data. A preflight or restore validation error MUST occur before runtime side effects. Only a failure during selected-item evaluation after complete key assignment has a first failing `ReadyKey`; every selected raw receipt MUST carry that stable selected-item identity even if its item was not evaluated. A pre-key admission/freeze/allocation/trace/gate-planning failure MUST instead use `DriverPhaseError` with phase and exact available coordinate, no `ReadyKey`, and §9.8 zero-governed-effect/fatal-settlement semantics. Terminal-tail and future-submit failures MUST share the stable post-End category while remaining distinguishable by settlement/admission context. These errors MUST reuse the existing public error container; M3 MUST NOT add a public error variant.

## 13. Observability

Crate-private status MAY expose, per binding:

- stable binding identity and ordinal;
- normalized mode name;
- `Active`/`Idle`/`Ended`;
- last accepted source watermark;
- generated maximum observed event time;
- last generated/effective watermark;
- pending logical timer deadlines;
- admission gate generation/state and close cut;
- stale-timer suppression count;
- watermark advancement/suppression count.

Aggregate status MAY expose last aggregate watermark, live/active/idle/ended ingress counts, idle epoch/latch, current/next drain epoch, next admission/gate-close/trace coordinates, consumed execution-trace position, last completed fence coordinates, unsettled receipt count, timer-heap size, and terminal state. These fields are observational copies of driver-owned semantic state; they MUST NOT allocate or alter a sequence.

Metrics MUST have bounded cardinality and MUST NOT use raw user data, event-time values, or unbounded binding strings as labels. No status, metric, log, or error may classify/count/tag rows as late in M3. Status and metrics are observational and MUST NOT participate in semantic decisions.

## 14. Acceptance criteria and named-test evidence

Each criterion is mandatory. Equivalent test names are allowed only when the PR provides an explicit AC-to-test mapping.

| AC | Acceptance criterion | Suggested named test |
|---|---|---|
| AC-01 | `compile_stream()` succeeds with graph context only and has no binding/catalog dependency. | `compile_stream_remains_binding_agnostic` |
| AC-02 | any preflight failure leaves connector opens, spawned tasks, registered timers, and pulls at zero. | `preflight_failure_has_no_runtime_side_effects` |
| AC-03 | generated mode rejects a dynamic/unknown schema before open. | `generated_policy_rejects_unknown_schema_before_open` |
| AC-04 | missing generated event-time column reports its stable binding-qualified path. | `generated_policy_reports_missing_column_path` |
| AC-05 | unsupported Arrow column type fails preflight. | `generated_policy_rejects_unsupported_event_time_type` |
| AC-06 | zero/unrepresentable policy durations fail preflight. | `watermark_policy_rejects_invalid_durations` |
| AC-07 | every D6 policy/capability combination is covered by a table-driven test. | `watermark_policy_capability_matrix_is_exhaustive` |
| AC-08 | `RuntimeToggleable` is accepted only with an existing private hook in the required state. | `runtime_toggleable_requires_existing_private_hook` |
| AC-09 | generated and connector-provided watermarks are never silently merged. | `watermark_modes_never_silently_merge` |
| AC-10 | seconds, milliseconds, microseconds, and nanoseconds Arrow timestamps normalize correctly. | `generated_watermark_supports_all_timestamp_units` |
| AC-11 | generated maximum ignores nulls and uses the maximum non-null value. | `generated_watermark_uses_non_null_max` |
| AC-12 | empty/null-only batches do not advance generated progress. | `empty_or_null_batch_does_not_advance_watermark` |
| AC-13 | watermark arithmetic overflow/underflow returns a controlled error. | `generated_watermark_checks_event_time_arithmetic` |
| AC-14 | generated watermarks strictly increase; duplicates/regressions are not emitted. | `generated_watermark_never_regresses_or_duplicates` |
| AC-15 | source-provided regression is a protocol error and equality is suppressed. | `source_watermark_is_monotonic` |
| AC-16 | connector watermark in generated/disabled mode is a protocol error. | `illegal_connector_watermark_is_rejected` |
| AC-17 | queued input/control wins over timers at the same logical instant. | `queued_data_precedes_timers_at_same_deadline` |
| AC-18 | watermark timers win over idle timers across bindings at the same instant. | `watermark_timer_precedes_idle_timer_across_bindings` |
| AC-19 | only the driver assigns binding ordinal/local sequence and they provide stable deterministic timer order. | `driver_owned_ordinals_and_sequences_break_timer_ties` |
| AC-20 | timer generation suppresses stale entries but never changes semantic sort order. | `timer_generation_is_validation_only` |
| AC-21 | each drain records exact per-inbox fences and selected membership; work readied afterward is deferred to the next finite snapshot except successful-End terminal-tail rejection. | `new_ready_work_waits_for_next_arbitration_snapshot` |
| AC-22 | `Active(None)` blocks aggregate progress. | `active_ingress_without_watermark_holds_progress` |
| AC-23 | aggregate progress is the minimum of all known active ingresses. | `multi_input_watermark_is_active_minimum` |
| AC-24 | idle and ended ingresses are excluded from the active minimum. | `idle_and_ended_ingresses_are_excluded_from_minimum` |
| AC-25 | all live idle ingresses emit one plain `Idle` per epoch. | `multi_input_emits_idle_once_per_idle_epoch` |
| AC-26 | repeated ingress `Idle` does not emit another aggregate `Idle`. | `repeated_idle_is_idempotent_within_epoch` |
| AC-27 | Data reactivates before processing and clears the idle latch. | `data_reactivation_precedes_processing` |
| AC-28 | a mode-legal connector WM reactivates before aggregation and clears the latch. | `legal_watermark_reactivation_precedes_aggregation` |
| AC-29 | reactivation followed by all-live-idle starts exactly one new idle epoch. | `reactivation_starts_a_new_idle_epoch` |
| AC-30 | reactivation never regresses the already emitted aggregate watermark. | `reactivation_cannot_regress_aggregate_watermark` |
| AC-31 | an end-induced minimum advance emits watermark before end. | `final_watermark_advancement_precedes_end` |
| AC-32 | all-ended emits one `EndOfInput`, no `Idle`, and no sentinel/MAX watermark. | `all_ended_emits_no_idle_or_sentinel_watermark` |
| AC-33 | selected connector/source input/control after end returns the first-key-ordered stable protocol error, causes its whole finite snapshot to commit no state/emission/timer/cursor acknowledgement, and failure-settles every selected raw receipt. | `post_end_input_aborts_whole_ready_snapshot_atomically` |
| AC-34 | snapshot capture is accepted only at a quiescent event boundary. | `progress_snapshot_requires_quiescent_boundary` |
| AC-35 | a non-trivial snapshot round-trips all binding/timer/aggregate/latch/epoch/clock/frontier state, full execution-trace prefix/position, gate generation/state/close cut, every next allocation coordinate, and zero unsettled receipts. | `progress_snapshot_roundtrips_complete_logical_coordinate` |
| AC-36 | prepared-job fingerprint mismatch fails restore before open. | `progress_restore_rejects_prepared_job_mismatch` |
| AC-37 | binding identity/ordinal mismatch fails restore before open. | `progress_restore_rejects_binding_identity_mismatch` |
| AC-38 | normalized config mismatch fails restore before open. | `progress_restore_rejects_normalized_config_mismatch` |
| AC-39 | any unequal driver instant, delivery/replay cursor, control frontier, execution-trace prefix/position, gate generation/state/close cut, last fence, any next allocation, fence-grouping config, or timer coordinate fails restore before governed effects. | `progress_restore_requires_exact_captured_coordinate` |
| AC-40 | from the exact coordinate, the same ordered raw attempts, logical-clock trace, and full `ProgressExecutionTrace` replayed at least 100 times yields identical admission decisions, outputs, receipt dispositions/results, errors, gate transitions, and terminal outcomes. | `progress_replay_is_deterministic_from_exact_coordinate` |
| AC-41 | cancellation records exact gate closes/cuts, extracted identities, and settlements; closes admission, unregisters timers/wakes, settles every accepted receipt, and produces no later emission. | `cancel_unregisters_all_progress_timers` |
| AC-42 | terminal completion occurs only after its transition/settlement trace records match, all accepted envelopes settle, and admission/timer/wake/detached work is clean. | `terminal_end_leaves_no_progress_work` |
| AC-43 | old/equal/new event-time rows relative to current WM all remain unclassified and forwarded by M3. | `m3_does_not_classify_or_drop_late_rows` |
| AC-44 | no late-row counter appears in M3 metrics/status. | `m3_observability_has_no_late_row_metric` |
| AC-45 | implementation and tests have no dependency on M4 window execution. | `m3_builds_without_window_operator` |
| AC-46 | public Rust API baseline has no additions/removals/changes. | `m3_preserves_public_rust_api` |
| AC-47 | Python and REST/OpenAPI baselines have no changes. | `m3_preserves_external_surface_baselines` |
| AC-48 | M2 lifecycle, backpressure, cancellation, and terminal invariants remain green. | `m2_runtime_regression_suite_remains_green` |
| AC-49 | adapters can submit only raw input/control; attempting to supply logical time, `ReadyKey`, ordinal, sequence, or timer coordinates is impossible or rejected. | `adapter_cannot_supply_or_forge_ready_key` |
| AC-50 | all input/control in one drain uses the driver current instant and no input/control API accepts caller `now`. | `input_control_uses_driver_owned_logical_instant` |
| AC-51 | first generated-WM deadline is exactly running instant plus interval. | `first_watermark_deadline_is_running_instant_plus_interval` |
| AC-52 | watermark cadence remains anchored to prior scheduled deadline, not delayed callback time. | `watermark_cadence_is_phase_anchored` |
| AC-53 | crossing multiple ticks evaluates exactly one coalesced expiry and schedules the smallest phase-aligned deadline after current driver time. | `missed_watermark_ticks_coalesce_deterministically` |
| AC-54 | idle deadline/rearm is exactly qualifying-activity driver instant plus timeout, including reactivation. | `idle_deadline_uses_last_driver_activity_instant` |
| AC-55 | only the driver can checked-increment timer generation/sequence and register/replace/unregister timers. | `timer_lifecycle_is_driver_owned` |
| AC-56 | a protocol error after an earlier valid item leaks no prefix effect; the driver stops at the first key-ordered error and reports that identity deterministically. | `ready_snapshot_protocol_error_discards_valid_prefix` |
| AC-57 | a successful finite snapshot retains every sender, commits all staged state/timers/cursor/fence coordinates, then success-settles every selected receipt before returning its complete emission batch. | `ready_snapshot_success_commits_atomically` |
| AC-58 | only a driver-internal generation-proven stale timer may be ignored after end and it has no effect. | `post_end_stale_internal_timer_is_effect_free` |
| AC-59 | restore-capable snapshot creation rejects a non-pausable or non-replayable source. | `progress_snapshot_rejects_non_replayable_source` |
| AC-60 | upstreams must be paused and exactly positioned before restore; missing cursor/frontier fields fail before mutation. | `progress_restore_requires_paused_exact_upstream_position` |
| AC-61 | local/global sequence exhaustion is fatal, deterministic, non-wrapping, snapshot-atomic, and leaves no accepted receipt unsettled before exit. | `sequence_exhaustion_aborts_without_mutation` |
| AC-62 | timer generation/sequence exhaustion is fatal, deterministic, non-wrapping, and snapshot-atomic. | `timer_counter_exhaustion_aborts_without_mutation` |
| AC-63 | idle epoch exhaustion is fatal, deterministic, non-wrapping, and snapshot-atomic. | `idle_epoch_exhaustion_aborts_without_mutation` |
| AC-64 | binding ordinal/deadline arithmetic exhaustion fails before work admission or state/output/timer/cursor commit and settles every already-accepted envelope before fatal exit. | `ordinal_or_deadline_exhaustion_is_atomic_fatal_error` |
| AC-65 | replay records and reproduces every admission, drain/fence/selected/timer/empty-epoch, terminal/gate-close, settlement, and driver-phase record in the complete execution trace. | `progress_replay_reproduces_full_progress_execution_trace` |
| AC-66 | identical raw attempts and clock with any different fence/admission/terminal/settlement trace are reported as different executions; no grouping/outcome promise crosses traces. | `same_raw_and_clock_with_different_fences_is_trace_inequality` |
| AC-67 | every selected raw sender is retained; success receipt is observable only after state/timer/frontier/fence commit, never before. | `commit_receipt_succeeds_only_after_atomic_commit` |
| AC-68 | a drain with multiple invalid items selects the first error by total key order, runs no suffix semantic hook, commits nothing, and settles every selected raw receipt once with that error. | `failed_drain_settles_all_selected_with_first_error` |
| AC-69 | an envelope accepted after a final End fence but before commit is recorded at the exact admission cut, atomically extracted, trace-linked, and receives one stable post-End failure without a next drain/emission/cursor acknowledgement. | `final_end_atomically_rejects_post_fence_tail` |
| AC-70 | submission after committed End consumes an admission-attempt record, observes the recorded closed generation/cause, rejects immediately, allocates no receipt/inbox identity, and enqueues nothing. | `submit_after_end_is_rejected_immediately` |
| AC-71 | dropping a receipt receiver before commit does not roll back or duplicate the semantic commit and leaves no sender/work retained. | `dropped_receipt_receiver_does_not_rollback_commit` |
| AC-72 | all-ended does not exit until terminal/gate/settlement trace records are consumed, selected successes and tail failures are attempted, and all timers/wakes/gates are clean. | `all_ended_waits_for_settlement_and_cleanup` |
| AC-73 | cancellation with selected/unselected accepted envelopes records the exact admission-race cut and settles each once with the stable cancellation error before exit. | `cancel_settles_every_accepted_receipt` |
| AC-74 | receipt, inbox, fence, drain, trace-record/cursor, admission, gate-generation, and gate-close next-value overflow never wraps or partially accepts/commits; all accepted envelopes deterministically settle before fatal exit. | `receipt_fence_epoch_and_inbox_counter_overflow_is_atomic` |
| AC-75 | a regression matrix proves the four original conflict resolutions, all prior B/N repairs, no M3 late metric, and M4/M5/post-M5 A6 boundaries remain intact after full-trace changes. | `m3_delta_prior_decisions_and_scope_firewall_remain_intact` |
| AC-76 | replay reproduces the End-tail race exactly: the recorded submit is accepted at its attempt/inbox identity, the End gate closes at the recorded generation/ordinal/cut, and the exact tail receives post-End settlement. | `progress_replay_reproduces_terminal_tail_linearization` |
| AC-77 | replay reproduces both sides of a fatal admission race, including exact accepted-versus-immediate-rejected decisions, fatal gate cuts, extracted membership, and settlements. | `progress_replay_reproduces_fatal_cleanup_settlements` |
| AC-78 | replay reproduces both sides of a cancellation admission race and the exact cancellation gate/accepted/settlement membership even without a drain epoch. | `progress_replay_reproduces_cancellation_settlements` |
| AC-79 | immediate post-End rejection replays with the same attempt ordinal, observed gate generation/cause, stable reason, no accepted identity, and no settlement record. | `progress_replay_reproduces_immediate_rejection` |
| AC-80 | commit success, selected transaction error, post-End tail rejection, cancellation, and fatal dispositions replay per accepted identity; receiver drop does not change the disposition. | `progress_replay_reproduces_settlement_dispositions` |
| AC-81 | missing, extra, reordered, or wrong-owner settlement/admission/terminal record fails before that record's governed effect. | `progress_replay_rejects_missing_or_extra_settlement_record` |
| AC-82 | structurally too-short/too-long trace, missing/extra attempt, or unconsumed expected suffix is rejected by complete replay-plan validation. | `progress_replay_requires_complete_expected_trace_suffix` |
| AC-83 | snapshot/restore round-trips and exactly validates trace prefix/position, next trace/admission/gate-close coordinates, and per-binding gate generation/state/close cut. | `progress_snapshot_roundtrips_execution_trace_coordinate` |
| AC-84 | trace-record/cursor, admission-attempt, gate-generation, and gate-close exhaustion is checked, non-wrapping, phase-atomic, and settlement-complete. | `execution_trace_admission_and_gate_counter_overflow_is_atomic` |
| AC-85 | admission/freeze/fence/drain/key/trace failure before complete item keys reports `DriverPhaseError` with exact phase coordinate, no fabricated `ReadyKey`, zero governed partial effect, and complete fatal settlement. | `pre_key_driver_phase_error_has_no_ready_key` |

Tests for logical ordering MUST use paused/fake time. A test whose semantic assertion depends on wall-clock scheduling or real `sleep` does not satisfy an AC.

## 15. Milestone repartition and delivery order

### 15.1 Revised milestone ownership

| Milestone | Owns | Explicitly does not own |
|---|---|---|
| M3 | policy/preflight, progress driver, complete admission/fence/terminal/settlement trace and receipt protocol, generated/native WM normalization, idle/reactivation, multi-input minimum, transient snapshot seam, progress status/metrics | late-row behavior, window execution, durable recovery, public A6 |
| M4 | state backend and window assignment/execution; final-only `LateDataPolicy::Drop` using concrete `window_end <= WM_in` | allowed lateness, side output, retraction, durable checkpoint/exactly-once |
| M5 | versioned durable progress/state encoding, checkpoint barriers, manifests, restart restore, timer deadline rebasing, exactly-once semantics | public A6 until the M5 gate is complete |
| post-M5 | public source-driven runner A6 and supported public integration surfaces | weakening M3–M5 guarantees |

The main plan MUST be updated so the former M3 late-data task becomes “progress status/metrics and transient snapshot,” the assignment-level late-drop task moves to M4, and durable progress recovery moves to M5.

### 15.2 Required delivery sequence

Implementation MUST proceed in dependency order:

1. **M3.0 documents** — merge this delta, API note, and zero-block critique.
2. **M3.1 policy/preflight** — descriptors, capability matrix, normalized modes, immutable prepared job, side-effect-free failures.
3. **M3.2 driver/progress** — job-scoped arbiter, full `ProgressExecutionTrace`, receipt/admission/gate-close/terminal-tail/cancel/fatal protocol, phase-correct errors, logical timer heap, generated/source-provided WM, idle/reactivation, multi-input progress.
4. **M3.3 snapshot/observability** — complete transient snapshot/restore with execution-trace/gate/allocation coordinates, status/metrics, deterministic complete-trace replay.
5. **M3.4 integration** — AC mapping, regression suite, 100-seed stress, 20-minute soak, documentation reconciliation.
6. **M4** — state/window execution and assignment-level late drop.
7. **M5** — durable checkpoint/exactly-once recovery.
8. **post-M5** — public A6.

Later steps MUST NOT be used to waive an earlier step's acceptance criteria.

## 16. Merge gates

M3 is mergeable only when all of the following are true:

1. this spec, its API note, and its critique are reconciled; critique reports `BLOCKS REMAINING: 0`;
2. every AC in §14 maps to automated named evidence in the PR;
3. all unit, integration, doctest, and public-API baseline checks pass;
4. the existing M2 lifecycle/backpressure/cancellation regression matrix passes;
5. deterministic stress passes for at least 100 recorded seeds, with every artifact recording its seed, ordered raw attempts, logical-clock trace, full `ProgressExecutionTrace` including admissions, empty drains, gate transitions/cuts, selected/tail/cancel/fatal settlements and driver-phase failures, exact queue/fence configuration, and terminal result; an exact captured coordinate plus the same complete trace replays item-identically at least 100 times;
6. the standardized **20-minute** soak passes and stores structured evidence sufficient to identify commit, seed/config, duration, admission decisions, drain/fence/timer counts, gate transitions/cuts, per-path settlement records, maximum unsettled receipts, progress transitions, cancellation/fatal/terminal outcome, and resource/error summary;
7. CI is green;
8. Codacy has no unresolved issue introduced or exposed by the M3 change set;
9. all Copilot and human review threads are resolved with code/test evidence or an explicit accepted rationale;
10. a `cf-reviewer` or designated human reviewer approves the final diff;
11. no public API/A6, Python, REST/OpenAPI, M4 window, or M5 durable-recovery scope has leaked into the change;
12. documentation consistently uses “snapshot-ready and deterministically replayable progress state” and does not claim M3 durable/crash recovery.
13. every accepted envelope in success, first-error, End-tail, cancel, fatal-overflow, receiver-drop, and all-ended paths has exactly one settlement attempt, and no gate/inbox/sender/timer/wake remains at exit;
14. the AC-75 regression evidence confirms that the four original conflict resolutions, all prior B/N repairs, no-late M3 rule, and M4/M5/post-M5 A6 boundaries did not regress;
15. AC-76–AC-85 prove End-tail, fatal/cancel admission races, immediate rejection, settlement disposition, incomplete/mismatched trace, exact snapshot coordinates, new counter exhaustion, and pre-key no-`ReadyKey` errors.

The 20-minute duration is the default soak standard for this and future milestone gates unless a later controlling specification explicitly replaces it. Shorter local smoke runs MAY be used during development but do not satisfy the merge gate.

## 17. Risks and required mitigations

| Risk | Required mitigation |
|---|---|
| connector open happens before policy validation | descriptor-only preflight counters proving zero opens/spawns/timers on failure |
| wall-clock races make replay flaky | one driver, logical timer heap, finite ready-set snapshot, paused/fake time |
| identical raw/clock inputs race between accepted-tail and immediate rejection | record/replay every admission decision, gate generation/close cut, terminal transition, and settlement in `ProgressExecutionTrace` |
| replay trace is missing/extra or ends with an unconsumed suffix | prevalidate the complete raw-attempt/trace plan and consume each record before its governed effect |
| completion sender is dropped before commit/error settlement | retain the owned sender through scratch and prepared commit; exactly one settlement attempt on every lifecycle path |
| final End commits while an after-fence envelope is already accepted | linearize admission close and terminal-tail extraction/failure settlement inside the End commit; no next drain required |
| adapter forges timestamps or order keys | raw unkeyed submission only; driver owns instant, ordinal, sequence, and `ReadyKey` allocation |
| delayed callbacks drift cadence | phase-anchor WM deadlines to prior scheduled deadline and deterministically coalesce missed ticks |
| approximate restore silently skips/repeats input | exact captured coordinate, paused exact positioning, deterministic rejection of non-replayable sources |
| post-end error leaks an earlier prefix emission or reports a scheduler-dependent error | scratch evaluation to the first total-order error, all-or-nothing finite-snapshot commit, and common failure receipts |
| pre-key failure is misattributed to an input item | disjoint `DriverPhaseError` taxonomy with phase coordinate and no fabricated `ReadyKey` |
| u64 wrap reuses trace/admission/gate/order/liveness/receipt/fence identity | checked increment and atomic deterministic fatal cleanup/settlement at every exhaustion boundary |
| idle reactivation regresses output WM | retain effective and aggregate WM; strict advancement checks |
| all-idle or all-ended emits duplicate/misordered controls | explicit idle latch/epoch and terminal state machine |
| generated/native modes both emit | exhaustive capability matrix and protocol error for illegal native WM |
| incomplete snapshot passes simple tests | non-trivial full execution-trace/gate/allocation state round-trip and mismatch tests |
| M3 accidentally implements row-level lateness | data-neutral ACs and absence of late-row metrics |
| internal seam becomes accidental public API | crate-private visibility and API-baseline gates |

## 18. Definition of done

M3 is complete when a fully prepared job can deterministically coordinate source-provided or generated event-time progress, phase-anchored/coalesced timers, idle/reactivation, multi-input watermarks, phase-correct atomic errors, receipt settlement, and terminal ordering under a single driver-owned logical arbiter; every admission, drain/fence, gate close, terminal transition, settlement disposition, and driver-phase failure is recorded and consumed in a complete `ProgressExecutionTrace`; End/cancel/fatal races replay exact accepted-versus-rejected outcomes; its complete progress state and exact upstream/trace/gate/fence/allocation coordinate can be captured and restored in memory at a quiescent boundary for deterministic same-job replay; checked exhaustion cannot wrap, fabricate a key, partially commit, or strand accepted work; it preserves all M2 runtime guarantees; and every gate in §16 passes.

Completion of M3 does not mean windows drop late assignments, durable recovery exists, or public A6 is available. Those claims are reserved for M4, M5, and post-M5 respectively.
