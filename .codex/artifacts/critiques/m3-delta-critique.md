# Continuous Streaming 3.0 — M3 Delta Critique, Round 4

| Field | Value |
|---|---|
| Status | **Approved for M3.0 document merge; implementation evidence remains pending** |
| Review type | Full fourth-round adversarial review after B-R3-01 and N-R3-01 repair |
| Baseline | `main@88243e9565795cd0bce01a155f75e735f4b47728` |
| Spec input | `m3-delta-spec.md`, 807 lines, SHA-256 `ab7ff946d674b08340448d989224e89773ded865ac4a4dd32420f9638140a46b` |
| API input | `m3-delta-api-note.md`, 2,364 lines, SHA-256 `f8f3f8730a234b8c53f514191aaad1053c5b2bbcd2652456904724c3aad5ca16` |
| Prior critique | `m3-delta-critique.md`, round 3, 518 lines, pre-rewrite SHA-256 `036e4c7c544f126ce7c2cfe973fbd583006179630914751de9161b39eb5d3481` |
| Scope | D1–D9, FR-001–FR-079, AC-01–AC-85, 85 named tests, concrete ownership/signatures/flows, and delivery gates |
| Reviewer verdict | **Zero document blockers; M3.0 spec/API/critique set is internally implementable and testable** |

## 1. Executive summary

The fourth-revision specification and API note close the final third-round blocker and wording defect without weakening any earlier decision. The complete `ProgressExecutionTrace`, rather than its drain-only `DrainFenceTrace` projection, is now the sole replay contract. It losslessly represents:

- every raw submission attempt and its accepted-versus-immediately-rejected decision;
- every finite drain, inbox fence, selected identity/key range, due timer, required empty epoch, and outcome;
- every successful End, cancellation, and fatal gate-close transition, including the logical instant, generation transition, close ordinal/cut, and exact extracted accepted tail;
- exactly one settlement disposition and stable result for every accepted envelope, with no settlement record for an immediate rejection;
- every recordable driver-phase failure, separate from selected-item failures.

The repaired replay boundary is sufficient: exact prepared/config state and captured coordinate, plus the same ordered raw/control attempts, driver logical-clock trace, and complete post-capture `ProgressExecutionTrace`, determine admission results, emissions, receipt dispositions/results, errors, gate transitions, and terminal outcome item for item. A different complete trace is explicitly a different logical execution. `DrainFenceTrace` remains only a diagnostic projection and is never called sufficient for receipt replay.

The error taxonomy is also repaired. Only evaluation after every selected item has a complete `ReadyKey` may produce `SelectedItemError { first_error_key, ... }`. Admission, allocation, freeze, fence, key construction, replay, gate-close planning, settlement planning, snapshot validation, and other pre-key failures produce `DriverPhaseError { phase, coordinate, ... }` with no fabricated key. Both paths preserve zero partial effect for the failed transaction/phase and perform deterministic, exactly-once settlement cleanup for all accepted envelopes.

The full audit found no blocker, major, or minor document defect. Eight implementation advisories remain; none changes the normative contract or blocks M3.0 document merge.

| Severity | Count |
|---|---:|
| Blocker | 0 |
| Major | 0 |
| Minor | 0 |
| Non-blocking advisory | 8 |

This verdict approves the documents, not the future implementation. It does not claim that the 85 tests, 100-seed stress, 100 replay repetitions, 20-minute soak, M2 regression matrix, CI, Codacy, Copilot/human review resolution, designated approval, or final PR merge have run.

## 2. Authority, method, and meaning of coverage

The review applied this precedence:

```text
m3-delta-spec.md
> m3-delta-api-note.md
> compatible baseline implementation details
```

The fourth pass read the three documents in full and performed:

1. a four-layer closure audit—decision/FR semantics, concrete API ownership/signature, flow/error behavior, and AC/named test—for the original four conflicts and every prior B/N finding;
2. a D1–D9 and FR-001–FR-079 identifier, completeness, and cross-reference audit;
3. an item-by-item AC-01–AC-85 and 85-name audit;
4. adversarial traces for accepted End tails, immediate post-End rejection, cancellation and fatal admission races, required empty drains, missing/extra/reordered/short/long trace suffixes, wrong settlement ownership, dropped receivers, and all-ended exit;
5. snapshot equality checks for execution-trace cursor, admission/gate state, exact upstream coordinates, inbox/fence/drain coordinates, settlement ownership, timers, and every semantic counter;
6. first-selected-error versus pre-key phase-error separation and zero-partial-effect cleanup checks;
7. regression checks for ordering, timer cadence, Idle/reactivation, capability/preflight, no-late M3 behavior, M4/M5/post-M5 scope, and public surface stability;
8. merge-gate executability checks for 100 recorded seeds, exact replay, the standardized 20-minute soak, CI, Codacy, Copilot/human review, designated approval, and final merge.

In this critique, **Covered** means the documents provide a coherent, implementable, testable obligation with a named evidence target. It does not mean implementation evidence already exists. **Closed** means the previous document defect is repaired. **Advisory** means implementation should make the stated choice explicit, but the controlling contract already determines the correct externally observable behavior.

## 3. Original four-conflict audit

| Original conflict | Result | Decision/FR layer | Concrete API, flow, and acceptance layer |
|---|---|---|---|
| M3 late drop refers to an M4-only window operator | **Closed** | D1, FR-041–FR-043, FR-067, FR-079 prohibit M3 classification/count/tag/split/drop and reserve assignment-level `window_end <= WM_in` for M4. | `DriverEmission::ForwardData` is neutral; no late field/counter exists; AC-43–AC-45 and AC-75 test the firewall. |
| M3 “recoverable” depends on M5 durability | **Closed** | D2/D7 and FR-032–FR-040 restrict M3 to exact, same-process, in-memory replay and prohibit serialization, manifests, restart, deadline rebasing, and exactly-once recovery claims. | `StreamProgressSnapshot`, `PausedExactUpstreams`, `ProgressReplayRequest`, and validated restore are crate-private; AC-34–AC-40, AC-59–AC-60, AC-83 cover the exact seam. |
| binding policy is invisible to pure `compile_stream()` | **Closed** | D3, FR-001–FR-010 place policy/schema/capability validation in whole-job preflight before side effects. | `prepare_stream_job` consumes binding specs and a side-effect-free descriptor catalog; AC-01–AC-09 prove graph-only compilation and pre-open rejection. |
| multi-input Idle propagation is undefined | **Closed** | D4, FR-019–FR-028 define `Active(None)`, active minimum, Idle epochs, reactivation, end ordering, all-ended, and post-End behavior. | `IngressActivity`, `InputProgress`, and `MultiInputProgress` encode the state machine; AC-22–AC-33 exercise every edge. |

No fourth-round repair reintroduces a public API change, M3 late metric, M4 window dependency, M5 durability claim, or public A6.

## 4. D1–D9 decision checklist

| Decision | Status | Review conclusion |
|---|---|---|
| D1 — late execution moves to M4 | **Conformant** | M3 forwards accepted data regardless of row time versus watermark and exposes no lateness observation. M4 alone applies the concrete-assignment predicate `window_end <= WM_in`. |
| D2 — exact transient replay only | **Conformant** | Capture is receipt-quiescent and exact; replay depends on ordered attempts, clock, and the complete lifecycle trace. Serialization, process restart, compatibility, rebasing, and durability remain excluded. |
| D3 — whole-job preflight owns policy validation | **Conformant** | `compile_stream()` remains pure. Descriptor/schema/capability/policy/config failures occur before connector open/toggle/spawn/timer/pull. |
| D4 — complete multi-input/Idle/transaction behavior | **Conformant** | `Active(None)` blocks; Idle/Ended are excluded; one Idle emits per epoch; legal reactivation precedes processing; WM precedes terminal End; first selected error aborts the entire finite snapshot. |
| D5 — one driver owns semantics | **Conformant** | One driver owns logical time, ordinals, fences, keys, timers, aggregation, trace, status, snapshot, and cleanup. Ordering and phase-anchored timer cadence are unique and checked. |
| D6 — explicit capability matrix | **Conformant** | All 12 policy/capability cells have exactly one result. Unknown rejects; toggleable requires an existing private route; native/generated emission is never merged. |
| D7 — all M3 surfaces remain private | **Conformant** | Every new type/function is `pub(crate)` at most; public Rust, connector/open, Python, REST/OpenAPI, and A6 surfaces stay unchanged. |
| D8 — admission/receipts/tails are one protocol | **Conformant** | Acceptance transfers sender ownership; success follows commit; End closes and extracts the exact tail; future calls reject; cancel/fatal/all-ended settle all accepted work before exit. |
| D9 — full trace and phase-correct errors | **Conformant** | Admissions, drains, terminals, settlements, and phase failures are lossless and consumed before their governed effects. Pre-key failures have no `ReadyKey`; selected errors alone use the first key. |

Decision coverage: **9/9 conformant**.

## 5. Severity findings

No blocker, major, or minor finding remains.

The two issues that could otherwise have blocked fourth-round approval were checked explicitly:

1. `ProgressExecutionTrace` is now lossless across both selected and outside-selected lifecycle boundaries. It distinguishes accepted terminal tails from immediate rejections and represents cancellation/fatal cleanup even without a drain.
2. `DriverPhaseError` is structurally disjoint from `SelectedItemError`; no API requires or permits a sentinel `ReadyKey` for a pre-key failure.

The advisories in §13 are implementation hardening notes, not unresolved normative contradictions.

## 6. Final-blocker adversarial closure proof

### 6.1 Trace sufficiency

The concrete trace has all five required record families:

```text
AdmissionAttemptRecord
DrainEpochRecord
TerminalTransitionRecord
SettlementRecord
DriverPhaseFailureTraceRecord
```

The information is sufficient because:

- `AdmissionAttemptRecord` fixes attempt position/ordinal, binding, event kind, upstream position, observed gate state/generation, and exactly one decision;
- `Accepted` fixes binding/ordinal, attempt, receipt, inbox, and upstream identity; `ImmediateRejected` contains no accepted identity;
- `DrainEpochRecord` fixes epoch, clock, every inbox fence, selected membership/order/key range, due timers, required empty epochs, and committed/selected-error/phase-error outcome;
- `TerminalTransitionRecord` fixes End/cancel/fatal cause, clock, owning drain when present, every gate generation/close ordinal/cut, and exact extracted accepted tail;
- `SettlementRecord` fixes exactly one owner, disposition, and stable outcome for each accepted identity;
- `DriverPhaseFailureTraceRecord` fixes pre-key phase and exact available coordinate without inventing an item key.

`DrainFenceTrace` is explicitly derived through `ProgressExecutionTrace::drain_projection()` and omits lifecycle records by design. Both documents state that it is insufficient for receipt replay.

### 6.2 Successful End race

For `End(A)` selected below fence `s` and `Data(A)` racing the commit, the full trace distinguishes all outcomes:

| Linearization | Admission record | Terminal record | Settlement record |
|---|---|---|---|
| Data accepted before End close | `Accepted(receipt r, inbox s+1, generation g)` | `EndCommit`, `g -> g+1`, close cut `s+2`, extracted tail `{r}` | selected End `CommitSuccess`; `r` `PostEndTailReject` |
| End closes before Data admission | `ImmediateRejected(observed ClosedAfterEnd, generation g+1)` | `EndCommit`, `g -> g+1`, close cut `s+1`, empty tail | selected End `CommitSuccess`; no Data settlement |
| End and Data selected in one earlier fence | both accepted and selected | no committed End gate transition because transaction fails | every selected identity receives the same `TransactionError(first ReadyKey)` |

The drain projections can be equal for the first two rows, but the full traces cannot. Replay waits for every recorded accepted tail identity, rejects an unexpected tail/cut/generation, consumes the terminal and canonical settlement records, and only then exposes closure or sends receipts. B-R3-01 is closed.

### 6.3 Fatal and cancellation races

Fatal cleanup records the selected-item or `DriverPhaseError` cause through the owning drain/phase-failure record and fatal settlement outcome, then records all gate transitions/cuts and every unselected accepted identity. Cancellation uses the same explicit terminal vocabulary even when no drain exists. In both cases:

- an attempt before the recorded cut is either selected or extracted and has exactly one settlement;
- an attempt after closure is an immediate rejection with no accepted identity or settlement;
- accepted/rejected membership is fixed by the trace rather than rediscovered from live scheduling;
- terminal replay validates exact tail membership and settlement ownership before closure/send visibility.

AC-77 and AC-78 directly cover both admission-race sides.

### 6.4 Settlement dispositions and receiver drop

The allowed dispositions are exactly:

```text
CommitSuccess
TransactionError
PostEndTailReject
Cancelled
Fatal
```

Every accepted identity owns exactly one of them. Immediate rejections own none. Canonical equality order is `(binding_ordinal, inbox_sequence, receipt_sequence)`; cross-task send/wakeup order is not semantic. A dropped receiver changes only observation of the one-shot send and never changes disposition, cursor acknowledgement, commit, trace, or cleanup. AC-71 and AC-80 cover this boundary.

### 6.5 Deterministic guarantee boundary

The final promise is coherent:

```text
same prepared job and normalized/fence configuration
+ exact captured logical coordinate
+ same ordered raw/control attempts
+ same driver logical-clock trace
+ same complete post-capture ProgressExecutionTrace
=> item-identical admissions, emissions, receipt dispositions/results,
   selected-item/driver-phase errors, gate transitions, terminal outcome
```

A different full trace is explicitly a different execution. This is not an escape hatch for a missing lifecycle record: admissions, terminal cuts/tails, and settlements are now inside the trace. AC-40, AC-65–AC-66, and AC-76–AC-82 make both the equality promise and inequality boundary observable.

## 7. Replay validation, consumption, and atomicity audit

### 7.1 Complete-plan prevalidation

`ProgressReplayRequest::prevalidate` runs before driver construction, timer installation, connector resume/open, task spawn, admission, receipt, cursor acknowledgement, state mutation, or emission. It must prove:

- exact captured prefix and consumed position;
- complete and contiguous trace coordinates;
- one ordered admission record per raw attempt;
- unique accepted identities;
- one settlement per acceptance and none per rejection;
- valid and existing settlement owners;
- disjoint and complete selected, End-tail, cancel, and fatal membership;
- self-consistent gate generations, close ordinals, cuts, and extracted tails;
- complete logical-clock records;
- an expected suffix ending exactly at the declared completion position.

Thus a structurally missing/extra/reordered/too-short/too-long/wrong-owner plan fails before runtime effects when detectable.

### 7.2 Record-by-record consumption

After prevalidation, the trace cursor validates the next exact record before the corresponding effect is visible:

| Record | Must be validated/consumed before |
|---|---|
| Admission | enqueue/receipt return or immediate rejection |
| Drain | live state/timer/cursor/fence commit and emission visibility |
| Terminal | admission-gate closure and tail extraction |
| Settlement | one-shot receipt send |
| Driver-phase failure | fatal cleanup transition/effects |

Any missing, extra, reordered, short, long, or mismatched record returns deterministic `DriverPhaseError`. Earlier fully matched and committed records remain committed; the mismatched record and owning phase/epoch/transition commit none of their governed effects. `finish()` rejects an unconsumed expected suffix.

### 7.3 Transaction and trace-cursor atomicity

The spec requires trace validation/consumption to be staged with the same scratch/guard transaction as the governed semantic effect. Therefore a precheck of a drain's fence/membership does not make a completed trace-cursor advance visible before the actual drain outcome is validated. The complete drain/terminal/settlement cursor advance commits with the corresponding live state and gate transition. This resolves the otherwise dangerous case in which a replay cursor could advance past a failed outcome while state did not.

The API note's `freeze_replay_epoch` helper is read under that controlling rule: it validates the expected pre-key portion before extraction, while §12.3 performs the final drain/terminal/settlement consumption in the atomic commit. Implementations should make that staged-versus-committed distinction explicit in type names; see advisory A-04.

## 8. Snapshot/restore exact-coordinate audit

| Required information | Spec/API coverage | Result |
|---|---|---|
| prepared job and runtime fence configuration | prepared-job, normalized binding, and runtime-fence fingerprints | Exact |
| binding table | identity, immutable order, driver ordinal, cardinality | Exact |
| logical clock | trace fingerprint plus current logical instant | Exact |
| upstream position | equality-comparable delivery/replay cursor and control frontier under a pause guard | Exact |
| admission state | gate state, generation, close cause/ordinal/cut | Exact |
| admission coordinates | next attempt, receipt, inbox, and gate-close coordinates | Exact |
| trace state | complete trace prefix, consumed position, next trace coordinate, replay suffix plan | Exact |
| settlement state | quiescent proof of zero unsettled accepted envelopes; next internal settlement coordinate where the concrete API retains one | Exact |
| drain/fence state | next drain, next per-binding fence, and exact last completed fence | Exact |
| ordering state | next binding ordinal, local, global, timer/local sequence coordinates | Exact |
| progress state | Active/Idle/Ended, native/generated/effective WMs, generated maximum | Exact |
| aggregate state | ingress table, last aggregate WM, Idle latch/epoch, terminal state | Exact |
| timers | current and stale heap entries, deadline, binding, kind, generation, timer/local sequence; absence distinguished | Exact |
| queue/fence grouping | capacity and `AllVisible` policy included in fingerprints; no restore override | Exact |

Capture is legal only at a quiescent boundary after every selected receipt is settlement-attempted and the emission batch returned, before the next fence, with no scratch mutation, accepted queue, or owned sender. A non-pausable/non-seekable source rejects restore-capable capture. Restore validates all duplicate copies and exact coordinates with upstreams still paused, then installs state and one heap-minimum wake without recomputing or rebasing deadlines.

No snapshot type has serde/bytes/manifest/version/restart semantics. The mandated phrase remains “snapshot-ready and deterministically replayable progress state.” M5 retains sole ownership of durable encoding, checkpoint barriers, restart, rebasing, and exactly-once recovery.

## 9. Phase-error and counter-exhaustion audit

### 9.1 Disjoint error identity

| Domain | Identity | Examples | Required behavior |
|---|---|---|---|
| selected-item evaluation after complete keys | `SelectedItemError { first_error_key, failure }` | illegal connector WM, post-End selected input, source WM regression, selected arithmetic failure | lowest failing `ReadyKey`; no suffix semantic hook; whole finite snapshot rolls back; all selected receipts get the same error |
| pre-key driver phase | `DriverPhaseError { phase, coordinate, failure }` | admission/trace allocation, drain/fence allocation, inbox freeze, key construction, replay validation, gate/settlement planning, snapshot validation | no `ReadyKey` or sentinel; zero governed partial effect; recorded fatal close/settlement cleanup when representable |

If drain epoch allocation or trace-record allocation itself exhausts, the error uses the last exact counter coordinate outside an invented epoch/record/key. Replay beyond that terminal exhaustion is unsupported, as explicitly stated; prior exact trace remains valid.

### 9.2 Checked semantic counters

The documents require checked allocation for every counter capable of changing ordering, liveness, admission, fencing, receipt identity, trace equality, or terminal ownership, including:

- trace-record ordinal and consumed trace position;
- admission-attempt, gate generation, and gate-close ordinal;
- receipt, inbox, fence, and drain coordinates;
- binding ordinal, local/global sequence, and the API's internal settlement coordinate;
- timer generation/sequence and deadline arithmetic;
- idle epoch and equivalent semantic counters.

Admission probes all needed successors atomically and consumes none if acceptance cannot be represented. Post-acceptance exhaustion closes admission and fatal-settles all accepted work before exit. Wrap, saturation, reuse, fabricated identities, partial state/emission/timer/cursor/fence commit, and stranded senders are forbidden. Observational counters alone may saturate because they never feed semantics.

AC-61–AC-64, AC-74, AC-84, and AC-85 form the counter and phase-error matrix.

## 10. Full receipt ownership and terminal-exit audit

| Path | Admission/ownership | Trace disposition | Settlement point | Cursor/effect rule |
|---|---|---|---|---|
| success | accepted envelope owns one sender through scratch/commit | `CommitSuccess`, owner drain | after state/timer/frontier/fence/trace commit | cursor acknowledged; complete emission batch returned after sends |
| selected transaction error | every selected sender retained, including unevaluated suffix | `TransactionError(first key)`, owner drain | after failed outcome/settlement trace commits | no state/timer/cursor/fence/emission commit |
| End post-fence tail | gate guard extracts exact same-binding accepted tail | `PostEndTailReject`, owner End terminal transition | after atomic gate/state/trace commit | no tail data/progress emission or cursor acknowledgement |
| future post-End submit | gate observes closed generation/cause | immediate rejection admission record; no settlement | returned synchronously; no accepted sender | no receipt/inbox identity, enqueue, cursor, or emission |
| cancellation | transition takes every selected/unselected accepted sender | `Cancelled`, owner cancellation transition | after exact cuts/tails/settlement records commit | clears gates, queues, timers, wake; no later emission |
| fatal cleanup | transition takes all remaining accepted senders | `Fatal(cause)`, owner fatal transition; selected failures retain transaction owner | after exact fatal records commit | zero failed-phase partial effect; clears all progress work |
| receiver dropped | sender ownership is still consumed once | unchanged semantic disposition | one send attempt; send error ignored | no rollback, retry, trace change, or retained work |
| all-ended | selected End successes and extracted tail failures retained | End terminal plus settlement records | every send attempted before exit | all gates closed; queues/heap/wake/detached work empty |

Submission and terminal close serialize on the same binding admission state. Capacity waiters are required to wake on terminal closure and then reject without allocation. All-ended cannot exit merely because `EndOfInput` was buffered; it must first consume the terminal/settlement trace suffix, attempt every settlement, close every gate, and prove there is no queued, timed, waking, or detached progress work.

## 11. Prior finding closure ledger

### 11.1 B-01 — driver-owned finite snapshots and semantic keys

| Layer | Closure evidence |
|---|---|
| Decision/FR | D5; FR-029–FR-031 and FR-045–FR-046 require raw-only input and exclusive driver allocation of time, ordinal, sequence, fences, and keys. |
| Concrete API | `RawIngressSender::submit(binding,event,position)` cannot express logical time, `ReadyKey`, ordinal, fence, or timer identity. Constructors remain private. |
| Flow/error | One clock coordinate is read per drain; exact inbox fences are frozen; driver keys/sorts a finite set; above-fence work is deferred except terminal-tail extraction. |
| AC/test | AC-17–AC-21, AC-49–AC-50 prove class order, stable ties, unforgeability, and finite-fence deferral. |

**Closure:** closed and preserved.

### 11.2 B-02 — timer ownership and phase cadence

| Layer | Closure evidence |
|---|---|
| Decision/FR | D5; FR-047–FR-050 freeze initial deadline, prior-deadline phase anchor, missed-tick coalescing, idle activity anchor, and exclusive driver ownership. |
| Concrete API | `TimerIntent` carries no coordinates; private `TimerAllocator` and the one driver create/replace/unregister heap entries. |
| Flow/error | First generated deadline is `running + interval`; one expiry is evaluated; next deadline is the smallest checked phase point after current time; idle is `activity + timeout`. |
| AC/test | AC-51–AC-55 plus no-candidate/suppressed-candidate/restore-wake refinement tests cover the seam. |

**Closure:** closed and preserved.

### 11.3 B-03 — complete exact restore coordinate

| Layer | Closure evidence |
|---|---|
| Decision/FR | D2; FR-032–FR-039, FR-053–FR-054, and FR-073 require quiescence, exact source positions, complete trace/gate/fence/allocation/timer state, and no approximate compatibility. |
| Concrete API | `PausedExactUpstreams`, complete snapshot structs, consuming `RestoreRequest`, prevalidated replay token, and no free capacity/fence override. |
| Flow/error | Every field compares before installation; upstreams remain paused; restore copies timers/counters and installs one wake without rebasing. |
| AC/test | AC-34–AC-40, AC-59–AC-60, AC-83 cover exact round-trip and mismatch rejection. |

**Closure:** closed and strengthened with full execution-trace/gate coordinates.

### 11.4 B-04 — post-End atomicity and zero-prefix failure

| Layer | Closure evidence |
|---|---|
| Decision/FR | D4/D8; FR-028, FR-051–FR-052, FR-062–FR-064 define first-error rollback, End closure/tail extraction, future rejection, and terminal cleanup. |
| Concrete API | scratch owns selected senders; `AdmissionCommitGuard` owns gate closure/tail extraction; committed settlement handles send only after trace/state commit. |
| Flow/error | same-fence post-End input fails the whole snapshot; successful End atomically closes/extracts; all-ended waits for every settlement and empty work. |
| AC/test | AC-33, AC-56–AC-58, AC-68–AC-72 cover every edge. |

**Closure:** closed and now replay-lossless.

### 11.5 N-01 — checked exhaustion

| Layer | Closure evidence |
|---|---|
| Decision/FR | D5/D9; FR-055–FR-056 and FR-074 prohibit wrap/saturation/reuse and require phase-correct fatal cleanup. |
| Concrete API | checked semantic allocators require a representable successor; deadline arithmetic is checked; observational counters alone may saturate. |
| Flow/error | admission failure accepts nothing; post-acceptance failure closes gates, records exact cleanup when possible, and settles all owned envelopes. |
| AC/test | AC-61–AC-64, AC-74, AC-84–AC-85 cover all counter families and no-fabricated-key identity. |

**Closure:** closed and preserved.

### 11.6 B-R2-01 — replay omitted finite drain partition

| Layer | Closure evidence |
|---|---|
| Decision/FR | D2/D5; FR-008, FR-030, FR-033–FR-038, FR-057–FR-058 make fence grouping semantic and exactly fingerprinted. |
| Concrete API | every drain records all inbox fences, selected identities/key range, due timers, outcome, and required empties; capacity/fence policy has no restore override. |
| Flow/error | replay validates exact membership/keys/timers and defers above-fence work; different drain trace is explicitly another execution. |
| AC/test | AC-21, AC-40, AC-65–AC-66 cover exact grouping and trace inequality. |

**Closure:** closed and preserved.

### 11.7 B-R2-02 — sender ownership and terminal admission edge

| Layer | Closure evidence |
|---|---|
| Decision/FR | D8; FR-059–FR-065 require sender transfer, success-after-commit, exact End tail, future rejection, cancel/fatal settlement, and clean exit. |
| Concrete API | sender moves from envelope through frozen/scratch/prepared/committed settlement values and is consumed once; no sender-less acknowledgement side map exists. |
| Flow/error | End/submit serialize; every racing attempt is selected, extracted tail, or immediate rejection; cancellation/fatal use the same gate protocol. |
| AC/test | AC-57, AC-67–AC-74 prove commit order, first error, tail, future reject, receiver drop, terminal/cancel, and allocation atomicity. |

**Closure:** closed and preserved.

### 11.8 N-R2-01 — first error versus suffix evaluation

| Layer | Closure evidence |
|---|---|
| Decision/FR | D4; FR-051–FR-052 and FR-066 require evaluation only to the first ordered selected error and no suffix semantic hook. |
| Concrete API | `PreparedDrainFailure` retains all selected senders while identifying one `SelectedItemError`. |
| Flow/error | valid prefix remains scratch-only; all selected receipts, including unevaluated suffix, get the same lowest-key error. |
| AC/test | AC-33, AC-56, AC-68 directly assert zero prefix leakage, first-key identity, no suffix hook, and complete settlement. |

**Closure:** closed and preserved.

### 11.9 B-R3-01 — lifecycle events missing from replay trace

| Layer | Closure evidence |
|---|---|
| Decision/FR | D2/D8/D9; FR-038, FR-040, FR-057–FR-060, FR-062–FR-072, FR-076–FR-078 require a full admission/drain/terminal/settlement trace. |
| Concrete API | `ProgressExecutionTrace` contains admission, drain, terminal, settlement, and phase-failure records; `DrainFenceTrace` is projection-only. |
| Flow/error | complete-plan validation proves admission/settlement ownership; End/cancel/fatal replay waits for exact tails and consumes transitions/settlements before effects. |
| AC/test | AC-40, AC-65–AC-66, AC-76–AC-82 and AC-83 cover tail/fatal/cancel races, immediate rejection, dispositions, mismatch, and exact trace coordinate. |

**Closure:** closed. The accepted-tail versus immediate-rejection adversarial pair now necessarily has different full traces.

### 11.10 N-R3-01 — pre-key failure misattributed to `ReadyKey`

| Layer | Closure evidence |
|---|---|
| Decision/FR | D9; FR-052, FR-066, FR-074–FR-075 separate selected-item and driver-phase identities. |
| Concrete API | disjoint `SelectedItemError` and `DriverPhaseError`; drain outcome has selected and phase variants; phase coordinate has no key field. |
| Flow/error | freeze/allocation/key/replay/gate/settlement failures commit no governed effect, perform fatal cleanup, and invent neither epoch nor key when allocation failed. |
| AC/test | AC-74, AC-84, AC-85 assert checked boundaries, exact phase coordinate, no fabricated key, and complete settlement. |

**Closure:** closed.

Prior-finding coverage: **10/10 closed**.

## 12. AC-01–AC-85 and named-test audit

Every AC identifier occurs exactly once in the spec acceptance table and exactly once in the API evidence table. Every criterion has one explicit named test; no identifier is missing or duplicated. The table below audits document coverage, not future test results.

| AC | Named test | Fourth-round conclusion |
|---|---|---|
| AC-01 | `compile_stream_remains_binding_agnostic` | Covered: pure compiler has no binding/catalog/config dependency. |
| AC-02 | `preflight_failure_has_no_runtime_side_effects` | Covered: open/spawn/timer/pull counters remain zero. |
| AC-03 | `generated_policy_rejects_unknown_schema_before_open` | Covered: Generated plus unknown schema rejects pre-open. |
| AC-04 | `generated_policy_reports_missing_column_path` | Covered: stable binding-qualified column path. |
| AC-05 | `generated_policy_rejects_unsupported_event_time_type` | Covered: only four timestamp units resolve. |
| AC-06 | `watermark_policy_rejects_invalid_durations` | Covered: zero/unrepresentable durations reject. |
| AC-07 | `watermark_policy_capability_matrix_is_exhaustive` | Covered: all 12 cells have one outcome. |
| AC-08 | `runtime_toggleable_requires_existing_private_hook` | Covered: toggleable requires the baseline private route. |
| AC-09 | `watermark_modes_never_silently_merge` | Covered: one producer only; illegal native input errors. |
| AC-10 | `generated_watermark_supports_all_timestamp_units` | Covered: second/milli/micro/nano normalize exactly. |
| AC-11 | `generated_watermark_uses_non_null_max` | Covered: maximum representable non-null value. |
| AC-12 | `empty_or_null_batch_does_not_advance_watermark` | Covered: progress-neutral empty/null batch. |
| AC-13 | `generated_watermark_checks_event_time_arithmetic` | Covered: conversion/subtraction overflow is controlled. |
| AC-14 | `generated_watermark_never_regresses_or_duplicates` | Covered: strict generated advancement. |
| AC-15 | `source_watermark_is_monotonic` | Covered: equality suppresses; regression errors. |
| AC-16 | `illegal_connector_watermark_is_rejected` | Covered: Generated/Disabled connector WM errors before reactivation. |
| AC-17 | `queued_data_precedes_timers_at_same_deadline` | Covered: input/control class wins. |
| AC-18 | `watermark_timer_precedes_idle_timer_across_bindings` | Covered: WM timer class wins. |
| AC-19 | `driver_owned_ordinals_and_sequences_break_timer_ties` | Covered: driver-only stable tie break. |
| AC-20 | `timer_generation_is_validation_only` | Covered: generation never enters semantic `ReadyKey`. |
| AC-21 | `new_ready_work_waits_for_next_arbitration_snapshot` | Covered: exact fence deferral with End-tail exception. |
| AC-22 | `active_ingress_without_watermark_holds_progress` | Covered: `Active(None)` blocks. |
| AC-23 | `multi_input_watermark_is_active_minimum` | Covered: known active minimum. |
| AC-24 | `idle_and_ended_ingresses_are_excluded_from_minimum` | Covered: Idle/Ended exclusion. |
| AC-25 | `multi_input_emits_idle_once_per_idle_epoch` | Covered: one checked epoch emission. |
| AC-26 | `repeated_idle_is_idempotent_within_epoch` | Covered: latch suppresses repetition. |
| AC-27 | `data_reactivation_precedes_processing` | Covered: Data reactivates first. |
| AC-28 | `legal_watermark_reactivation_precedes_aggregation` | Covered: legal connector WM reactivates first. |
| AC-29 | `reactivation_starts_a_new_idle_epoch` | Covered: latch clear permits one later epoch. |
| AC-30 | `reactivation_cannot_regress_aggregate_watermark` | Covered: aggregate strict monotonicity. |
| AC-31 | `final_watermark_advancement_precedes_end` | Covered: end-induced WM precedes End. |
| AC-32 | `all_ended_emits_no_idle_or_sentinel_watermark` | Covered: one End, no Idle/MAX. |
| AC-33 | `post_end_input_aborts_whole_ready_snapshot_atomically` | Covered: lowest keyed protocol error, zero effects, all selected receipts fail. |
| AC-34 | `progress_snapshot_requires_quiescent_boundary` | Covered: no scratch or unsettled sender at capture. |
| AC-35 | `progress_snapshot_roundtrips_complete_logical_coordinate` | Covered: full job/clock/frontier/gate/trace/timer/allocator/fence round-trip. |
| AC-36 | `progress_restore_rejects_prepared_job_mismatch` | Covered: mismatch before construction/open. |
| AC-37 | `progress_restore_rejects_binding_identity_mismatch` | Covered: identity/ordinal/order mismatch. |
| AC-38 | `progress_restore_rejects_normalized_config_mismatch` | Covered: config/fence fingerprint mismatch. |
| AC-39 | `progress_restore_requires_exact_captured_coordinate` | Covered: every coordinate mismatch rejects before effects. |
| AC-40 | `progress_replay_is_deterministic_from_exact_coordinate` | Covered: 100 identical full-trace replays include admissions, receipts, errors, gates, terminal result. |
| AC-41 | `cancel_unregisters_all_progress_timers` | Covered: exact cancellation records, settlement, timer/wake cleanup, no later emission. |
| AC-42 | `terminal_end_leaves_no_progress_work` | Covered: terminal waits for trace suffix, settlements, and empty work. |
| AC-43 | `m3_does_not_classify_or_drop_late_rows` | Covered: old/equal/new row times all forward unchanged. |
| AC-44 | `m3_observability_has_no_late_row_metric` | Covered: no late field/metric/log/error. |
| AC-45 | `m3_builds_without_window_operator` | Covered as future gate: no M4 dependency. |
| AC-46 | `m3_preserves_public_rust_api` | Covered as future gate: public Rust baseline unchanged. |
| AC-47 | `m3_preserves_external_surface_baselines` | Covered as future gate: Python/REST/OpenAPI unchanged. |
| AC-48 | `m2_runtime_regression_suite_remains_green` | Covered as future gate: all M2 lifecycle guarantees remain mandatory. |
| AC-49 | `adapter_cannot_supply_or_forge_ready_key` | Covered: raw signature cannot express semantic fields. |
| AC-50 | `input_control_uses_driver_owned_logical_instant` | Covered: one driver instant, no caller `now`. |
| AC-51 | `first_watermark_deadline_is_running_instant_plus_interval` | Covered: exact checked initial deadline. |
| AC-52 | `watermark_cadence_is_phase_anchored` | Covered: recurrence uses prior scheduled deadline. |
| AC-53 | `missed_watermark_ticks_coalesce_deterministically` | Covered: one expiry and smallest future phase. |
| AC-54 | `idle_deadline_uses_last_driver_activity_instant` | Covered: exact qualifying-activity anchor. |
| AC-55 | `timer_lifecycle_is_driver_owned` | Covered: driver-private allocator owns all timer mutation. |
| AC-56 | `ready_snapshot_protocol_error_discards_valid_prefix` | Covered: no prefix effect, lowest selected error. |
| AC-57 | `ready_snapshot_success_commits_atomically` | Covered: actual sender success follows state/timer/frontier/fence/trace commit. |
| AC-58 | `post_end_stale_internal_timer_is_effect_free` | Covered: generation-proven internal stale timer is sole silent exception. |
| AC-59 | `progress_snapshot_rejects_non_replayable_source` | Covered: non-exact source cannot create restore-capable snapshot. |
| AC-60 | `progress_restore_requires_paused_exact_upstream_position` | Covered: pause guard and exact source position precede install. |
| AC-61 | `sequence_exhaustion_aborts_without_mutation` | Covered: local/global exhaustion is non-wrapping and settlement-complete. |
| AC-62 | `timer_counter_exhaustion_aborts_without_mutation` | Covered: timer generation/sequence exhaustion is atomic. |
| AC-63 | `idle_epoch_exhaustion_aborts_without_mutation` | Covered: idle epoch exhaustion is atomic. |
| AC-64 | `ordinal_or_deadline_exhaustion_is_atomic_fatal_error` | Covered: pre-admission/scratch failure and complete cleanup. |
| AC-65 | `progress_replay_reproduces_full_progress_execution_trace` | Covered: admissions, drains/empties, terminals/cuts, settlements, phase failures all replay. |
| AC-66 | `same_raw_and_clock_with_different_fences_is_trace_inequality` | Covered: any different full trace has no cross-trace result promise. |
| AC-67 | `commit_receipt_succeeds_only_after_atomic_commit` | Covered: sender retained and success follows complete commit. |
| AC-68 | `failed_drain_settles_all_selected_with_first_error` | Covered: lowest key, no suffix hook, every selected sender fails once. |
| AC-69 | `final_end_atomically_rejects_post_fence_tail` | Covered: exact cut/tail, trace-linked post-End settlement, no later drain. |
| AC-70 | `submit_after_end_is_rejected_immediately` | Covered: closed gate record, no receipt/inbox/enqueue/settlement. |
| AC-71 | `dropped_receipt_receiver_does_not_rollback_commit` | Covered: one send attempt, no rollback/retry/retained work. |
| AC-72 | `all_ended_waits_for_settlement_and_cleanup` | Covered: trace consumed and gates/queues/heap/wake clean before exit. |
| AC-73 | `cancel_settles_every_accepted_receipt` | Covered: exact race cut and all selected/unselected settlements. |
| AC-74 | `receipt_fence_epoch_and_inbox_counter_overflow_is_atomic` | Covered: all listed counters checked; fatal cleanup settles all accepted work. |
| AC-75 | `m3_delta_prior_decisions_and_scope_firewall_remain_intact` | Covered: four conflicts and every prior repair remain intact. |
| AC-76 | `progress_replay_reproduces_terminal_tail_linearization` | Covered: exact accepted tail, gate generation/ordinal/cut, and post-End result. |
| AC-77 | `progress_replay_reproduces_fatal_cleanup_settlements` | Covered: both fatal race sides, cuts, membership, and settlements. |
| AC-78 | `progress_replay_reproduces_cancellation_settlements` | Covered: cancellation race replays even without a drain. |
| AC-79 | `progress_replay_reproduces_immediate_rejection` | Covered: attempt/gate/cause/reason replay with no accepted identity/settlement. |
| AC-80 | `progress_replay_reproduces_settlement_dispositions` | Covered: all five dispositions by identity; receiver drop changes none. |
| AC-81 | `progress_replay_rejects_missing_or_extra_settlement_record` | Covered: missing/extra/reordered/wrong-owner record fails before governed effect. |
| AC-82 | `progress_replay_requires_complete_expected_trace_suffix` | Covered: short/long/missing/extra attempt and unconsumed suffix reject pre-run. |
| AC-83 | `progress_snapshot_roundtrips_execution_trace_coordinate` | Covered: prefix/position, next trace/attempt/close coordinates, gate snapshots exact. |
| AC-84 | `execution_trace_admission_and_gate_counter_overflow_is_atomic` | Covered: new trace/admission/gate counters are checked and cleanup-complete. |
| AC-85 | `pre_key_driver_phase_error_has_no_ready_key` | Covered: exact phase coordinate, no key, zero effect, complete fatal settlement. |

Coverage count:

| Classification | Count |
|---|---:|
| Covered / covered as a future evidence gate | 85 |
| Blocked or incomplete | 0 |
| Named tests mapped | 85 |
| Total criteria | 85 |

## 13. Residual non-blocking implementation advisories

### A-01 — startup silence policy

A binding receives no runtime idle deadline before its first qualifying activity. A source silent from startup therefore remains `Active(None)` unless it reports Idle. This is coherent with the current contract but should be product-confirmed and documented in the implementation PR.

### A-02 — exact cursor must imply stable replay identity

`RawAttemptTraceIdentity` uses binding, event kind, and exact upstream cursor/control position rather than embedding arbitrary payload bytes. A connector advertising `ExactPauseReportAndSeek` must guarantee that a cursor/control identity re-delivers the same logical record/control payload. A connector unable to prove this must remain `Unsupported`; `RawUpstreamPosition::Unavailable` must never enter restore-capable replay.

### A-03 — optional settlement ordinal

The API note introduces a checked internal `SettlementOrdinal`, while D9/FR-071 say canonical envelope-identity order is sufficient and no settlement counter is required. This does not violate a `MUST NOT`, is included in snapshot equality, and is implementable, but it creates one extra exhaustion surface. The implementation may remove it and use record position plus canonical identity, or retain and fully test it as specified.

### A-04 — staged drain precheck versus committed cursor consumption

The helper prose says `freeze_replay_epoch` “consumes” the expected drain before extraction, while the final outcome is known only after evaluation. The controlling atomicity rules require this to be a staged/rollback-safe precheck, with visible trace-cursor advancement committed only after the complete outcome matches. Use distinct peek/stage/commit types so an early error cannot advance the live replay cursor.

### A-05 — prepared admission state

The lifecycle prose constructs a driver with closed gates and opens them in `start_running`, while the illustrative `AdmissionGateState` enum lists only Open and terminal-closed states. Because concrete spelling is non-normative, implementation may make `new`+`start_running` one unobservable atomic construction or add a private `PreparedClosed` state. It must not expose a sender that can accept before initial timer setup commits.

### A-06 — frozen-envelope RAII ownership

Fallible work after inbox extraction should use an owned guard/handoff so a `?` path or panic boundary cannot drop selected senders before recorded fatal settlement. The documents mandate the right outcome; the implementation should make it structurally unavoidable.

### A-07 — trace, scratch, and stale-heap growth

Lossless in-memory trace plus `AllVisible` scratch batches and stale timer entries can grow. Stress and soak evidence should report peak trace bytes/records, fence size, heap size, unsettled receipts, and settlement latency. Any compaction must preserve exact trace equality and remains an M3 internal optimization, not durable encoding.

### A-08 — output/backpressure and terminal waiter integration

The driver commits and receipt-settles before the run loop returns the emission batch. Integration must preserve M2 output-sink/backpressure failure semantics. Capacity waiters must wake when End/cancel/fatal closes a gate and reject without allocation; otherwise a semantically closed gate could still leave a task hung.

None of these advisories authorizes a public surface, a late metric, an M4 window dependency, M5 durability, or post-M5 A6.

## 14. Merge-gate audit

| Gate | Fourth-round document result |
|---|---|
| reconciled spec/API/critique | **Pass on document content:** this critique reports zero blocks and both upstream documents describe the same D1–D9 contract. |
| D/FR/AC mapping | **Pass on paper:** 9 decisions, 79 FRs, 85 ACs, and 85 named tests are present and cross-mapped. |
| 100-seed deterministic stress | **Executable, not yet run:** artifacts must include seed, attempts, clock, full trace, config, admissions, gates/cuts, settlements, phase failures, terminal result. |
| 100 exact replay repetitions | **Executable, not yet run:** exact coordinate plus same complete trace must reproduce all listed outcomes item-identically. |
| different-trace boundary | **Executable, not yet run:** same raw/clock with any different full trace is explicit inequality. |
| phase/transaction atomicity | **Executable, not yet run:** selected first-error and pre-key phase-error matrices are disjoint and testable. |
| receipt ownership | **Executable, not yet run:** success, transaction error, End tail, immediate reject, cancel, fatal, receiver drop, and all-ended paths are specified. |
| counter exhaustion | **Executable, not yet run:** all semantic coordinates and deadline arithmetic have named boundary tests. |
| M2 regression | **Pending:** lifecycle, bounded backpressure, cancellation, receipt, and terminal suite must remain green. |
| standardized soak | **Pending:** one 20-minute run with structured trace/settlement/resource evidence is mandatory. The same duration is the future default unless superseded. |
| CI | **Pending:** must be green. |
| Codacy | **Pending:** no unresolved issue introduced or exposed by M3. |
| Copilot/human review | **Pending:** every thread resolved with evidence or accepted rationale. |
| designated review | **Pending:** `cf-reviewer` or designated human approval required. |
| public/scope firewall | **Executable, not yet run:** Rust/Python/REST/OpenAPI baseline diffs empty; no M4/M5/A6 leak. |
| final PR merge | **Pending:** only after every implementation/evidence gate passes; PR URL and merge commit are required delivery evidence. |

The document set is ready for M3.0 merge. The eventual implementation PR is not merge-ready merely because this critique has zero blockers.

## 15. Recommended implementation order

1. **M3.1 preparation:** side-effect-free descriptor catalog, schema/policy validation, exhaustive capability table, existing toggle-route verification, immutable binding order, exact capacity/fence fingerprints, and zero-side-effect failure counters.
2. **M3.2a admission and trace:** checked attempt/receipt/inbox/trace allocation, prepared/open/terminal gate lifecycle, raw sender ownership, full admission record, replay prevalidation, and staged cursor consumption.
3. **M3.2b finite drains:** checked epochs/fences/keys, frozen-envelope RAII guard, exact drain and empty-epoch records, total ordering, and first-selected versus pre-key phase errors.
4. **M3.2c timers and progress:** one logical heap/wake, generated/native normalization, phase-anchored coalescing, idle deadlines/reactivation, aggregate minimum/Idle epoch/terminal ordering, and checked arithmetic.
5. **M3.2d settlement and terminal transitions:** scratch commit, common first-error settlement, End close/extract, immediate future reject, cancel/fatal admission races, receiver drop, and all-ended proof.
6. **M3.3 snapshot and replay:** exact paused source/frontier coordinator, complete in-memory snapshot/restore, full trace/gate/counter round-trip, mismatch rejection, and bounded no-late status.
7. **M3.4 verification:** all 85 named AC tests, refinement tests, 100 recorded seeds, 100 exact replays, M2 regression matrix, API/scope baselines, and standardized 20-minute soak.
8. Resolve Codacy, Copilot, and human findings, obtain final reviewer approval, merge the PR, and record the PR URL and merge commit.

M4 concrete-assignment late drop, M5 durable checkpoint/exactly-once recovery, and post-M5 public A6 remain strictly later work.

## 16. Final verdict

The fourth revision is internally coherent, implementable, and testable. It closes the original four conflicts and every first-, second-, and third-round finding at all four required layers: controlling decisions/FRs, concrete crate-private API ownership/signatures, happy/adversarial lifecycle flows, and named AC evidence.

Most importantly, the complete execution trace now records the acceptance and settlement boundaries that the third revision omitted. End-tail, fatal, and cancellation races can no longer collapse into the same trace as immediate rejection; missing/extra/reordered/short/long records fail before their governed effect; snapshots carry the exact trace/gate/allocation coordinate; selected errors and pre-key phase errors are disjoint; and every accepted sender has one settlement path before terminal exit.

There are **no blocker, major, or minor document findings**. The eight residual advisories concern implementation hardening and evidence quality only. The M3.0 documents may proceed. The M3 implementation and final PR must still satisfy all 85 ACs, the 100-seed/100-replay gates, M2 regression suite, 20-minute soak, CI, Codacy, Copilot/human review, designated approval, and final merge requirements.

BLOCKS REMAINING: 0
