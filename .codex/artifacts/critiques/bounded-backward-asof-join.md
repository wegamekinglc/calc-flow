# Bounded backward ASOF Join - Critic Critique

## Target

- Spec: [bounded-backward-asof-join.md](../specs/bounded-backward-asof-join.md).
- API note: [bounded-backward-asof-join.md](../api-notes/bounded-backward-asof-join.md).
- Plan: [2026-09-09-bounded-backward-asof-join.md](../../../docs/plans/2026-09-09-bounded-backward-asof-join.md).
- Baseline: PR #259 branch `feature/python-expression-api-refactor`,
  `eda1583751abbd1ca4d246fcb8ee6b70f57d9b09`.
- Scope: pre-implementation contract critique. Read the complete artifacts,
  applicable guidance, Introduction, existing inner Join/runtime code and focused
  analogue tests. No implementation edits, builds or benchmark runs were made.

## Verdict

**Proceed with caveats.**

Approve the chosen first-version scope, matching/finality contract, independent
wire/state identity and Python application surface. There are no blocking
semantic findings. One diagnostic wording correction is recorded as C1 below;
it does not require changing matching semantics or the existing progress
protocol. C2 and C3 are implementation evidence gates already required by
FR13-FR17 and AC9-AC11, not requests to expand the feature.

Approval of a contract is not proof that its native state, workspace or recovery
implementation exists or passes acceptance.

## Findings

### Blocking Issues

None found.

The strict watermark boundary, conservative downstream frontier, permanent
suppression of ASOF output idle before dual EOF, separate identity lifetime,
atomic Batch admission and independent old-inner compatibility vectors are
sufficiently explicit to implement without inventing semantics.

### Significant Concerns

- **C1 - An upstream protocol rejection is not an ASOF handler error.**
  The API note's diagnostic example originally mapped all data-after-EOF cases
  to `asof_protocol_error`. In the managed runtime,
  [operator_task.rs](../../../crates/calc-flow/src/runtime/streaming/operator_task.rs)
  lines 539-550 evaluate ingress progress before calling the data handler;
  [aggregate.rs](../../../crates/calc-flow/src/runtime/streaming/progress/aggregate.rs)
  lines 206-210 reject post-EOF traffic there.
  `OperatorInputProgress::evaluate` at operator-task lines 1157-1179 converts the
  typed progress failure into its existing error envelope.
  A direct ASOF handler test can therefore pass while a real runtime test
  legitimately reports a different category.
  - **Suggested fix:** Preserve existing source/runtime categories when those
    layers reject first; use the new ASOF reason only for checks performed by the
    ASOF boundary itself. Say so in the API error table and cover both layers
    independently. Do not recover a reason by parsing message strings or globally
    rewrite existing progress errors.
  - **Disposition:** Closed. The API author added layer precedence and separate
    boundary tests at lines 592-604, with distinct table entries at lines 627-628.
    This critic reread the revision and verified that it preserves the existing
    structured error path without text parsing.

- **C2 - A finite DataFusion pool proves only its own reservations.**
  The stage-zero
  probe, now retained in `crates/calc-flow/tests/asof_support/materialization.rs`
  and run by the `stream_asof_join_properties` integration target,
  creates Arrow candidate arrays before registering them in the session, and its
  64-byte test exercises `MemoryConsumer::try_grow`. These useful tests establish
  key ordering, bounded candidate cardinality and pool failure behavior. They do
  not reserve the arrays, row encodings, IPC writers, query outputs or restore
  scratch described by API note Accounting version 1.
  A large repeated right payload can fit retained state once and exceed the
  workspace when copied into many selected-row candidates.
  - **Suggested fix:** Enforce one aggregate workspace reservation across those
    ownership domains, including DataFusion reservations. Add actual ASOF
    fault/limit vectors for a repeated wide right payload, compacting a small
    slice of a large backing allocation, bounded encoding and restore scratch.
    Verify rejection occurs before the charged allocation and leaves committed
    input state unchanged. Exercise chunk reduction when a smaller valid chunk
    fits; a one-row failure must remain explicit. Keep state charge, workspace,
    runtime/checkpoint ownership and process RSS separate.
  - **Disposition:** Already required by spec FR13/FR14 and AC9, and API note
    lines 419-455. No additional public limit or alternate table engine is needed.

- **C3 - Repeated checkpoint/restore cycles must preserve both bounded ownership and progress.**
  Existing
  [stream_join_state.rs](../../../crates/calc-flow/tests/stream_join_state.rs)
  lines 541-605 explicitly restore a compacted base, checkpoint immediately
  without new input, restore again, and then match against the restored history.
  A single ASOF snapshot round-trip could miss a lost carried base, ever-growing
  dead delta descriptors or a restored output frontier inconsistent with pending
  left rows.
  The public
  [StreamOperator checkpoint contract](../../../crates/calc-flow/src/operator/stream.rs)
  lines 495-498 also prohibits a bulk executor-thread encode during capture.
  - **Suggested fix:** Carry this multi-cycle analogue into ASOF, including an
    identity-only entry and progress-driven eviction. Verify unchanged captures
    reuse owned segment allocations, repeated steady-state epochs release
    obsolete inventory under the chosen compaction scheme, and restore rejects
    an impossible pending/frontier pair before readiness acknowledgement.
    Encoding/compaction belongs in bounded handler work; capture may not hide a
    full state scan merely because the final snapshot is small.
  - **Disposition:** Already required by plan section 3.5, ASOF-03, spec
    FR15/FR16 and AC10/AC11. The immutable payload, encoded capacity, segment
    descriptor and workspace charging table is adequate; implementation must
    supply ownership evidence.

### Minor / Style Notes

- The public trait calls its capture method `checkpoint`, not `snapshot`.
  API note line 122's lifecycle shorthand should use the actual method name in
  final public rustdoc and examples.
- Keep the plan's older “candidate” wording subordinate to the frozen spec/API
  note, as its opening already states. It must not become permission to drop the
  explicit mixed-join or logical-name stream acceptance rows.

## Axis Audit

- **Correctness: OK.** Introduction's immutable Batch and native table-execution
  boundaries are preserved. Rust/Arrow indexing chooses a temporal candidate;
  DataFusion performs typed key equality, bounded left association and output
  projection. No Python matching, array-provider work or executable project
  payload is introduced. See Introduction “The basic vocabulary” and
  “Supported boundaries,” spec FR3-FR9 and its Non-Functional Requirements.
- **Hidden assumptions: OK.** The declaration explicitly requires exact schemas,
  non-null typed identities and replay-stable sequence values. It does not
  require dense sequence values or sorted physical arrival. Runtime duplicate
  validation remains necessary after explicit key overrides and after inner
  Join; neither source inputs nor existing inner Join prove global uniqueness.
- **Missing edge cases: OK with C3.** Empty sides/Batch, unmatched nullable output,
  inclusive zero tolerance, i64 extremes, no initial watermark, idle,
  reactivation, one-sided EOF, corrupt state and sink failure are covered by
  AC1-AC12. C3 identifies a focused analogue needed to make recovery evidence
  stronger than one round-trip.
- **Backwards compatibility: OK.** Separate kind/spec/capability/layout avoids
  tightening the old inner event-time/null rules. The baseline targets project
  format 3 and managed manifest 3, as the current Introduction requires.
  Existing inner defaults, generated subdefinitions, native/project
  fingerprints and symbolic v1/v2 vectors must each remain unchanged; one
  generic serialization test cannot replace that inventory.
- **Performance: OK with C2/C3.** Per-left predecessor selection avoids the old
  all-key-pairs path; candidate rows are bounded by finalization chunk size.
  Reused DataFusion session and controlled compaction limit obvious repeated
  work. The 1000-key/100000-total-row scenario is correctly a resource/failure
  validation task, not a throughput promise. No local throughput run is needed
  for this contract review.
- **Surface and ergonomics: OK with C1.** The new root namespace and fluent
  forms share a declaration and reuse temporal metadata while allowing explicit
  matching keys. Advanced callers retain the full immutable spec. Logical
  input-name watermark mapping, temporary convenience state and source/sink
  delivery limits follow Introduction “Choose how to declare and execute a
  calculation.” Dedicated status strings avoid JavaScript integer rounding
  without changing project safe-integer numbers or old inner metrics.
- **Test plan: OK with C2/C3.** The independent oracle checks values and canonical
  order, not only row counts. Real operator-task, downstream rolling/CS,
  watermark/idle, managed publish-cut and terminal restore tests prevent an
  eager-inner substitute or no-output stub from passing. Each composition row
  requires compile/execution evidence or a stable rejection; the rejection list
  is not a license to drop approved mixed directions.
- **Risk and scope: OK.** Finality and bounded checkpoint ownership are the
  load-bearing work. Stages 00-03 put them before public capability claims.
  Independent event-window branches remain supported while unsupported
  event-window/SQL temporal chains remain rejected. No forward/nearest,
  unbounded lookup, batch extension, editor project or new exactly-once promise
  was added.

## Counter-Proposals

No alternate operator shape is warranted. Retain the independent ASOF operator,
whole-Batch admission, typed predecessor index and bounded DataFusion output
stage. Reusing old inner Join's all-matches execution or identity semantics
would undermine both finality and compatibility.

For diagnostics, use C1's existing-layer precedence instead of adding a new
cross-runtime protocol translation mechanism.

## Questions for the Author

No unanswered user-facing semantic or scope question blocks implementation.
The implementation handoff should name the concrete workspace reservation owner
and checkpoint compaction/capture representation, then attach C2/C3 evidence to
the existing ACs. These are implementation decisions within the approved scope.

## Review Closure

- C1: corrected by the API author and verified against the runtime dispatch
  order; no open contract correction remains.
- C2/C3: retained as implementation/final-review checks under the existing
  acceptance criteria. They do not block starting the approved native work.
- No source changes, builds, tests, remote writes or goal-status changes were
  performed by this critic.
