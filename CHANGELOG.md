# Changelog

This file records fundamental changes to Calc Flow's public surfaces and
engine or Studio capabilities.

## 2026-08

- 2026-08-13: Add crate-private stable operator checkpoint identities and
  explicit stateless, versioned stateful, or unproven capabilities. Stream
  fingerprints now freeze the capability contract, checkpoint admission
  rejects unproven operators before task registration, restore fails closed on
  state-layout version mismatch, and per-output delivery proof records both
  requested and effective guarantees without silently upgrading at-least-once.

- 2026-08-10: Add the crate-private M5 epoch-checkpoint runtime: whole-job and
  per-output capability preflight, globally aligned source/operator cuts,
  manifest-last state publication and strict recovery selection,
  transactional and epoch-idempotent sink completion, mixed ended/live and
  terminal recovery, retention/orphan cleanup, owner-settled cancellation, and
  payload-safe checkpoint status and metrics. Add the 48-case fault matrix,
  scoped M5-D12-E1 benchmark orchestrator, and three-process 20-minute
  checkpoint/restart soak harness. Existing public v2 runner and checkpoint
  APIs, project formats, Python binding, and Studio routes remain unchanged.

- 2026-08-08: Add the public v3 state and window surface: strict bounded
  checkpoint manifests, lineage-exclusive `StateBackend` sessions,
  immutable checksum-verified `LocalStateBackend` segments, deterministic
  compaction, and the stream-only `WindowAggregateOperator` with fixed UTC
  tumbling/hopping windows, ordered aggregates, late-assignment accounting,
  deterministic close output, and incremental Arrow IPC snapshots. The
  source-driven runtime remains crate-private, and barrier coordination,
  durable restart, Python bindings, and Studio contracts remain unchanged.

- 2026-08-08: Add crate-private job-scoped stream progress coordination with
  deterministic source policies, watermark aggregation, idle/reactivation,
  timer ordering, completion receipts, progress snapshots, and replayable
  traces. This capability prepares continuous execution without changing the
  public v2 runners or adding a Python or Studio surface.

- 2026-08-06: Complete the crate-private M2 continuous-runtime skeleton with
  whole-job preflight, bounded source/operator/sink tasks, stable supervision,
  private runner/job/reaper lifecycle, deterministic metrics, stress coverage,
  and the universal 1,200-second soak gate (10-second cadence, 120 samples,
  30-sample/300-second warm-up). Failed or cancelled launch now closes every
  begun resource in stable order with an independent five-second timeout per
  resource; typed cleanup diagnostics preserve the primary outcome. Add the
  non-exhaustive `CalcFlowError::TaskPanicked` variant as this slice's sole
  public API item. Without changing `EdgeBudget` or `edge_channel` signatures,
  `(R, B)` now independently caps queued envelopes and charged rows at `R`, and
  charged bytes at `B`; direct callers must choose
  `R >= max(required_row_limit, required_simultaneous_messages)`. Existing
  public v2 runners remain unchanged until the separately reviewed post-M5 A6
  integration.

- Unify batch memory metering ahead of bounded stream channels. `Batch` and
  `TableBatch` gain `estimated_bytes()`: table batches are charged the Arrow
  memory size of their visible slices (sliced arrays sharing a larger backing
  allocation are charged only their visible window), and `ExternalPayload`
  now requires an exact or conservative `estimated_bytes()` with no opt-out —
  a breaking change for payload implementors. Row and byte sums use checked
  arithmetic and report overflow as a typed `InvalidArgument` error. Python
  NumPy and JAX payloads report their exact visible bytes (views are charged
  their visible window, empty arrays zero); host objects without an `nbytes`
  report receive a logical per-element charge. All charges are logical queue
  occupancy, not process RSS. `BatchingSource` over-limit behavior converges
  on the v3 fail-before-enqueue rule: a single source item exceeding the row
  or byte limit now fails with a latched typed error and the source must be
  reopened, instead of the item being emitted alone as before.

- Split the operator and plan surface by lifecycle — `BatchOperator` /
  `BatchExecutionPlan` for finite one-shot graphs, `StreamOperator` /
  `StreamExecutionPlan` for continuously running graphs — and replace the
  crate-internal opaque runtime markers with the typed v3 stream contract:
  `StreamMessage` (public data plus crate-private watermark, barrier, idle,
  and end-of-input control), strongly typed `EventTime` (signed UTC
  microseconds with checked, uniformly floor-rounded Arrow conversions), and
  `Epoch` (starts at 1, strictly increasing). Stream-rule violations fail at
  compile time before any source opens. This is a breaking Rust API change
  with no v2 compatibility layer; the PyO3 `ExecutionPlan` keeps its name
  over the batch plan, so Python and Studio surfaces are unchanged in this
  step. `docs/runtime-envelope.md` is rewritten as the normative v3
  contract.

## 2026-07

- Add a large, responsive Studio Data Source editor dialog with temporary
  per-source drafts. Inline JSON is validated and applied only on **Confirm**;
  Cancel, close, Escape, and backdrop dismissal discard edits. Keyboard focus
  starts in the editor, remains contained, and returns to the opener. The top
  action toolbar now uses consistently sized controls and intentional
  narrow-screen wrapping. Project, REST, OpenAPI, and engine contracts remain
  unchanged.
- Add frozen native Python `ExecutionOptions(settings, deadline)` for blocking
  and async plan execution. Its constructor remains positional-or-keyword,
  while plans receive it through the keyword-only `options=` parameter.
  Strict settings accept `settings=None` as empty and copy nested mappings in
  one pass with closed, redacted JSON validation. Any valid timezone-aware
  deadline is normalized to UTC with microseconds preserved. An absolute
  deadline continues while a same-plan run waits; queued cancellation remains
  isolated from the active run, and an observed deadline or accepted
  cancellation wins over a later provider error. Deadline expiry raises
  `calc_flow.CancelledError`; asyncio task cancellation remains
  `asyncio.CancelledError`, waits for native cleanup, and linearizes against a
  native result at handler entry. Existing two-argument providers remain
  compatible, while `accepts_context=True` opts into a frozen native
  `ProviderContext`; the public Python API does not expose the native
  cancellation token. These execution-options changes leave project and
  checkpoint formats unchanged.
- Add versioned Studio runtime and preview-worker discovery at
  `GET /api/v2/capabilities`, and model validation, run-state, and table/array
  result responses as closed unions. This is an intentional generated
  TypeScript client source break that requires consumers to regenerate and
  migrate; `/api/v2/catalog` remains a UDF-only compatibility route.
