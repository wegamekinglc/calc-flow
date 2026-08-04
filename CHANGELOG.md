# Changelog

This file records fundamental changes to Calc Flow's public surfaces and
engine or Studio capabilities.

## 2026-08

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
