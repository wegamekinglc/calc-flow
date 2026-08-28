# Changelog

This file records fundamental changes to Calc Flow's public surfaces and
engine or Studio capabilities.

## 2026-08

- 2026-08-28: Add event-time duration frames and the rolling aggregates
  `min`, `max`, `covariance`, and `correlation` to `RollingOperator` across
  Rust, Python, and the regenerated contracts. Aggregate outputs now declare
  either a `rows` frame or a `duration` frame — the exact-width event-time
  interval `(t − micros, t]`, open at the lower bound — with `min_periods`
  capped at the frame size for `rows` frames only; per-entity retention
  keeps the largest row demand and the widest duration bound together, so
  mixed row/duration outputs retain by both bounds. `min`/`max` read a
  monotonic extrema queue over the total-order column types (booleans,
  integers, floats, strings, dates, `timestamp[us]` with no timezone or
  UTC), preserve the input type, and order floating samples by the IEEE
  total order, so −0.0 and +0.0 are distinguishable — a self-consistent
  deterministic choice, not SQL `MIN`/`MAX` semantics.
  `covariance`/`correlation` declare left/right
  input columns and read a reversible West co-moment accumulator over
  pairwise-valid positions; their classification order is frozen — the
  `min_periods` and divisor null gates first, any ±inf on either side reads
  NaN, and only a finite zero-variance window reads null for correlation —
  so an infinity window never reads null. Checkpoint segments still store
  only retained history and buffered rows, with the extrema queues and pair
  state rebuilt by the same ordered fold on restore. The Python constructors
  `ts.min`, `ts.max`, `ts.covariance`, and `ts.correlation` and
  duration-frame windows lower into rolling nodes; the JSON Schema, Studio
  OpenAPI document, and TypeScript contract are regenerated in the same
  commit.

- 2026-08-28: Reject wall-clock built-in functions in stream query
  compilation (DAL-166). `compile_stream` now rejects read-only expression
  and SQL queries that call `now`, `current_date`, or `current_time`:
  DataFusion declares these built-ins `Volatility::Stable`, so the
  volatile-function guard let them compile while checkpoint replays of the
  same input produced different output, breaking the replay-safety the
  guard exists to protect. The check resolves each call against the
  built-in default function registry and matches the canonical name, so
  the `current_timestamp` and `today` aliases and the no-parenthesis
  keyword spellings are rejected too, with the same error shape as the
  volatile rejection. This is a user-visible tightening: existing stream
  queries that call these functions now fail at compile time. Batch query
  compilation is unchanged.

- 2026-08-26: Make epoch checkpoint capture cost proportional to the dirty
  set, not the retained state (DAL-160). `OperatorStateSnapshot.segments`
  now maps segment IDs to the new public `StateSegment` type — an
  allocation-shared (`Arc`) byte buffer with a SHA-256 computed once at
  construction — replacing `BTreeMap<String, Vec<u8>>` in a breaking Rust
  API change; custom `StreamOperator` checkpoint implementations and
  snapshot consumers must wrap segment bytes in `StateSegment::new`. The
  stream Join encodes dirty upserts from the records its dirty log carries
  at admission instead of scanning live state per dirty row, eliminating a
  quadratic capture path that reached 45–78.5 s per epoch at ~1.04M
  retained rows, and carries prepared base/delta segments across epochs by
  sharing allocations rather than copying bytes. The manifest transaction
  reuses the already-committed handle for a segment whose content is
  unchanged since the current session committed it, skipping the per-epoch
  re-hash, re-write, and re-validation of carried state; recovery-time
  manifest selection still revalidates every referenced segment byte on
  every fresh session (AC-08), while same-session publish and retention
  skip that redundant re-read. Manifests may now reference handles
  committed at earlier epochs, and recovery loads validate handle ownership
  without requiring one shared epoch. A resident `stream_join_perf`
  Criterion suite preserves the frozen AC19 join baseline (nine scenarios
  plus a capture-independence canary pair) for future 0.80×/1.20× gate
  comparisons; the scheduled nightly workflow stays contract-frozen to the
  `core` harness.
- 2026-08-27: Add the native row-window rolling operator for lag and delta
  across Rust, Python, and the generated contracts. `RollingOperator`
  evaluates ordered `lag`/`delta` outputs over entity-partitioned, event-time
  rows in both batch and final-only stream graphs: stream evaluation buffers
  rows until the input watermark passes each row's closing coordinate, then
  emits final rows in canonical order, classifying late rows with an
  envelope-scoped `error` policy or a metrics-recorded `drop` policy; its
  per-entity histories checkpoint at the aligned epoch cut with state version
  1, and batch and stream produce the same ordered rows across segmentation
  and recovery. The data-only `RollingSpec` joins project format v3 with nine
  required fields — `configuration_version`/`state_layout_version` 1, ordered
  `partition_by`/`sequence_by` keys, a non-null UTC `timestamp[us]`
  `event_time` column, `outputs` (kind `lag` or `delta`,
  `primitive_version` 1, input/output names, positive `periods`),
  `allowed_lateness_micros`, `late_policy`, and the frozen
  `stateful_numeric_v1` value policy — with the JSON Schema, Studio OpenAPI
  document, and TypeScript contract regenerated in the same commit.
  `Program.compile_batch` and `Program.compile_stream` now lower
  `ts.lag`/`ts.delta` declarations into rolling nodes, requiring declared
  `entity_by`/`event_time`/`sequence_by` ordering keys and, in this release,
  plain input-column arguments; `compile_stream`'s validated
  `allowed_lateness_micros` and `late_policy` are written into every lowered
  node. The capability snapshot reports `rolling@1` as a batch/stream,
  per-row-final, micro-batch-invariant, checkpointed-stateful operator with
  state version 1. Rolling aggregates, duration frames, correlation, and
  cross-section declarations remain loudly rejected.
- 2026-08-28: Add the native rolling aggregates `count`, `sum`, `mean`,
  `variance`, and `stddev` to `RollingOperator` across Rust, Python, and the
  regenerated contracts. Each aggregate output declares a `rows` frame and
  `min_periods`, with `ddof` 0 or 1 on the two statistical kinds; valid
  samples are non-null, non-NaN values, infinities count as numeric samples,
  and the minimum-period gate reads null so a computed NaN stays distinct.
  Output types follow the frozen table: `uint64` count, checked `int64` /
  `uint64` integer sums, and `float64` for floating sums and every
  statistical output. Floating sums follow IEEE arithmetic; integer sums
  slide through a wide transient accumulator with a checked narrowing at
  readout, so a window whose true sum is representable never reports a false
  overflow while genuine integer overflow fails the run loudly. Mean,
  variance, and standard deviation read a reversible West accumulator that
  also counts the window's infinities: a mean over one infinity sign is
  that infinity, over both signs it is NaN, and variance and standard
  deviation over any infinity window are NaN behind the minimum-period and
  divisor null gates — the ±inf readout semantics were adjudicated in
  `.codex/artifacts/critiques/sce-07-rolling-inf-mean.md`. Outputs over the
  same input column and frame share one accumulator group, and the
  checkpoint segment still stores only history and buffered rows, with
  every accumulator rebuilt by the same ordered fold on restore. The Python
  constructors `ts.count`, `ts.sum`, `ts.mean`, `ts.variance`, and
  `ts.stddev` lower into rolling nodes; `ts.min`, `ts.max`, correlation,
  and duration frames remain rejected.

- 2026-08-26: Compile symbolic row-local programs into execution plans.
  `Program.compile_batch(runtime)` and `Program.compile_stream(runtime, *,
  allowed_lateness_micros=0, late_policy="error")` lower literal, field,
  arithmetic, comparison, boolean, `where`, `coalesce`,
  `log`/`exp`/`sqrt`/`abs`/`clip`/`cast`, and
  `select`/`filter`/`with_columns` declarations into strict project-v3
  `expression` nodes. Row-local chains fuse into one node per program
  output, and structurally shared non-trivial subexpressions materialize
  exactly once through deterministically named CSE tiers. Node IDs and plan
  fingerprints are deterministic, the lowered project carries strict JSON
  only, the Rust graph compiler keeps final validation, and no symbolic
  Python runs while a compiled plan executes. Declarations outside the
  row-local lowerer fail loudly at compile time. Stream compilation now
  also rejects read-only `expression`/`sql` queries that call volatile
  built-in functions such as `random()`, resolved against the built-in
  default function registry, so the deterministic and replay-safe
  capability claims of those operators stay truthful.

- 2026-08-26: Add the complete Python publication boundary: a cross-platform
  local package builder, five-platform abi3 matrix and metadata verification,
  SHA-256 release manifests, annotated-tag and PyPI immutability checks, and
  separate OIDC Trusted Publisher jobs for `calc-flow` and
  `calc-flow-studio`. Studio now publishes a nonempty package README, and its
  frontend helper modules use case-distinct filenames so release builds also
  work on Windows.

- 2026-08-25: Extend the runtime capability snapshot to lifecycle-aware
  schema version 2 across Python and Studio. `OperatorCapability` and
  `ProviderCapability` report execution modes, output finality, statefulness,
  micro-batch invariance, watermark requirement, checkpoint support with a
  positive state version exactly when checkpointed state is declared,
  determinism, and replay safety; providers additionally report static-input
  support, a partition contract, and closed `CapabilityRule` /
  `ProviderArrayRules` identities. The vocabularies and rule identities are
  closed and validated fail-closed at construction, in the Studio `/api/v3`
  response models, and in the browser decoder: unknown identities, forged
  lifecycle keys on registration records, and inconsistent state claims are
  rejected. Existing provider registrations stay source compatible and
  report batch-only conservative values until an approved stream proof
  exists. `GET /api/v3/capabilities` serves `schemaVersion` 2 with the
  nested runtime version removed; clients consuming the version 1 shape must
  regenerate their contract.

- 2026-08-24: Add the public symbolic declaration surface
  `calc_flow.symbolic`. Typed immutable column, table, array, and parameter
  expressions carry canonical v1 digests; an ordered `FeatureSet` of uniquely
  named features feeds `TableExpr.with_columns` as derived columns; and an
  immutable `Program` over declared inputs and outputs carries a
  runtime-independent v1 fingerprint that conforming implementations reproduce
  byte-for-byte. A complete program verifies, fingerprints, and explains with
  no data or runner: `Program.analyze(runtime, mode=...)` and
  `Program.explain(runtime, mode=...)` take one explicit runtime capability
  snapshot and statically infer value types, domains and lineages, symbolic
  dimensions, attachment compatibility, state requirements, and stream safety,
  reporting immutable `AnalysisIssue` findings on stable output- or
  input-rooted paths. Type inference proves only what the capability snapshot
  proves, so cross-type arithmetic requires an explicit `row.cast`. The
  package has no data execution path; execution remains owned by the existing
  execution plans and runners.

- 2026-08-22: Release the 4.0 bounded event-time stream Join across Rust,
  Python, project v3, and Studio. The inner equi-Join has explicit asymmetric
  time bounds, required state and match limits, deterministic recovery, typed
  failure reasons, and independently checkpointed output-watermark progress.

- 2026-08-21: Add a maintained cross-language example runner, Python terminal
  checkpoint recovery and Rust event-time window examples, a complete
  executable-example inventory, a continuous streaming usage guide, and a
  component-level design and architecture guide. Reconcile current docs with
  the public project-v3, connector, recovery, and Studio `/api/v3/jobs`
  surfaces.

- 2026-08-21: Close every P0 finding from the 2026-08-19 audit. The canonical
  Rust/PyO3/Python project contract is v3, registered connectors own their
  deferred source/sink lifecycles, PostgreSQL CDC and delivery preflight are
  complete, Studio jobs own the native continuous runtime, and the 3.0 release
  and verification gates are restored. This supersedes the temporary M6/M7
  withdrawal below.

- 2026-08-19: Withdraw M6/M7 acceptance pending the
  [current-main audit](docs/superpowers/audits/2026-08-19-continuous-streaming-v3-current-main.md).
  The 2026-08-18 entries record merged, release-labelled slices, not completed
  milestone gates: the canonical project/PyO3/Python surface is still v2,
  registered connector lifecycles and database protocols are incomplete, and
  Studio jobs do not yet own the native continuous runtime. Repair the 3.0
  release trigger and release-helper unit-test coverage, but do not publish
  3.0 until every P0 audit finding and the exact-head M7 evidence are closed.

- 2026-08-18: Complete the M6 connector milestone. The
  `calc-flow-connectors` workspace crate ships six transports behind
  feature gates — file/Parquet (default), Kafka, PostgreSQL,
  ClickHouse, and HTTP/WebSocket — each registering through the
  trusted `ConnectorRegistry` with the eight-axis capability
  vocabulary. Project format v3 introduces `runtime.mode`,
  connector-bound source/sink/database bindings, `SecretRef`
  credentials, and `StateConfig`; `schemas/project-v3.schema.json` is
  the canonical contract and the v2 schema artifact is removed. The
  Python surface exposes `registered_connectors()` and the
  `ConnectorCapability` types; Studio serves the `/api/v3` continuous
  job API with checkpoint, shutdown, cancel, SSE events, and the
  `ResourceLimits` endpoint replacing the v2 worker timeout.
- 2026-08-18: Add the M7 release gates.
  `scripts/verify_perf_gates.py` evaluates paired Python statistics and Rust
  Criterion confidence intervals at the 5% regression threshold and documents
  the opt-in 1,200-second soak
  procedures; `scripts/verify_security_gates.py` publishes the
  18-entry threat-model coverage checklist linking each security
  boundary to its named enforcement evidence.

- 2026-08-15: Harden the public continuous-runtime delivery. Python blocking
  start now reclaims its dedicated event-loop thread after blocking or async
  terminal operations, observer cancellation during cleanup, and abandoned-job
  GC; a native terminal outcome that has already linearized retains precedence.
  Studio POSIX checkpoint replace and unlink paths fsync the parent directory.
  Public diagnostics fail closed and redact non-portable source IDs. The
  48-case fault matrix and three-process soak now run through the crate-root
  public facade and read real fault, cancellation, checkpoint-failure, and
  registry probes, while a real POSIX directory-sync failure test strengthens
  indeterminate manifest-publication evidence.

- 2026-08-14: Complete the A6 continuous-runtime cutover. Rust now exports the
  source-driven connector, binding, managed-checkpoint, one-shot
  `StreamingRunner`, and owning `StreamingJob` surface at the crate root.
  Python exposes that lifecycle as the sole `StreamingRunner`. The v2
  `Source`/`Sink`, `MicroBatchRunner`, push runner, public checkpoint store,
  and their PyO3 compatibility classes are removed with no aliases. Studio's
  unchanged v2 checkpoint inspection routes use a private async document
  store; project format v2, checkpoint manifests v3, REST, and OpenAPI remain
  unchanged.

- 2026-08-13: Add crate-private stable operator checkpoint identities and
  explicit stateless, versioned stateful, or unproven capabilities. Stream
  fingerprints now freeze the capability contract, checkpoint admission
  rejects unproven operators before task registration, restore fails closed on
  state-layout version mismatch, and per-output delivery proof records both
  requested and effective guarantees without silently upgrading at-least-once.

- 2026-08-12: Preserve full-range late-window accounting when the watermark
  and window end span more than `i64::MAX` microseconds. Valid `EventTime`
  inputs now record the exact `u64` lateness metric instead of aborting the
  input batch with a signed-subtraction overflow.

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
