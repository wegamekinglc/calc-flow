# Continuous Streaming Runtime - API Note

Artifact slug: `continuous-streaming-runtime` (shared with the M0.1
specification and the M0.2 critique; this note freezes names, signatures,
ownership, error variants, JSON layouts, and packaging decisions without
contradicting the specification).

## Sources and status

- Semantics:
  [`.codex/artifacts/specs/continuous-streaming-runtime.md`](../specs/continuous-streaming-runtime.md)
  (authoritative; cited as "spec" with its D1-D9, S1-S10, I1-I10, NG1-NG13,
  FR/NFR identifiers).
- Plan task: M0.2 of
  [`docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`](../../../docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md)
  (cited as "plan"; its M0.2 checkbox list is covered end to end in Appendix A).
- Current surfaces this replaces per plan section 1.3:
  `crates/calc-flow/src/{lib.rs,operator.rs,io.rs,checkpoint.rs}`,
  `crates/calc-flow-python/src/`, `python/calc_flow/{runtime,pipeline,store}.py`,
  `python/calc_flow/_native.pyi`, `web-ui/backend/src/calc_flow_studio/`,
  `schemas/project-v2.schema.json`.
- Status: proposed public surface for Calc-Flow 3.0. Every section below is a
  numbered, quotable decision (A1-A8, B1-B5, C1-C3, D1-D3, E1-E5, F1, G1-G3).
  No TBDs. Where the plan's sketches were explicitly non-binding ("目标形态，
  不是预先冻结的最终签名"), deviations are stated and justified inline.

## Audiences

- Rust users: new `BatchOperator`/`StreamOperator` traits, runner/job
  lifecycle, source/sink bindings, state/checkpoint types, error variants.
- Python users: replaced `calc_flow.runtime` module (async-first runner/job,
  source/sink protocols), `_native.pyi` additions, exception behavior.
- Studio clients: `/api/v3` routes, request/response models, and the SSE event
  model consumed by generated TypeScript types.

## Naming map (v2 replacement table, implements plan 1.3 and NG9)

| v2 name (removed, no alias)                   | v3 replacement                                                      |
| --------------------------------------------- | ------------------------------------------------------------------- |
| `Operator`                                    | `OperatorMetadata` + `BatchOperator` (A1) and `StreamOperator` (A2) |
| `OperatorContext`                             | `BatchOperatorContext` (A1)                                         |
| `ExecutionPlan`                               | `BatchExecutionPlan` / `StreamExecutionPlan` (A1.2)                 |
| `PipelineBuilder::compile()`                  | `compile_batch()` / `compile_stream()` (A1.2)                       |
| `Source`, `SourceItem`, `BatchingSource`      | `StreamSource`, `SourceEvent`, `SourceBinding` (A4)                 |
| `Sink`, `SinkRouter`                          | `StreamSink`, `TransactionalSink`, `SinkBinding` (A5)               |
| `MicroBatchRunner`                            | bounded-source stream run on `StreamingRunner` (A6)                 |
| `StreamingRunner` (push `step()`)             | `StreamingRunner` (source-driven) + `StreamingJob` (A6)             |
| `Checkpoint`, `CHECKPOINT_FORMAT_VERSION = 2` | `CheckpointManifest`, `MANIFEST_FORMAT_VERSION = 3` (G3)            |
| `ExternalOperatorFactory`                     | `BatchOperatorFactory` / `StreamOperatorFactory` (A1.3)             |
| `SignalAwareOperator` (crate-private)         | deleted; covered by `StreamOperator` (A2)                           |
| project `format_version: 2`                   | project `format_version: 3` (C1)                                    |
| Studio `/api/v2`                              | Studio `/api/v3` (D1)                                               |

**Same-name replacements (louder than a rename, so listed explicitly).**
`CheckpointStore` keeps its v2 trait name with a new manifest-oriented
method set (A7); v2's document-oriented methods are gone, not overloaded.
`StreamSink::write` drops v2 `Sink::write`'s `context: &RunContext`
parameter (job context reaches the sink through its lifecycle, not per
batch). `StreamingRunner` keeps the v2 name with the source-driven
lifecycle (A6).

**Catch-all.** Every v2 public name not listed in the replacement table or
the same-name list above keeps its v2 meaning and disposition: `Port`,
`Batch`, `BatchKind`, `BatchMetadata`, `TableBatch`, `ExternalPayload`,
`UdfReference`, `UdfRegistry`, `UdfRegistrySnapshot`, `ProviderRegistry`,
`RunContext`, `CancellationToken`, `ExecutionOptions`, `RunResult`,
`RunMetadata`, `NodeTiming`, `DataFusionRuntime`, `DataFusionConfig`,
`DataFusionQueryMetric`, `compile_project`, `validate_project`,
`canonical_json`, `JsonMap`, `ExternalOperatorSpec`,
`validate_selected_udfs`, `FileProjectStore`, the `MAX_*` constants, and
the `CalcFlowError` enum (additions only, A8). The
`ProjectSpec`/`PipelineSpec`/`NodeSpec` model family is re-derived for
format v3 (C1/C3). `OperatorDefinition` is replaced by the batch/stream
factory split (A1.3).

---

## A. Rust public types

### A1. `OperatorMetadata` supertrait and `BatchOperator` (implements plan 2.2; constrained by I3, I9)

```rust
/// Metadata every compiled node exposes to the graph compiler.
/// Method signatures are byte-identical to v2 `Operator`.
pub trait OperatorMetadata: Send + Sync {
    fn name(&self) -> &str;
    fn input_ports(&self) -> &[Port];
    fn output_ports(&self) -> &[Port];
    fn configuration(&self) -> JsonMap;
    fn udf_references(&self) -> Vec<UdfReference> {
        Vec::new()
    }
}

#[async_trait]
pub trait BatchOperator: OperatorMetadata {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>>;

    fn snapshot(&self) -> Result<Value> {
        Ok(Value::Null)
    }

    fn restore(&mut self, state: &Value) -> Result<()> {
        if state.is_null() {
            Ok(())
        } else {
            Err(CalcFlowError::Format {
                message: "stateless operator state must be null".into(),
            })
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.restore(&Value::Null)
    }
}

pub struct BatchOperatorContext<'a> {
    pub run: &'a RunContext,
}
```

**Source compatibility of existing `Operator` implementations.** v3 removes
the `Operator` name (plan 1.3: replacement, not aliasing). A v2 implementation
ports mechanically with zero behavioral change:

1. Rename `impl Operator for T` to `impl BatchOperator for T`.
2. Move the five metadata methods (`name`, `input_ports`, `output_ports`,
   `configuration`, `udf_references`) into a new `impl OperatorMetadata for T`
   block; signatures and bodies are unchanged.
3. `process`, `snapshot`, `restore`, `reset` keep their signatures; the
   context type is renamed `BatchOperatorContext` with the same
   `run: &RunContext` field.

No other edits are required: port/kind/schema validation, JSON configuration,
and UDF reference shapes are untouched. The compiler depends only on
`OperatorMetadata`, so `compile_batch()` and `compile_stream()` share port,
schema, one-writer, cycle, UDF, topology, and fingerprint validation
(plan M1.1).

**Why this shape.** A supertrait (not duplicated method sets on two traits)
guarantees the two compilers cannot drift on metadata semantics; a supertrait
(not a free struct) lets external operators keep computing configuration
lazily, as v2 providers do. Rejected alternative: keep one `Operator` trait
with a `mode()` discriminator - it would re-mix the two lifecycles that plan
1.3 requires separating, and the `process` signatures genuinely differ
(whole-map vs per-ingress).

#### A1.2 Plans and compilers

```rust
impl PipelineBuilder {
    pub fn compile_batch(
        self,
        udfs: &UdfRegistrySnapshot,
    ) -> Result<BatchExecutionPlan>;

    pub fn compile_stream(
        self,
        udfs: &UdfRegistrySnapshot,
        requirements: &StreamRequirements,
    ) -> Result<StreamExecutionPlan>;
}

/// Delivery requests recorded into the compiled stream plan.
#[derive(Clone, Debug, Default)]
pub struct StreamRequirements {
    /// Per graph-output delivery request; outputs absent from the map
    /// default to `DeliveryGuarantee::AtLeastOnce`.
    pub delivery: BTreeMap<String, DeliveryGuarantee>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryGuarantee {
    AtLeastOnce,
    ExactlyOnce,
}
```

- `BatchExecutionPlan` keeps the v2 `ExecutionPlan` surface renamed:
  `name()`, `fingerprint()`, `execute(inputs, ExecutionOptions) ->
  Result<RunResult>`, `snapshot/restore/reset`, and the exclusive runner
  lease. Batch-mode requested delivery does not exist; batch runs are
  one-shot with v2's at-least-once runner contract preserved through bounded
  stream runs (plan 1.3).
- `StreamExecutionPlan` exposes `name()`, `fingerprint()` (semantic
  fingerprint per NFR-5), `runtime_config_hash()`, `requirements()`,
  `source_binding_ids()`, and `sink_binding_ids()` (the stable binding slots
  of plan M1.1). It never executes directly; a `StreamingRunner` owns
  execution (A6).
- `compile_stream()` enforces, before any source opens: unary expression,
  single-alias SQL, explicit stream providers, union, and window operators
  only (plan 2.2 matrix); multi-alias SQL fails (NG6); volatile UDFs in an
  exactly-once plan fail with the node path (S9.3); window/session/early/
  allowed-lateness configuration fails (NG4, D8.4); edge budget
  pre-validation runs when source batch bounds are known (project path,
  A4.1).

**Why one `compile_stream` with a `StreamRequirements` parameter rather than a
default-arg convenience overload:** one way to do it. `StreamRequirements` is
a data-only struct with `Default` (all outputs at-least-once), so the common
call is `compile_stream(&udfs, &StreamRequirements::default())`.

#### A1.3 Factories and provider registry

```rust
pub trait BatchOperatorFactory: Send + Sync {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>>;
}

pub trait StreamOperatorFactory: Send + Sync {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn StreamOperator>>;
}
```

`ProviderRegistry` gains `register_batch` / `resolve_batch` and
`register_stream` / `resolve_stream` over the same
`(provider, name, version)` key space, with atomic duplicate rejection per
(identity, mode) pair. A provider may implement one or both modes;
`compile_stream()` rejects a spec whose provider offers no stream factory
(plan 2.2: "factory 必须显式构造 StreamOperator"). Rejected alternative: one
factory trait with both `create_*` methods defaulting to an error - two
traits make an unsupported mode a compile-time absence instead of a runtime
default-error, and keep the registry queryable for Studio capabilities.

### A2. `StreamOperator`, `StreamOperatorContext`, and `StreamCollector` (implements plan 2.2, M2.3; constrained by S1.3, S5.2, S7.3)

```rust
#[async_trait]
pub trait StreamOperator: OperatorMetadata {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()>;

    /// Synchronously captures dirty state for `epoch`. See A2.2 for why this
    /// is synchronous and for the capture-cost rule.
    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let _ = epoch;
        Ok(OperatorStateSnapshot::default())
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            Ok(())
        } else {
            Err(CalcFlowError::Format {
                message: "stateless operator state must be empty".into(),
            })
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct StreamOperatorContext<'a> {
    job: &'a StreamJobContext,
    operator_id: &'a str,
    input_watermark: Option<EventTime>,
}

impl StreamOperatorContext<'_> {
    pub fn job(&self) -> &StreamJobContext;
    pub fn operator_id(&self) -> &str;
    /// Current input watermark `WM_in`; `None` while undefined (S5.2).
    pub fn input_watermark(&self) -> Option<EventTime>;
    pub fn check_cancelled(&self) -> Result<()>;
    /// Records dropped row-window assignments for the batch currently being
    /// processed (D2.5). The runtime derives `affected_batches` (any call
    /// with `dropped > 0`) and the running maximum lateness; row payloads
    /// are never accepted. Only window operators call this.
    pub fn record_late_rows(
        &self,
        dropped: u64,
        max_lateness: Option<Duration>,
    ) -> Result<()>;
}

/// The only way an operator emits data. Construction is crate-private;
/// control messages can never be emitted through it (S1.3).
#[async_trait]
pub trait StreamCollector: Send {
    /// Validates port name, batch kind, and optional exact schema, then
    /// enqueues onto the port's edge channel, awaiting capacity under the
    /// edge budget (S10.1).
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()>;
}
```

#### A2.1 `StreamCollector` object safety (M0.2 checkbox: prove it)

`StreamCollector` is used exclusively as `&mut dyn StreamCollector`. It is
object-safe because:

1. No generic type parameters on the trait or any method.
2. No associated types, no associated constants.
3. No method takes `self` by value, returns `Self`, or uses `Self` in an
   argument position outside the receiver.
4. The only supertrait is `Send`, which has no methods; `Send`/`Sync` bounds
   are permitted on trait objects (`dyn StreamCollector` is `Send` because
   the trait requires it).
5. With `#[async_trait]` (the pattern already used by v2 `Operator`,
   `Source`, and `Sink` behind `Box<dyn ...>`), `emit` desugars to:

   ```rust
   fn emit<'life0, 'life1, 'async_trait>(
       &'life0 mut self,
       port: &'life1 str,
       batch: Batch,
   ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'async_trait>>
   where
       'life0: 'async_trait,
       'life1: 'async_trait;
   ```

   Late-bound lifetimes do not affect object safety; the return type is a
   concrete boxed future. A vtable for `dyn StreamCollector` therefore
   contains exactly one entry.

`Sync` is deliberately not required: the operator task holds
`&mut dyn StreamCollector` with exclusive access; requiring `Sync` would
forbid implementations holding a `RefCell`-style scratch buffer for no
benefit.

**Error propagation through the collector (M2.3 RED lines).** `emit` fails
before any downstream observation with `CalcFlowError::Compile` when the port
is unknown, the kind mismatches, or the exact schema mismatches; with
`CalcFlowError::EdgeClosed` (A8) when the edge receiver is closed during job
convergence; with `CalcFlowError::InvalidArgument` when a single batch
exceeds the edge byte budget (S10.3). Operators propagate with `?`; a failed
handler never forwards a partial control event because control forwarding is
runtime-owned (S1.3) and happens only after the handler returns `Ok`.

**Why async `emit` and not a synchronous staging buffer.** Backpressure must
propagate sink -> operator -> source (S10.4, plan 2.1). An async `emit` parks
the operator task at the edge budget without blocking a Tokio worker (I6). A
sync staging buffer drained between handler calls would create an unbounded
in-memory staging area inside the operator (a window close can emit many
windows), violating NFR-3's steady-state bound, and would delay backpressure
to handler granularity.

#### A2.2 Deviation from the plan's sketch: synchronous `checkpoint`/`restore`

The plan's section 2.2 sketch (explicitly non-binding) shows
`async fn checkpoint` / `async fn restore`. This note freezes **synchronous**
signatures. Rationale: the snapshot is an in-memory capture of dirty state;
durable staging, validation, and manifest publication are runtime-owned
(D4.1) and run off the alignment path on blocking pools (I6). An async
snapshot signature would invite implementors to perform I/O on the barrier
path, directly attacking the D3.4/D7 latency budget. `restore` is synchronous
for symmetry: segment bytes are delivered to the operator already loaded by
the runtime (A3). This is a signature-level decision this note owns (spec
section 9); it does not change any frozen semantics.

**Capture-cost rule (I6's intent, stated explicitly).** `checkpoint()` MUST
be O(dirty-key metadata), not O(dirty-key bytes): the built-in window
operator maintains its segment buffers pre-encoded (Arrow IPC written
incrementally as accumulators mutate), so the synchronous capture is a
cheap handoff of already-formed buffers, never a bulk encode on the
executor thread. Any residual encoding runs on the blocking pool with the
operator task's ingresses paused (they are already blocked at alignment,
S7.3). M4.4 RED line: with paused Tokio time and a large dirty set, the
operator task processes a following control message within a bounded number
of executor steps.

#### A2.3 `OperatorStateSnapshot`

```rust
/// Operator-private state captured at one epoch. The runtime wraps it with
/// input progress and segment handles into the manifest entry (G3).
#[derive(Clone, Debug, Default)]
pub struct OperatorStateSnapshot {
    /// Small bounded JSON placed inline in the manifest (validated for
    /// depth/size; D4.4). No keyed row state, no payloads, no secrets (I4).
    pub inline_metadata: JsonMap,
    /// `segment_id -> bytes`. Segment IDs are portable identifiers; the
    /// runtime assigns paths, lengths, and checksums (D4.1). The built-in
    /// window operator encodes Arrow IPC (D6); custom operators own their
    /// byte format, which the segment envelope's length/SHA-256 validation
    /// covers opaquely (D6 rationale).
    pub segments: BTreeMap<String, Vec<u8>>,
}
```

`checkpoint` on a stateless operator returns the default (empty) snapshot;
the default `restore` rejects a non-empty snapshot with
`CalcFlowError::Format` (mirrors the v2 stateless-reject rule); the default
`reset` returns `Ok(())`.

#### A2.4 Built-in stream operators

`ExpressionOperator` and single-alias `SqlOperator` implement both
`BatchOperator` and `StreamOperator` (per-batch stateless processing; plan
2.2 matrix). The two new stream-only operators:

```rust
/// Multi-ingress forwarding operator (S4). At least two input ports, all
/// of one kind and one exact schema (or all schema-less); the single
/// `output` port carries the same kind and schema. Stream-only: batch mode
/// already composes multi-input logic through SQL aliases.
pub struct UnionOperator { /* ... */ }

impl UnionOperator {
    pub fn new(name: &str, input_ports: Vec<Port>) -> Result<Self>;
}

/// Final-only tumbling/hopping aggregate (D8, S6). Stream-only.
pub struct WindowAggregateOperator { /* ... */ }

#[derive(Clone, Debug)]
pub struct WindowSpec {
    pub event_time_column: String,
    pub group_by: Vec<String>,
    pub geometry: WindowGeometry,
    pub aggregates: Vec<AggregateSpec>,
}

#[derive(Clone, Copy, Debug)]
pub enum WindowGeometry {
    /// `slide == size` (D8.4 special case).
    Tumbling { size: Duration },
    /// `size = m x slide` for an integer `m >= 1`, checked exactly (D8.4).
    Hopping { size: Duration, slide: Duration },
}

#[derive(Clone, Debug)]
pub struct AggregateSpec {
    pub function: AggregateFunction,
    pub column: String,
    pub output: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateFunction {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

impl WindowSpec {
    /// Durations must be exact multiples of one microsecond and greater
    /// than zero; violations name the field. Output rows carry
    /// `window_start`, `window_end`, the group keys, and the named
    /// aggregates in deterministic schema order (S6.4).
    pub fn tumbling(event_time_column: &str, size: Duration) -> Self;
    pub fn hopping(event_time_column: &str, size: Duration, slide: Duration) -> Self;
    pub fn group_by<I, S>(self, columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>;
    pub fn aggregate(self, function: AggregateFunction, column: &str, output: &str) -> Result<Self>;
}

impl WindowAggregateOperator {
    /// Validates the spec, derives the output schema, and rejects
    /// output-name collisions with window/group columns (plan M4.3).
    pub fn new(name: &str, spec: WindowSpec) -> Result<Self>;
}
```

The supported aggregate type matrix, and null/NaN/overflow/decimal/avg
semantics, are M4.3's to freeze (spec section 9); the shapes above are
closed under whatever matrix M4.3 selects because the type matrix constrains
column types, not these signatures.

### A3. Stream-operator state entry semantics: `reset()` / restart / recovery (implements D4, D5, S7.3, S8.3; constrained by S5.2, S6.3, D2.5)

Three and only three entry points bring an operator instance to a runnable
state; each runs inside the operator task owned by the job supervisor (D5.1),
completes before the task opens its ingresses, and before any source opens
(fail-closed ordering, S2.5 analogue).

| Entry point | When (exactly once per job, per operator)                                                   | Receives                                                                                                                                                                              | Guarantees on `Ok` return                                                                                                                                                                                                          | On `Err`                                                                         |
| ----------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `reset()`   | Job starts a **new lineage** (no manifest selected) with a plan-owned, reused instance      | Nothing                                                                                                                                                                               | Operator state equals a freshly constructed instance: no keyed state, no open windows, no held batches; runtime separately clears input progress so `WM_in` is undefined (S5.2) and late metrics start at zero for the new lineage | Job converges to `failed` before any source opens (S8.3: logic class)            |
| Restart     | Same as `reset()` when the instance is freshly constructed (first job on a plan)            | Nothing (no call)                                                                                                                                                                     | Same as `reset()`; construction is the entry point                                                                                                                                                                                 | n/a (construction failures surface at `start()`)                                 |
| Recovery    | Job starts from the latest valid manifest at epoch `R` (S8.3: recovery is always a new job) | `&OperatorStateSnapshot` reassembled exclusively from segments referenced by that manifest (D4.2), plus runtime-restored input progress delivered through subsequent watermark events | For the replayed input prefix, all downstream-observable results converge to the pre-failure lineage's values (S6.3); watermarks and late metrics do not regress and do not double-count committed values (D2.5)                   | Job converges to `recovery-required` before any source opens (infra class, S8.3) |

Mapping to the frozen decisions:

- **D5 (task ownership).** All three entries execute inside the registered
  operator task before it is first polled on data; `start()` does not return
  until every entry has succeeded or the job has converged. No entry spawns
  work outside the supervisor registry (I5).
- **D4 (commit order).** Recovery reads manifests only and reconstructs state
  exclusively from manifest-referenced, validated segments (D4.2); orphaned
  segments never reach `restore` (their presence must not fail recovery).
  `checkpoint(epoch)` output is staged and validated per D4.1 before the
  manifest references it; the operator itself never touches the filesystem.
- **S7.3.** `checkpoint(epoch)` is invoked only after full barrier alignment
  for `epoch`; its success precedes barrier forwarding (F1); its failure
  fails the job without forwarding.
- **Idempotence.** `reset()` is safe to call repeatedly with identical
  effect. `restore` applied twice with the same snapshot converges to the
  same state (implementations replace, not merge, held state).
- **Lineage context.** `checkpoint` receives the epoch for diagnostics and
  segment tagging; `restore` and `reset` deliberately receive no lineage
  context. A custom operator never distinguishes "restored at R" from
  "fresh": `restore` replaces held state wholesale from the snapshot, and
  replay convergence (S6.3) is a property of state plus replayed input, not
  of epoch bookkeeping inside the operator.

### A4. `StreamSource`, `SourceEvent`, capabilities, and `SourceBinding` (implements D3, S2, S5.6, S9; constrained by I3, I4)

```rust
/// Source-defined replay position with a runtime-comparable ordering key.
///
/// `order` is a connector-encoded, lexicographically ordered byte key:
/// bytewise comparison MUST equal the source's own position order.
/// Connectors build it with the G1 order-preserving encoding (PostgreSQL
/// LSN as a big-endian `u64`; Kafka `(partition, offset)` as a G1
/// composite; file identity plus row-group ordinal likewise). The runtime
/// compares `order` bytewise and fails closed on repeat or regression
/// before enqueue (S2.2, S5.4). `order` MUST be non-empty and is bounded
/// (16 KiB) so the manifest stays bounded (D4.4).
///
/// `payload` is the connector's opaque, human-debuggable JSON form; it is
/// stored in the manifest and handed back to `open` unchanged. It carries
/// no secrets (I4) and is depth/size validated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cursor {
    pub order: Vec<u8>,
    pub payload: JsonMap,
}
```

**Spec-side wording this contract requires (for the spec owner's concurrent
revision, stated generically so S2.2/S5.4 do not name this Rust type).**
S2.2's monotonicity rule should read over the ordering key: "Within one
run, the ordering key reported with each item MUST strictly increase in the
source's own order; a repeated or regressed ordering key is detected by the
runtime by bytewise comparison and fails closed before enqueue." S5.4
correspondingly names "cursor ordering-key regression (S2.2)". Enforcement
stays runtime-side, before any side effect - nothing else in S2.2/S5.4
changes. The rejected alternative (regression detection as a source-side
obligation, critique BLOCK 3 option b) would repeat D3.3's mistake of
trusting per-connector diligence with a guarantee the runtime can check
mechanically.

Serialization: in the manifest, `order` is a lowercase hex string and
`payload` is the JSON object (G3). Across PyO3, a cursor crosses as
`{"order": <hex str>, "payload": <mapping>}`; the B1 adapter validates the
hex and defensively copies the payload.

```rust
/// Events a source may produce. Sources never construct barriers or epochs
/// (S1.3); the runtime validates watermark and cursor monotonicity before
/// enqueue (S5.4, S2.2).
#[derive(Clone, Debug)]
pub enum SourceEvent {
    Data { batch: Batch, cursor: Cursor },
    Watermark(EventTime),
    Idle,
}

#[derive(Clone, Copy, Debug)]
pub struct SourceCapabilities {
    /// `false` marks a non-replayable source (S9.1 best-effort tier).
    pub replayable: bool,
    /// Conservative upper bounds a single emitted `Data` batch can reach;
    /// used by runner-time edge budget pre-validation (S10.3).
    pub max_batch_rows: usize,
    pub max_batch_bytes: usize,
}

#[async_trait]
pub trait StreamSource: Send {
    /// Positions the source at the resume cursor before any polling (S2.5).
    /// `None` means the configured beginning. Failing to honor the cursor
    /// fails closed here, before any item is emitted.
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()>;

    /// Returns the next event, or `None` at end of input. The runtime turns
    /// the first `None` into exactly one ordered `EndOfInput` (S1.6).
    ///
    /// Contract (D3.5, normative in rustdoc):
    /// 1. MAY block awaiting external data for an unbounded time; is NOT
    ///    required to be cancellation-safe; the runtime never cancels it
    ///    mid-stream.
    /// 2. The runtime guarantees exactly one in-flight call per instance.
    /// 3. A pending call is dropped only at teardown; after such a drop the
    ///    runtime NEVER calls `next` again on that instance.
    /// 4. The source MUST NOT treat an item as consumed until the call
    ///    returning it has completed; the durable cursor advances only via
    ///    the checkpoint protocol (S2.3, S7.4).
    async fn next(&mut self) -> Result<Option<SourceEvent>>;

    /// Releases external resources without requiring a dropped `next` call
    /// to complete, and tolerates being called with a poll dropped
    /// mid-flight (D3.5.5).
    async fn close(&mut self) -> Result<()>;

    fn capabilities(&self) -> SourceCapabilities;
}

pub enum BackpressurePolicy {
    /// Default and only implicit policy (S10.4).
    Block,
    /// Explicit lossy source; observable in metrics; makes exactly-once
    /// compilation fail (S10.4, S9.2).
    DropOldest,
}

pub struct SourceBinding {
    pub source: Box<dyn StreamSource>,
    pub backpressure: BackpressurePolicy,
}
```

Notes:

- `WatermarkPolicy` (S5.6) is source-binding configuration, not a source
  method, because the generated-watermark machinery is runtime-owned:

  ```rust
  pub enum WatermarkPolicy {
      SourceProvided,
      BoundedOutOfOrderness {
          event_time_column: String,
          delay: Duration,
          emit_interval: Duration,
          idle_timeout: Duration,
      },
      Disabled,
  }
  ```

  Durations convert to internal microseconds with an exact-microsecond check
  (error naming the field otherwise); the policy is validated at compile
  time, including the timestamp-column check with the source and column path
  (S5.6, D1.6).
- The v2 `SourceItem.sequence` field is gone: the per-source sequence is
  runtime-owned (S2.1). Sources supply only the cursor; the runtime writes
  binding ID and sequence into `BatchMetadata` immutably.
- D3's prefetch-slot model (pump + one-item slot + control-responsive source
  task) is a runtime structure, invisible in this trait; the trait's rustdoc
  contract above is the entirety of the connector implementor's obligations.
- **Connector task rule (I5).** Connectors MUST NOT spawn Tokio tasks: this
  trait surface provides no supervisor registration path, so a spawned task
  would escape the D5 ownership chain. Blocking client work runs on the I6
  mechanisms (blocking pools or dedicated threads); client-internal OS
  threads (e.g., a bundled client's background thread) are external
  resources released by `close()`, not runtime tasks. The same rule binds
  sink and operator implementations (A5, A2).
- Rejected alternative: keep v2's `Source` trait shape (`next() ->
  Option<(Batch, cursor, sequence)>`). It cannot express watermarks or idle,
  lets sources forge sequences, and its name would shadow the semantics
  change. New names make the break loud.

#### A4.1 Edge-budget pre-validation

Compilation (project path, inside `compile_stream`) or runner construction
(builder path, inside `StreamingRunner::new`) validates every binding's
declared `max_batch_rows`/`max_batch_bytes` against its first-hop edge
budgets - always before any source opens (S10.3). Both checks name the
binding and the edge in the error.

### A5. `StreamSink`, `TransactionalSink`, and `SinkBinding` (implements S7.5, S9; constrained by I3, I4)

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SinkDelivery {
    Ordinary,
    /// The sink deduplicates re-delivery of one epoch's batches by a stated
    /// mechanism. The declaration is sampled ONCE at binding validation and
    /// frozen into the compiled plan's per-sink delivery record (a sink
    /// must not vary its answer mid-job); it is recorded with its basis
    /// into the manifest's `sinks.*.delivery` (G3) and into per-sink
    /// guarantee reporting (S9.4). A `Bounded` retention class cannot
    /// satisfy `delivery = "exactly_once"` (compilation fails naming the
    /// configuration path) and is reported as retry-deduplicated within a
    /// bounded window - never as unqualified exactly-once (plan M6.5's
    /// dedup-token rule).
    EpochIdempotent {
        /// Portable identifier of the dedup mechanism, e.g.
        /// `"clickhouse_insert_dedup_token"`.
        mechanism: String,
        retention: RetentionClass,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionClass {
    Unbounded,
    Bounded,
}

#[async_trait]
pub trait StreamSink: Send {
    /// Sampled exactly once, at binding validation (A6); the returned value
    /// is frozen into the plan.
    fn delivery_capability(&self) -> SinkDelivery;

    /// Delivers one batch. The batch is borrowed; the sink must not retain
    /// or mutate it (I9). Ordering per edge is FIFO (S1.1).
    async fn write(&mut self, batch: &Batch) -> Result<()>;

    /// Epoch-boundary durability point. Returning `Ok` acknowledges that
    /// all pre-barrier data for `epoch` is staged per the sink's delivery
    /// capability (S7.5). Called at most once per epoch, only after the
    /// epoch's barrier and all pre-barrier data.
    async fn flush(&mut self, epoch: Epoch) -> Result<()>;

    /// Graceful teardown at drain, cancel, or failure convergence (D5.2).
    async fn close(&mut self) -> Result<()>;
}

/// Bounded, data-only pre-commit metadata recorded in the manifest (G3).
/// Never carries secrets (I4) or row payloads (D4.4).
#[derive(Clone, Debug, Default)]
pub struct SinkPreCommit {
    pub metadata: JsonMap,
}

#[async_trait]
pub trait TransactionalSink: StreamSink {
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()>;
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<SinkPreCommit>;
    /// External commit, strictly after the epoch's manifest is durable
    /// (S7.5); retried idempotently under the job's normal error rules.
    async fn commit(&mut self, epoch: Epoch, state: &SinkPreCommit) -> Result<()>;
    async fn abort(&mut self, epoch: Epoch, state: Option<&SinkPreCommit>) -> Result<()>;
    /// Completes durable pre-commits without rewriting data (S7.5). MUST be
    /// able to determine, from the manifest's `SinkPreCommit` and the
    /// external system alone, whether each recorded epoch was externally
    /// committed, completing those that were not (the commit-outcome query
    /// S7.5's derivation rule rests on). MUST also abort or remove any
    /// locally staged epoch artifacts NOT recorded in that manifest -
    /// orphan cleanup for crash-abandoned epochs (S7.5: "a failure before
    /// manifest publication aborts staged sink output"; a crash cannot run
    /// `abort`, so recovery owns this).
    async fn recover(&mut self, manifest: &CheckpointManifest) -> Result<()>;
}

/// Type-level capability declaration: a sink is transactional only by
/// implementing the full `TransactionalSink` protocol.
pub enum SinkBinding {
    Ordinary(Box<dyn StreamSink>),
    Transactional(Box<dyn TransactionalSink>),
}
```

**Sink teardown-drop contract (mirrors D3.5 for sources).** Sink method
futures (`write`, `flush`, `begin_epoch`, `pre_commit`, `commit`, `abort`)
are dropped only at teardown - graceful drain, cancel, or failure
convergence (D5.2) - never mid-epoch-and-resume. After such a drop the
runtime calls no method other than `close()` or `abort()` on that instance.
`close()` MUST tolerate a mid-flight drop of any prior method and MUST
release external resources without requiring the dropped call to complete
(the D3.5.5 hygiene rule, applied to sinks). A dropped `commit()` is the
commit-ack-loss case, covered at the protocol level by S7.5's idempotent
retry and the `recover` contract above. The A4 connector task rule (no
Tokio tasks inside connectors, I5) applies identically to sinks.

**Capability spoofing resistance (M0.2 adversarial checkbox).** Transactional
capability is a type, not a flag: there is no boolean a sink can set to
claim 2PC. The `EpochIdempotent` declaration is a claim the runtime cannot
verify, so it carries its basis - mechanism plus retention class - into the
plan record, the manifest, and S9.4 reporting: a false claim damages the
*user-visible guarantee* (S9.4 reports what the plan records), which is
exactly why the basis is recorded and why a bounded-retention mechanism is
never reported as unqualified exactly-once. Exactly-once plans still
require the full S9.1 matrix (replayable source, deterministic operators,
`Block` everywhere, durable aligned epochs, and a transactional or
unbounded epoch-idempotent sink) verified by `StreamingRunner::new`/the
project compiler before any source opens (S9.2), with the diagnostic
naming the precise configuration path.

### A6. `StreamingRunner` and `StreamingJob` lifecycle (implements D5, S8, D7, D9.5; constrained by I5, I6)

```rust
/// The runner handle. The runner core - plan, bindings, stores, config,
/// reaper, and the live-job registry - is `Arc`-shared between this handle
/// and every live `StreamingJob`, so the reaper target outlives this
/// handle and a job handle's Drop always has a live transfer target.
pub struct StreamingRunner { /* handle over Arc<RunnerCore> */ }

impl StreamingRunner {
    pub fn new(
        plan: StreamExecutionPlan,
        sources: BTreeMap<String, SourceBinding>,
        sinks: BTreeMap<String, Vec<SinkBinding>>,
        state_backend: Arc<dyn StateBackend>,
        checkpoint_store: Arc<dyn CheckpointStore>,
        config: StreamRuntimeConfig,
    ) -> Result<Self>;

    /// Starts a new job. Fails with `CalcFlowError::Conflict` when a job is
    /// still active on this runner, or when the lineage (pipeline
    /// fingerprint plus state root, D9.3) is already locked by an active
    /// job on any runner or in any process: the managed state root and
    /// checkpoint store are exclusively locked for the job's lifetime, and
    /// the check completes before any source opens. Before spawning, the
    /// runner-scoped reaper lazily joins any previously transferred
    /// registries (D5.4 point i); `JoinError`s the reaper observes there
    /// are logged and counted, and never fail the new job. Recovery from
    /// the latest manifest is the default; a fresh lineage starts when no
    /// manifest exists (D9.3). The returned handle shares the runner core.
    pub async fn start(&self) -> Result<StreamingJob>;

    /// Idempotent and resumable shutdown: cancels and joins every live job
    /// first (the runner core tracks them), then joins the reaper, and only
    /// then reports closed (D5.4 point ii). A caller-cancelled `shutdown`
    /// leaves a live runner that can be shut down again; a completed
    /// `shutdown` leaves no task alive (the D5.4 no-outlive guarantee binds
    /// on this path, and only on this path).
    pub async fn shutdown(&mut self) -> Result<()>;
}

impl Drop for StreamingRunner {
    /// Triggers the cancellation token only. Never joins, never blocks.
    /// Dropping the handle without a completed `shutdown` is the sanctioned
    /// abandon-joins escape hatch: tasks are cancelled via the token and
    /// stay registered to their supervisor/reaper owner while any handle
    /// lives, but their joins are abandoned. This path is observable
    /// (metric plus log) and is asserted in tests via token state and
    /// task-completion flags, never sleeps.
    fn drop(&mut self);
}

/// The owning job handle; shares the runner core (`Arc`). `Drop` cancels
/// but never joins (D5.3).
pub struct StreamingJob { /* owns the supervisor registry */ }

impl StreamingJob {
    pub fn id(&self) -> u64;

    /// Deterministic snapshot (S8.5): state, current epoch, per-edge queue
    /// state, per-source progress, watermarks, per-sink delivery. Stable
    /// ingress/task ordering (I2). Cheap, synchronous, never fails.
    pub fn status(&self) -> JobStatus;

    /// Queues a FIFO checkpoint request (D9.5: serialized, never coalesced,
    /// never rejected) and resolves with the injected epoch once its
    /// manifest is published. Errors with `CalcFlowError::CheckpointTimeout`
    /// on expiry (D7) or with the terminal error if the job converges first
    /// (S7.6). Dropping the returned future does NOT dequeue the request:
    /// it stays queued and injects its epoch; the caller merely loses the
    /// resolution.
    pub async fn trigger_checkpoint(&self) -> Result<Epoch>;

    /// Graceful stop: `running -> draining -> completed` (S8.2). Stops
    /// source polling, drains accepted data, flushes sinks, publishes the
    /// final epoch manifest, joins every task (D5.2), and returns the
    /// terminal outcome. Idempotent: on an already-terminal job returns the
    /// recorded outcome.
    pub async fn shutdown(&self) -> Result<JobOutcome>;

    /// Explicit cancellation from any non-terminal state: no drain promise,
    /// abandons any in-flight checkpoint (S7.6), joins every task (D5.2).
    /// Converges deterministically with deadline expiry to one terminal
    /// state (S8.4). Idempotent like `shutdown`.
    pub async fn cancel(&self) -> Result<JobOutcome>;

    /// Idempotent terminal await: the same outcome to every caller (S8.5).
    /// Dropping this FUTURE does not cancel the job; dropping the HANDLE
    /// does (D5.3).
    pub async fn wait(&self) -> Result<JobOutcome>;
}

impl Drop for StreamingJob {
    /// "Cancel and release ownership": triggers the cancellation token and
    /// transfers the supervisor registry to the runner-scoped reaper (alive
    /// via the shared core). Never joins, never blocks a Tokio worker
    /// (D5.3).
    fn drop(&mut self);
}
```

**Ownership guarantee (mirrors the spec-side D5.4 restatement).** The
no-task-outlives-shutdown guarantee binds only on the
completed-`shutdown(&mut self)` path. I5's "no detached tasks" reads: no
task without a cancellation token and a registered owner while any handle
is alive - the repaired model enforces that by construction, because the
`Arc` core keeps the reaper (the transfer target of every job-handle Drop)
alive as long as any job or runner handle exists. The bare-drop path is the
sanctioned escape hatch, observable and testable per the `Drop` rustdoc
above. A borrowed-job alternative (`StreamingJob<'a>`) was rejected: it
would force an explicit `drop(job)` before `runner.shutdown()` in the
plan's own usage sketch.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Draining,
    Completed,
    Cancelled,
    Failed,
    RecoveryRequired,
}

#[derive(Clone, Debug)]
pub struct JobOutcome {
    /// Always terminal (S8.1).
    pub state: JobState,
    /// Latest completed epoch at termination, if any.
    pub completed_epoch: Option<Epoch>,
    /// S8.4 ordering: first is primary; the rest attached in stable
    /// task-ID order. Empty on `Completed`/`Cancelled`.
    pub errors: Vec<CalcFlowError>,
}
```

`StreamRuntimeConfig` (data-only, serde, all fields with defaults; runtime
config hash per NFR-5 - changes here never invalidate checkpoints):

```rust
pub struct StreamRuntimeConfig {
    pub checkpoint_interval: Duration,   // default 60s; exact-micros check
    pub checkpoint_timeout: Duration,    // default 10min; job-fatal on expiry (D7)
    pub edge_budget: EdgeBudget,         // default per-edge budget
    pub retained_epochs: usize,          // completed manifests kept, default 2
}

pub struct EdgeBudget {
    pub max_rows: usize,                 // > 0 (S10.1)
    pub max_bytes: usize,                // > 0 (S10.1)
}
```

**`pause` is deliberately absent.** S8.1 freezes exactly six job states; a
pause surface would require a seventh state and new transition semantics the
spec does not define. The plan's chosen set (construct, start, status,
trigger_checkpoint, shutdown, cancel, wait) is adopted verbatim.

**Constructor argument count.** Six arguments, matching the plan's section
2.3 sketch order. Grouping `state_backend` + `checkpoint_store` into one
struct was rejected: they are distinct seams (segment I/O vs manifest I/O)
with independent test doubles.

### A7. `EventTime`, `Epoch`, and the state/checkpoint types (implements D1, D9, D6, D4; plan M4.1 sketch adopted)

```rust
/// UTC instant, microseconds since the Unix epoch (D1.1). The inner `i64`
/// is crate-private; serde exposes the exact microsecond value (D1.2).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash,
         Serialize, Deserialize)]
pub struct EventTime(i64);

impl EventTime {
    pub const fn from_micros(micros: i64) -> Self;
    pub const fn as_micros(self) -> i64;
    // Arrow import/export conversions live on the ingestion/emission side
    // and implement D1.3/D1.4 (checked, floor toward negative infinity,
    // column path in errors, D1.6 timezone rule).
}

/// Checkpoint identifier (D9.1). `0` is reserved and unconstructable.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash,
         Serialize, Deserialize)]
pub struct Epoch(u64);

impl Epoch {
    pub const INITIAL: Epoch;            // 1
    pub fn new(value: u64) -> Option<Self>;  // None for 0
    pub const fn as_u64(self) -> u64;
    pub fn next(self) -> Result<Self>;   // +1, checked (D9.2)
}

/// Plan M4.1's handle, adopted verbatim.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateHandle {
    pub operator_id: String,
    pub epoch: Epoch,
    pub segment_id: String,
    pub relative_path: String,
    pub byte_len: u64,
    pub sha256: String,                  // lowercase hex, 64 chars
}

#[async_trait]
pub trait StateBackend: Send + Sync {
    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()>;
    async fn validate_segment(&self, handle: &StateHandle) -> Result<()>;
    /// Atomic rename from staging to committed, including an fsync of the
    /// committed parent directory, always BEFORE the manifest that
    /// references the segment is published; the manifest's `relative_path`
    /// always denotes the committed path (D4.1's stage -> durable write ->
    /// validate -> rename -> manifest order).
    async fn publish_segment(&self, handle: &StateHandle) -> Result<()>;
    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>>;
    /// Removes unreferenced segments; their presence never fails recovery
    /// (D4.2). Collection is serialized against checkpoint commits by the
    /// coordinator, and the caller's `retained` set MUST include: (i) every
    /// segment referenced by a retained manifest, (ii) every segment
    /// belonging to an in-flight epoch, and (iii) every segment created
    /// after the latest retained manifest - otherwise collection could
    /// delete a segment an in-flight manifest is about to reference.
    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize>;
}

pub struct LocalStateBackend { /* canonicalized managed root; D6 Arrow IPC envelope */ }

pub const MANIFEST_FORMAT_VERSION: u32 = 3;
pub const MAX_MANIFEST_DOCUMENT_BYTES: usize = 10 * 1024 * 1024;

#[async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn load_latest(&self, pipeline_name: &str) -> Result<Option<CheckpointManifest>>;
    async fn load(&self, pipeline_name: &str, epoch: Epoch) -> Result<Option<CheckpointManifest>>;
    /// Atomic publish: temporary name, fsync, rename, parent fsync (D4.1).
    async fn publish(&self, manifest: &CheckpointManifest) -> Result<()>;
    async fn list_epochs(&self, pipeline_name: &str) -> Result<Vec<Epoch>>;
    async fn delete_before(&self, pipeline_name: &str, keep_latest: usize) -> Result<()>;
}

pub struct FileCheckpointStore { /* bounded atomic JSON; rejects format_version != 3 with UnsupportedVersion (NG9) */ }

pub struct CheckpointManifest { /* exact JSON layout frozen in G3 */ }
```

### A8. Error variants (implements D7, S8.4, S10.3; v2 variants otherwise unchanged)

Additions to `CalcFlowError`:

```rust
pub enum CalcFlowError {
    // ... all v2 variants unchanged ...
    /// Checkpoint attempt exceeded the configured timeout; the epoch is
    /// abandoned with no manifest (D7). Class: infrastructure ->
    /// `recovery-required` (S8.3).
    #[error("checkpoint epoch {epoch} exceeded the configured timeout")]
    CheckpointTimeout { epoch: u64 },

    /// A supervised task panicked; carries the stable task identity (S8.4).
    /// Class: logic -> `failed`.
    #[error("task {task_id} panicked: {message}")]
    TaskPanicked { task_id: u64, message: String },

    /// `StreamCollector::emit` on an edge whose receiver closed during job
    /// convergence.
    #[error("edge {edge:?} is closed")]
    EdgeClosed { edge: String },
}
```

Reused with frozen meaning: `UnsupportedVersion { expected: 3, found }` for
v2 project/checkpoint documents (NG9); `Compile` for capability violations,
naming the precise configuration path (S9.2); `Compile` for collector
validation failures; `InvalidArgument` for edge-budget and constructor
violations with the field path; `Conflict { resource: "streaming job", .. }`
for `start()` reentrancy; `Cancelled` remains the v2 run-cancellation
variant (job-level cancellation is a `JobOutcome`, not an exception, per
A6/B3). Python mapping: `CheckpointTimeout`/`TaskPanicked`/`EdgeClosed` map
to new `CheckpointTimeoutError`, `TaskPanickedError`, and `EdgeClosedError`
subclasses of the existing `ExecutionError` family (B4).

---

## B. Python API (implements plan M6.8; constrained by I3, I4, D5)

### B1. Module layout and protocols

`python/calc_flow/runtime.py` is replaced (plan M6.8). It hosts, in one
cohesive module: `StreamSource`, `StreamSink`, `TransactionalSink` protocols;
`Data`, `Watermark`, `Idle` event types; `StreamRuntimeConfig`;
`StreamingRunner`, `StreamingJob`; `JobState`, `JobStatus`, `JobOutcome`.
`MicroBatchRunner` and the push-`step` `StreamingRunner` are deleted without
aliases (plan 1.3).

```python
@dataclass(frozen=True, slots=True)
class Data:
    batch: Batch
    cursor: Mapping[str, object]
    # {"order": <lowercase hex str>, "payload": <strict JSON mapping>} -
    # the A4 Cursor shape; bytewise comparison of the decoded `order` must
    # equal the source's own position order (S2.2).

@dataclass(frozen=True, slots=True)
class Watermark:
    at: datetime  # timezone-aware; normalized to UTC; microseconds preserved

@dataclass(frozen=True, slots=True)
class Idle:
    pass

type SourceEvent = Data | Watermark | Idle

class StreamSource(Protocol):
    def capabilities(self) -> Mapping[str, object]: ...
    def open(self, cursor: object) -> object: ...
    def next(self) -> object: ...   # SourceEvent | None; None ends input once
    def close(self) -> object: ...

class StreamSink(Protocol):
    def write(self, batch: Batch) -> object: ...
    def flush(self, epoch: int) -> object: ...
    def close(self) -> object: ...

class TransactionalSink(StreamSink, Protocol):
    def begin_epoch(self, epoch: int) -> object: ...
    def pre_commit(self, epoch: int) -> Mapping[str, object]: ...
    def commit(self, epoch: int, state: Mapping[str, object]) -> object: ...
    def abort(self, epoch: int, state: Mapping[str, object] | None) -> object: ...
    def recover(self, manifest: Mapping[str, object]) -> object: ...
```

- Sync or async method returns are both accepted (the v2 `_resolve`
  awaitable pattern); `open`/`next`/`close` mirror the Rust contract in A4,
  including D3.5 across the PyO3 boundary (teardown cancellation is
  delivered once, never mid-stream-and-resume).
- **Synchronous Python sources (D3.5 mechanics across PyO3).** For an
  *async* Python `next()`, dropping the Rust-side awaitable cancels the
  Python task. For a *sync* Python `next()` blocking on a worker thread,
  there is no future to drop: teardown abandons the blocked call. The
  contract for sync sources: `close()` MUST be callable while a `next()`
  call is blocked on another thread, and MUST cause that blocked call to
  return or raise (close the queue, set the event); the runtime never calls
  `next()` again once teardown starts (D3.5.3), and it joins the abandoned
  thread before `close()` returns. M6.8 RED line: a sync mock source
  blocked in `next()` is released by `close()`, observed via flags, never
  sleeps.
- `cursor` mappings are validated on every crossing: `order` must be a
  non-empty lowercase hex string decoding to at most 16 KiB, `payload` a
  strict JSON mapping defensively copied (v2 `_copy_json_value` rule); a
  malformed shape raises `TypeError` naming the field. Watermarks cross
  only as timezone-aware `datetime`; naive datetimes raise `TypeError`. No
  Python object other than `Batch` ever crosses the edge: no source text,
  callables, or import paths appear in events, cursors, manifests, or
  configurations (I3).
- `capabilities()` validation (adapter-side, before the native crossing):
  a missing key raises `TypeError` naming the key; a non-`bool`
  `replayable` or a non-`int` bound raises `TypeError`; a negative or zero
  bound raises `ValueError`. The adapter never invents defaults.
- Transactional capability detection is structural, matching A5: the
  binding adapter checks that all five protocol methods exist and are
  callable before classifying the sink; a Python object with a partial
  protocol is bound as ordinary and cannot satisfy an exactly-once plan
  (M6.8 RED). A Python sink may also declare
  `delivery_capability() -> Mapping[str, str]`, validated to the A5 shapes
  (`{"kind": "ordinary"}`,
  `{"kind": "epoch_idempotent", "mechanism": ..., "retention":
  "unbounded"|"bounded"}`); absent the method, the sink is ordinary.
- Epoch values cross as `int` validated into `u64`; job-facing types:

```python
class JobState(StrEnum):
    RUNNING = "running"
    DRAINING = "draining"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RECOVERY_REQUIRED = "recovery_required"

@dataclass(frozen=True, slots=True)
class JobOutcome:
    state: JobState
    completed_epoch: int | None
    error: CalcFlowError | None          # primary, S8.4
    attached_errors: tuple[CalcFlowError, ...]

# JobStatus is a strict JSON-compatible dict snapshot (S8.5), the same
# document shape the Studio SSE `job.status` event carries (D3).
```

### B2. Runner and job

```python
class StreamingRunner:
    def __init__(
        self,
        plan: StreamExecutionPlan,
        sources: Mapping[str, StreamSource | tuple[StreamSource, str]],
        sinks: Mapping[str, Sequence[StreamSink | Callable[[Batch], object]]],
        state_backend: FileStateBackend,
        checkpoints: FileCheckpointStore,
        config: StreamRuntimeConfig | None = None,
    ) -> None: ...
    # tuple form: (source, "block" | "drop_oldest"); bare source = "block".
    # A sink is bound Transactional when it structurally satisfies the
    # protocol (B1), else Ordinary. A bare callable is adapted to a
    # write-only ordinary sink (no-op flush/close), preserving the v2
    # callback ergonomic; sync and async callables are both accepted.
    async def start_async(self) -> StreamingJob: ...
    def start(self) -> StreamingJob: ...
    async def shutdown_async(self) -> None: ...
    def shutdown(self) -> None: ...

class StreamingJob:
    @property
    def id(self) -> int: ...
    async def status_async(self) -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...
    async def trigger_checkpoint_async(self) -> int: ...
    def trigger_checkpoint(self) -> int: ...
    async def shutdown_async(self) -> JobOutcome: ...
    def shutdown(self) -> JobOutcome: ...
    async def cancel_async(self) -> JobOutcome: ...
    def cancel(self) -> JobOutcome: ...
    async def wait_async(self) -> JobOutcome: ...
    def wait(self) -> JobOutcome: ...
```

`StreamExecutionPlan` (native) exposes `name`, `fingerprint`,
`runtime_config_hash`, `requirements`, `source_binding_ids`,
`sink_binding_ids`; `BatchExecutionPlan` keeps the v2 `ExecutionPlan`
adapter surface renamed. `PipelineBuilder` gains
`compile_batch(runtime=None)` and
`compile_stream(runtime=None, delivery: Mapping[str, str] | None = None)`
(`delivery` values `"at_least_once" | "exactly_once"`, keyed by output name);
v2 `compile()` is removed (plan 1.3). New builder steps `union(...)` and
`window(...)` mirror the Rust operators. Plan exclusivity matches the v2
lease: constructing a second `StreamingRunner` over the same plan raises
`calc_flow.CompileError`; the lease is native, taken at runner construction
and released at runner shutdown.

### B3. Blocking rejection in a running event loop (exact error)

Every blocking convenience method (`start`, `status`, `trigger_checkpoint`,
`shutdown`, `cancel`, `wait`, and store methods unchanged from v2) checks
`asyncio.get_running_loop()` FIRST, before any argument validation (v2
`execute()` ordering rule), and raises:

```
RuntimeError: "<method>() cannot run inside an event loop; use <method>_async()"
```

The exception type is the builtin **`RuntimeError`** (v2 parity:
`python/calc_flow/pipeline.py` `execute()` and `store.py` `_run_blocking`
both raise `RuntimeError`); the message pattern is frozen as
`"<verb>() cannot run inside an event loop; use <verb>_async()"`.

Async cancellation is per-method, not a blanket rule (a long-lived job
surface cannot reuse v2's one-shot `execute_async` sentence). When the
surrounding asyncio task awaiting one of these methods is cancelled:

| Awaiter cancelled while pending  | Effect                                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `status_async`, `wait_async`     | Observe-only: no job effect; a later call re-attaches to the same deterministic snapshot/outcome (S8.5).           |
| `trigger_checkpoint_async`       | The request stays queued and injects its epoch (D9.5); the caller merely loses the resolution.                     |
| `shutdown_async`, `cancel_async` | The native transition runs to convergence; a re-call re-attaches and returns the same idempotent outcome.          |
| `start_async`                    | The call either returns a live job or converges to none - never a half-started job (D5.2 join rules on the abort). |
| runner `shutdown_async`          | Native runner shutdown is resumable (A6): the runner stays live; a re-call resumes and completes.                  |

In every case native cleanup completes before `asyncio.CancelledError`
propagates, and a native outcome that was already terminal when the
cancellation was handled wins (v2's precedence rule, preserved).

Store adapters keep their v2 pattern unchanged: async methods are
unsuffixed (`get`, `save`) with `*_blocking` convenience variants raising
`RuntimeError` carrying v2's store message ("blocking store operation
cannot run inside an event loop; await {method}()"). The
`<verb>()`/`<verb>_async()` pairing and the frozen message above apply to
the new v3 runner/job surface only.

### B4. Exceptions and the native stub

`_native.pyi` additions: `_StreamingRunner`, `_StreamingJob`,
`_FileStateBackend`, `StreamExecutionPlan`, `BatchExecutionPlan`. The stub
uses `int` for epochs (u64-validated) and `datetime` for watermarks (B1);
`EventTime`/`Epoch` microsecond values never cross as ambiguous bare
integers without the adapter-level units documented above. New exception
classes in `calc_flow.errors`:
`CheckpointTimeoutError`, `TaskPanickedError`, `EdgeClosedError` (subclasses
of `ExecutionError`, mapped from A8). `capabilities.py` reports project
format versions `(3,)` only, and its connector enumeration lists exactly the
connectors compiled into the running wheel (E2) - never connectors a project
v3 document could not actually resolve (M6.8 RED).

### B5. No-executable-object rule (I3, I4)

Python connector/source/sink configuration is plain `dict[str, JSONValue]`
data; UDF selection remains `(provider, name, version)` tuples;
`SecretReference` appears only as the C2 shape. Registration callbacks
(UDFs, providers) live only in the process-local `Runtime`, exactly as v2;
nothing callable enters a project document, cursor, manifest, status
snapshot, or error message.

---

## C. Project format v3 (implements plan M6.7; constrained by I2, I3, I4, NG9)

### C1. Top-level document

```json
{
  "format_version": 3,
  "id": "pg-cdc-to-clickhouse",
  "name": "pg-cdc-to-clickhouse",
  "description": "",
  "runtime": {
    "mode": "stream",
    "stream": {
      "checkpoint_interval_micros": 60000000,
      "checkpoint_timeout_micros": 600000000,
      "edge_budget": { "max_rows": 10000, "max_bytes": 67108864 },
      "retained_epochs": 2
    },
    "batch": null
  },
  "pipeline": {
    "name": "pg-cdc-to-clickhouse",
    "nodes": [],
    "edges": [],
    "datafusion": { "batch_size": 8192, "target_partitions": 1 }
  },
  "sources": [],
  "sinks": [],
  "state": { "backend": "local", "root": ".calc-flow-state" },
  "checkpoint": { "store": "file", "directory": ".calc-flow-checkpoints" }
}
```

Rules:

- `format_version` is the constant `3`; any other value fails with
  `UnsupportedVersion` (NG9). Exactly one of `runtime.stream` /
  `runtime.batch` is non-null and MUST match `runtime.mode`.
- Strict data-only contract unchanged from v2: unknown fields and duplicate
  keys rejected at every level, bounded depth/size, canonical JSON storage,
  YAML import/export only. `position` metadata stays per node for the
  Studio.
- Source binding shape (one per external input, keyed by `input`):

  ```json
  {
    "input": "events",
    "connector": { "provider": "calc_flow_connectors", "name": "postgresql", "version": "1" },
    "format": { "name": "pgoutput", "options": {} },
    "mode": "logical_cdc",
    "options": { "publication": "calc_flow_pub", "slot": "calc_flow_slot" },
    "watermark": {
      "kind": "bounded_out_of_orderness",
      "event_time_column": "commit_time",
      "delay_micros": 5000000,
      "emit_interval_micros": 1000000,
      "idle_timeout_micros": 30000000
    },
    "max_batch_rows": 5000,
    "max_batch_bytes": 33554432,
    "backpressure": "block"
  }
  ```

- Sink binding shape (one per graph output, ordered list per output):

  ```json
  {
    "output": "output",
    "connector": { "provider": "calc_flow_connectors", "name": "clickhouse", "version": "1" },
    "format": { "name": "native", "options": {} },
    "mode": "append",
    "options": { "table": "events_hourly" },
    "delivery": "exactly_once"
  }
  ```

- `runtime.stream` values are runtime-tunable (NFR-5: they feed the runtime
  config hash, never the semantic fingerprint). Graph, operator
  configuration, UDF catalog, connector identities, watermark/window/state
  semantics feed the semantic fingerprint.
- Window operator node configuration (referenced by S6/S8):
  `kind: "window"`, `geometry: {"kind": "tumbling"|"hopping",
  "size_micros", "slide_micros"}`, `event_time_column`, `group_by`,
  `aggregates: [{"function": "count"|"sum"|"min"|"max"|"avg", "column",
  "output"}]`. Session/early/allowed-lateness/update keys do not exist in
  the schema, so strict validation rejects them (NG4).

### C2. Secret reference representation (I4)

Exactly one shape, valid anywhere a connector `options` value may appear:

```json
{
  "kind": "secret_reference",
  "provider": "env",
  "name": "POSTGRES_PASSWORD"
}
```

- `provider` is a portable identifier selecting a resolution mechanism from
  the process-local `SecretResolver` registry (built-in: `"env"`); `name` is
  the provider-local key: 1-512 printable ASCII characters, control
  characters forbidden, and resolvers MUST reject names containing `..`
  path segments.
- A secret **reference** is data, not a secret: it appears in connector
  options and stored documents in the clear, exactly like a `UdfReference`,
  and it feeds NEITHER the semantic fingerprint nor the runtime-config hash
  nor the source identity hash (credential rotation must not invalidate
  checkpoints or fail recovery). Connector options stay out of the semantic
  fingerprint (NFR-5); instead, options split into two declared classes -
  **data-semantics options** (publication, slot, query, topic, table, URL,
  file paths, format choice) and **credential/transport options** (timeouts,
  pool sizes, TLS mode flags) - and each connector declares a **source
  identity hash** over its data-semantics options, recorded in the manifest
  (`sources.*.identity_hash`, G3). Recovery fails closed on an identity-hash
  mismatch: two projects differing in these options read different data and
  must not share a lineage (D9.3), and the hash is what enforces it without
  invalidating checkpoints on transport-only changes. Sink-side target
  identity travels inside `sinks.*.pre_commit` per S7.5 (ledger keys and
  staged-target records already carry it). This note owns the two-class
  rule and the manifest placement; the per-connector declared list is owned
  by M6.1's connector capability surface. A secret **value** never enters
  documents, fingerprints, identity hashes, logs, metrics, state segments,
  manifests, status snapshots, SSE events, or error messages (I4):
  resolution happens at source/sink open time only, inside the connector.
- **Error sanitization at the resolution boundary.** Connector errors
  raised after secret resolution MUST be wrapped so that client-library
  messages - which may echo resolved values, connection URLs, or auth
  headers - never propagate raw into `CalcFlowError`; the wrapped error
  carries the connector identity and a sanitized message. (Spec-side
  companions owned by the spec owner: the I4 extension sentence and an
  AC-I canary-secret redaction test; this note freezes the
  connector-facing obligation.)
- Structural validation: any object carrying `"kind": "secret_reference"`
  must have exactly the three fields above; any other `kind` value at that
  position is rejected. There is no string-interpolation scheme - magic
  substrings inside ordinary strings cannot be validated or redacted
  reliably. Python: `calc_flow.config.SecretReference(provider, name)`
  frozen dataclass serializing to this exact shape.

### C3. Schema artifact

`schemas/project-v3.schema.json` is generated from the Rust v3 model and
checked in (replacing the canonical role of the v2 schema; the v2 schema
moves to historical docs per plan 1.3). The generated schema, Rust
`ProjectSpec`, Python `ProjectDocument`, FastAPI models, and generated
TypeScript contract describe one structure, as v2 does today.

---

## D. Studio `/api/v3` (implements plan M6.9; constrained by I2, I4, NFR-6)

### D1. Route list

`/api/v2` is deleted (plan 1.3). All v3 routes are loopback-only as today;
request/response models are strict Pydantic with snake_case JSON (v2
convention), generated into `web-ui/openapi.json` and
`web-ui/src/api/schema.d.ts` in the same commit.

Projects (v2 parity, v3 documents):

| Method | Route                                    | Request model          | Response model                     |
| ------ | ---------------------------------------- | ---------------------- | ---------------------------------- |
| GET    | `/api/v3/projects`                       | -                      | `tuple[ProjectSummary, ...]`       |
| POST   | `/api/v3/projects`                       | `ProjectDocument` (v3) | `ProjectDocument`, 201             |
| POST   | `/api/v3/projects/import?format&replace` | bounded body           | `ProjectDocument`, 201             |
| GET    | `/api/v3/projects/{project_id}`          | -                      | `ProjectDocument`                  |
| PUT    | `/api/v3/projects/{project_id}`          | `ProjectDocument` (v3) | `ProjectDocument`                  |
| DELETE | `/api/v3/projects/{project_id}`          | -                      | 204                                |
| GET    | `/api/v3/projects/{project_id}/export`   | -                      | `text` (json/yaml)                 |
| POST   | `/api/v3/projects/{project_id}/validate` | -                      | `ValidationReport`                 |
| GET    | `/api/v3/catalog`                        | -                      | UDF catalog (unchanged)            |
| GET    | `/api/v3/capabilities`                   | -                      | `CapabilitiesResponse` (schema v3) |
| GET    | `/api/v3/schema/project`                 | -                      | project-v3 JSON Schema             |

Jobs (continuous lifecycle; the plan's exact list plus `GET /api/v3/jobs`
for listing, required by the Studio's job browser):

| Method | Route                              | Request model      | Response model                                           |
| ------ | ---------------------------------- | ------------------ | -------------------------------------------------------- |
| POST   | `/api/v3/jobs`                     | `JobCreateRequest` | `JobResponse`, 202                                       |
| GET    | `/api/v3/jobs`                     | -                  | `tuple[JobSummary, ...]`                                 |
| GET    | `/api/v3/jobs/{job_id}`            | -                  | `JobResponse`                                            |
| GET    | `/api/v3/jobs/{job_id}/events`     | SSE (D2)           | `text/event-stream`                                      |
| POST   | `/api/v3/jobs/{job_id}/checkpoint` | -                  | `CheckpointResponse` (200) or `CheckpointAccepted` (202) |
| POST   | `/api/v3/jobs/{job_id}/shutdown`   | -                  | `JobResponse`                                            |
| POST   | `/api/v3/jobs/{job_id}/cancel`     | -                  | `JobResponse`                                            |

The checkpoint route never parks a client for the full
`checkpoint_timeout`: the server awaits completion up to a route bound
(`checkpoint_route_wait_seconds`, server constant, default 60, always at or
below the job's checkpoint timeout). Completion within the bound returns
200 `CheckpointResponse`; otherwise 202 `CheckpointAccepted`, and the
client observes completion through the SSE `job.epoch` event (D2). A client
disconnect does not dequeue the queued request (A6's trigger-cancellation
rule).

Removed without replacement: `/api/v2/projects/{id}/runs*`, and
`/api/v2/projects/{id}/checkpoint*`. Bounded previews (`runs`) are re-homed
as batch-mode jobs (`POST /api/v3/jobs` over a `runtime.mode: "batch"`
project); v2's project-level checkpoint inspect/delete routes are replaced
by job-scoped epoch/status visibility - destructive state-root operations are
deliberately not exposed in the loopback Studio in 3.0 (a fresh lineage is a
filesystem-level decision, plan M4.2's managed root).

Models (additive over the v2 style; all strict, frozen, `extra="forbid"`):

```python
class JobCreateRequest(StrictModel):
    project_id: str = Field(min_length=1, max_length=128)
    config: StreamRuntimeConfigModel | None = None   # document overrides

class CheckpointResponse(StrictModel):
    job_id: str
    epoch: int = Field(ge=1)
    duration_ms: int = Field(ge=0)

class CheckpointAccepted(StrictModel):
    job_id: str
    status: Literal["in_flight"] = "in_flight"

class JobSummary(StrictModel):
    id: str
    project_id: str
    state: JobState
    created_at: datetime

class JobError(StrictModel):
    message: str
    attached: tuple[str, ...] = ()        # S8.4 order preserved

class JobResponse(StrictModel):
    id: str
    project_id: str
    pipeline_fingerprint: str
    state: Literal["running", "draining", "completed", "cancelled",
                   "failed", "recovery_required"]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    current_epoch: int | None              # latest completed epoch
    status: dict[str, JSONValue]           # S8.5 snapshot: per-edge queues,
                                           # per-source progress, watermarks,
                                           # per-sink delivery
    error: JobError | None                 # present iff state is failed /
                                           # recovery_required
```

`JobResponse` is ONE model with documented nullability invariants rather
than v2's six-way discriminated union: the six states share a large,
identical snapshot payload, and repeating it per variant would multiply the
generated TypeScript surface for no discriminable benefit; `state` remains
the discriminator clients switch on.

Studio server bounds replacing the v2 worker timeout (plan M6.9: an
equivalent limit set is mandatory, never a null timeout): server
configuration constants `max_jobs` (default 4 concurrent jobs),
`job_memory_limit_mb` (default 1024), `global_memory_limit_mb` (default
3072), `max_state_bytes` (default 10 GiB), and jobs run until explicitly
stopped. `POST /api/v3/jobs` beyond `max_jobs` fails 409.

### D2. SSE event model (`GET /api/v3/jobs/{job_id}/events`)

Wire format follows the v2 run-events precedent: `id: {sequence}\n`,
`event: {type}\n`, `data: {compact-json}\n\n`, an initial `retry: 500\n\n`
hint, `: keep-alive\n\n` comment lines when idle, and `Last-Event-ID`
resume. `Cache-Control: no-cache`, `X-Accel-Buffering: no`.

Event envelope (the `data` payload):

```json
{
  "sequence": 41,
  "timestamp": "2026-08-03T00:00:00.000000Z",
  "type": "job.metrics",
  "data": { "..." : "..." }
}
```

Event types:

| Type           | Emitted                                          | Payload                                                                                                                 | Collapsible |
| -------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ----------- |
| `job.status`   | On connect, on resync (D2 resume), on request    | Full S8.5 snapshot (same document as `JobResponse.status`)                                                              | yes         |
| `job.state`    | On every state transition                        | `{from, to, at}`                                                                                                        | no          |
| `job.epoch`    | On every completed checkpoint                    | `{epoch, duration_ms}`                                                                                                  | no          |
| `job.metrics`  | Periodically (default every 5s, server constant) | Rollups: rows/bytes in/out, blocked sends/durations, queue high-water marks, late-row counters (D2.5), per-edge charges | yes         |
| `job.terminal` | Exactly once, as the final event                 | `{state, completed_epoch, error}` mirrors `JobOutcome`                                                                  | no          |

Guarantees:

- **Ordering.** Per-job `sequence` is a u64 starting at 0, strictly +1 per
  event, total order per job (I2).
- **Resume.** The server retains a per-job ring of the latest 1024 events.
  On reconnect with `Last-Event-ID`: replay from `last+1`; if that sequence
  fell out of the ring, emit a fresh `job.status` full snapshot at the
  current sequence first (deterministic resync), then continue live. No
  error response for stale IDs - EventSource auto-reconnect must always
  converge.
- **Termination.** After `job.terminal`, the server closes the stream; a
  reconnect to a terminal job replays the retained tail ending in
  `job.terminal` and closes.
- **Backpressure.** Each connection owns a bounded outbox of 256 events.
  When full, collapsible types (`job.status`, `job.metrics`) coalesce to
  the latest pending value; non-collapsible types are never dropped. If
  non-collapsible events alone overflow the outbox (possible only under a
  pathological checkpoint cadence below the drain rate), the server closes
  that connection; the client reconnects and resyncs via `Last-Event-ID`.
  SSE delivery NEVER blocks the job, the coordinator, or any edge (NFR-6).
- **Redaction (I4).** Events carry no row payloads, no secrets, no batch
  IDs (no high-cardinality labels), and no cursor values beyond what the
  `job.status` snapshot already exposes.

### D3. Consistency with the Python surface

`JobStatus` dict (B1) and `JobResponse.status` (D1) are the same JSON
document produced by the same Rust snapshot type; `JobOutcome` and
`job.terminal` carry the same fields. One snapshot type, three projections
(Rust `JobStatus`, Python dict, REST/SSE JSON).

---

## E. `calc-flow-connectors` packaging (plan section 3; M0-level decision)

### E1. Dependency edge: `calc-flow-python` DEPENDS on `calc-flow-connectors`

Decision: **`calc-flow-python` takes an optional dependency on
`calc-flow-connectors` with one mirrored cargo feature per connector, and
the published wheel's native module is built with the full connector
feature set (E2)**, so project v3 documents can resolve every shipped
connector from Python and Studio. Cargo-level optionality keeps
connector-less development builds possible; distribution always includes
the full set.

Justification: the alternative (no edge) makes the plan's M6.7/M6.8
acceptance gates unreachable - "Python/Studio 运行 PostgreSQL CDC 到
ClickHouse" and "Python 侧能够注册可用 connector" both require the native
extension to link the connector factories. Rewriting those gates would gut
the plan's section 1.1 deliverable of three complete operating surfaces
(Rust, PyO3/Python, Studio) and leave project v3's data-only connector
selection unexecutable from Python. The costs of depending (rdkafka, TLS,
compression libraries enter the wheel; manylinux/macOS/Windows builds and
the SBOM must cover them) are already scoped by plan M6.3-M6.5 and the M7.2
supply-chain gate, which must audit "启用全部 connector feature 的配置"
regardless. Both options carry the build risk; only one delivers the
acceptance gates.

### E2. Feature gating: one cargo feature per connector, all default-off

`calc-flow-connectors` features: `file`, `csv`, `json`, `parquet`, `kafka`,
`postgresql`, `clickhouse`, `http`, `websocket` - one per module in the
plan's section 3 layout, each default-off, each mapping 1:1 to its client
dependencies. `calc-flow-python` mirrors them (`connectors-kafka` etc.)
with a `connectors-all` meta-feature; the maturin wheel build enables
`connectors-all`. Cargo builds of the core crate alone never pull connector
clients (the core has no dependency on the connectors crate; connector
identities resolve through the registry seam, plan M6.1).

**Hard rule for M6 connector client selection (M6.3/M6.4/M6.5):** every
connector's client stack MUST build from a clean Rust toolchain with no
system libraries beyond a C compiler and cmake, and MUST use pure-Rust TLS
(rustls). Bundled builds (e.g., rdkafka's cmake librdkafka) are permitted;
linking system OpenSSL or system librdkafka is not. This keeps
`--all-features` builds hermetic.

### E3. `--all-features` CI interaction

The clippy and rustdoc commands stay byte-identical to today's AGENTS.md;
the single coverage line splits into two lines (the connectors crate moves
to its own floor, E4):

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo llvm-cov --workspace --all-features --exclude calc-flow-connectors --fail-under-lines 90
cargo llvm-cov -p calc-flow-connectors --all-features --fail-under-lines 75
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

With all connector features enabled, CI needs no network credentials and
no live external systems: unit tests use
mocks and byte fixtures only, and container integration tests are a
separate opt-in harness that never enters `cargo test` default runs (I8).
The only new documented developer prerequisites are a C compiler and cmake
(E2 rule). `cargo audit` / `cargo deny` run over the `--locked` workspace
with all features resolved, as today.

### E4. Coverage gate treatment

Decision: **exclude `calc-flow-connectors` from the workspace 90% line gate
and enforce a separate crate-level floor of 75% lines** (single fixed
number, not per-feature). Justification: the plan forbids deciding "later"
(section 3: "必须在 M0 决定"), and per-feature thresholds are not
measurable - llvm-cov attributes lines to the crate, not to feature
combinations, and features compile additively into one coverage run. 75%
reflects the crate's structure: connector protocol I/O paths are only
meaningfully testable against live systems in container tests (excluded
from ordinary runs by I8), while everything else - configuration
validation, cursor ordering, identifier quoting, type matrices, redaction,
format codecs over byte fixtures, capability computation, dedup-token
derivation - is pure and unit-testable. A floor of zero (plain exclusion)
was rejected because it invites untested pure logic in exactly the modules
where SQL-injection and redaction bugs live. M7.2's existing checklist item
re-verifies the final configuration passes.

### E5. Versioning and release

`calc-flow-connectors` joins the workspace version lockstep: it releases as
`3.0.0` together with `calc-flow`, the Python core, Studio, and the
frontend (release invariant), is published to crates.io (Rust users depend
on it directly for connector use without Python), and its package contents
follow the core crate's rules: license, sources, and no repository-only
integration/container tests. The PyO3 crate's dependency on it is an exact
version requirement, same as the existing `calc-flow` edge. M7.4 gains one
checklist line: connectors crate packaging inspection.

---

## F. Barrier forwarding timing (spec section 9 deferral; constrained by S7.3, D4.2, D7, D9.4)

### F1. Decision: forward immediately after the local synchronous snapshot completes

On full alignment for epoch `E`, an operator task executes in order:

1. `checkpoint(E)` returns `Ok(OperatorStateSnapshot)` (synchronous, A2.2) -
   **snapshot success precedes forwarding** (S7.3's hard constraint).
2. The barrier is forwarded on all output edges immediately, and all
   blocked ingresses resume.
3. The snapshot is handed to the coordinator as the operator's
   acknowledgement (in-memory handoff), which stages, validates, and
   publishes per D4.1 off the per-operator barrier path (I6).

The acknowledgement is NOT awaited before forwarding; there is no
coordinator round trip per graph level.

**Latency cost recorded (required by the deferral).** End-to-end barrier
traversal is O(graph depth x per-edge traversal) with zero coordinator
round trips. The rejected alternative (forward only after the coordinator
acknowledges this operator's ack) adds O(graph depth x coordinator round
trip) to every checkpoint - each level's barrier release waits on a
controller response, serializing depth-wise - and its only enforceable
bound remains the same configured checkpoint timeout (D7), so the wait buys
latency, not safety.

**Failure-semantics implications.** A forwarded barrier can belong to an
epoch that never commits (the coordinator times out per D7, the job is
cancelled per S7.6, or a crash lands between forwarding and manifest
publication). Downstream snapshots for such an epoch are simply never
referenced by any manifest: they are discarded per D4.2 (manifest is sole
truth) and D9.4 (an epoch without a manifest never happened), and the
lineage is torn down by job convergence, so no downstream effect escapes.
This imposes one requirement, already implied by D4.3's crash table:
operator snapshots must be side-effect-free until manifest publication -
true by construction here, because staging is not publication (D4.1) and
operators perform no I/O (A2.2). Reconciliation with S7.3 (settled in
critique round 2): S7.3 revision 2 explicitly disclaims an ack/forward
ordering and defers it to this decision; the one constraint it freezes -
the snapshot precedes both the acknowledgement and the forwarding - is
satisfied by step 1.

---

## G. Smaller owned deferrals

### G1. Group-key byte encoding (constrained by the frozen S6.4 ordering contract)

The encoding is per key column, concatenated in declared `group_by` order;
within one close event rows sort by `(window_start, window_end, encoding)`
byte order (S6.4). It is total, deterministic across runs and recoveries,
and identical bytes for identical values. Per column, one tag byte then an
order-preserving payload:

| Key value                                                    | Encoding                                                                                                                    |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| null (any type)                                              | `0x00` (sorts before every non-null value)                                                                                  |
| non-null                                                     | `0x01` followed by the typed payload below                                                                                  |
| `bool`                                                       | `0x00` (false) or `0x01` (true)                                                                                             |
| signed ints (`i8..i64`), `Date32`, `Date64`, `Timestamp(us)` | big-endian two's-complement bytes with the sign bit flipped (XOR `0x80` on the first byte)                                  |
| unsigned ints (`u8..u64`)                                    | big-endian bytes                                                                                                            |
| `f32`, `f64`                                                 | IEEE-754 total-order map: if the sign bit is set, invert all bits; otherwise set the sign bit (deterministic including NaN) |
| `Utf8`/`LargeUtf8`                                           | UTF-8 bytes with `0x00` escaped as `0x00 0xFF`, terminated by `0x00 0x00`; byte order equals scalar order                   |

Composite key = concatenation of per-column encodings in `group_by`
declaration order; because every non-null encoding begins `0x01` and every
null is the single byte `0x00`, concatenation is unambiguous and memcmp on
the composite implements the row ordering. Properties and motivations:

- Byte comparison reproduces each column type's natural order (chosen over
  length-prefixed encodings, which are deterministic but scramble string
  lexicographic order and make replay-diff debugging hostile).
- Deterministic across runs and recoveries: pure function of key values and
  the compiled schema; independent of platform endianness.
- The same encoding is used wherever deterministic group-key order is
  serialized or observed: window emission order (S6.4) and state segment
  row order (S6.5 compaction stability).
- Group-key columns outside {bool, integer, float, utf8, date,
  timestamp(us)} types are rejected at compile time with the column path;
  timestamps must satisfy D1.6 (naive-as-UTC or `"UTC"`).
- **Bounded keys (M4.3 note).** A composite key's encoded size is bounded
  (M4.3 freezes the bound, 64 KiB unless its review finds otherwise);
  oversize keys fail at the operator boundary with the column path, so the
  sort path never materializes unbounded encodings.
- **Float grouping (M4.3 note).** The encoding's total order distinguishes
  `-0.0` from `+0.0` and orders NaN deterministically. M4.3 must state
  whether grouping equality matches the encoding order; this note
  recommends byte-equality of the encoding (so `-0.0` and `+0.0` are
  distinct groups and each NaN payload its own group), which keeps
  grouping and emission order consistent by construction.

### G2. Window bound column timezone annotation (constrained by S6.4, D1.4, D1.6)

`window_start` and `window_end` are emitted as Arrow
`timestamp(us, "UTC")`: full internal microsecond precision (S6.4) with the
explicit `"UTC"` timezone string. Rationale: self-describing in Studio and
Parquet sinks, and re-importing the column satisfies D1.6's acceptance rule
(naive-as-UTC or explicit `"UTC"`) with an identity conversion - a naive
column would also be accepted but forces every consumer to re-derive the
UTC convention. Internal `EventTime` -> export conversion is exact at
microseconds (D1.4), so no flooring ever applies to window bounds.

### G3. Checkpoint manifest JSON field layout (implements D4, D9, S2, S5, S7.5, S9.4; plan M5.1 required list)

```json
{
  "format_version": 3,
  "pipeline_name": "pg-cdc-to-clickhouse",
  "pipeline_fingerprint": "<sha256 hex>",
  "runtime_config_hash": "<sha256 hex>",
  "epoch": 7,
  "created_at": "2026-08-03T00:00:00.000000Z",
  "recovery_status": "final",
  "sources": {
    "events": {
      "cursor": { "order": "0000000001a2b3c8", "payload": { "lsn": "0/1A2B3C8" } },
      "identity_hash": "<sha256 hex>",
      "sequence": 15233,
      "ended": false,
      "watermark_policy": {
        "kind": "bounded_out_of_orderness",
        "observed_max_micros": 1785000000000000,
        "last_emitted_micros": 1784999995000000,
        "idle": false
      }
    }
  },
  "operators": {
    "hourly": {
      "progress": {
        "input": { "state": "active", "watermark": 1784999995000000 }
      },
      "inline_metadata": { "open_windows": 3 },
      "segments": [
        {
          "operator_id": "hourly",
          "epoch": 7,
          "segment_id": "accumulators",
          "relative_path": "pg-cdc-to-clickhouse-9f2c/hourly/0000000000000007-accumulators.arrow",
          "byte_len": 4194304,
          "sha256": "<64 lowercase hex>"
        }
      ]
    }
  },
  "sinks": {
    "warehouse": {
      "delivery": { "kind": "transactional" },
      "pre_commit": { "ledger_key": "9f2c...:warehouse:7", "staged_path": "...", "row_count": 12044 }
    }
  },
  "state_checksum": "<sha256 hex>"
}
```

Field rules:

- **Validation.** `format_version != 3` -> `UnsupportedVersion`; unknown
  fields, missing nullable fields, duplicate keys, over-depth, and
  over-size (`MAX_MANIFEST_DOCUMENT_BYTES`) all fail closed (M5.1 RED);
  source/operator/sink IDs must equal the plan's binding slots exactly;
  every segment handle must belong to this pipeline fingerprint and epoch;
  a fingerprint mismatch fails before any restore side effect.
- **Cursor shape.** `sources.*.cursor.order` is a lowercase hex string
  (the A4 ordering key); `cursor.payload` is connector-owned strict JSON.
- **Source identity hash.** `sources.*.identity_hash` is the connector's
  declared SHA-256 over its data-semantics options (C2's two-class rule,
  NFR-5), computed identically at compile time and at recovery; recovery
  fails closed on a mismatch before any source opens. Sink-side target
  identity travels inside `sinks.*.pre_commit` (S7.5).
- **Delivery shape.** `sinks.*.delivery` is one of `{"kind": "ordinary"}`,
  `{"kind": "epoch_idempotent", "mechanism": "<id>", "retention":
  "unbounded"|"bounded"}`, or `{"kind": "transactional"}` - exactly the
  binding-time sample (A5). A bounded-retention epoch-idempotent record is
  reported as retry-deduplicated within a bounded window, never as
  exactly-once (S9.4).
- **EventTime fields** (`watermark`, `observed_max_micros`,
  `last_emitted_micros`) are exact i64 microseconds (D1.2). `epoch`,
  `sequence` are u64. `created_at` is RFC 3339 UTC.
- **D4 uphold.** `segments` entries appear only after staging, durable
  write, length/SHA-256 validation, and atomic rename into the committed
  area (with an fsync of the committed parent directory); every
  `relative_path` denotes the committed path, and the manifest that
  references the segments is published last (D4.1's stage -> write ->
  validate -> rename -> manifest order). The manifest is written to a
  temporary name, fsynced, atomically renamed, and the parent directory
  fsynced. The manifest is bounded: handles and metadata only, never keyed
  rows (D4.4).
- **`recovery_status`.** Present and required, with exactly one valid value
  in 3.0: `"final"`. It satisfies plan M5.1's required "manifest status"
  field without creating a second source of truth: any other value fails
  closed at load, so in 3.0 manifest existence remains equivalent to
  completeness (D4.2), and per-sink completion is derived from
  `sinks.*.pre_commit` per S7.5. The enum reserves the extension point the
  spec left open ("whether the manifest additionally carries an explicit
  recovery-status field" - section 9) without a format bump.
- **`state_checksum`.** SHA-256 hex over the canonical JSON serialization
  of the object `{"operators": ..., "sinks": ..., "sources": ...}` (keys in
  that fixed order), satisfying M5.1's deterministic state-metadata
  checksum.
- **Determinism (I2).** All maps serialize in `BTreeMap` key order through
  the existing canonical JSON writer; repeated serializations are
  byte-identical.
- **Redaction (I4).** `cursor`, `pre_commit`, and `watermark_policy`
  contain connector-owned data validated as strict JSON; connectors are
  responsible for never placing secrets there, and the redaction tests of
  AC-I cover the serialized document.

---

## Example (happy path, Rust)

```rust
use std::{collections::BTreeMap, sync::Arc, time::Duration};

use calc_flow::{
    DeliveryGuarantee, EdgeBudget, ExpressionOperator, FileCheckpointStore,
    LocalStateBackend, PipelineBuilder, SinkBinding, SourceBinding,
    StreamRequirements, StreamRuntimeConfig, StreamingRunner, UdfRegistry,
    UnionOperator, WindowAggregateOperator, WindowSpec,
};

#[tokio::main]
async fn main() -> calc_flow::Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    let mut requirements = StreamRequirements::default();
    requirements
        .delivery
        .insert("output".into(), DeliveryGuarantee::ExactlyOnce);

    let plan = PipelineBuilder::new("hourly_totals")?
        .add_node("normalize", Box::new(ExpressionOperator::new(
            "normalize", "total = a + b", Vec::new(), None, Vec::new(),
        )?))?
        .add_node("hourly", Box::new(WindowAggregateOperator::new(
            "hourly",
            WindowSpec::tumbling("event_time", Duration::from_secs(3600))
                .group_by(["symbol"])?
                .aggregate(AggregateFunction::Sum, "total", "total_sum")?,
        )?))?
        .connect("normalize", "hourly")?
        .compile_stream(&udfs, &requirements)?;

    let mut runner = StreamingRunner::new(
        plan,
        BTreeMap::from([("input".into(), SourceBinding {
            source: Box::new(my_kafka_source()),
            backpressure: calc_flow::BackpressurePolicy::Block,
        })]),
        BTreeMap::from([("output".into(), vec![
            SinkBinding::Transactional(Box::new(my_parquet_sink())),
        ])]),
        Arc::new(LocalStateBackend::new(".calc-flow-state")?),
        Arc::new(FileCheckpointStore::new(".calc-flow-checkpoints")?),
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_secs(60),
            checkpoint_timeout: Duration::from_secs(600),
            edge_budget: EdgeBudget { max_rows: 10_000, max_bytes: 64 << 20 },
            retained_epochs: 2,
        },
    )?;

    let job = runner.start().await?;
    println!("epoch {}", job.trigger_checkpoint().await?);
    let outcome = job.shutdown().await?; // drain, final epoch, join
    assert_eq!(outcome.state, job.wait().await?.state); // idempotent read
    runner.shutdown().await
}
```

## Example (happy path, Python)

```python
from datetime import datetime, timezone

from calc_flow import Batch, PipelineBuilder
from calc_flow.runtime import (
    Data,
    Idle,
    StreamingRunner,
    StreamRuntimeConfig,
    Watermark,
)


class TickSource:
    def capabilities(self):
        return {"replayable": True, "max_batch_rows": 1000,
                "max_batch_bytes": 1 << 20}

    def open(self, cursor):
        self._offset = 0 if cursor is None else int(cursor["payload"]["offset"])

    def next(self):
        if self._offset >= 10:
            return None
        batch = make_batch(self._offset)  # one formed calc_flow.Batch
        self._offset += 1
        # order: big-endian u64 hex; bytewise order == source order (A4).
        order = self._offset.to_bytes(8, "big").hex()
        return Data(batch, {"order": order, "payload": {"offset": self._offset}})

    def close(self):
        pass


plan = (
    PipelineBuilder("hourly_totals")
    .expression("normalize", "total = a + b")
    .compile_stream(delivery={"output": "at_least_once"})
)
runner = StreamingRunner(
    plan,
    sources={"input": TickSource()},
    sinks={"output": [print_batch]},
    state_backend=FileStateBackend(".calc-flow-state"),
    checkpoints=FileCheckpointStore(".calc-flow-checkpoints"),
    config=StreamRuntimeConfig(),
)
job = runner.start()                  # blocking; raises RuntimeError in a loop
epoch = job.trigger_checkpoint()
outcome = job.wait()                  # bounded sources drain to completed
assert outcome.state == "completed"
# The drain publishes a final epoch numbered per D9 (S8.2, D9.2):
assert outcome.completed_epoch > epoch
```

## Design answers to the M0.2 adversarial concerns

- **Cancellation safety.** Sources are never cancelled mid-stream (D3 model
  b, A4 rustdoc contract); operator/sink teardown observes the token at
  await boundaries (D5.2); Python teardown cancellation is delivered once
  (D3.5.6).
- **Reentrancy.** One active job per runner AND per lineage (pipeline
  fingerprint plus state root): the lineage is exclusively locked via the
  state backend/checkpoint store for the job's lifetime, and a conflicting
  `start()` fails with `Conflict` before any source opens. Checkpoint
  requests serialize FIFO without coalescing (D9.5); `shutdown`/`cancel`/
  `wait` are idempotent-convergent (S8.4/S8.5); `status` is always safe.
- **Task ownership.** Supervisor registry plus runner-scoped reaper per
  D5; the runner core is `Arc`-shared with every live job handle, so the
  reaper target always exists. `StreamingJob::drop` and
  `StreamingRunner::drop` fire the cancellation token and never join.
  `StreamingRunner::shutdown(&mut self)` is idempotent and resumable:
  it cancels and joins live jobs first, then the reaper; the D5.4
  no-outlive guarantee binds on that completed path only, with the
  bare-drop path observable (metric plus log) and asserted via token state
  and task-completion flags. Connectors never spawn Tokio tasks (A4 rule,
  I5).
- **Checkpoint recovery.** Restore reads manifests only (D4.2); operator
  entries run before any source opens; recovery is always a new job (S8.3);
  `TransactionalSink::recover` completes durable pre-commits without
  rewriting data and removes staged artifacts no manifest records (S7.5).
- **Capability spoofing.** Transactional is a type-level protocol (A5);
  epoch-idempotency is a once-sampled declaration carrying its mechanism
  and retention class into the plan record, the manifest, and S9.4
  reporting; bounded retention is never reported as unqualified
  exactly-once; exactly-once verification runs before any source opens
  with the failing path named (S9.2); `DropOldest` anywhere on the path
  fails exactly-once compilation (S10.4).
- **Secret persistence.** One structured reference shape (C2); values
  resolve at open time only; connector errors after resolution are wrapped
  and sanitized; documents, fingerprints, manifests, metrics, logs, SSE,
  and error messages carry references at most (I4).

## Appendix A. Plan M0.2 checkbox coverage map

| Plan M0.2 checkbox                                                   | Sections                      |
| -------------------------------------------------------------------- | ----------------------------- |
| 冻结 Rust 类型名、所有权、async 方法和错误 variant                                  | A1-A8                         |
| 验证 `StreamCollector` 的 object safety、生命周期和 error propagation         | A2.1                          |
| 冻结 `StreamingRunner` 的 start/wait/shutdown/cancel/checkpoint/status  | A6                            |
| 冻结 Python async/blocking API；blocking API 在 running event loop 中必须拒绝 | B1-B4                         |
| 冻结 project v3 顶层结构和 secret reference                                 | C1-C3                         |
| 冻结 Studio `/api/v3` job route 与 SSE event model                      | D1-D3                         |
| `StreamSource` 取下一项的取消安全契约                                           | A4 (D3.5 rustdoc contract)    |
| `Drop` 只取消不 join 的所有权模型                                              | A6 (Drop contract, D5.3/D5.4) |
| barrier 转发时机二选一，记录延迟代价与上限                                            | F1                            |
| spec section 9 deferrals: 类型名/collector/Python split                 | A1-A8, B1-B5                  |
| spec section 9 deferrals: connectors 依赖边、feature、CI、覆盖率              | E1-E5                         |
| spec section 9 deferrals: project v3 与 secret、Studio 路由与 SSE         | C1-C3, D1-D3                  |
| spec section 9 deferrals: group-key 编码、window 列时区                    | G1, G2                        |
| spec section 9 deferrals: manifest JSON 布局与 recovery-status          | G3                            |

Items on the M0.2 list owned by the critique artifact (adversarial pass,
`Block` verdicts) are out of scope here; the "Design answers" section above
supplies the positions it will attack.

## Open questions

None blocking. Non-blocking notes with named owners: (1) the
`SecretResolver` trait/registry seam - the resolution and redaction
boundary behind C2, built-in `"env"` resolver - is owned by M6.1; (2)
whether a `vault`-style resolver ships in 3.0 (C2's registry admits either
without a format change); (3) the per-connector declared split of options
into data-semantics vs credential/transport is owned by M6.1's capability
surface under C2's two-class rule; (4) Studio fresh-lineage escape hatch:
M7.3 documents the manual procedure (stop the job, remove or repoint the
state root and checkpoint directory); (5) M6.9 verifies the SSE route
survives `npm run sync:api` into `web-ui/openapi.json` and
`web-ui/src/api/schema.d.ts` with a frontend-consumable type, including
`Last-Event-ID` resume; (6) Studio batch-mode job parity for v2 preview
ergonomics is an M6.9 UI concern, not an API shape concern; (7) the
operator-scoped lazy `DataFusionRuntime` (one per table stream-operator
task, none for array-only chains, plan section 2.2) is deliberately absent
from this note as an implementation detail - M2.3's RED list is its
intended home.
