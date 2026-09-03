# Rust API

The `calc-flow` crate is the implementation of Calc Flow 4.0. Its public
surface is re-exported from `calc_flow`; no Python process is required.

## Build and document

```bash
cargo add calc-flow@4.0.0
cargo test -p calc-flow --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow --no-deps
```

## Expression pipeline

This is the canonical first example, exercised by
[`crates/calc-flow/examples/expression_pipeline.rs`](../crates/calc-flow/examples/expression_pipeline.rs)
and a true twin of the Python quickstart:

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ExpressionOperator, PipelineBuilder, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("totals")?
        .add_node(
            "calculate",
            Box::new(ExpressionOperator::new(
                "calculate",
                "total = a + b",
                Vec::new(),
                None,
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let input = RecordBatch::try_from_iter(vec![
        (
            "a",
            Arc::new(Int64Array::from(vec![1, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("b", Arc::new(Int64Array::from(vec![2, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![input], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let totals = output.batches()[0]
        .column_by_name("total")
        .expect("expression output contains total")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("total is an Int64 column");

    assert_eq!(totals.values(), &[3, 7]);
    println!("calculated totals: {totals:?}");
    Ok(())
}
```

`PipelineBuilder` consumes and returns itself, so partially built graphs cannot
be mutated through aliases. Compilation validates ports and topology and binds
an immutable `UdfRegistrySnapshot`. `BatchExecutionPlan::execute` is
asynchronous and accepts only `Batch` values.

### Engine boundary

`ExpressionOperator`, `SqlOperator`, and `RollingOperator` are built-in
operators that implement both `BatchOperator` and `StreamOperator`;
`UnionOperator`, `WindowAggregateOperator`, and `StreamJoinOperator` are
stream-only. Add them through `PipelineBuilder::add_node`, which accepts a
`NodeOperator` conversion from a boxed built-in, custom batch operator, or
custom stream operator.

Every operator implements `OperatorMetadata`. Custom finite operators implement
`BatchOperator`, whose `BatchOperatorContext` carries the run-scoped context;
custom continuous operators implement `StreamOperator`, whose
`StreamOperatorContext` carries the stream-job context, operator identity,
current input watermark, late-row counters, and read-only access to declared
job-static batches through `static_input(name)`. External operators resolve
through `ProviderRegistry` and lifecycle-specific factories, covered below.

`BatchExecutionPlan::datafusion_config()` returns `Option<DataFusionConfig>`:
`None` means the plan is external-only and owns no table resources, so its runs
create no DataFusion session or UDF snapshot.
`BatchExecutionPlan::requires_datafusion()` returns the classification alone.

## SQL operators

`SqlOperator::new` declares the input aliases that become operator ports. It
accepts one read-only `SELECT` or CTE. Connect upstream node outputs to those
aliases with `Edge`/`PortEndpoint`, or leave them unconnected as graph inputs.
DDL, DML, utility commands, and multiple statements are rejected before
execution. The canonical SQL example is
[`crates/calc-flow/examples/sql_join.rs`](../crates/calc-flow/examples/sql_join.rs),
which joins `orders` to `fees` for `net = amount - fee`:

The run-scoped physical planner has one fail-closed rolling extension. A
bounded ascending `AVG(Float64) OVER (PARTITION BY simple_columns ORDER BY
non_null_timestamp_us, non_null_sequence... ROWS n PRECEDING ... CURRENT
ROW)` may execute through the crate-private `CalcFlowRollingExec`; it reuses
the same typed `RollingKernelPlan` transition as `RollingOperator`. Filters,
`DISTINCT`, explicit null treatment, descending or expression order keys,
`RANGE`/`GROUPS`, future or unbounded frames, other aggregates and unsupported
types remain on DataFusion's standard window executor. `DataFusionQueryMetric`
reports the candidate/rewrite counts, stable fallback reasons, configured and
effective partition counts, and the physical plan. Configured parallelism is
adapted downward below 65,536 input rows per useful partition so small queries
stay single-partition; it is never increased above `target_partitions`.

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, PipelineBuilder, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("orders-and-fees")?
        .add_node(
            "join",
            Box::new(SqlOperator::new(
                "join",
                "SELECT orders.order_id, orders.amount - fees.fee AS net \
                 FROM orders JOIN fees ON orders.order_id = fees.order_id \
                 ORDER BY orders.order_id",
                vec!["orders".to_string(), "fees".to_string()],
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let orders = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("amount", Arc::new(Int64Array::from(vec![75, 120, 40])) as _),
    ])?;
    let fees = RecordBatch::try_from_iter(vec![
        (
            "order_id",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("fee", Arc::new(Int64Array::from(vec![5, 12, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([
                (
                    "orders".into(),
                    Batch::table(vec![orders], BatchMetadata::default())?,
                ),
                (
                    "fees".into(),
                    Batch::table(vec![fees], BatchMetadata::default())?,
                ),
            ]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let net = output.batches()[0]
        .column_by_name("net")
        .expect("sql output contains net")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("net is an Int64 column");

    assert_eq!(net.values(), &[70, 108, 36]);
    println!("net amounts: {net:?}");
    Ok(())
}
```

## Rolling windows

`RollingOperator` evaluates native lag, delta, EWMA, and aggregate outputs over
entity-partitioned, event-time-ordered rows. Its data-only `RollingSpec`
declares ordered `partition_by` and `sequence_by` keys, a non-null UTC
`timestamp[us]` `event_time` column, and one `RollingOutputSpec` per rolling
column: kind `lag` or `delta` — `primitive_version` 1, input and output
column names, and a positive `periods` distance; kind `count`, `sum`, `mean`,
`min`, `max`, `variance`, or `stddev` — `primitive_version` 1, input and
output column names, a positive frame, and `min_periods`, with `variance` and
`stddev` adding `ddof`; or the pair kinds `covariance` and `correlation` —
`primitive_version` 1, left and right input column names, an output column
name, a positive frame, `min_periods`, and `ddof`. Kind `ewma` carries
input/output names, a positive `span`, and positive `min_periods`; it emits nullable `float64` from the
unadjusted `alpha = 2 / (span + 1)` recurrence and shares constant state by
`(input, span)`. Kind `difference` embeds two `mean`, `variance`, `stddev`, or
`ewma` readouts and one output name. The leaves create and share state but are
not output columns; the operator writes their nullable `float64` difference
directly.

A frame is `rows(size)` —
the `size` rows through the current row of the entity total order — or
`duration(micros)`: the exact-width event-time interval `(t − micros, t]`,
open at the lower bound and closed at the upper bound, with equal-time rows
ordered by `sequence_by`. `configuration_version` and `state_layout_version`
must equal `ROLLING_CONFIGURATION_VERSION` (1). Existing rolling declarations
use `ROLLING_STATE_LAYOUT_VERSION` (1), while any EWMA declaration requires
`ROLLING_EWMA_STATE_LAYOUT_VERSION` (2) so checkpoint segments persist the
valid count and exact binary64 accumulator. These declaration versions remain
stable in project documents and configuration fingerprints. Current operators
write `ROLLING_COLUMNAR_STATE_LAYOUT_VERSION` (3): one deterministic entity
dictionary, projected retained history, a columnar reorder buffer, recurrence
state, and kernel/numerical fingerprints. Restore continues to read the
declaration's v1/v2 layout, while newly emitted descriptors and pipeline
capabilities report writer layout v3. `allowed_lateness_micros` and a
`LatePolicySpec` of `Error` (envelope scope) or `Drop` (metrics version 1)
classify late rows
against the input watermark, and `RollingValuePolicy` is the frozen
`stateful_numeric_v1`, which preserves a null or NaN current or referenced
value.

`numerical_profile` is optional. Its default `stable_v1` is omitted from the
canonical configuration, preserving existing project and checkpoint hashes.
The explicit `stable_v2` value is a preview: floating numeric and pair windows
periodically rebuild from retained values with shifted compensated sums. The
per-entity transition count is stored in the existing nullable columnar-state
position field, so the cadence survives checkpoint recovery without changing
the layout-v3 field schema. The profile is part of both configuration and
kernel fingerprints, and restore rejects a profile mismatch.

Validation rejects unknown fields, empty or duplicate keys, duplicate
or input-colliding output names, nullable or floating sequence columns,
non-numeric `delta`, `ewma`, `sum`, `mean`, `variance`, `stddev`, `covariance`,
and `correlation` inputs, `min` and `max` inputs outside the total-order types
(booleans, integers, floats, strings, dates, and `timestamp[us]` with no
timezone or UTC), a `min_periods` below one or — for `rows` frames only — above
the frame size,
a `ddof` outside 0 or 1, and ambiguous column references; `count` accepts any
input column. The output schema is the input schema followed by the declared
outputs in declaration order, each nullable. Lag, delta, `min`, and `max`
keep the input type; the other aggregates follow the frozen output-type
table:

| Kind                           | Input            | Output type       |
| ------------------------------ | ---------------- | ----------------- |
| `ewma`                         | numeric          | `float64`         |
| `count`                        | any column       | `uint64`          |
| `sum`                          | signed integer   | `int64`, checked  |
| `sum`                          | unsigned integer | `uint64`, checked |
| `sum`                          | floating         | `float64`         |
| `mean` / `variance` / `stddev` | numeric          | `float64`         |
| `min` / `max`                  | total order      | input type        |
| `covariance` / `correlation`   | numeric pair     | `float64`         |
| `difference`                   | float readouts   | `float64`         |

Aggregates count valid samples — values that are neither null nor NaN;
infinities are numeric samples and count toward `min_periods`. A window with
fewer than `min_periods` valid samples reads null, and a computed NaN stays
observably distinct from that gate. `min` and `max` read a monotonic queue of
valid samples and preserve the input type. Floating sums follow IEEE
arithmetic: a window holding both signs of infinity sums to NaN, and a window
holding one sign sums to that infinity. Integer sums stay exact and checked:
the slide runs through a wide transient accumulator and narrows with a
checked conversion at readout, so a window whose true sum is representable
never reports a false overflow, while a genuine `int64` or `uint64` overflow
fails the run loudly. Mean, variance, and standard deviation read a
reversible West accumulator that also counts the window's positive and
negative infinities: a mean over both signs is NaN (the undefined ∞ − ∞), a
mean over one sign is that infinity, and a finite window keeps the
incremental West value. Variance and standard deviation over any window
holding an infinity are NaN — a pinned contract — and their divisor is
`valid_count − ddof`: a non-positive divisor reads null after the
`min_periods` gate and before the NaN classification. Negative `M2` drift
from reversible removal clamps to zero at readout.

`covariance` and `correlation` read a reversible West co-moment accumulator
over the pairwise-valid positions — rows where both operands are non-null
and non-NaN — so `min_periods` counts pairwise samples and their divisor is
that count minus `ddof`. Their classification order is frozen: the
`min_periods` and divisor null gates come first, then a window holding ±inf
on either side reads NaN, and only a finite window with a zero-variance side
reads null for `correlation` — an infinity window therefore never reads null.

Outputs over the same input column and frame share one accumulator group —
numeric, or one monotonic extrema group per direction for `min` and `max` —
and pair outputs over the same operand pair and frame share one pair group;
`min_periods` and `ddof` are readout parameters, not extra state. Each entity
retains the largest `periods` distance or row-frame size across every output
and, when duration frames are declared, every row inside the widest duration
bound, so mixed row and duration outputs retain by both bounds together. The
same kernel serves both lifecycles. Batch evaluation
orders the complete input canonically and classifies no late rows: every row
is final at end-of-input. Stream evaluation buffers rows until the input
watermark passes each row's closing coordinate — its event time plus the
allowed lateness — then emits the ordered final rows, rejecting the whole
envelope under the `error` policy or dropping each late row and recording
the late metrics under `drop`. Duplicate row identities are rejected. Rolling
stream state checkpoints at the aligned epoch cut as an Arrow IPC segment
with state version 1, and restore or reset reproduces the same ordered
output rows: the segment stores only the retained history and buffered rows,
and every accumulator — the sums, the West state, the infinity counts, the
extrema queues, and the pair state — is rebuilt by the same ordered fold
over those rows, so live, refolded, and restored states read identically.
Sliding arithmetic that becomes non-finite re-folds the current window,
keeping the classification exact at an O(frame) cost per row for those
windows.

Three settled corner properties of this algorithm family are identical on
every path: the incremental mean can overflow opposite-sign extremes
({−1e308, +1e308} reads +inf though the true mean is 0); a floating sum can
overflow its finite partial sum before reaching an infinity ({−1e308,
−1e308, +inf} sums to NaN while the mean reads +inf); and `min`/`max` order
floating samples by the IEEE total order, so −0.0 and +0.0 are
distinguishable extrema — a self-consistent deterministic choice, not SQL
`MIN`/`MAX` semantics, which compare them equal.

## Cross-section statistics

`CrossSectionOperator` evaluates native complete-group rank, percentile,
z-score, demean, winsorize, top/bottom selection, and mean-fill outputs. Its
data-only `CrossSectionSpec` declares
`configuration_version` and `state_layout_version` (both 1 in this release),
a non-null UTC `timestamp[us]` `event_time` column, ordered non-empty
`entity_by` and `sequence_by` keys, an ordered `partition_by` group key —
optional, and empty means one global group per grouping coordinate — an
`ExactTime` or `FixedBucket { width_micros }` grouping (one group per exact
event time, or per UTC `[start, start + width)` bucket with the Unix-epoch
origin flooring toward negative infinity), one `CrossSectionOutputSpec` per
output, `allowed_lateness_micros`, a `LatePolicySpec` of `Error` (envelope
scope) or `Drop` (metrics version 1), and the frozen
`CrossSectionValuePolicy` `NanExcludePreserveV1`. A row identity is the
`(event_time, entity_by, sequence_by)` value.

Each output declares `primitive_version` 1, input and output column names, and
`min_samples`; the per-variant rules below define which values are gated when
a group has fewer valid samples. `rank` and `percentile` add `direction`
(`ascending`/`descending`),
`tie_method` (`average`/`min`/`max`), and `null_placement`
(`exclude`/`first`/`last`); `zscore` adds `ddof` of 0 or 1. Rank, percentile,
demean, and z-score produce nullable `float64`: rank is the one-based position
after the tie method, percentile is
`(rank - 1) / (ordered_count - 1)` with a single ordered value reading
exactly `0.5`, demean subtracts the valid-sample mean, and z-score divides
by the standard deviation over the divisor `valid_count − ddof` — null when
the divisor is not positive or the deviation is zero.

`winsorize` accepts only float32/float64, carries finite `lower` and `upper`
probabilities satisfying `0 <= lower <= upper <= 1`, and clamps to the
Hyndman-Fan type-7 quantiles while preserving the input type. `top` and
`bottom` carry a positive `count` and `include_ties`: they produce nullable
boolean masks over the exact scalar total order, include every boundary tie
when requested, and otherwise use canonical row identity to select exactly
`min(count, valid_count)` rows. `mean_fill` accepts float32/float64, replaces a
null with the complete valid sample's mean when `min_samples` is met, and
preserves valid values, NaN, and the input type. All compatible outputs share
one grouping, sample classification, and measured-value sort per input column.

Under `nan_exclude_preserve_v1`, NaN is excluded from every sample and stays
NaN at its own row, infinity is numeric, and null handling comes only from
`null_placement` or the variant's fixed non-ordering rule: excluded nulls read
null
and leave the ordering, included nulls form one tied class at the requested
end; a null `demean`, `zscore`, or `winsorize` input reads null; top/bottom
map both null and NaN to null; and mean-fill replaces only null. The mean and
deviation classify infinities exactly as the rolling aggregates do: a sample
over one sign of infinity has that infinity as its mean and over both signs
reads NaN, and any infinity in the group makes the variance — and with it
the z-score — NaN. Validation rejects
unknown fields, zero bucket widths, empty keys or output lists, duplicate key
or output names, input-colliding output names, non-numeric input columns,
nullable or floating sequence columns, unsupported key types, and ambiguous
column references. The output schema is the input schema followed by the
declared outputs in declaration order.

One micro-batch is never evidence of completeness. Batch evaluation treats
every group as complete at end-of-input, classifies no late rows, and emits
all groups once, ordered by finality coordinate then group key, with rows
inside a group ordered by `(event_time, entity, sequence)`. Stream
evaluation accumulates groups across envelopes: a group closes when the
input watermark reaches its finality coordinate — the exact event time or
the bucket end, plus the allowed lateness, with equality closing — and emits
once in the same canonical order, releasing its rows and identity index.
End of input flushes every open group once without synthesizing a sentinel
watermark; a repeated end is a no-op and data after end is rejected. A row
whose group has closed is late: the `error` policy rejects the whole
envelope before any state changes, and the `drop` policy discards the row
and records the late metrics. Duplicate row identities are rejected
transactionally — within one envelope across partition groups, and across
envelopes while the identity is still open. Open groups checkpoint at the
aligned epoch cut as an Arrow IPC base segment with state version 1 —
configuration-hash and schema-fingerprint metadata plus bounded inline
manifest fields — and a restored operator reproduces the same ordered
output, watermark frontier, output sequence, and metrics.

## Stream compilation and continuous runtime

`PipelineBuilder::compile_stream` produces an immutable
`StreamExecutionPlan`. The plan records stable edges, source and sink binding
slots, stream requirements, and the semantic fingerprint. Pass it to the
crate-root `StreamingRunner` with `SourceBinding`, `SinkBinding`, and
`ManagedCheckpointRuntime` values.

A plan compiled from a project that declares static side inputs exposes them
through `StreamExecutionPlan::static_input_ids()` and `static_inputs()`;
`source_binding_ids()` excludes the static names. Supply the immutable
`Batch` values with the additive builder, leaving `StreamingRunner::new`
source-compatible:

```rust
let runner = StreamingRunner::new(plan, sources, sinks, checkpoints)?
    .with_static_inputs(BTreeMap::from([("weights".to_owned(), weights)]))?;
```

`with_static_inputs` copies the mapping and is `#[must_use]`; validation,
latching, and digest computation are deferred to `start`, where they happen
exactly once before any source or operator lifecycle runs. The complete frozen
static-input crate-root export set is `StaticInputSpec`, `StaticInputDigest`,
`StaticMutability`, `STATIC_INPUT_DIGEST_VERSION`, `StaticArraySnapshot`, and
`StaticArrayValues`. The complete SCE-13 export delta is exactly the last two
types; `Batch::static_array_snapshot()` and its five read-only accessors are
the associated inherent API. Declarations join the plan fingerprint;
`CheckpointManifest` records one `StaticInputDigest` per name under its
`static_inputs` field, omitted when empty. See
[static inputs](streaming-guide.md#static-inputs) for the full per-job,
digest, and recovery contract.

The source-driven runtime consumes that plan after a whole-job
preflight. It runs one task per source, compiled operator, and graph output;
preserves per-ingress FIFO; routes source control through a job-scoped progress
driver; and converges source, operator, sink, queue, reservation, driver, and
reaper state on every terminal path. The progress driver owns watermark
generation, idle/reactivation, deterministic timers, and multi-ingress
progress.

`WindowSpec` declares fixed UTC tumbling or hopping windows and ordered
aggregates. `WindowAggregateOperator` is stream-only: it updates incremental
state, classifies late row-window assignments against the current input
watermark, emits closed windows in deterministic order, and snapshots retained
Arrow IPC deltas. The complete
[`windowed_streaming.rs`](../crates/calc-flow/examples/windowed_streaming.rs)
example wires a source-provided watermark through this operator to a sink.
`StateBackend` opens an exclusive lineage session;
`LocalStateBackend` provides immutable checksum-verified segment publication.
`CheckpointManifest` is the strict bounded v3 state-manifest contract. Barrier
coordination cuts live sources, aligns operator ingresses, stages operator
state, and publishes the manifest last as the sole durable recovery truth.
Transactional sinks commit only after manifest durability. Recovery restores
mixed ended/live jobs, and terminal checkpoints capture post-`on_end` state
without forwarding a post-end barrier. Exactly-once compatibility is proved
for each requested graph output before lifecycle work; checkpointed
ordinary-sink outputs remain at least once.

Launch failure and cancellation settle source, operator, sink, queue,
progress, state, checkpoint transaction, and reaper ownership. Cleanup
failures and panics remain typed secondary diagnostics and do not replace the
primary outcome.

`StreamingRunner::start(self)` is one-shot and returns the sole
`StreamingJob` owner. The job exposes synchronous status plus async checkpoint,
shutdown, cancel, and wait operations. The old formed-batch push runner,
`MicroBatchRunner`, v2 source/sink traits, and checkpoint-document store are
removed without aliases. Public continuous lifecycle failures surface as
`CalcFlowError::Streaming(StreamingError)`. Contained task panics use
`StreamingErrorCategory::TaskPanicked`; internal task IDs and panic messages
never cross the facade.

`edge_channel` retains its public signature and `EdgeBudget` retains the fields
`max_rows` and `max_bytes`. For `EdgeBudget::new(R, B)`, envelope count and
charged rows are each independently capped at `R`, while charged bytes are
capped at `B`. Direct callers must choose
`R >= max(required_row_limit, required_simultaneous_messages)`.

## Batches

- `Batch::table(Vec<RecordBatch>, BatchMetadata)` creates an Arrow table batch.
- `Batch::external(Arc<dyn ExternalPayload>, BatchMetadata)` creates an
  explicitly registered external-provider batch.
- `Batch::static_array_snapshot()` returns `Some(StaticArraySnapshot)` only for
  an engine-latched static array. It makes one detached `O(rank + bitmap +
  non-null values)` copy with read-only backend, dtype, shape, null-bitmap, and
  compact-value accessors; tables and general external arrays return `None`.
  `StaticArraySnapshot` and `StaticArrayValues` are non-exhaustive, deliberately
  non-`Clone`/non-`Debug`, and preserve `Send + Sync + Unpin` plus unwind-safe
  auto traits.
- `Batch::kind`, `num_rows`, `metadata`, `table_payload`, and
  `external_payload` expose immutable observations.
- `Batch::with_metadata` returns a new envelope.

`BatchMetadata::new(source, sequence, attributes)` validates the source and stores
JSON-compatible attributes. Its sequence is descriptive batch metadata and is
independent of continuous source cursors and checkpoint epochs.

Python static placement creates this snapshot once per static input inside a
blocking worker and caches only the provider-owned batch after successful
placement and a post-worker cancellation check. The transient peak may include
the engine latch, snapshot carriers, Python host list, NumPy storage, and JAX
result simultaneously. `static_placement_bytes` counts only the logical
provider transfer, not that peak memory or the internal snapshot copy.

## UDFs and external providers

Create an application-owned `UdfRegistry`, register native DataFusion scalar
UDFs, then compile against `registry.snapshot()`. Operators hold serializable
`UdfReference` values; they never own source code or import paths.

`ProviderRegistry` binds `ExternalOperatorSpec` values to trusted
`BatchOperatorFactory` and `StreamOperatorFactory` implementations through
separate `register_batch`/`resolve_batch` and `register_stream`/`resolve_stream`
paths. External provider state and payloads must satisfy Rust's `Send + Sync`
boundaries.

An external `StreamOperator` reports its proof through `lifecycle()`. The
default is `StreamOperatorLifecycle::Unproven`. A trusted factory may return
`StreamOperatorLifecycle::Stateless` only when the operator owns no checkpoint
state; its `microbatch_invariant` flag must also be true so splitting or
combining input micro-batches cannot change the output rows.

Stream planning keeps lifecycle proof and checkpoint capability distinct.
`PipelineBuilder::add_node` classifies a proven external stateless operator as
`Stateless`, and `compile_stream` revalidates its lifecycle proof. The
stateless proof keeps `deterministic` and `replay_safe` separate: best-effort
and at-least-once plans may compile without those two claims, but a plan
requesting exactly-once delivery rejects the provider unless both are true.
Operators classified as `CheckpointedStateful` instead qualify through their
positive checkpoint state version and do not need a stateless lifecycle claim.

An `Unproven` external operator is not rejected merely by `compile_stream`, so
ordinary plan construction retains the existing two-stage contract. If a
`StreamingRunner` is configured with a checkpoint runtime, job admission
rejects that operator before any connector opens. Exactly-once delivery
requires a checkpoint runtime, so an unproven operator cannot be admitted on
that route either.

## Continuous execution and recovery

The complete checked example is
[`continuous_runtime.rs`](../crates/calc-flow/examples/continuous_runtime.rs).
Its central lifecycle is:

```rust
let runner = StreamingRunner::new(
    plan,
    BTreeMap::from([(source_id, SourceBinding::new(source))]),
    BTreeMap::from([(
        output_id,
        vec![SinkBinding::ordinary("archive", sink)?],
    )]),
    ManagedCheckpointRuntime::new(".calc-flow-continuous")?,
)?;
let job = runner.start().await?;
let outcome = job.wait().await;
```

A `StreamSource` declares replay, delivery, schema, watermark, and batch-bound
capabilities, then implements async `open`, `next`, and `close`. An ordinary
`StreamSink` implements async `open`, `write`, and `close`; transactional sinks
also expose epoch commit and recovery. Managed checkpoints bind source cursors,
operator state, and sink evidence to the plan fingerprint.

Run the checked examples:

```bash
cargo run -p calc-flow --example expression_pipeline
cargo run -p calc-flow --example sql_join
cargo run -p calc-flow --example continuous_runtime
cargo run -p calc-flow --example windowed_streaming
```

The [continuous streaming guide](streaming-guide.md) explains cursor,
watermark, window, delivery, checkpoint, and recovery behavior across both
language surfaces.

## Projects and stores

`ProjectSpec` is the strict v3 data model. Use `validate_project` for a
`ValidationReport`, `compile_project` for a `BatchExecutionPlan`, and
`project_json_schema` for the generated schema.

`FileProjectStore` is the public async atomic local project store. Project JSON
is canonical; YAML is safe import/export only. Continuous recovery storage is
owned by `ManagedCheckpointRuntime`, while `LocalStateBackend` and
`CheckpointManifest` expose the data-only v3 state contract. Print the
canonical project schema with
[`crates/calc-flow/examples/export_schema.rs`](../crates/calc-flow/examples/export_schema.rs).

## Errors and cancellation

Public operations return `calc_flow::Result<T>`. `CalcFlowError` defines typed
variants for invalid arguments/documents, compilation errors,
execution/provider failures, checkpoint errors, I/O paths, cancellation,
closed stream edges, and private runtime task-panic capture. Private capture
keeps `TaskPanicked.message` valid UTF-8 and at most 1,024 bytes including its
ellipsis; non-string panic payloads use a fixed message. Fallible
`ManagedCheckpointRuntime`, `StreamingRunner`, and `StreamingJob` operations
expose only `CalcFlowError::Streaming(StreamingError)`; contained panics use
`StreamingErrorCategory::TaskPanicked` and retain neither the internal task ID
nor the panic message.

`ExecutionOptions` carries cancellation/deadline controls. Operators receive a
`RunContext` and must check cancellation at safe work boundaries.
`RunContext::settings()` returns the immutable run settings and
`RunContext::deadline()` returns the optional absolute UTC deadline by
reference, allowing providers to observe the authoritative run context without
duplicating it.
