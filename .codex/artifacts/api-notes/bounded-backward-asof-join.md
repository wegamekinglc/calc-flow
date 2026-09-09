# Bounded backward ASOF Join - API Note

Date: 2026-09-09. Source baseline: PR #259 branch
`feature/python-expression-api-refactor`, commit
`eda1583751abbd1ca4d246fcb8ee6b70f57d9b09`.
Status: public surface selected for the blocking specification/API review;
implementation and final review remain required. The dependent implementation
PR targets that feature branch unless its changes have already reached `main`.

## Audiences

- Python application users: attach the latest valid historical information to
  each left event, using immutable `TableExpr` declarations and owned `.stream()`
  results. Avoid repeating temporal metadata already declared on inputs.
- Rust runtime users: construct one strict stream operator, validate exact Arrow
  schemas, inspect bounded state, and recover through the existing managed runner.
- Studio clients: import/export the same data-only project-v3 operator and inspect
  its progress and structured failures through existing `/api/v3` routes.

## Surface Today

[Introduction](../../../docs/introduction.md) now leads with Python expressions,
`compute`, `Program`, and owned async stream results. The Rust crate remains the
single native runtime. This note follows that vocabulary and application path.

Existing bounded inner Join has these entry points:

```rust
StreamJoinSpec::inner(left_keys, right_keys, left_event_time,
                      right_event_time, bounds, limits) -> Result<StreamJoinSpec>
StreamJoinOperator::new(name, left_schema, right_schema, spec)
    -> Result<StreamJoinOperator>
```

```python
cf.table.stream_join(left, right, *, left_keys, right_keys,
                     left_event_time, right_event_time, bounds, limits, ...)
PipelineBuilder.stream_join(name, *, left_schema, right_schema, ...)
```

It emits every bounded inner match and retains its existing `stream_join@1`
native capability, symbolic primitive versions 1/2, configuration defaults,
output nullability, fingerprint, and checkpoint readers. ASOF adds a separate
operator; it does not add members to `StreamJoinType` or fields to `StreamJoinSpec`.

Relevant existing surfaces are [Rust exports](../../../crates/calc-flow/src/lib.rs),
[inner Join](../../../crates/calc-flow/src/operator/join.rs),
[shared Python inner spec](../../../python/calc_flow/join_spec.py),
[TableExpr](../../../python/calc_flow/symbolic/expr.py),
[table namespace](../../../python/calc_flow/symbolic/ops.py),
[advanced builder](../../../python/calc_flow/pipeline.py),
[native stub](../../../python/calc_flow/_native.pyi),
[stream input ownership](../../../python/calc_flow/_stream_inputs.py),
[OpenAPI](../../../web-ui/openapi.json), and the existing
[inner Join example](../../../examples/12_symbolic_stream_join.py).

## Proposed Surface

### Rust runtime declarations

Export the following types from `calc_flow`. Each configuration value owns its
strings and collections, exposes read-only accessors, validates construction and
strict deserialization, and derives the applicable serde/JSON Schema traits.
Signatures below omit repetitive generic `Into<String>` bounds and accessors.

```rust
pub enum AsofLatePolicy {
    Error,
    Drop,
}

pub struct AsofStateLimits { /* validated, private fields */ }
impl AsofStateLimits {
    pub fn new(max_state_rows: u64, max_state_bytes: u64) -> Result<Self>;
    pub const fn max_state_rows(&self) -> u64;
    pub const fn max_state_bytes(&self) -> u64;
}

pub struct AsofJoinSide { /* keys, event_time, sequence_by, prefix */ }
impl AsofJoinSide {
    pub fn new(
        keys: Vec<String>,
        event_time: String,
        sequence_by: Vec<String>,
        prefix: String,
    ) -> Result<Self>;
}

pub struct StreamAsofJoinSpec { /* validated, private fields */ }
impl StreamAsofJoinSpec {
    pub fn new(
        left: AsofJoinSide,
        right: AsofJoinSide,
        tolerance: std::time::Duration,
        limits: AsofStateLimits,
    ) -> Result<Self>;
    pub fn with_late_policy(self, policy: AsofLatePolicy) -> Self;
}

pub struct StreamAsofJoinOperator { /* owned native state */ }
impl StreamAsofJoinOperator {
    pub fn new(
        name: impl Into<String>,
        left_schema: arrow::datatypes::SchemaRef,
        right_schema: arrow::datatypes::SchemaRef,
        spec: StreamAsofJoinSpec,
    ) -> Result<Self>;
    pub fn spec(&self) -> &StreamAsofJoinSpec;
    pub fn status(&self) -> StreamAsofJoinStatus;
}
```

`new` selects backward, left-preserving, final append-only semantics and defaults
late policy to `Error`. `Duration` must be exactly representable as integral
microseconds; positive sub-microsecond remainder is rejected. There is no
additional `from_micros` constructor: strict project deserialization is the
wire-level integer entry point. Getter `tolerance_micros()` returns the resolved
`u64` value. Side getters return slices/borrowed strings; spec getters return
borrowed sides, copied limits and policy.

The operator implements `OperatorMetadata` and `StreamOperator`, with the
existing `snapshot`/`restore`/`reset` lifecycle; it does not implement
`BatchOperator`. `NodeOperator` and project `OperatorSpec` get an additive
`StreamAsofJoin` variant. `PipelineBuilder::add_node` remains the Rust graph
construction entry point. No public control-injection API or new public
`StreamOperator` trait requirement is introduced.

### Shared immutable Python declarations

Add `python/calc_flow/asof_join_spec.py`, imported directly by the application
expression layer and advanced builder. Export the three data containers below
from root `calc_flow`; they contain no schemas, callbacks, providers, or mutable
runtime state.

```python
@dataclass(frozen=True, slots=True)
class AsofStateLimits:
    max_state_rows: int
    max_state_bytes: int

@dataclass(frozen=True, slots=True)
class AsofJoinSide:
    keys: Sequence[str]
    event_time: str
    sequence_by: Sequence[str]
    prefix: str

@dataclass(frozen=True, slots=True)
class AsofJoinSpec:
    left: AsofJoinSide
    right: AsofJoinSide
    tolerance: timedelta
    limits: AsofStateLimits
    late_policy: Literal["error", "drop"] = "error"
```

Side sequence inputs are copied into tuples during construction. Strings are
not accepted in place of a sequence; elements are non-empty exact strings;
keys and sequence lists are non-empty and contain no duplicate column names.
The same column may occupy different roles, such as key and sequence; uniqueness
is checked within each list. Values are immutable after construction, including
when callers later mutate the lists originally passed in.

Require exact `int` for both limits and exact `datetime.timedelta` for tolerance;
`bool`, float, and implicit duration strings are rejected. Convert timedelta
using integer days/seconds/microseconds, never floating `total_seconds()`.
`late_policy` accepts only the two exact strings. Prefixes are distinct non-empty
ASCII identifiers and must not produce any output-field collision.

These containers express the same configuration as Rust. There is one private
wire encoder and one expression-to-spec resolver. Python does not execute
matching, watermark finalization, or checkpoint state transitions.

### Python application entry points

```python
class _AsofJoinOptions(TypedDict, total=False):
    keys: tuple[Sequence[str], Sequence[str]] | None
    late_policy: Literal["error", "drop"]
    prefixes: tuple[str, str]


# cf.table namespace; existing calc_flow.symbolic.table shares this object.
def stream_asof_join(
    left: TableExpr,
    right: TableExpr,
    /,
    *,
    tolerance: timedelta,
    limits: AsofStateLimits,
    **options: Unpack[_AsofJoinOptions],
) -> TableExpr: ...

# TableExpr method delegates to exactly the same declaration function.
def stream_asof_join(
    self,
    right: TableExpr,
    /,
    *,
    tolerance: timedelta,
    limits: AsofStateLimits,
    **options: Unpack[_AsofJoinOptions],
) -> TableExpr: ...
```

Implementation correction after the contract review: typed `Unpack` groups the
three optional settings so both signatures satisfy the repository's five-argument
complexity limit without a waiver. Calls still use `keys=`, `late_policy=` and
`prefixes=` directly; omitted values resolve to `None`, `"error"` and
`("left", "right")`. Unknown option names raise `TypeError` immediately. One
frozen internal options value owns validation and defensive copying. Signature
introspection shows `**options`; this does not require a caller migration or
permit arbitrary parameters. Final implementation review covers this correction.

Both forms create the same primitive and digest; the fluent form follows the
existing `TableExpr.filter`/namespace convention rather than introducing another
execution path. Two positional inputs and grouped keyword settings keep the
normal call short. Validators and lowering helpers remain small; do not add
argument-count or complexity exemptions for this feature.

Resolve metadata from the analyzed operand at the ASOF boundary:

- `keys=None` selects left and right `entity_by`, respectively. An explicit
  `keys=(["symbol"], ["ticker"])` selects a different matching key without
  changing the input's entity grouping. Copy both supplied lists.
- Both operands must provide non-empty `event_time` and `sequence_by` metadata
  that still names columns in the operand's exact current schema. The resolved
  wire spec records those exact values. Do not guess names or silently repair
  missing metadata after a projection or SQL stage.
- Sequence identity uses each side's effective join keys, selected time and
  declared sequence. If a sequence was unique only under the original entity
  grouping, changing keys may expose a duplicate; native admission rejects it.
- A forged primitive whose resolved time/sequence differs from analyzed operand
  metadata fails analysis. Key overrides are intentional and need not equal
  `entity_by`. Keys and prefixes must each have exactly two sides.
- Output entity metadata is the prefixed effective **left join key**, output
  event time is prefixed left event time, and output sequence is prefixed left
  sequence. The caller does not supply independent output-ordering overrides.

### Advanced Python builder

```python
def stream_asof_join(
    self,
    name: str,
    *,
    left_schema: Sequence[ArrowFieldSpec],
    right_schema: Sequence[ArrowFieldSpec],
    spec: AsofJoinSpec,
) -> PipelineBuilder: ...
```

Return a new builder and defensively copy schemas and configuration. Create
required exact-schema `left` and `right` table ports and one `output` table port.
Use the same private wire encoder as expression lowering. Native project
validation remains authoritative for Arrow schema compatibility and runtime
mode. Advanced callers supply explicit side declarations because their graph
ports do not carry expression temporal metadata.

Do not add a PyO3 `AsofJoinOperator` class or Python per-batch matching callback.
The existing `ProjectDocument`/`Runtime` project compiler already supplies the
native operator. PyO3 changes cover required capability, status and structured
error projection; the native stub records those changes.

### Strict project-v3 wire shape

```json
{
  "kind": "stream_asof_join",
  "spec": {
    "left": {
      "keys": ["symbol"],
      "event_time": "trade_time",
      "sequence_by": ["trade_sequence"],
      "prefix": "trade"
    },
    "right": {
      "keys": ["symbol"],
      "event_time": "quote_time",
      "sequence_by": ["quote_sequence"],
      "prefix": "quote"
    },
    "tolerance_micros": 4000000,
    "limits": {
      "max_state_rows": 100000,
      "max_state_bytes": 67108864
    },
    "late_policy": "error"
  }
}
```

All shown fields are required in raw JSON, including prefixes and late policy.
Constructor defaults are materialized before wire encoding. Every ASOF object
rejects unknown fields. There are no `direction`, `join_type`, `allowed_lateness`,
`max_matches_per_input_batch`, or generic operator `version` fields.
`tolerance_micros` is an integer in `0..=9_007_199_254_740_991`; each limit is an
integer in `1..=9_007_199_254_740_991`. Fractional tokens and booleans fail raw
validation, before typed deserialization loses their original JSON shape.

The new serde variant generates additive `AsofJoinSide`, `AsofStateLimits`,
`AsofLatePolicy` and `StreamAsofJoinSpec` definitions. Project format remains 3;
capability/operator version 1 and checkpoint state/layout version 1 are separate
identities. All ordered field lists and resolved settings participate in the new
primitive digest and fingerprint. Old inner defaults, canonical configuration
bytes, fingerprint, schema, capability, and state readers remain byte-for-byte
or behaviorally unchanged as specified by their frozen compatibility vectors.

### Schema and supported types

Both ports require table batches and exact Arrow schemas. Keys and time/sequence
fields must have non-null schema declarations and actual non-null values. Left
and right key counts are equal and positive; corresponding types are exactly
equal, including timestamp unit/timezone. No implicit casts apply.

The key set is Boolean, signed/unsigned integers of 8/16/32/64 bits, Utf8,
LargeUtf8, Date32, Date64, and Timestamp with exactly matching type parameters.
The sequence set is signed/unsigned integers of 8/16/32/64 bits, Utf8 and
LargeUtf8. Sequence tuples compare lexicographically by typed field values;
strings use binary UTF-8 ordering. Sequence types need not match across sides.
Event time is only `Timestamp(Microsecond, Some("UTC"))`. Unsupported key or
sequence types fail compile/analysis even if both schemas use the same type.
Native v1 payloads use an explicit flat Arrow allowlist: Null, Boolean,
Int8/16/32/64, UInt8/16/32/64, Float16/32/64, Date32/64, valid Time32/64,
Timestamp, Duration, Interval, Decimal32/64/128/256, Utf8/LargeUtf8,
Binary/LargeBinary, and FixedSizeBinary. Nested List/FixedSizeList/Struct/Map/
Union, Dictionary, RunEndEncoded, and view types are rejected at construction
with `invalid_type` at `left_schema.<field>` or `right_schema.<field>`.
This bounds materialization scratch without promising unproved nested logical
expansion accounting. Python/project declarations keep their existing narrower
portable schema vocabulary. Being an accepted payload type does not make a
field a valid key or sequence type.

Output fields are all left fields followed by all right fields, named
`{prefix}__{original_name}`. Preserve each field's Arrow type and field metadata.
Preserve left nullability; set every right output field nullable, including keys,
event time and sequence. Preserve the existing project schema representation's
metadata limits; do not promise new metadata serialization capabilities in
project-v3. No extra boolean match column is emitted: a null right event time
identifies an unmatched result. Output row identity is left-derived, and
canonical order is `(left_time, left_key_tuple, left_sequence_tuple)`.
Physical output Batch boundaries are not stable API.

### Finality, limits and recovery visible to callers

A left event at `t` selects the greatest `(right_time, right_sequence)` at the
same key within inclusive `[t - tolerance, t]`. Emit exactly one logical row only
after each side has watermark strictly greater than `t` or has ended. Equal
watermark does not close `t`; idle is not EOF. No update or retraction follows.

Rows with time strictly below their own accepted watermark are late. Validate
schema/types/nulls first, then late policy, then on-time duplicate identity,
then budgets, before committing the input Batch. For a structurally valid
Batch, late classification counts all late rows before applying policy; if
error policy rejects it, duplicate validation is not attempted. Duplicate
classification counts occurrences whose identity was already retained or
appeared earlier in that on-time Batch. It rejects the complete Batch before
accepted-row/state changes; the reported attempt metrics remain observable. Identity is
`(side, key, time, sequence)`, regardless of payload equality. A late historical
duplicate follows late policy; finite state does not retain an eternal seen-set.

Do not expose input-frontier `C` directly as the output watermark. After draining
all `t < C`, the safe output frontier is `C - 1 microsecond`, with no emission
below the representable minimum. EOF is independent of maximum EventTime. Output
idle is suppressed before dual EOF; downstream stateful stages may consequently
wait for watermark progress. `tolerance` limits historical distance, not elapsed
wall-clock wait. A stall may eventually cause a state-limit failure.

`max_state_rows` and `max_state_bytes` are total per-operator charged state
limits. They include pending left rows, retained right payloads, identity-only
entries, indexes and persisted checkpoint representations as defined in the
accounting version 1 table below. Row/payload eviction never removes
information that can still determine an answer or an on-time duplicate error.

There is a separate transient workspace ceiling equal to `max_state_bytes`;
this is additional to charged persistent state, not an additional public knob.
Finalization chunks also obey the existing positive edge row/byte budgets. The
implementation must check allocation charges before work and avoid retaining a
large Arrow backing buffer via a small slice. `state_bytes` is a deterministic
charge, not a process RSS limit. State, workspace, edge queues, and runtime
allocations must be distinguished in resource evidence.

Rust's ordered index chooses at most one candidate per left row, using Arrow's
typed row encoding. DataFusion performs the bounded candidate-table left join,
projection and ordinal ordering. This split is internal: the public API remains
one native operator and one table engine. No all-pairs intermediate is permitted.

The new checkpoint identity is independent of inner Join. Operator state
contains admitted rows, identity state, emitted-output sequence, counters,
layout/accounting version, and terminal state. The existing runtime wrapper
owns the snapshot of both ingress watermarks/idle/EOF and emitted output
frontier; ASOF cross-validates that snapshot on restore rather than introducing
a competing control-state owner. Restore checks
configuration/schema/limits before installing state. Failed restore is atomic.
Reset clears only operator-owned in-memory state. The managed manifest version
and source/sink protocols remain unchanged.

### Accounting version 1

This section owns the concrete public charging contract; the specification owns
the requirement to enforce it. All sums use checked integer arithmetic. The
accounting version is stored in ASOF checkpoint metadata independently of the
existing inner accounting/layout versions.

```text
state_rows = live_identity_count
state_bytes = 256 * live_identity_count
            + encoded_allocation_capacity
            + retained_arrow_memory
            + retained_segment_capacity
            + 64 * (index_entry_count + segment_entry_count)
            + retained_result_and_cursor_charge
workspace_limit_bytes = max_state_bytes
```

A live identity is counted once whether it owns a pending left payload, right
history payload, or only a duplicate-detection entry. Payload-to-identity-only
conversion does not decrement rows. Releasing that identity decrements rows.
The byte components have these exact meanings:

| Component                          | Version 1 charge                                                                                               |
|------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Identity bookkeeping               | 256 bytes per live identity, including fixed owned record/bookkeeping overhead                                 |
| Encoded key/sequence allocations   | Retained allocation capacity for each independent encoding; multiple references to one shared Arc count once   |
| Retained Arrow arrays              | Sum of `get_array_memory_size()` for each independently owned retained array                                   |
| Prepared/base/delta segments       | Allocation capacity of every distinct segment byte allocation retained by the operator, once                   |
| Index and segment metadata         | 64 bytes per occupied index entry and per retained segment descriptor                                          |
| Retained output payload/cursor     | Same Arrow/encoding allocation rules plus 64 bytes per independently retained output cursor                    |

The index count includes each occupied map/set slot in the operator's identity,
pending-left, key-bucket and ordered-right indexes; two independent references
stored in different indexes each count. Empty container capacity and fixed
operator/config/schema references are covered by fixed bookkeeping or existing
runtime ownership; they are not an excuse to retain an unbounded empty
container. Implementations using a different index representation must document
the same conservative inventory before changing this accounting version.

For stored Arrow rows, use compact owned arrays with no cross-row shared backing
buffers. If an implementation shares a buffer within a single stored row, count
it conservatively through every array's `get_array_memory_size()`; overcharging
is permitted by this formula and never silently switches to `.nbytes`. Do not
retain an input slice while charging only its logical row size. An IPC-only
retained payload representation has zero retained Arrow charge while its owned
byte allocation remains fully charged as retained segment/payload bytes.

Each distinct encoding/segment allocation gets one owner for accounting. Sharing
an `Arc` within the operator does not duplicate that allocation charge; an
independent encoded copy does. Segment descriptors are separately charged even
when their bytes are shared. Snapshot publication transfers/clones ownership
according to the existing runtime; only allocations still retained by the
operator remain in its state charge. This does not remove the runtime's own
checkpoint allocation budget.

Prefer keeping finalization chunks and their cursors wholly in bounded transient
workspace until collector acceptance, while the original left rows remain
pending. Any result/cursor retained between handlers moves into the persistent
charge above. A snapshot never skips such retained pending work.

Admission copies and typed encodings, Arrow candidate tables, DataFusion output
materialization, snapshot/compaction serialization and restore decoding all
reserve finite workspace **before** allocation. An active operation's aggregate
workspace reservation may not exceed `max_state_bytes`. Use a bounded writer
for encodings and a finite DataFusion memory pool whose reservation is part of
that same aggregate. An after-allocation `.len()` check alone does not satisfy
the contract. Existing positive output edge budgets add row/byte constraints;
a single output row that cannot fit produces `asof_output_limit_exceeded`.

Restore does not trust serialized charges. It checks declared lengths before
allocating, decodes within workspace, rebuilds indexes, recomputes this inventory,
and revalidates current limits before replacing live state. During replacement,
old committed state and new scratch state may coexist: old state is charged to
persistent state, and new work is charged to the single workspace ceiling until
atomic installation. Checkpoint serialization must reserve both transient bytes
and any retained replacement segment capacity without uncharged double copies.

This is a versioned conservative logical model, not a bound on allocator RSS.
The resource scenario records charged state and retained Arrow/segment
ownership separately from workspace evidence. The v1 fixed traces record the
enforced workspace ceiling and workspace failures; they do not measure the
shared pool's high-water mark. Focused native lifecycle tests verify reservation
failure, release, cancellation, and pool reuse. Instrumented thread heap peaks
must remain separately named and cannot substitute for a workspace peak or
relabel state + workspace + runtime queues as a single 64 MiB process cap.
Any implementation-driven change to the formula requires explicit artifact
review and updated accounting vectors.

### Root stream integration and delivery

ASOF remains stream-only: `.collect()`, `compute`, and `compile_batch()` reject
it; a finite iterable does not make it a batch ASOF operator.

`TableExpr.stream` and `Program.stream` bind both original dynamic inputs by
logical declaration name, including through prefixes, joins and projections.
With more than one dynamic input, a `watermarks` selection is a mapping such as
`{"trades": ..., "quotes": ...}`. Physical operator IDs and port names are not
valid replacements. For a supplied `SourceBinding`, keep the binding's watermark
policy and reject a conflicting convenience override.

Ordinary event-time iterables keep #259's nondecreasing-time default and
`max_seen - 1 microsecond` watermark policy. This can keep the latest timestamps
pending until later progress or EOF. Out-of-order examples must explicitly use
`BoundedOutOfOrderness` or `SourceProvidedWatermarks`; source-policy rejection
happens before ASOF late policy when the source contract itself is violated.
All ASOF routes still undergo native source-watermark preflight.

`async with` owns and awaits the native job and cleanup; `async for` yields
Arrow tables or named `StreamOutput` events with bounded backpressure. Iterable
sources offer no replay. Temporary checkpoints used by convenience streams do
not promise durable restart or exactly-once application delivery. Exported
projects contain data-only physical graph/binding names. Durable restart uses
`Program.compile_stream()` plus explicit replay-capable sources, sink bindings
and a stable managed checkpoint root. Logical one-result-per-left does not
prevent an ordinary at-least-once sink from observing fault-recovery replays.

### Capability, analysis and explain

The added native capability has exact facts:

```text
kind=stream_asof_join version=1
inputs=(left:required table, right:required table)
outputs=(output:required table)
modes=(stream,)
finality=group_final_append_only
requires_datafusion=true stateful=true microbatch_invariant=true
requires_watermark=true checkpoint_support=checkpointed_stateful
state_version=1 state_layouts=(1,) deterministic=true replay_safe=true
```

The symbolic primitive is independently `stream_asof_join@1`. Its gate checks
every listed fact, including exact stream-only mode, finality, layout and state
version; unknown/incompatible capabilities fail closed. `microbatch_invariant`
means equal logical outputs/canonical order for successful runs with equivalent
legal data/control traces; it does not promise identical Batch sizes, metrics
that count Batches, or successful admission for every possible batching under a
fixed finite workspace.

Each ASOF digest has one physical state owner across all output references.
ASOF is a temporal/finality boundary: transformations may not cross it if they
change the candidate set, left preservation, closure or derived ordering.

The first-version combination contract is:

- Accept direct inputs, supported row-local transformations retaining the
  required metadata, multiple outputs/fan-out, independent ASOF nodes, and ASOF
  chains retaining non-null left-derived identity/time.
- Accept post-ASOF rolling/cross-section and row-local projections/filters under
  their existing analysis rules. Validate the conservative output frontier in
  a real downstream runtime graph.
- Accept mixed inner/ASOF graphs in both directions when the operand provides
  complete non-null time/sequence metadata and the existing inner Join output
  ordering rules are satisfied. ASOF admission still validates actual identity
  uniqueness. The terminal inner result retains its existing finality/ordering
  contract; an inner node does not inherit ASOF finality merely through an edge.
- Do not newly enable event-window-to/from-join or matrix attachment/stateful
  combinations already rejected by the expression layer. SQL drops ordering;
  an SQL-derived operand without supported temporal lineage is rejected.
- Do not claim arbitrary stateful stages before ASOF. Only combinations listed
  as supported in the jointly reviewed specification may lower successfully.

`explain` shows kind/version, effective key/time/sequence on each side,
backward/inclusive tolerance, late policy, left preservation/right nullability,
strict double-watermark closure, output-time source/frontier lag, state/workspace
limits, state identity/layout, unique state-owner count and source/sink delivery
requirements. A default-inferred key is displayed as its concrete resolved list.

### Status and errors

Add `stream_asof_joins: dict[str, StreamAsofJoinStatus]` to Python job status,
keyed by node ID. Keep `stream_joins` and every existing inner status field
unchanged. Rust exports independent `StreamAsofJoinStatus` and
`StreamAsofJoinSideStatus`; micros fields project `Option<EventTime>` to Python
`int | None`. The proposed shape is:

```text
left/right:
  accepted_rows, late_rows, duplicate_rows, watermark_micros, idle, ended
pending_left_rows, retained_right_rows, identity_only_rows
state_rows, state_bytes
emitted_left_rows, matched_rows, unmatched_rows, evicted_right_rows
state_limit_failures, workspace_limit_failures, output_limit_failures
output_watermark_micros
```

All count fields are non-negative `u64` in Rust and `int` in Python; booleans
and nullable time fields retain their declared types. `state_limit_failures`,
`workspace_limit_failures` and `output_limit_failures` increment once when the
corresponding failed operation is reported, using the relevant reason code.
Counter overflow instead fails with `asof_counter_overflow` before wrapping.

Gauges represent retained committed state; successful output counters advance
only after collector acceptance. `late_rows` counts rows rejected/dropped as
late, regardless of policy; it does not include source-rejected rows. At
successful terminal drain, `left.accepted_rows = emitted_left_rows = matched_rows
+ unmatched_rows` and `pending_left_rows = 0`. Failure
counters describe attempts and are not logical-output counts. Counters are
checked, non-wrapping and checkpointed; gauges are revalidated/recomputed on
restore. Node metrics expose no keys, sequence values, identity bytes or payload.

Faults raised by the ASOF operator use existing `CalcFlowError::OperatorReason`
and Python `StreamingRuntimeError`, preserving category `operator`, node/component ID,
epoch and stable reason code. Extend the corresponding Literal and generated
OpenAPI enum additively:

```text
asof_invalid_input
asof_duplicate_identity
asof_late_row
asof_state_limit_exceeded
asof_workspace_limit_exceeded
asof_output_limit_exceeded
asof_counter_overflow
asof_protocol_error
```

`asof_protocol_error` applies only to protocol violations detected inside ASOF.
The operator task calls `MultiInputProgress.evaluate` before dispatching data;
post-EOF input is normally rejected there with the existing
`runtime.progress.aggregate.post_end` field, before the ASOF handler runs.
Source/runtime-first rejection retains its existing structured category and
field/reason projection. Schema validation and source watermark-policy checks
can likewise reject before ASOF admission. Do not translate these errors by
parsing message text or change existing source/runtime error behavior.

Boundary tests must distinguish ASOF-local detection from real operator-task
post-EOF rejection and source/schema-first rejection. They assert the layer's
actual structured diagnostics and verify that the ASOF handler/counters are not
entered when an earlier layer rejects the input.

Compile/import errors continue using structured validation issues and their
field paths; declaration type/range errors use `TypeError`/`ValueError`. Corrupt
or incompatible checkpoints use existing checkpoint mismatch/error categories.
Stable public messages name the side, column or limit but never row payload or
identity values. Representative cases follow; native project paths are prefixed
with `graph.nodes[i].operator.spec`.

| Input violation                             | Field / reason code                                                       | Message text                                                                                     |
|---------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Negative tolerance                          | `tolerance_micros`                                                        | `tolerance_micros must be an integer in 0..=9007199254740991`                                    |
| Zero state rows                             | `limits.max_state_rows`                                                   | `max_state_rows must be an integer in 1..=9007199254740991`                                      |
| Nullable sequence declaration               | `left.sequence_by[0]`                                                     | `left sequence column 'trade_sequence' must be non-null and have a supported total-order type`   |
| Missing expression event-time metadata      | `left.event_time`                                                         | `left input requires declared event_time metadata for stream_asof_join`                          |
| Event time in milliseconds or without UTC   | `right.event_time`                                                        | `right event-time column 'quote_time' must have exact type timestamp[us, UTC]`                   |
| Different corresponding key types           | `left.keys[0]`                                                            | `ASOF key pair 0 requires identical supported Arrow types; left is Int64 and right is UInt64`    |
| Actual null required value                  | `asof_invalid_input`                                                      | `left sequence column 'trade_sequence' contains null values`                                     |
| On-time duplicate identity                  | `asof_duplicate_identity`                                                 | `right input contains a duplicate key/event-time/sequence identity`                              |
| Late row under error policy                 | `asof_late_row`                                                           | `left input contains event time below its accepted watermark`                                    |
| Exceeded total state bytes                  | `asof_state_limit_exceeded`                                               | `stream_asof_join.limits.max_state_bytes exceeded`                                               |
| One row cannot fit transient workspace      | `asof_workspace_limit_exceeded`                                           | `ASOF finalization workspace exceeds max_state_bytes`                                            |
| One output row cannot fit its edge          | `asof_output_limit_exceeded`                                              | `ASOF output row exceeds the output edge budget`                                                 |
| Post-EOF data rejected before ASOF          | Existing runtime/source category; `runtime.progress.aggregate.post_end`   | `input/control after EndOfInput is forbidden`                                                    |
| ASOF-local post-EOF detection               | `asof_protocol_error`                                                     | `right input received data after end-of-input`                                                   |
| Batch compilation                           | `unsupported_mode`                                                        | `stream_asof_join is available only in stream mode`                                              |

Raw import preserves existing validation issue envelopes. For the new ASOF
branch, use `missing_field`, `unknown_field`, `invalid_type`, `out_of_range`,
`invalid_asof_keys`, `incompatible_key_type`, `invalid_event_time`,
`invalid_asof_sequence`, `nullable_identity_field`, `invalid_output_prefix`,
`invalid_asof_late_policy`, `invalid_asof_ports`, and `unsupported_mode`, with
the exact offending nested field path. Existing inner issue codes/messages
remain unchanged. Symbolic validation retains its established codes, including
`ordering_required`, `type_mismatch`, `capability_mismatch`, and `unsupported_mode`.

Reason and component/node identity are structured. Side and column paths use
existing validation issues or the safe diagnostic message; ASOF does not add
generic streaming error attributes. Consumers branch on reason codes, not by
parsing message text. `duplicate_rows` counts on-time identities rejected at the
duplicate validation stage; invalid schema/null or earlier late-policy rejection
does not enter that stage. An input Batch rejected by any admission stage does
not increment `accepted_rows` or modify retained row state. The specification
and implementation tests must use these same validation/counter boundaries.

### Studio and generated contracts

Use existing project create/import/update/export/validate endpoints and existing
job start/status/events endpoints. Add no ASOF-specific route and no new Python
worker matching engine. Regenerate project JSON Schema, OpenAPI and TypeScript
from their existing generators, then verify a second generation is clean.

Studio `RunEvent` gains optional `stream_asof_joins` using a new list of metrics
objects carrying `node_id` plus the ASOF status fields. Job status projection,
worker messages, Pydantic models and hand-authored frontend event types must
agree. Existing inner metrics and error enum members are preserved.

For **new ASOF Studio metrics only**, encode every integer counter, gauge and
`watermark_micros` field as a canonical decimal **string**. Nullable watermarks
stay JSON `null`; `idle` and `ended` stay booleans. Native/Python status remains
`u64`/`i64` projected as Python `int`. The backend explicitly validates exact
integer type and its signed/unsigned range, then performs decimal serialization;
it must not pass the value through a JSON number or float first. Unsigned strings
match `0|[1-9][0-9]*`; signed strings match `0|-?[1-9][0-9]*`, subject to their
native numeric range. No leading plus sign, leading zero, or `-0` is canonical.

Generated OpenAPI/TypeScript ASOF metric fields therefore use `string` or
`string | null`, and the UI displays the exact strings. This additive projection
preserves all i64 event-time values and u64 counter values through JavaScript.
Existing inner/general job status numeric representations remain unchanged.
Test Python/native values at i64 minimum/maximum, u64 maximum, zero and null
against semantically equal Studio strings, along with unchanged booleans.
Configuration tolerance and limits remain JSON-safe **numbers**, with no float
conversion; this status projection does not change project JSON.

The first version requires lossless generic project import/save/export and
readable status/errors. A dedicated graphical ASOF node editor is not required.
Round-trip tests must cover unknown-kind fallback handling so the generic editor
does not drop the new nested spec or right nullable output schema.

## Why This Shape

- Separate operator/state identity prevents the new left-preserving/finalizing
  behavior from changing old inner configuration, schema or recovery semantics.
- Temporal metadata belongs to immutable input expressions. Reusing it removes
  four repeated time/sequence arguments while explicit `keys` keeps independent
  matching-key selection possible.
- Side data objects keep advanced entry points small and wire validation local.
  The normal expression API needs only tolerance and state limits beyond inputs.
- Fixed direction and final output make invalid forward/nearest/update requests
  impossible to express through a generic mode flag.
- A separate status mapping and reason-code family let existing clients continue
  distinguishing inner matches from ASOF left results.

## Example

This is the seed for a 20–50-line runnable trade/quote example. `trades_source`
and `quotes_source` denote async Arrow iterables with the declared schemas; a
runnable example supplies their tiny fixtures without changing the API shown.

```python
from datetime import timedelta
import pyarrow as pa
import calc_flow as cf

schema = pa.schema([
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("sequence", pa.uint64(), nullable=False),
    pa.field("price", pa.float64(), nullable=False),
])
trades = cf.table_input("trades", schema=schema, entity_by=["symbol"],
                       event_time="time", sequence_by=["sequence"])
quotes = cf.table_input("quotes", schema=schema, entity_by=["symbol"],
                       event_time="time", sequence_by=["sequence"])
matched = trades.stream_asof_join(
    quotes,
    tolerance=timedelta(seconds=4),
    limits=cf.AsofStateLimits(100_000, 64 * 1024 * 1024),
    prefixes=("trade", "quote"),
)
result = matched.select("trade__symbol", "trade__price", "quote__price")
async with result.stream({"trades": trades_source, "quotes": quotes_source}) as stream:
    async for batch in stream:
        print(batch.to_pydict())
```

For out-of-order fixtures the shipped example supplies explicit named policies
or source watermarks; it must show no output at equal watermarks, then one final
left result after both sides advance, as well as an unmatched left row. The
narrative distinguishes tolerance from waiting time and convenience delivery
from durable managed recovery. A separate focused recovery test uses replayable
bindings and proves the committed terminal checkpoint does not flush again.

## Open Questions

The public entry-point/wire direction is selected, including explicit alternate
matching keys. No user approval is required to implement it within the requested
scope. Before implementation, `cf-critic` must close blocking findings in the
joint specification/API review, particularly accounting/workspace enforcement, precise Studio metric projection,
atomic admission/restore, downstream idle/frontier tests, and supported
stateful combinations. Changes from that review update both artifacts before
`cf-implementer` proceeds. This note is not evidence that these behaviors have
already been implemented or tested.
