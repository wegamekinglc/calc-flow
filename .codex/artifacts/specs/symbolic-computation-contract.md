# Symbolic Computation Contract Freeze

| Field             | Value                                                                  |
| ----------------- | ---------------------------------------------------------------------- |
| Status            | APPROVED — semantic freeze; implementation not started                 |
| Issue             | GitHub #167 / SCE-00                                                   |
| Baseline          | `feature/symbolic-contract@f6b8a6f90b7a978de1976f5a163ea689b989caee`   |
| Artifact slug     | `symbolic-computation-contract`                                        |
| Intended audience | symbolic API, compiler, operator, runtime, state, and test owners      |

## 1. Authority and completion boundary

This document freezes the behavior required by SCE-00. It is the controlling
semantic delta over:

1. `docs/superpowers/specs/2026-08-22-symbolic-computation-engine-design.md`;
2. `docs/superpowers/plans/2026-08-22-symbolic-computation-engine.md`;
3. the project-v3, streaming, state, checkpoint, and capability contracts on
   the baseline above.

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** are normative.

This artifact freezes meanings, defaults, failure behavior, compatibility,
finality, and recovery. The same-slug API note still owns exact Python
signatures, Rust type signatures, JSON tag/field spelling, capability object
nesting, and the byte-level static-input digest encoding. Those choices MUST
represent the semantic model here one-to-one and MUST NOT introduce new
defaults or lifecycle behavior. The paired critique was the final SCE-00 gate
and approved this contract after one blocker-only correction round.

No symbolic computation engine is implemented by this document. Until its
dependent implementation issues land, public documentation MUST continue to
describe the surface as proposed rather than available.

## 2. Scope and firewall

SCE-00 freezes:

- the public namespace and declaration/execution boundary;
- immutable expression identity and Python comparison behavior;
- table and array type promotion plus null and NaN behavior;
- entity, event-time, sequence, and observable output ordering;
- row-count and duration rolling frames;
- cross-section membership, completeness, ties, statistics, and row order;
- watermark finality, allowed lateness, and late-row policies;
- the semantic contents of the two project-v3 operator variants;
- the lifecycle capability vocabulary and schema revision rule;
- static-input ownership, content identity, and recovery compatibility;
- durable rolling/cross-section state obligations; and
- per-primitive equivalence tolerances and testable acceptance conditions.

SCE-00 does not authorize a Python evaluator, mutable value-holder tree,
serialized callable, formula parser, runtime operator, project-schema change,
checkpoint migration, or public runner change. Dynamic stream parameters,
retractions, update streams, joins, unbounded windows, and cross-row array
reductions remain deferred.

## 3. D1 — Public boundary and namespace

The symbolic declaration package MUST live only under `calc_flow.symbolic`.
Its initial semantic concepts are:

- typed immutable expressions for columns, tables, arrays, and parameters;
- an ordered immutable `FeatureSet` of unique named expressions;
- an immutable `Program` with unique named inputs and outputs;
- the `row`, `ts`, `cs`, `window`, `table`, and `linalg` namespaces; and
- explicit constructors for table inputs, external parameters, row-count
  frames, duration frames, and event-time buckets.

The exact exported class and function spellings are an API-note decision, but
the namespaces above and their separation are frozen. In particular, rolling
per-input-row features and cardinality-changing event windows MUST remain
different concepts.

`Program` compilation is declaration processing only. Batch compilation MUST
return the existing `BatchExecutionPlan`; stream compilation MUST return the
existing `StreamExecutionPlan`. Compilation MUST capture one immutable
runtime-capability snapshot, lower one strict project-v3 document, and invoke
the Rust graph compiler for final graph validation. A program MUST NOT accept
or execute a `Batch`, Arrow object, array, source, or sink during compilation.

The symbolic package MUST NOT expose `eval`, `push`, `value`, `transform`, a
preview evaluator, or any convenience path that silently executes data.
Execution remains owned by the existing plans and runner. Ordinary Python
functions may compose expressions, but no callable, closure, module, class,
import path, or executable object is retained or serialized.

Constructors MUST defensively copy caller-owned mappings and sequences.
Expression, feature, program, analysis, capability, and explain results MUST
be immutable from the caller's perspective.

## 4. D2 — Node identity and symbolic comparison

Each declaration node has a versioned primitive identity, ordered child list,
and strict JSON attributes. Its canonical identity MUST satisfy all of the
following:

1. primitive name and primitive version are identity inputs;
2. ordered arguments, output declarations, projection columns, partition
   columns, sequence columns, and array dimensions retain declaration order;
3. JSON object insertion order is ignored by sorting keys deterministically;
4. JSON scalar types remain distinct: a boolean is not an integer, an integer
   is not a float, and positive and negative floating zero remain distinct;
5. only finite JSON floats are accepted as declaration attributes;
6. inferred types, Python object identity, memory addresses, runtime session
   IDs, and compile-cache state are not declaration-node identity inputs; and
7. the node digest is lowercase SHA-256 over the versioned canonical
   declaration encoding.

### 4.1 Canonical declaration values

The byte format version is exactly
`calc_flow.symbolic.declaration.v1`. `U64(x)` is the eight-byte unsigned
big-endian representation of `x`. `BYTES(x)` is `U64(len(x)) || x`.
`TEXT(s)` is `BYTES(UTF-8(s))` with no Unicode normalization. Every count,
length, magnitude, rank, and argument index MUST fit `u64`.

Canonical declaration values use these exact tags and payloads:

```text
0x00 null
0x01 false
0x02 true
0x03 integer  || SIGN || U64(magnitude)
0x04 float64  || IEEE_BITS
0x05 string   || TEXT(value)
0x06 bytes    || BYTES(value)
0x07 enum     || TEXT(enum_family) || TEXT(variant)
0x08 sequence || U64(item_count) || VALUE*
0x09 map      || U64(entry_count) || (TEXT(key) || VALUE)*
0x0a shape    || U64(rank) || DIMENSION*
0x0b dtype    || TEXT(canonical_dtype)

SIGN = 0x00                                  # non-negative integer
     | 0x01                                  # negative integer
DIMENSION = 0x00 || U64(size)                # known dimension
          | 0x01 || TEXT(symbol)             # symbolic dimension
```

Integer values are restricted to the existing portable JSON range
`[-2^63, 2^64 - 1]`; negative values encode their positive magnitude, so an
integer has one representation. A declaration float is Python/IEEE binary64.
Every NaN encodes as big-endian bits `0x7ff8000000000000`; every other value
uses its exact big-endian IEEE bits. Positive and negative zero and infinity
sign are therefore retained, while NaN sign, payload, and signaling/quiet
differences are discarded.

Map keys are unique strings sorted by their raw UTF-8 bytes. Sequences retain
their declared order. An enum family and variant are the exact case-sensitive
identifiers frozen by the selected versioned primitive. Dtype values use the
exact canonical Arrow/provider spelling after aliases have been rejected.
Shape sizes are non-negative; symbolic dimensions use their exact validated
identifier and are not Unicode-normalized.

Primitive-catalog metadata assigns each normalized attribute its value kind,
so, for example, an enum is not encoded as a string and a dtype is not encoded
as an arbitrary string. All semantic defaults are materialized before
encoding; omission and an explicit default MUST produce identical bytes.
Unknown attributes, values without a catalog kind, map-key collisions after
UTF-8 encoding, and cyclic values fail before hashing.

The bytes and NaN tags make the canonical encoder byte-exact and independently
testable, but they do not broaden the public declaration language: initial
public scalar literals and strict JSON attributes continue to reject bytes,
NaN, and infinity as required by D3.

### 4.2 Node digest

The domain separator and one normalized node are encoded exactly as:

```text
MAGIC = ASCII("calc_flow.symbolic.declaration.v1") || 0x00
NODE_BYTES = 0x20
             || TEXT(primitive_name)
             || TEXT(primitive_version)
             || U64(argument_count)
             || CHILD_DIGEST*
             || VALUE(normalized_attributes)
NODE_DIGEST_INPUT = MAGIC || 0x01 || BYTES(NODE_BYTES)
NODE_DIGEST = SHA256(NODE_DIGEST_INPUT)
```

Each child digest is the raw 32 SHA-256 bytes, in argument order. The argument
order is therefore semantic; map insertion order is not. A declaration graph
MUST be acyclic. Structurally identical nodes are represented once after
normalization. If equal digest bytes ever identify unequal `NODE_BYTES`, the
program fails with a digest-collision error rather than treating the nodes as
identical.

The public `Expr.digest` is the lowercase 64-character hexadecimal encoding of
`NODE_DIGEST`. It is stable across processes and conforming Python/Rust
implementations for this encoding version. Changing any tag, normalization
rule, primitive attribute kind, or byte order requires a new declaration
encoding version; it MUST NOT silently change v1 digests.

### 4.3 Program fingerprint

A program includes every unique node reachable from a declared input or
output. Node records sort by raw node-digest bytes. Edge records contain one
record per child occurrence and sort lexicographically by raw parent digest,
numeric argument index, then raw child digest. Input and output records retain
their declaration order.

```text
PROGRAM_BYTES = 0x21
                || TEXT(program_name)
                || U64(input_count)
                || (TEXT(input_name) || INPUT_NODE_DIGEST)*
                || U64(output_count)
                || (TEXT(output_name) || OUTPUT_NODE_DIGEST)*
                || U64(node_count)
                || (NODE_DIGEST || BYTES(NODE_BYTES))*
                || U64(edge_count)
                || (PARENT_DIGEST || U64(argument_index) || CHILD_DIGEST)*
PROGRAM_FINGERPRINT_INPUT = MAGIC || 0x02 || BYTES(PROGRAM_BYTES)
PROGRAM_FINGERPRINT = SHA256(PROGRAM_FINGERPRINT_INPUT)
```

Declared input names are the names owned by their table-input/parameter nodes;
output names are the explicit `Program` output names. Duplicate names are
rejected before encoding. Node/edge order never depends on traversal, mapping
insertion, memory address, construction history, or the optimizer. CSE follows
node digest plus exact `NODE_BYTES`; no algebraic rewrite is part of the v1
declaration fingerprint.

The public `Program.fingerprint` is the lowercase 64-character hexadecimal
encoding of `PROGRAM_FINGERPRINT`, stable across conforming implementations.
A compile-cache key additionally includes execution mode, exact input schemas,
the capability schema version, capability session/revision, and every selected
operator, provider, and UDF identity/version. Those compile-time facts are not
added to `Program.fingerprint`, and neither identity may depend on Python
object identity.

### 4.4 Cross-implementation golden vectors

The following values are normative v1 fixtures. Hex is lowercase and contains
no separators. The standalone `VALUE` vector is:

```text
sequence(
  null,
  integer(-1),
  float64(-0.0),
  float64(NaN),
  string("é"),
  bytes(0x00ff),
  enum("batch_kind", "array"),
  shape(2, symbol("n")),
  dtype("float64"),
)
```

```text
VALUE_BYTES_HEX = 0800000000000000090003010000000000000001048000000000000000047ff8000000000000050000000000000002c3a906000000000000000200ff07000000000000000a62617463685f6b696e64000000000000000561727261790a00000000000000020000000000000000020100000000000000016e0b0000000000000007666c6f61743634
SHA256(VALUE_BYTES) = b3ec4d3d06b466e01f5e9de9fe2e9d2f77a48257a5bfeda79e9d6b8deee92008
```

The node fixture is primitive `table_input` version `1`, no arguments, and
these normalized attributes:

```text
{
  "entity_by": sequence(),
  "event_time": null,
  "name": string("quotes"),
  "schema": sequence({
    "data_type": dtype("float64"),
    "name": string("x"),
    "nullable": true,
  }),
  "sequence_by": sequence(),
}
```

```text
NODE_BYTES_HEX = 20000000000000000b7461626c655f696e70757400000000000000013100000000000000000900000000000000050000000000000009656e746974795f6279080000000000000000000000000000000a6576656e745f74696d650000000000000000046e616d6505000000000000000671756f7465730000000000000006736368656d610800000000000000010900000000000000030000000000000009646174615f747970650b0000000000000007666c6f6174363400000000000000046e616d650500000000000000017800000000000000086e756c6c61626c6502000000000000000b73657175656e63655f6279080000000000000000
NODE_DIGEST = 961b52bfdfb340125fa0b241b312521a43c9dce63dcc6b92717e1f1f2cdb7772
```

The program fixture has name `p`, declared input `quotes` targeting that node,
declared output `signals` targeting the same node, one node record, and no edge
records:

```text
PROGRAM_BYTES_HEX = 210000000000000001700000000000000001000000000000000671756f746573961b52bfdfb340125fa0b241b312521a43c9dce63dcc6b92717e1f1f2cdb7772000000000000000100000000000000077369676e616c73961b52bfdfb340125fa0b241b312521a43c9dce63dcc6b92717e1f1f2cdb77720000000000000001961b52bfdfb340125fa0b241b312521a43c9dce63dcc6b92717e1f1f2cdb777200000000000000fa20000000000000000b7461626c655f696e70757400000000000000013100000000000000000900000000000000050000000000000009656e746974795f6279080000000000000000000000000000000a6576656e745f74696d650000000000000000046e616d6505000000000000000671756f7465730000000000000006736368656d610800000000000000010900000000000000030000000000000009646174615f747970650b0000000000000007666c6f6174363400000000000000046e616d650500000000000000017800000000000000086e756c6c61626c6502000000000000000b73657175656e63655f62790800000000000000000000000000000000
PROGRAM_FINGERPRINT = f09929c7be3d368981565aca0cfd1a3c5becaba3927d06cc25e330912c1e6888
```

Python and Rust implementations MUST reproduce all three byte vectors and both
public hashes before either implementation exposes v1 digest/fingerprint
values.

### 4.5 Symbolic comparison

Public `==`, `!=`, `<`, `<=`, `>`, and `>=` MUST construct symbolic comparison
expressions. Converting any expression to `bool` MUST fail with an actionable
message directing the user to symbolic boolean composition. Public expression
objects MUST be unhashable. Structural identity is exposed by a boolean
`identical(other)` operation that compares the complete normalized structure,
not algebraic equivalence and not data values. Compiler-internal maps MAY use
the canonical encoding/digest as an ordinary value key.

## 5. D3 — Types, promotion, nulls, NaNs, and failures

### 5.1 Table and row-local values

Table scalar/column coercion MUST use the exact DataFusion 54 / Arrow common
type rules captured by the selected runtime. Symbolic analysis MUST ask the
capability snapshot for supported types and MUST reject a coercion it cannot
prove. It MUST NOT implement a competing Python promotion table. An explicit
cast is required between values for which the selected runtime has no safe
common type.

Row-local null behavior is SQL/Arrow behavior:

- arithmetic, comparison, cast, and scalar functions propagate null unless
  the primitive explicitly consumes nulls;
- boolean operators use SQL three-valued logic;
- coalesce selects the first non-null value and does not treat NaN as null;
- a conditional selects the true branch only for true, the false branch for
  false or null, matching a DataFusion `CASE WHEN ... ELSE ...`; and
- invalid or overflowing checked casts/arithmetic fail with the output and
  primitive path; integer wraparound is forbidden.

Floating NaN is a non-null IEEE value. Row-local primitives retain the
selected DataFusion/Arrow NaN and infinity behavior. JSON attributes and
scalar literals MUST be finite, so NaN and infinity enter only through typed
runtime data, not project JSON.

### 5.2 Stateful numeric values

For rolling and cross-section statistics, null and NaN values are excluded
from numeric samples. A rolling aggregate still produces the frame result at
a row whose current input is null or NaN when its minimum count is met; lag and
delta instead preserve a null/NaN current or referenced operand. A
cross-section transform preserves null or NaN at that input row except for the
explicit rank/percentile null-placement modes, `mean_fill`'s null replacement,
and the nullable boolean result of top/bottom selection in D6. Null and NaN
therefore remain observably distinct wherever the output type can represent
both, and other rows in the same frame/group are not poisoned by an excluded
NaN. Positive and negative infinity are numeric sample values and follow IEEE
arithmetic; an undefined result is NaN, not null.

`min_periods` and `min_samples` count non-null, non-NaN values. Pairwise
covariance/correlation count only positions where both operands are non-null
and non-NaN. A statistical result is null when its minimum count is not met,
its divisor is not positive, or correlation has zero variance on either side.

The initial native output types are frozen semantically:

| Primitive                          | Output type                                                |
| ---------------------------------- | ---------------------------------------------------------- |
| lag, min, max                      | input type, nullable                                       |
| integer delta                      | DataFusion subtraction type, nullable, checked             |
| floating delta                     | input floating type, nullable                              |
| count                              | nullable `uint64`                                          |
| signed / unsigned integer sum      | nullable `int64` / `uint64`, checked                       |
| floating sum                       | nullable `float64`                                         |
| mean, variance, stddev             | nullable `float64`                                         |
| covariance, correlation            | nullable `float64`                                         |
| rank, percentile, z-score, demean  | nullable `float64`                                         |
| winsorize                          | input floating type, nullable                              |
| top, bottom                        | nullable `bool`                                            |
| mean_fill                          | input floating type, nullable                              |

Decimal inputs and outputs are outside the first native rolling/cross-section
catalog. Supporting them later requires a separate overflow, scale, and
serialization decision.

### 5.3 Arrays

Array promotion MUST use the selected provider's reported safe dtype rules.
The analyzer MUST retain backend, dtype, rank, symbolic dimensions, and row
lineage. Table-to-array and array-to-table conversion is always explicit.
Cross-backend coercion is rejected unless an explicit, capability-advertised
conversion primitive is selected. A provider that cannot prove a requested
dtype, including JAX `float64` when x64 is disabled, is rejected at compile
time.

## 6. D4 — Input identity and deterministic order

Any temporal or cross-section input MUST declare:

- one non-null UTC `timestamp[us]` event-time column;
- an ordered, non-empty entity key used for row identity and temporal
  partitioning;
- an ordered, non-empty sequence key; and
- exact field names, Arrow types, and nullability.

Sequence fields MUST be non-null and use an Arrow type with a portable total
order; floating sequence fields are forbidden. Entity and cross-section group
fields MAY be nullable and use Arrow total order with null before non-null.

The row identity is the tuple:

```text
(event_time_micros, entity_key..., sequence_key...)
```

It MUST be unique within one logical input. A duplicate identity is a data
error, not a duplicate to drop, and MUST fail before the offending envelope
changes state or emits output. Arrival order and micro-batch boundaries are
never tie breakers.

Within an entity, temporal evaluation order is `(event_time, sequence_key...)`.
The canonical observable output order for row-preserving stateful work is
`(event_time, entity_key..., sequence_key...)`. Cross-section groups are
ordered by their finality coordinate and then group key; rows within a group
are ordered by `(event_time, entity_key..., sequence_key...)`. Exact batch and
final stream results MUST use these orders regardless of arrival,
interleaving, or segmentation.

## 7. D5 — Rolling temporal frames

Rolling output is row preserving: every accepted input row produces exactly
one output row after it becomes final. Derived output fields are appended in
declaration order; duplicate names or collisions fail at compile time.

A row-count frame `rows(n)` requires `n > 0`. For row `i` in one entity's
total order, it contains rows `[max(0, i - n + 1), i]`, including the current
row. A duration frame of `d` exact positive microseconds contains rows whose
event times are in `(t - d, t]`; among equal-time rows, only rows whose
sequence key is not greater than the current row's sequence key are included.
The open lower and closed upper boundary are fixed and use checked event-time
arithmetic.

`min_periods` is a positive integer and defaults semantically to `1`. For a row
frame it MUST NOT exceed `n`. It counts valid values as defined in D3 rather
than physical rows. Lag and delta use a positive integer `periods`; they return
null until that many earlier ordered rows exist. Null/NaN in the referenced
lag position is preserved.

Variance, standard deviation, covariance, and correlation expose a semantic
degrees-of-freedom choice restricted initially to `0` or `1`; the default is
`1`. Their divisor is `valid_count - ddof`. Sum/mean/variance state MUST use a
deterministic ordered algorithm with reversible removal; segmentation may not
select a different algorithm.

The initial temporal primitive semantics are lag, delta, count, sum, mean,
min, max, variance, standard deviation, covariance, and correlation. Batch
and stream lifecycles MUST call the same calculation kernels over the same
canonical row order. A rolling operator MUST NOT be lowered to one
micro-batch-local SQL/window expression.

## 8. D6 — Cross-section groups and statistics

A cross-section group is either:

- one exact event-time value plus the ordered `partition_by` key; or
- one fixed UTC bucket `[start, end)` plus the ordered `partition_by` key.

Bucket width is an exact positive number of microseconds, the origin is Unix
epoch zero, and negative timestamps use floor division toward negative
infinity. Batch input is treated as complete only at end-of-input. Stream input
is complete only by the watermark rules in D7; an incoming micro-batch is
never evidence of completeness.

Cross-section output is row preserving. Transform values are calculated over
the complete valid sample, then restored to the canonical row identity order;
sorting by the measured value MUST NOT reorder output rows.

The initial order/statistic rules are:

- direction is explicitly ascending or descending and defaults to ascending;
- supported rank tie methods are `average`, `min`, and `max`, defaulting to
  `average`; ranks are one-based and returned as `float64`;
- rank/percentile null handling is explicitly `exclude`, `first`, or `last`
  and defaults to `exclude`; excluded nulls produce null, while included nulls
  form one tied class at the requested end of the final sort order;
- NaNs are always excluded from rank/statistic samples and produce NaN at
  their own rows; they are never silently treated as null or as infinity;
- percentile is `(rank - 1) / (ordered_count - 1)` after the selected tie
  method, with one ordered value defined as `0.5`; `ordered_count` includes
  nulls only when `first` or `last` includes them in the ordering;
- `min_samples` is positive and defaults to `1`;
- demean subtracts the arithmetic mean of the valid sample;
- z-score uses the selected `ddof` (`0` or `1`, default `0`) and returns null
  for valid rows when the divisor is not positive or standard deviation is
  zero; and
- winsorization uses lower/upper probabilities satisfying
  `0 <= lower <= upper <= 1` and the Hyndman-Fan type-7 linear quantile;
- top/bottom selection declares a positive `count`, `include_ties` defaults to
  true, and the output is a nullable boolean mask. Valid values are ordered by
  the exact scalar total order; null and NaN rows produce null. When
  `include_ties` is true, every value equal to the count boundary is selected.
  When false, exactly `min(count, valid_count)` rows are selected and canonical
  row identity breaks the boundary tie without reordering output rows; and
- `mean_fill` accepts only `float32` or `float64`, preserves every valid or NaN
  input, and replaces a null with the complete valid sample's arithmetic mean
  only when `min_samples` is met. Its output preserves the input floating type.

Null placement other than `exclude` is valid only for order-statistic
primitives. Applying it to demean, z-score, winsorize, top, bottom, or
mean_fill is a compile error. Unsupported direction/tie arguments on
non-ordering primitives are likewise errors rather than ignored options.

## 9. D7 — Watermarks, lateness, and final-only output

Allowed lateness `L` is an exact non-negative number of microseconds and
defaults semantically to zero. The compile contract supports only `error` and
`drop` late policies. Any other policy, including update, retract, side output,
or early trigger, is rejected.

Given the current aggregate input watermark `W`:

- a temporal row at event time `t` becomes final when `t + L <= W`;
- an exact-time cross section at `t` becomes final when `t + L <= W`; and
- a bucketed cross section becomes final when `bucket_end + L <= W`.

Equality closes the row/group. All additions are checked; an unrepresentable
finality coordinate is a compile/data error. Multi-input watermark aggregation
continues to use the runtime's minimum-known-active-input rule.

A row is too late if it arrives after the relevant row/group has reached the
closed condition above. Each stateful operator samples its aggregate input
watermark `W` once immediately before classifying one input envelope; every row
in that envelope uses that same `W`. When `W` is undefined, no row is late and
the envelope does not change late metrics. For a dropped row whose normalized
event time is `t`, its metric lateness is exactly:

```text
lateness_micros = checked_u64(i128(W) - i128(t))
```

`L` participates only in the closing test (`t + L <= W`, or the corresponding
cross-section group test); it is deliberately not subtracted from the metric.
For a bucketed cross section, the drop decision uses `bucket_end + L <= W`,
while the dropped row's metric still uses its own event time `t`, not the
bucket end. The widened subtraction and conversion are checked and cannot
wrap.

Each operator's late metrics start as `dropped_rows = 0`,
`affected_envelopes = 0`, and `max_lateness_micros = None`. `None` means that
the operator has observed no dropped row; it is distinct from `Some(0)`, which
is possible when `L = 0` and `W = t`. For one envelope containing `n` dropped
rows, the operator atomically:

1. adds `n` to `dropped_rows` with checked arithmetic;
2. adds exactly one to `affected_envelopes` when `n > 0`; and
3. replaces `max_lateness_micros` with the maximum of its prior value and all
   `W - t` values for those `n` rows.

An overflow or invalid subtraction rejects the envelope without changing any
metric, calculation state, or output. Job-level `dropped_rows` and
`affected_envelopes` are checked sums of the per-operator counters;
job-level `max_lateness_micros` is the maximum of all present per-operator
values and is `None` only when every operator value is `None`. These totals
count operator-input drop decisions: on a branched graph, the same source row
MAY contribute once at each operator that independently drops it; no
cross-operator row deduplication is implied.

The three per-operator values are checkpointed semantic state. Restore
reinstalls them exactly before source replay, and job totals are re-derived
from the restored operator values. Metrics observed after the selected epoch
but before a failed attempt are rolled back with that attempt; replay adds the
same post-epoch drop decisions once and MUST NOT merge the failed attempt's
in-memory counters. A terminal checkpoint retains the final values, and
terminal recovery does not change them.

Under `drop`, a late row has no calculation-state or output effect beyond the
metric transaction above. Under `error`, the complete input envelope is
transactional: it emits nothing, installs no calculation or metric state
change, and fails with the row and primitive path. The runtime progress driver
continues to forward data; the owning stateful operator performs this
classification.

Stateful outputs are append-only and final-only. Each accepted row is emitted
at most once, no emitted value is revised or retracted, and output is emitted
before the closing watermark is forwarded. For finality purposes only,
`EndOfInput` closes every buffered accepted row/group and emits it once in
canonical order; it MUST NOT synthesize or forward a sentinel watermark. The
flush participates in the existing terminal checkpoint rule. Batch evaluation
performs the same final flush without late-row classification.

## 10. D8 — Project-v3 semantic model and compatibility

Project format version remains `3`. Rolling and cross-section are additive,
strict operator variants; they MUST NOT overload `expression`, `sql`, the
existing cardinality-changing `window`, or `external`. Unknown fields and
unknown primitive/configuration versions are rejected.

The rolling serialized model MUST carry, directly or through unambiguous
nested values:

- one operator kind and semantic configuration version;
- ordered partition, event-time, and sequence field identities;
- exact input schema and derived output schema contract;
- ordered output declarations with primitive version, input field(s), output
  field, row/duration frame, minimum periods, and degrees of freedom where
  applicable;
- allowed lateness, late policy, and the frozen null/NaN policy; and
- every value needed to reproduce the configuration fingerprint and select a
  durable state-layout version.

The cross-section serialized model MUST likewise carry:

- one distinct operator kind and semantic configuration version;
- exact-time or fixed-bucket grouping plus ordered partition/entity/sequence
  fields;
- exact input and derived output schema contract;
- ordered output declarations with primitive version, inputs/output,
  direction, tie method, null placement, minimum samples, degrees of freedom,
  or quantile bounds where applicable;
- allowed lateness, late policy, and the frozen NaN policy; and
- every value needed for the configuration fingerprint and state-layout
  selection.

The API note MUST select one canonical JSON representation with no aliases and
no implicit omission that changes these defaults. Declaration order is
semantic where this document calls it ordered; maps used only for lookup are
canonically key sorted.

All previously valid project-v3 documents remain valid and canonicalize
unchanged. A runtime that predates the new variants may reject them as an
unsupported operator/capability with a stable node path. No Python
compatibility shim, format downgrade, or executable fallback is allowed.
Project schema, OpenAPI, and generated TypeScript artifacts MUST change
atomically with the Rust serialized model.

## 11. D9 — Runtime capability revision and lifecycle vocabulary

Adding lifecycle facts changes the capability schema from version `1` to
version `2`. Schema version `2` has the following frozen vocabulary:

- execution modes are exactly `batch` and `stream`;
- output finality is exactly `unproven`, `per_row_final`, or
  `group_final_append_only` for the first release;
- checkpoint support is `stateless`, `checkpointed_stateful`, or `unproven`;
- determinism and replay safety are independent booleans; and
- watermark requirement, statefulness, and micro-batch invariance are
  independent booleans rather than inferred from operator names.

Every operator and provider capability MUST report its exact identity/version,
input/output kinds and ports, supported modes, finality, statefulness,
micro-batch invariance, watermark requirement, checkpoint support and state
version when applicable, determinism, and replay safety. Provider capability
also reports safe dtype/shape rules and whether immutable static side inputs
are supported. The API note chooses exact nesting and data-class names.

`unproven` means that registration evidence does not establish any output
finality contract; it MUST NOT be interpreted as either proved finality value.
An `unproven` provider is never selectable for stream compilation, even if
another field or callback would otherwise appear stream capable. Stream
selection requires an explicit proved finality plus every other required
lifecycle fact.

Existing provider registrations remain source compatible and report
batch-only, stateless, no stream factory, and `unproven` finality unless their
registration supplies a separately approved proof. Such an existing provider
remains selectable in batch mode under its existing port/options contract;
`unproven` does not remove or narrow that batch capability. Registrations MUST
NOT become stream safe by omission, callback observation, or optimistic
inference. Built-in operators report only actually implemented modes.
Capability snapshots remain immutable, defensively copied, session scoped,
revisioned, and deterministically ordered. Compilation fails if a selected
capability is absent, stale, from another session/revision, or does not prove
every required lifecycle fact.

## 12. D10 — Immutable static inputs

A static input is declared by name, batch kind, exact schema or
backend/dtype/shape, and `static` mutability. Its payload is not serialized in
the project and is not emitted repeatedly as stream data. Dynamic parameters
are rejected in the first stream release.

The runner MUST defensively copy the supplied name mapping and acquire an
engine-owned immutable `Batch` handle before preflight completes. It validates
the exact expected name set, kinds, schemas, backends, dtypes, and shapes and
computes every content digest before any source, operator, sink, or provider
lifecycle method runs. Missing, extra, or mismatched values fail at the
parameter path.

The content digest is a versioned lowercase SHA-256 over a canonical logical
value encoding. It MUST include the input name, batch kind, ordered schema or
backend/dtype/shape, row/element shape, null positions, and every logical value
in canonical order. It MUST ignore allocation address, chunking, device
address, dictionary indices, and unrelated Arrow metadata. Floating identity
preserves dtype, signed zero, and infinity sign; all NaNs of one dtype use one
canonical NaN tag, so payload/sign differences are not semantic. The API note
freezes the exact tagged byte encoding and digest-version spelling.

Static declarations participate in the plan fingerprint. Sorted static-input
name/digest pairs participate in prepared-job and checkpoint recovery
identity. A checkpoint MUST carry enough bounded digest evidence to compare
the exact name/digest set without storing raw values. Starting recovery with a
missing, extra, or changed digest MUST fail before sources open and MUST NOT
silently select a fresh lineage. Status, metrics, logs, and errors expose names
and digests at most; they never expose raw payloads.

Each latched value is installed once before operator tasks start, is visible
read-only to every consuming operator/provider, survives until the complete
job teardown, and is released exactly once on success, cancellation, startup
failure, or recovery failure. A checkpoint never duplicates the payload.

## 13. D11 — Durable state and recovery

Rolling and cross-section operators are checkpointed-stateful in stream mode.
Each exposes a positive state-layout version in the compiled semantic
fingerprint. The first release has no state migration: an operator version,
layout version, normalized configuration, input/output schema, or primitive
version mismatch fails recovery before source open.

A rolling checkpoint MUST be sufficient to resume without replay-dependent
recalculation or duplicate output. Its semantic state includes:

- the last accepted watermark/finality frontier and checked output sequence;
- each entity's retained ordered history required by every shared frame;
- deterministic accumulator/queue state or enough retained rows to rebuild it
  exactly with the same algorithm;
- all accepted but unfinalized out-of-order rows and their complete row
  identities/required values; and
- bounded duplicate-identity evidence for every still-replayable coordinate.

A cross-section checkpoint similarly includes:

- the finality frontier and checked output sequence;
- every open exact-time/bucket group and its complete accepted rows/identities;
- shared sort/statistic state or enough rows to rebuild it exactly; and
- bounded duplicate evidence for every open/replayable group.

State is partitioned and serialized in deterministic key order. State size is
bounded by declared frames, allowed lateness, open groups, active entities,
and explicit runtime admission limits; an estimate is not permission to exceed
runtime limits. Raw rows or parameters MUST NOT appear in manifest inline
metadata, logs, status, or errors. Immutable state segments and checksums use
the existing `StateBackend` and `CheckpointManifest` v3 transaction.

Checkpoint capture includes unfinalized buffers and occurs at the existing
aligned epoch cut. Restore validates manifest, segment checksum/ownership,
participant sets, static-input digests, schemas, configurations, state
versions, and ordering invariants before installing any state or opening a
source. A terminal checkpoint captures post-`on_end` state so recovery neither
reopens sources nor repeats final output.

## 14. D12 — Errors, paths, and security

All declaration, analysis, capability, project, preflight, data, and recovery
failures MUST preserve a stable path beginning at a named program output or
input. Paths continue through the primitive and failing field, for example:

```text
outputs.signals.score.matmul.right.shape[0]
outputs.alpha.zscore.event_time
inputs.quotes.sequence_by[0]
static_inputs.weights.digest
```

Duplicate names, unresolved types/dimensions, implicit backend conversion,
unbounded stream state, missing ordering/watermark facts, volatile replay
paths, unsupported modes, and unknown primitive versions fail during
analysis/compilation. Duplicate row identity, overflow, and malformed runtime
data fail transactionally at the owning operator. Checkpoint incompatibility
uses the existing checkpoint-mismatch class with the precise identity field.

Errors and explain output MAY contain names, types, shapes, versions, digests,
counts, event-time coordinates, and bounded estimates. They MUST NOT contain
row payloads, array values, static-input contents, arbitrary metadata, secrets,
callable representations, or Python memory addresses.

## 15. D13 — Floating-point equivalence

Equivalence is evaluated elementwise as:

```text
abs(actual - expected) <= atol + rtol * max(abs(actual), abs(expected))
```

Null positions MUST match exactly. NaN positions compare equal only to NaN;
infinities require equal sign. Names, schemas, row count/order, group
membership, delivery count/status, and integer/boolean/string values are
always exact.

| Primitive family                                      | `rtol`  | `atol`  |
| ----------------------------------------------------- | ------- | ------- |
| row-local, lag/delta, min/max, rank/percentile        | exact   | exact   |
| cross-section top/bottom selection                    | exact   | exact   |
| rolling sum/mean (`float64` result)                   | `1e-12` | `1e-12` |
| variance/stddev/covariance/correlation                | `1e-10` | `1e-12` |
| cross-section demean/z-score/winsorize/mean-fill      | `1e-10` | `1e-12` |
| matrix/provider `float64` within one provider/version | `1e-12` | `1e-12` |
| matrix/provider `float32` within one provider/version | `1e-5`  | `1e-6`  |

The tolerance compares batch with final stream, different segmentations, and
checkpoint/recovery for the same primitive/provider version. It is not a
cross-provider equivalence promise. The same canonical row order and algorithm
are mandatory; tolerances MUST NOT excuse nondeterministic iteration,
different missing-value classification, overflow, or state loss.

## 16. Acceptance matrix

SCE implementation issues MUST turn the following into focused RED/GREEN
tests on the owning surface. SCE-00 itself is documentation-only and requires
only artifact consistency and `git diff --check`.

### 16.1 Declaration and compilation

- mapping insertion order does not change node identity; ordered declarations
  do change it;
- independent Python and Rust encoders reproduce the exact D2 v1 value bytes,
  node bytes/digest, and program bytes/fingerprint golden vectors;
- reordering graph construction without changing declarations preserves the
  fingerprint, while changing argument/input/output declaration order changes
  the corresponding node or program identity;
- symbolic comparisons build expressions, truth conversion fails, and
  `identical` is a boolean structural check;
- caller mappings/sequences are not mutated and non-JSON attributes fail;
- exact runtime capability session/revision/version participates in lowering
  and cache identity;
- an existing provider registration without finality evidence reports
  batch-only plus `unproven`, remains selectable by batch compilation, and is
  rejected by stream compilation before its factory/callback runs;
- no symbolic member accepts data or executes it; and
- generated project IDs, canonical JSON, fingerprints, and explain order are
  deterministic.

### 16.2 Rolling and cross section

- row and duration boundaries exercise exact lower/upper equality, negative
  event times, duplicate event times with unique sequence keys, and checked
  overflow;
- duplicate row identities fail transactionally;
- null, NaN, infinity, insufficient samples, all-missing samples, zero
  variance, and integer overflow follow D3;
- tie methods, direction, null placement, singleton percentile, bucket floor,
  and type-7 quantiles follow D6;
- one group split across batches and many groups in one batch produce the
  canonical order; and
- batch equals final stream across 1-, small-, and large-batch segmentations,
  ordered and bounded-out-of-order arrival, and entity/group interleaving.

### 16.3 Finality, checkpoint, and static inputs

- equality at `t + L == W` and `bucket_end + L == W` closes exactly once;
- a later row exercises both transactional `error` and metric-bearing `drop`;
- with `L = 5`, `W = 120`, and one envelope whose event times are
  `(115, 114, 117)`, `drop` rejects the first two rows with lateness values
  `(5, 6)` and produces `(dropped_rows, affected_envelopes,
  max_lateness_micros) = (2, 1, Some(6))`; after checkpoint/restore, a second
  envelope at `W = 123` with times `(118, 120)` produces `(3, 2, Some(6))`,
  and crashing before that second result is checkpointed then replaying the
  envelope produces the same tuple rather than double counting;
- a checkpoint captures open frames/groups and reorder buffers, then restore
  emits neither a duplicate nor a missing row;
- layout/config/schema/primitive/static-digest mismatch fails before source
  open;
- static values are latched before operators, transferred once, visible
  read-only, omitted from status/errors/manifests, and released once on every
  exit path; and
- terminal recovery does not reopen sources or repeat final rows/groups.

## 17. Gate disposition

The design's eight Decision Gates are resolved at the semantic layer:

1. namespace, expression concepts, and compile/execute boundary are D1;
2. row/duration interval and minimum-count rules are D5;
3. cross-section completeness, ties, missing values, and ordering are D6;
4. late policy and final-only guarantees are D7;
5. required project-v3 semantic contents and compatibility are D8;
6. capability revision and lifecycle vocabulary are D9;
7. static-input ownership, digest meaning, and recovery behavior are D10; and
8. primitive equivalence tolerances are D13.

There is no remaining specification-level semantic blocker. The time-boxed
API note selected the exact signatures and shapes, and the adversarial
critique approved them after one blocker-only correction round. SCE-00 is
approved for downstream implementation; this artifact does not itself
implement any symbolic runtime behavior.
