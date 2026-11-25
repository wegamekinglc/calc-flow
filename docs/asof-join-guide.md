# Bounded backward ASOF Join

[Documentation](README.md) / [Continuous streaming](streaming-guide.md) / ASOF Join

Use ASOF to attach the latest historical quote, status, or other right-side
value to each left-side event with the same key. Each accepted left row
produces one final result after both inputs establish event-time completeness.
An unmatched left row is preserved with null right-side fields.

ASOF supports finite batch and stateful stream execution with bounded backward
tolerance. It does not provide forward/nearest matching, unbounded history, dynamic database lookup,
general outer joins, updates, or retractions.

## Declare and run

Run [35_stream_asof_join.py](../examples/35_stream_asof_join.py):

```bash
uv run --no-sync python examples/35_stream_asof_join.py
```

The example declares both inputs using this exact schema and helper:

```python
SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("sequence", pa.uint64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)


def source(name: str) -> cf.TableExpr:
    return cf.table_input(
        name,
        schema=SCHEMA,
        entity_by=["symbol"],
        event_time="time",
        sequence_by=["sequence"],
    )
```

Its `main` function begins with the ASOF declaration:

```python
async def main() -> None:
    matched = source("trades").stream_asof_join(
        source("quotes"),
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(100_000, 64 * 1024 * 1024),
        prefixes=("trade", "quote"),
        late_policy="error",
    )
```

`cf.table.stream_asof_join` accepts the left and right declarations as
positional arguments; `left.stream_asof_join` takes the right declaration.
Both require keyword arguments `tolerance: timedelta` and
`limits: AsofStateLimits`. They also accept these optional keywords:

| Keyword       | Default             | Meaning                                                      |
|---------------|---------------------|--------------------------------------------------------------|
| `keys`        | `None`              | Use each input's entity keys, or supply two column sequences |
| `late_policy` | `"error"`           | Fail on late input; `"drop"` discards and counts late rows   |
| `prefixes`    | `("left", "right")` | Distinct ASCII identifiers used to name output columns       |

Event time and sequence come from each operand's declared temporal metadata.
An explicit `keys` pair changes the matching keys; the output entity metadata
uses those effective left keys. Declarations copy column sequences and remain
immutable. Unknown keywords are rejected.

Use `TableExpr.stream` or `Program.stream` with logical input names. In the
example these are `trades` and `quotes`; each has a
`SourceProvidedWatermarks` policy. Both watermarks first equal 105, and no
result is available. After they advance to 122, the two trades produce
`quote__price = [10.2, None]`. The example requires no external service.

`collect`, `compute`, and batch compilation consume one complete history per
input. The compiled plan owns no candidates between executions. Explicit runner
bindings and exported projects use the plan's physical binding names; see
[stream ownership](python-api.md#streaming-results) and
[project export](projects-guide.md#export-expressions-and-retain-the-right-names).

## Complete finite inputs and SQL

Run [finite_asof_join.py](../examples/finite_asof_join.py) or the independent
[Rust core example](../crates/calc-flow/examples/finite_asof_join.rs). Python
keeps the existing `stream_asof_join` name for both lifecycles; Rust uses
`left.asof_join(&right, spec)` with the existing `StreamAsofJoinSpec`.

Both languages also accept the narrow SQL dialect:

```sql
SELECT * FROM CF_ASOF_JOIN('a', 'b', '<configuration JSON>')
```

Bind exactly the two distinct aliases. The strict configuration object contains
only `spec`: `left` and `right` each require `keys`, `event_time`, `sequence_by`,
and `prefix`; also require `tolerance_micros`, `limits.max_state_rows`,
`limits.max_state_bytes`, and `late_policy`. Tolerance is a nonnegative exact
JSON-safe integer; both limits are positive exact JSON-safe integers. All
objects reject unknown fields. Clauses, projections, modifiers, CTEs and nested
SQL text are outside this dialect. Compose separate row-local relations instead.
Ordinary SQL retains its existing support domain.

Finite execution validates both envelopes and admits the complete left then
right history, across all RecordBatches, before finalization. Both sides share
the existing total state budget. No early eviction or fabricated watermark can
skip duplicates, identity checks, or budget charges. Empty right inputs produce
one unmatched row per left identity; empty left inputs preserve the full typed
output schema. Each execute owns fresh state, including after an error or
cancellation, and requires no reset. Finite plans do not resume stream checkpoints.

Shared declarations have one owner per job or finite execute. Nested ASOF and
inner↔ASOF paths require unchanged or purely renamed temporal identity. ASOF
uses the prefixed left keys, time and sequence; nullable right identity fields
cannot establish the next join's identity. A nested inner join still needs its
complete canonical pair ordering. Export uses the same project-v3 native ASOF
node and physical bindings; logical Python input names are not serialized.

Revision 7 analysis can prove `final_relation / accepted_history /
all_sources_eof / unordered` conditionally: both runs must succeed over the same
complete accepted history, within their own budgets, with no stream late discard.
Compare all append outputs at EOF against the finite relation. Stream intermediate
events and chunk boundaries need not match. Explicit latest semantics reject
snapshot inputs at either stream ingress before opening sources; legacy append
and existing checkpoint identities remain unchanged.

## Selection and exact input types

For a left event at time `t`, tolerance `T` selects candidates with equal keys
and `t - T <= right_time <= t`. The answer is the greatest pair
`(right_time, right_sequence)`. Both interval endpoints are included, and a
right row may serve multiple left rows.

For one key with right times 90, 100, and 110:

| Left time | Tolerance | Selected right time | Result                           |
|-----------|-----------|---------------------|----------------------------------|
| 105       | 10        | 100                 | Latest eligible historical row   |
| 105       | 4         | None                | Left row kept; right fields null |
| 100       | 0         | 100                 | Exact-time match                 |
| 105       | 0         | None                | No same-time right row           |

All values in this table are integer microseconds. If right time 100 has
sequence values 7 and 9, sequence 9 wins regardless of arrival order. Composite
sequences compare lexicographically by field: integers numerically and strings
by UTF-8 binary order. Sequence values need not be consecutive. Left and right
sequence types may differ because they are never compared across sides.

Both inputs require exact schemas. Key lists must be nonempty, have the same
length, and pair columns with exactly equal Arrow types. Keys support booleans,
signed/unsigned 8/16/32/64-bit integers, UTF-8/large UTF-8 strings, Date32/Date64,
and timestamps with matching type parameters. Floating, dictionary, and nested
keys are unsupported. Event time must be exactly non-null `timestamp[us, UTC]`.
Sequence lists must be nonempty and use non-null integer or UTF-8/large UTF-8
string fields. No key/time/sequence casts are implicit. Python declarations
also obey the [portable field-type and naming rules](python-api.md#table-expressions-and-names).

Key, time, and sequence fields must be declared non-null and contain no actual
nulls. Nullable payload fields are allowed. A complete identity is
`(side, key tuple, event time, sequence tuple)`. A duplicate on-time identity
fails even if its payload differs; identities must be stable across replay.
There is no implicit deduplication or arrival-generated sequence.

Native ASOF payloads must be flat Arrow types: null, boolean, integers,
Float16/32/64, dates, times, timestamps, durations, intervals, decimals,
UTF-8/large UTF-8, binary/large binary, and fixed-size binary. Nested lists,
structs, maps, unions, dictionaries, run-end encoding, and view types are
rejected during schema validation. Python/project declarations remain subject
to their existing, narrower portable schema vocabulary. Nullable payloads do
not relax the non-null identity requirements above.

Output columns preserve source order: all left fields, then all right fields,
named `<prefix>__<field>`. Left nullability is preserved; every right field is
nullable, including its key, time, and sequence. A null right event time
identifies an unmatched row. Output temporal metadata comes entirely from the
prefixed left fields. Final rows are ordered by
`(left_time, left_key_tuple, left_sequence_tuple)`; physical batch boundaries
are not part of the result contract.

A downstream column-only `select` can reduce the fields ASOF materializes.
The stream compiler combines the column requirements of all direct projection
consumers, including aliases and reordered selections. A consumer requiring
the complete row or another expression keeps the full intermediate output.
This physical optimization preserves the declared graph, source validation,
output order, and checkpoint identity. It reduces gathering, output workspace,
and edge traffic; retained input payloads still contain their declared columns.

## Watermarks, idle, and late input

A left event at `t` is final only when **both** sides have a watermark strictly
greater than `t`, or the corresponding side has permanently ended. A watermark
equal to `t` still allows more rows at `t`. Finding a candidate, filling a batch,
waiting for a timer, or marking a source idle cannot finalize that event.

ASOF includes idle inputs in its completeness calculation and keeps their last
watermark on reactivation. It suppresses output idle until both inputs end,
even with no pending left rows. A live input without a watermark blocks
finalization. All reachable sources must provide a valid source-provided or
generated watermark policy; disabling watermarks on an ASOF source fails
preflight before sources open. An unrelated branch may still disable them.

The [default iterable policy](streaming-guide.md#watermark-policies) requires
nondecreasing timestamps and emits `max_seen - 1 microsecond`. Because ASOF
requires strict passage, a latest observed time of 106 yields watermark 105
and cannot yet close a left row at 105. Use explicit source-provided progress
or a suitable generated policy for the source's actual completeness guarantee.
Tolerance limits quote age, not wall-clock wait: a stalled input can hold rows
until progress, EOF, cancellation, or an explicit resource failure.

Let `C` be the minimum watermark of unfinished inputs, including idle ones.
ASOF emits all pending left rows with `t < C` before forwarding at most
`C - 1 microsecond`. It emits no underflowing watermark when `C` is the minimum
representable event time. This lets downstream rolling/cross-section stages
accept a later legal row at `t = C`. Chained ASOF stages can consequently need
additional progress. Both inputs ending drains all remaining left rows and
then forwards EOF; EOF is not represented by a maximum-time sentinel.

A row is late when its time is **strictly less than its own side's accepted
watermark**. Equality remains on-time. `late_policy="error"` fails admission;
`"drop"` discards and counts the late rows on that side. ASOF has no separate
allowed-lateness setting. Program compile-time lateness options belong to
rolling/cross-section stages and do not override the ASOF declaration.

Admission validates schema/type/nulls, then lateness, on-time duplicate
identities, and resource limits before committing a complete batch. A historical
identity already below its own watermark follows the late policy; duplicate
detection does not retain an unbounded lifetime seen-set. Sources and the
runtime may reject invalid schemas or progress before ASOF sees them, using
their existing structured error category.

## Bounded state and workspace

`tolerance` is a finite nonnegative integral microsecond duration, no greater
than `9_007_199_254_740_991` microseconds. Both `AsofStateLimits` values are
required exact positive integers up to that same JSON-safe maximum. Booleans,
floats, and numeric strings are rejected.

Limits apply to the whole operator across both inputs. `state_rows` counts
each live identity once: pending left rows, retained right rows, and any
identity-only entries still needed to reject duplicates. Right history is
released only when it cannot influence a pending or future legal left row.
For right time `r`, the conservative condition is
`r + T < min(left_future_time_bound, earliest_pending_left_time)`.
The equality case remains eligible. Right EOF alone does not discard useful
history. An expired payload can leave a charged identity-only entry until its
own side's watermark passes the identity time or that side ends. When the last
live run retires, the operator releases its key dictionary and locator indexes.
Later inputs establish fresh group IDs and continue the existing run sequence.

Native accounting version `11` charges shared key storage, immutable run
payload and identity columns, payload/identity liveness, locator capacities,
and index owners. Source-replay records and anchors are also charged when that
managed path is active. Admission, selection, Arrow materialization, encoding,
and restore share a **separate workspace ceiling equal to `max_state_bytes`**.
Output is further bounded by runtime edge rows/bytes. A 64 MiB state limit is
not a 64 MiB process RSS limit; sources, edges, checkpoint publication, runtime
allocations, and caller-owned tables have their own ownership and budgets.

Accepted rows from one admitted batch form an immutable Arrow run. A run owns
an optional payload RecordBatch plus parallel group-ID, event-time, sequence,
and payload/identity liveness columns. Admission materializes the accepted rows
with Arrow `take` before retention, so dropped rows do not remain reachable
through the run. A stable `(run ID, row offset)` locator
identifies each retained row. Dropped late rows create no persistent state, and
a duplicate or budget rejection installs no partial key, run, or output.

Per-key right payload and identity indexes merge locators in `(time, sequence)`
order. Pending left locators merge in canonical `(time, key, sequence)` order,
using typed key ranks rather than group birth order. Out-of-order admission
merges locator columns without rewriting the Arrow payloads. The same run and
locator identities are used by duplicate validation, selection, checkpoint
capture, and restore.

For each finalizable left row, selection chooses the greatest right
`(time, sequence)` within the inclusive tolerance from that key's ordered
payload locators. A selection contains left and right locators plus right
validity. When a side selects complete, freshly admitted runs in their original
row and run order, materialization shares the columns of a single run or concatenates
columns from several runs. Restored payloads can share one IPC allocation
across columns, so a narrow projection copies its selected values. Other
selections use Arrow `take` for one run and
`interleave` for multiple runs, inserting nulls for unmatched right rows. Direct-consumer
projection limits the materialized columns without changing retained input
schemas or checkpoint identity.

Selection and output charges are preflighted before emission. A retryable
workspace or output-limit failure halves the attempted prefix; a single row
that cannot fit fails without committing it. After downstream acceptance, the
operator commits exactly that pending prefix, its counters, and the output
sequence. Previously accepted output is not withdrawn by a later failure.

Once the ready chunks in a progress tick have been accepted, right eviction
uses the earlier of the left future-time bound and earliest pending left time.
It cuts payload prefixes satisfying `right_time + tolerance < threshold` and
can also drop dominated payloads at or before the threshold except the newest
such row. Identity-only prefixes retire only when their time is strictly below
the right watermark or the right side has ended. Runs with no live payload or
identity are released. Cancellation leaves unaccepted output and its pending
state available for a compatible checkpoint.

When required state or a single output row cannot fit, execution fails with a
structured resource reason. It never discards a still-useful candidate or
chooses a worse match to fit a limit. No throughput or latency guarantee follows
from the configured resource bounds. See the
[ASOF benchmark boundary and workloads](benchmark-suite.md#asof-settlement-measurements)
for separate settlement, allocation, and RSS diagnostics.

## Composition and explanation

| Composition                                              | Contract                                                         |
|----------------------------------------------------------|------------------------------------------------------------------|
| Source or row-local transformation into ASOF             | Supported with exact non-null temporal facts and source progress |
| ASOF fan-out, independent ASOF nodes, unrelated branches | Supported; identical declarations share one native state owner   |
| ASOF into ASOF                                           | Supported using valid left-derived identity and time fields      |
| Inner Join into ASOF, or ASOF into inner Join            | Supported with complete valid ordering and schema evidence       |
| ASOF into rolling/cross-section                          | Supported under the downstream operator's own type rules         |
| ASOF into projection/filter or single-alias stream SQL   | Supported; SQL output has its usual new lineage and no ordering  |
| SQL into ASOF                                            | Rejected because SQL does not retain temporal ordering           |
| Event window into ASOF, or ASOF into event window        | Rejected; independent window branches may coexist                |
| Matrix attachment around ASOF                            | Unsupported                                                      |

Optimization preserves ASOF's finality boundary and candidate set. Use
`Program.analyze(mode="stream")` and `Program.explain(mode="stream")` to inspect
resolved keys/time/sequence, backward tolerance, late policy, strict closure,
output ordering, limits, and physical state ownership. These are compile-time
facts; resource occupancy and delivery guarantees come from the running job.

## Recovery, status, and delivery

ASOF uses its own `stream_asof_join@1` identity with semantic state version `3`.
Native snapshots require layout/accounting version `11` and
`row_encoding="arrow-columnar-59.2.0"`; other native layouts are rejected even
when their semantic version is `3`. Checkpoints contain canonical shared keys,
per-run identity and optional payload segments, and a control descriptor with
run inventory, the next run ID, and locator-index capacities. Metadata includes
logical counters, output sequence, and terminal state. The runtime wrapper owns
ingress watermarks/idle/EOF and the forwarded output frontier. Restore validates
the exact segment inventory, checksums, Arrow schemas, key/group coverage,
payload/identity agreement, run identifiers, progress, counters, and recomputed
resource charges before installing state. Unknown or incompatible versions fail
closed. Compatible restore preserves pending rows and the next output sequence;
subsequent output and a recapture at the same epoch retain their deterministic
records and bytes. `reset` only clears operator-owned memory; it does not
delete shared checkpoints, reset sources, or operate sink transactions.

Finalization validates selection and output charges before emitting a chunk,
then installs the selected prefix, counters, and output sequence after the sink
accepts it. Cancellation or a blocked emit leaves that chunk pending and
preserves any earlier committed prefix. A captured checkpoint resumes the
remaining rows and sequence without losing or repeating the committed operator
prefix. The strict dual-watermark boundary is preserved.

Eligible managed jobs with immutable source history and replay factories on
both ASOF input paths can instead use `source_replay` version `3`. That
representation stores callback order, source cursors, progress, output limits,
and bounded replay frames, with native anchors when required. Recovery reads
the validated source history and rebuilds state without delivering reconstructed
outputs to sinks. It validates the restored counters, output sequence, state
charges, and progress before readiness. Missing source replay support or
incompatible control evidence fails closed; this path does not give ordinary
async iterables durable replay.

A managed checkpoint aligns source cursors, operator state, and sink decisions.
A published terminal checkpoint resumes without repeating final output.
Ordinary at-least-once sinks can still receive replayed writes after an
interruption before a durable cut. Exactly-once external delivery depends on the
[complete source/operator/sink proof](streaming-guide.md#delivery-requirements).
One logical result per accepted left row does not upgrade a sink's guarantee.
The convenience `.stream()` iterator, with or without opt-in temporary
checkpoints, does not provide durable restart or application delivery
acknowledgements.

`job.status()["stream_asof_joins"]` maps node IDs to payload-free status values.
It is empty when the graph has no ASOF node. Side status includes accepted,
late, and duplicate counts plus watermark/idle/ended progress. Operator status
includes pending/retained/identity-only rows, state rows/bytes, matched/unmatched/
emitted-left counts, evictions, resource failures, and output watermark.
At successful terminal drain,
`left.accepted_rows == emitted_left_rows == matched_rows + unmatched_rows`
and `pending_left_rows == 0`.

Rust/Python counters remain integers and absent watermarks remain null/`None`.
Project tolerance/limits remain JSON-safe numbers. The existing inner Join
status format is independent.

ASOF-local runtime reasons distinguish invalid input, duplicate identity, late
rows, state/workspace/output limits, counter overflow, and protocol violations
with the `asof_` prefix. Checkpoint mismatch and earlier source/runtime failures
retain their own categories. No status or diagnostic includes row payloads.
See the [project representation](projects-guide.md#bounded-backward-asof-join)
and [API reference](api-reference.md).
