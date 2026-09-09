# Bounded backward ASOF Join

[Documentation](README.md) / [Continuous streaming](streaming-guide.md) / ASOF Join

Use ASOF to attach the latest historical quote, status, or other right-side
value to each left-side event with the same key. Each accepted left row
produces one final result after both inputs establish event-time completeness.
An unmatched left row is preserved with null right-side fields.

ASOF is a stream-only operation with finite backward tolerance. It does not
provide forward/nearest matching, unbounded history, dynamic database lookup,
general outer joins, updates, retractions, or batch ASOF execution.

## Declare and run

Run [22_stream_asof_join.py](../examples/22_stream_asof_join.py):

```bash
uv run --no-sync python examples/22_stream_asof_join.py
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

| Keyword           | Default                 | Meaning                                                           |
|-------------------|-------------------------|-------------------------------------------------------------------|
| `keys`            | `None`                  | Use each input's entity keys, or supply two column sequences      |
| `late_policy`     | `"error"`               | Fail on late input; `"drop"` discards and counts late rows        |
| `prefixes`        | `("left", "right")`     | Distinct ASCII identifiers used to name output columns            |

Event time and sequence come from each operand's declared temporal metadata.
An explicit `keys` pair changes the matching keys; the output entity metadata
uses those effective left keys. Declarations copy column sequences and remain
immutable. Unknown keywords are rejected.

Use `TableExpr.stream` or `Program.stream` with logical input names. In the
example these are `trades` and `quotes`; each has a
`SourceProvidedWatermarks` policy. Both watermarks first equal 105, and no
result is available. After they advance to 122, the two trades produce
`quote__price = [10.2, None]`. The example requires no external service.

`collect`, `compute`, and batch compilation reject ASOF. Explicit runner
bindings and exported projects use the plan's physical binding names; see
[stream ownership](python-api.md#streaming-results) and
[project export](projects-guide.md#export-expressions-and-retain-the-right-names).

## Selection and exact input types

For a left event at time `t`, tolerance `T` selects candidates with equal keys
and `t - T <= right_time <= t`. The answer is the greatest pair
`(right_time, right_sequence)`. Both interval endpoints are included, and a
right row may serve multiple left rows.

For one key with right times 90, 100, and 110:

| Left time     | Tolerance     | Selected right time     | Result                                 |
|---------------|---------------|-------------------------|----------------------------------------|
| 105           | 10            | 100                     | Latest eligible historical row         |
| 105           | 4             | None                    | Left row kept; right fields null       |
| 100           | 0             | 100                     | Exact-time match                       |
| 105           | 0             | None                    | No same-time right row                 |

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

Native ASOF v1 payloads must be flat Arrow types: null, boolean, integers,
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
own side's watermark passes the identity time or that side ends.

Accounting version 1 charges 256 bytes per live identity, the capacities of
independently owned key/sequence and segment allocations, retained Arrow array
memory, 64 bytes per index entry or segment descriptor, and any retained result
or cursor. Shared allocations count once; independent encoded copies are
charged. The implementation retains compact IPC row payloads and one prepared
state segment, so it does not pin a large source Arrow buffer for one retained
slice.

Admission, sorting, DataFusion materialization, encoding, and restore also share
a **separate workspace ceiling equal to `max_state_bytes`**. Output is further
bounded by runtime edge rows/bytes. A 64 MiB state limit therefore is not a
64 MiB process RSS limit; sources, edges, checkpoint publication, runtime
allocations, and caller-owned tables have their own ownership and budgets.

When required state or a single output row cannot fit, execution fails with a
structured resource reason. It never discards a still-useful candidate or
chooses a worse match to fit a limit. Failed batch admission leaves no partial
accepted state or corresponding output. Results previously accepted by a sink
are not withdrawn.

Native Rust/Arrow indexes choose at most one candidate per left row. One reused
DataFusion session performs the bounded ordinal-and-key left join, projection,
and output ordering. Python only declares and lowers the graph. The operator
prepares a complete compacted state segment during asynchronous handlers,
with budget checks and cancellation points. Each preparation is
`O(retained state)` and repeats for each accepted output chunk, so one handler's
total work includes those repeated preparations.
The `checkpoint` capture then shares the prepared segment and records metadata.
Small captures do not imply constant-cost admission or finalization. No
throughput or latency guarantee follows from the configured resource bounds.

## Composition and explanation

| Composition                                                 | Contract                                                           |
|-------------------------------------------------------------|--------------------------------------------------------------------|
| Source or row-local transformation into ASOF                | Supported with exact non-null temporal facts and source progress   |
| ASOF fan-out, independent ASOF nodes, unrelated branches    | Supported; identical declarations share one native state owner     |
| ASOF into ASOF                                              | Supported using valid left-derived identity and time fields        |
| Inner Join into ASOF, or ASOF into inner Join               | Supported with complete valid ordering and schema evidence         |
| ASOF into rolling/cross-section                             | Supported under the downstream operator's own type rules           |
| ASOF into projection/filter or single-alias stream SQL      | Supported; SQL output has its usual new lineage and no ordering    |
| SQL into ASOF                                               | Rejected because SQL does not retain temporal ordering             |
| Event window into ASOF, or ASOF into event window           | Rejected; independent window branches may coexist                  |
| Matrix attachment around ASOF                               | Unsupported                                                        |

Optimization preserves ASOF's finality boundary and candidate set. Use
`Program.analyze(mode="stream")` and `Program.explain(mode="stream")` to inspect
resolved keys/time/sequence, backward tolerance, late policy, strict closure,
output ordering, limits, and physical state ownership. These are compile-time
facts; resource occupancy and delivery guarantees come from the running job.

## Recovery, status, and delivery

ASOF uses its own `stream_asof_join@1` identity with state/layout/accounting
version 1. Checkpoints include pending left rows, right history, live identities,
logical counters, output sequence, and terminal state. The runtime wrapper owns
ingress watermarks/idle/EOF and the forwarded output frontier. Restore validates
these together with configuration, schema, segment integrity, and recomputed
resource charges before installing state. Repeated capture and restore retain a
self-contained segment. `reset` only clears operator-owned memory; it does not
delete shared checkpoints, reset sources, or operate sink transactions.

A managed checkpoint aligns source cursors, operator state, and sink decisions.
A published terminal checkpoint resumes without repeating final output.
Ordinary at-least-once sinks can still receive replayed writes after an
interruption before a durable cut. Exactly-once external delivery depends on the
[complete source/operator/sink proof](streaming-guide.md#delivery-requirements).
One logical result per accepted left row does not upgrade a sink's guarantee.
The convenience `.stream()` iterator and its temporary checkpoints do not
provide durable restart or application delivery acknowledgements.

`job.status()["stream_asof_joins"]` maps node IDs to payload-free status values.
It is empty when the graph has no ASOF node. Side status includes accepted,
late, and duplicate counts plus watermark/idle/ended progress. Operator status
includes pending/retained/identity-only rows, state rows/bytes, matched/unmatched/
emitted-left counts, evictions, resource failures, and output watermark.
At successful terminal drain,
`left.accepted_rows == emitted_left_rows == matched_rows + unmatched_rows`
and `pending_left_rows == 0`.

Rust/Python counters remain integers and absent watermarks remain null/`None`.
Studio SSE events expose `stream_asof_joins` as a list with node IDs: integer
counters, gauges, and watermark microseconds are canonical decimal strings to
preserve the full i64/u64 range in JavaScript. Missing watermarks are JSON null
and idle/ended remain booleans. Project tolerance/limits remain JSON-safe
numbers. The existing inner Join status format is independent.

ASOF-local runtime reasons distinguish invalid input, duplicate identity, late
rows, state/workspace/output limits, counter overflow, and protocol violations
with the `asof_` prefix. Checkpoint mismatch and earlier source/runtime failures
retain their own categories. No status or diagnostic includes row payloads.
Studio supports strict project import, inspection, and save without a dedicated
ASOF editor. See the [project representation](projects-guide.md#bounded-backward-asof-join)
and [API reference](api-reference.md).
