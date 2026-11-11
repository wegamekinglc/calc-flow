# Query history and portability

[Documentation](README.md) / Query analysis

`Program.analyze()` reports query facts without reading input rows or starting
a job. Its `query` field contains deterministic node and consumption-edge paths,
exact schemas where known, and separate batch/stream history, update, finality,
ordering and lifecycle support. `schema_version=2` describes the report format;
`analysis_revision=2` describes its proof rules, including the required
`update_semantics` field. Older v1 reports can be displayed as legacy only.
Runtime capability registration
revisions remain separate.

An explicit `PortabilityRequest` asks whether the same declaration can be
compared across batch and stream under stated input and cutoff assumptions:

```python
import pyarrow as pa
import calc_flow as cf

source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
program = cf.Program(
    "positive",
    engine="sql",
    outputs={
        "result": source.filter(source["x"] > cf.lit(0)).select("x"),
    },
)
request = cf.PortabilityRequest(
    comparison="per_batch",
    history="current_batch",
    cutoff="batch_boundary",
)
report = program.analyze(options=cf.AnalysisOptions(portability=request))
assert report.query.portability == "supported"
result = program.collect({"events": pa.table({"x": [-1, 2]})}, portability=request)
assert result["result"].to_pydict() == {"x": [2]}
```

`analyze` and `explain` return findings, including rejected and unproved requests.
`compile_batch`, `compile_stream`, `collect`, `collect_async`, `stream`, and
`execute` accept `portability=request` and require a supported result under the
current registration snapshot. A preparation-cache hit still checks the request.
Rejection raises `CompileError` with an immutable `query_analysis` report,
before input wrapping, source binding or job startup. Async collection and
stream startup perform planning on their compilation worker.

Rust users call `PipelineBuilder::analyze(&udfs, &options).await` and explicitly
select `compile_batch_checked(&udfs, &request).await` or
`compile_stream_checked(&udfs, &requirements, &request).await`. Original
compilers and their final lifecycle and recovery checks continue to apply.
There is no Rust typed DataFrame/SQL frontend in this slice.

`support` describes local lifecycle admission; `portability` describes the
requested whole-graph comparison. The latter is `None` without a request.
`unknown` means proof is missing; `unsupported` identifies a known limitation
or counterexample. Both reject strict execution. Human messages may improve;
paths and codes are the machine-readable interface for a fixed declaration and
analysis revision.

The supported requests are:

| Comparison         | History            | Cutoff            |
|--------------------|--------------------|-------------------|
| `per_batch`        | `current_batch`    | `batch_boundary`  |
| `final_relation`   | `accepted_history` | `all_sources_eof` |
| `final_relation`   | `latest_relation`  | `all_sources_eof` |
| `final_relation`   | `event_window`     | `closed_window`   |

Other combinations are invalid configuration. Batch must receive the same
accepted input history H, or the same individual batch for per-batch comparison.
Input order and late policy must agree; late-drop comparisons use the accepted
set and late-error comparisons use the successful prefix. EOF and closed-window
cutoffs must actually be reached. Analysis does not make an unbounded source
finite or generate watermarks.

`ordering="unordered"` compares multisets, including duplicates, schemas and
NULL behavior. `ordering="declared"` also requires proved result order and tie
handling. SQL `ORDER BY` does not create temporal identity for downstream
rolling/window calculations. Data-dependent overflow, budgets, cancellation and
exactly-once delivery remain execution constraints.

Ordinary stream SQL aggregates consume accepted history and emit cumulative
complete snapshots. A downstream ordinary aggregate appends those snapshots:
`SUM` of `[1]` followed by `[2, 3]`, then another `SUM`, still ends at 7.
A strict relation comparison identifies the snapshot-to-append consumption
edge with `update_interpretation_mismatch`; it does not switch to replacement
semantics. Nonaggregate DISTINCT/top-k/analytic queries retain their existing
execution scopes. Selection shapes without a row/tie proof, and complex SQL without a proof,
are reported as unknown. A simple aggregate can be
classified accurately while strict final-relation comparison remains unknown:
a job with zero input events has no last complete snapshot. This slice does
not invent an empty-history snapshot or equate absence of an event with an
explicit empty replacement.

New facts and requests do not enter project documents, declaration or native
fingerprints, preparation-cache identities, or checkpoint bytes. Stream state
continues across batches in one job; each batch execution remains independent.
This diagnostic API does not add batch query checkpoint/resume.

| Code                               | Meaning                                                  |
|------------------------------------|----------------------------------------------------------|
| `history_scope_mismatch`           | Known history scope cannot satisfy the request           |
| `update_interpretation_mismatch`   | A complete snapshot is consumed with append semantics    |
| `sql_semantics_unproven`           | No rule proves this SQL shape                            |
| `query_semantics_unproven`         | No rule proves this declaration or composition           |
| `finality_unproven`                | Cutoff or empty-history reduction is unproved            |
| `row_order_unproven`               | Requested row/tie order is unproved                      |

Existing `unsupported_mode`, `ordering_required`, `capability_mismatch` and
schema/type codes retain their meanings. Python paths use logical output/input
names and consuming roles, such as `outputs.total.sql.tables.input`. Special
names use JSON string subscripts. Shared nodes use their smallest reachable
logical path; adding or renaming outputs may therefore change that address.
Rust paths use quoted node IDs and endpoint quadruples, independent of insertion
order. Reports are data, and cannot be imported as execution authorization.
