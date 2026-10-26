# Native snapshot SQL

`SnapshotSqlOperator::new(name, query, aliases, udfs)` declares a stream-only
replacement query. Supply exact input and output Arrow schemas with
`with_ports`. Project format 3 represents this as
`{"kind":"snapshot_sql","query":"SELECT * FROM p","aliases":["p"],"udfs":[]}`.
The Python symbolic declaration is delivered separately.

Each data `Batch`, including all of its RecordBatches, replaces one alias's
complete relation. An empty table initializes and clears that relation. No
result is emitted until every alias is initialized. Each update then runs the
read-only DataFusion query over the latest relations and emits one complete
result, including empty results. JOIN and aggregate execution is planned anew
for each update. Caller-owned batches remain immutable; successful emission
precedes state commit. Failed emission or cancellation drops the candidate.

Ingress FIFO order is preserved. Cross-source ordering does not define an
event-time atomic cut. Idle does not initialize, clear, or recompute a relation.
An ended input retains its latest relation while other inputs continue; an
input that ends without data remains uninitialized. All inputs ending produces
no extra result. Outputs currently connect directly to sinks, and delivery
requests involving these outputs reject ExactlyOnce before opening sources.
Independent outputs retain their existing delivery proofs.

Input relations and complete results must fit their edge budgets. There is no
fragment reassembly protocol. The explicit Python runner rejects a known
`SourceBinding.from_arrow` source directly feeding a snapshot ingress because
that source slices tables. Use a source yielding one complete relation per data
event. Custom connectors are responsible for this message boundary.

`StreamRuntimeConfig.sql_snapshot_state_budget` applies independently to each
snapshot node. `max_rows` counts all aliases' latest rows after replacement.
`max_bytes` conservatively accounts for reachable Arrow backing, schema,
metadata, handles, candidate replacements, and retained encoded segments while
old state is still alive. Shared owners share leases. Slices can retain large
buffers and therefore cost more than their visible rows. State leases survive
in carried checkpoint segments and refund on final release. DataFusion execution,
IPC/JSON scratch reservations, and transport budgets remain separate; none is
an RSS limit. The two existing append-SQL budgets keep their meanings.

The independent snapshot state kind uses layout 1 and accounting 1. A control
segment is present even when all aliases are uninitialized. It binds alias and
schema identity, query/UDF configuration, DataFusion version/configuration,
initialization, relation rows, and data/metadata digests. Restore rejects unknown
versions, extra/missing/corrupt segments, incompatible schemas and current-budget
violations. It prepares and validates all relations before atomically installing
them. Managed live and terminal recovery use the job-owned cancellable SQL work
queue. Checkpoint publication retains capture → stage → ACK → barrier ordering.
Existing append-SQL layout 3/4/5 and fingerprints are unchanged.
