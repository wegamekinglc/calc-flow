# P1b Expression to compact SQL task fusion — specification

Date: 2026-11-08. Baseline: main
`8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.
P1a was withdrawn after its preregistered target did not demonstrate a
performance improvement. Implement P1b independently on this main baseline;
do not include P1a's unmerged Rolling/ASOF arithmetic/filter admission.

## Scope and problem

The user authorized the TA-Lib/Arroyo plan, one PR per implementation step,
autonomous CI/Codacy fixes, and merging only with real measured performance
improvement and passing checks/review. This step targets the actual stock
positions graph: filter trades and compute signed quantity in an Expression,
then grouped SUM in SQL, emitting a full cumulative snapshot after each batch.

The change combines task execution through the existing local-edge/ready-group
mechanism. It preserves the two logical operators, their DataFusion resources,
data batches, state ownership, checkpoints, metrics and error identities.
DataFusion 55 remains the table engine. It does not parallelize SUM, delegate
its delta calculation, merge SQL into the Expression query, coalesce batches,
change output protocol, or eliminate the computed Arrow arrays.

Use the existing [runtime contract](../../../docs/runtime-envelope.md),
[streaming guide](../../../docs/streaming-guide.md),
[introduction](../../../docs/introduction.md) and
[AGENTS verification](../../../AGENTS.md#verification).

## Source findings and minimal change boundary

- `runner/operator_fusion.rs:113` currently requires the downstream member to
  be an Expression. P1b needs an internal admission branch for Expression→SQL;
  it must not broaden all operator combinations indiscriminately. Its current
  `eligible_link` also assumes ingress `input`; SQL links must match their
  actual declared alias without renaming graph ports.
- `supervisor.rs:189` and `operator_task.rs:540` already bind ordinary retained
  logical operator tasks into a ready group. They are not direct recursive
  calls from one operator's collector into the next operator's callback.
- `runner.rs:3509` already gives every SQL member a job-owned recovery client
  regardless of group membership. `operator_task.rs:336,624` preserves the
  SQL restore loan/entry acknowledgement; `operator_task.rs:1654,1760` routes
  checkpoint preparation through owned work and restores the actual operator.
  These paths must be exercised in a group, not replaced by a second SQL
  lifecycle or synchronous checkpoint shortcut.
- `sql.rs:1345` performs its original lazy incremental planning and proofs;
  `sql.rs:965–1140` shows that `has_stream_aggregate` alone does not prove
  compact execution. Schema, normalized plans and sometimes input-dependent
  eager/grouped/global proofs still matter.
- `sql/compact.rs:15` prepares an update, emits its snapshot, then commits the
  native transaction and installs compact state. Fusion must preserve this
  exact per-node boundary.
- `local_edge.rs:24,44` reuses the existing EdgeQueueCore for local ownership,
  message-cost admission, queues and wakeups. Shared execution does not remove
  the internal edge or its budget/metric obligations.

The expected product changes are a small syntax/registration proof plus its
integration into fusion admission. The generic group/SQL lifecycle may require
targeted fixes if tests reveal assumptions; changing eligibility alone is not
evidence that lifecycle support works. Do not add a new task owner, SQL state
representation, executor, checkpoint protocol or generic framework.

## Admission and fallback model

**A1 — Structural proof.** Initially admit a single-input stateless Expression
with one output and exactly one consumer, directly connected to one-input
append SQL through an internal edge. The first SQL candidate covers the real
grouped-SUM holdings shape; it must include its actual filter and CASE-derived
signed quantity. Reject fan-out, late-derived branches, external providers,
SnapshotSql, joins/multi-input SQL, selected UDF resources and volatile or
wall-clock work. Preserve baseline exact-projection admission and group ownership. The new
admission branch is limited to Expression→SQL; it does not enable the withdrawn
P1a Rolling/ASOF arithmetic/filter combinations.

**A2 — Syntax proof without rows.** Prove the Expression's supported immutable
row-local syntax by minimally factoring the baseline dependency visitor's existing proof where applicable. For SQL, use a
conservative parsed candidate shape: one declared alias/table, grouping keys,
supported built-in grouped SUM and ordinary projected group/aggregate columns.
Reject unsupported syntax rather than guessing compact eligibility from a
substring or `has_stream_aggregate`. Resolve/limit function calls so unknown,
selected or volatile functions cannot masquerade as supported built-ins.
The candidate proof does not need to claim a successful compact plan.

**A3 — Optional schemas.** Existing exact declared schemas remain checked.
When an endpoint has no declared schema, preserve the current runtime port,
query and SQL input-schema validation; do not require callers to add one.
The Rust holdings example constructs both nodes using default ports with
no schema (`examples/30_streaming_stock_positions.rs:29`). Python SQL graph
lowering usually pins fragment and SQL schemas
(`python/calc_flow/symbolic/lower/sql.py:109,195,250`), whereas Expression
construction can omit output schema (`lower/segments.py:497`). Test those
actual emitted graphs. Do not modify their public documents or invent exact
schemas solely to make a benchmark eligible.

**A4 — Original lazy planning.** Keep the SQL callback's current planning,
compact-budget configuration, proof/rebuild and activation sequence. No new
preflight session, first-batch consumer, speculative state initialization,
query execution, or duplicate plan reservation is needed to select the task
layout. Planning failures still occur at their original node/callback boundary.

**A5 — Safe fallback inside the group.** If the syntax candidate passes but
SQL's ordinary lazy checks select retained-history fallback, the unchanged
generic SQL logic may continue inside the group. This is safe only because
the group runs the original SQL task/collector/lifecycle with the original
node context, ports, edge budgets and state limits; fusion is a scheduling
choice and does not assume a compact state representation. No data need be
replayed or moved to a newly spawned task. Test this case explicitly, including
its output and checkpoint/restore. If a grouping assumption actually depends
on compact state, that assumption must be removed or this candidate rejected;
do not invent dynamic regrouping to hide it.

**A6 — Evidence labels.** Report syntactic fusion eligibility and actual SQL
state mode separately. Target performance evidence must attest compact state
activation for the real holdings input; an unproved syntax candidate, a zero-row
first batch or a retained fallback is not proof of compact optimization.

## Preserved execution and resource requirements

1. Internal edge acceptance precedes upstream callback completion/commit, then
   SQL processes that accepted batch. SQL success cannot become a new upstream
   commit condition. Upstream expression failure does not reach SQL; SQL
   failure retains its own node identity. Already accepted upstream output is
   not retrospectively rolled back.
2. Keep complete message admission on both sides, row/byte/envelope limits,
   cancellation during a full local edge or blocked sink, edge observations,
   per-node input/output counters, and job status. Never enlarge a queue or
   bypass its cost accounting to manufacture a speedup.
3. Keep each SQL batch's complete cumulative snapshot, order, types, null and
   overflow behavior. An empty/fully filtered input follows the exact existing
   callback/emission rules. Do not suppress unchanged snapshots or defer them.
4. Preserve the separate logical node identities, fingerprint/project JSON,
   state capability, SQL resource/session ownership and manifest entries.
   Entry/reset/restore acknowledgements, Barrier/EOF/progress FIFO and sink
   checkpoint obligations remain unchanged. SQL capture failure cannot forward
   the barrier as successful. Recovery continues from the same source cursor.
5. Preserve the existing bounded cooperation/readiness and every awaited owned
   recovery/capture task. No detached worker or synchronous replacement of
   asynchronous lifecycle work. SQL's existing compute behavior remains its
   own callback; grouping is not CPU parallelism. Lost pipeline overlap between
   the Expression and SQL on different cores is a performance risk to measure.
6. Reject/fallback from an unproved topology before installing the layout.
   Cleanup must settle both logical members, local buffers, reservations and
   SQL loans after an error, cancellation, panic or launch failure.

## Focused acceptance and red/green sequence

Begin each new behavior with the smallest observed failing test. Reuse existing
fusion/task/checkpoint fixtures; no product-wide test matrix is needed locally.

- [ ] Admission: the real Rust example's filter/CASE→grouped SUM graph with
  absent port schemas fuses; the actual Python holdings graph also fuses.
  A non-`input` SQL alias is preserved. Baseline Rolling/ASOF exact-projection chains remain admitted.
  Fan-out, late, UDF/volatile, unsupported
  SQL and incorrect topology/schema candidates remain safely unfused/rejected.
- [ ] Data: fused/unfused executions of the same graph produce every identical
  snapshot, including multiple RecordBatches, empty/fully filtered batch,
  new groups, mixed buy/sell, arithmetic overflow and metadata. Assert logical
  metrics/errors and the acceptance→upstream completion→SQL processing order;
  do not use only final-table equality.
- [ ] Lazy initialization: no SQL runtime or compact state is initialized by
  fusion eligibility alone. First-empty then nonempty batch follows the old
  lazy path; actual compact activation is observable. A syntactic candidate
  whose types/plan require ordinary SQL fallback stays correct inside the
  group, without duplicate consumption or changed error timing.
- [ ] Refusal/cleanup: local-edge and sink budgets, slow/full sink, cancellation
  while blocked, SQL calculation/output failure and task panic preserve the
  existing commit boundary and settle owned resources and operator loans.
- [ ] Lifecycle: fused fresh entry and fused restored entry retain two logical
  identities/acks; a nonterminal barrier captures both nodes and restores the
  next batches to the same snapshots. Capture/restore failure and cancellation
  are attributed correctly, leave no stuck CheckpointLoan and do not forward
  a successful barrier. EOF before/after data and progress ordering agree.
- [ ] Focused Rust fusion/SQL lifecycle checks and Python holdings/lowering
  checks pass; required CI, coverage, Codacy and final specialist review pass.
  Only run changed-surface checks locally; full regression belongs to CI.

## Implementation decisions and evidence boundaries

- P1b is independent of the withdrawn P1a runtime change. The existing exact
  projection path retains its original admission rules; only Expression→SQL
  gains a new branch. No operator callback, collector, SQL state format or
  recovery owner implementation is replaced.
- The existing Expression dependency visitor accepts an optional schema for
  the new syntax proof. Known schemas still resolve column dependencies; an
  absent schema defers column/type errors to the original callback. Runtime
  and operator ports must agree in name, kind, requiredness and Some/None schema.
  When both ends declare a schema, the internal edge must agree exactly.
- Actual late ancestry is retained by `derive_late_outputs` and copied to every
  derived node's `late_output_ports`. The new admission branch rejects either
  member carrying this marker. A real SideOutput graph consumes both primary
  and late ports and verifies the derived Expression→SQL pair stays unfused.
- The Python fixture `holdings_project.json` is the unmodified data-only export of
  main `8d8be7d9`'s `examples/30_streaming_stock_positions.py::make_program`,
  using `lower_program_document(..., "stream")`. SHA-256:
  `ad764ad8c07437414df54a165a86e2521eda4a6a0e9778ab3db87cb82bfc4671`. Its four logical nodes retain
  their exact optional schemas; only the middle CASE/filter→SUM link groups.
- Successful data/recovery tests use the supported non-`input` alias `trades`.
  The baseline planner fails to register quoted uppercase `Trades`; a separate
  fused/unfused refusal test preserves that SQL-owned error. P1b does not repair
  that unrelated name-resolution behavior or advertise uppercase execution support.
- Existing Int64 CASE negation wraps `-i64::MIN`; the new snapshot comparison
  preserves the wrapped value. Invalid CAST inputs supply the deterministic
  upstream failure test without changing the engine's arithmetic contract.
- Native layout 3 and retained fallback layout 4 are captured while unfused,
  restored through the original SQL owner in a group and continued. Capture
  refusal/panic, paid-loan cancellation and restore refusal remain targeted
  lifecycle evidence; performance and required CI gates remain separate.

## Performance and stop conditions

Pre-register at most three workloads, using maintained stock/SUM tools rather
than a new framework:

1. The actual holdings filter/CASE→compact grouped SUM graph with small batches.
2. The same application with larger batches to expose lost pipeline overlap
   and snapshot cost; this is a distinct case.
3. One necessary unfused control, such as the same topology made ineligible by
   fan-out, if the first two cases do not already cover the relevant risk.

Keep row/batch counts, keys, CPU workers/partitions, full snapshot volumes and
correctness witnesses identical between revisions. Use modest fixtures that
fit the complete measurement budget; do not automatically reuse historical
10M-row studies. Both real Rust and Python graph admission need correctness
coverage; only the preregistered measured surface earns a speedup claim.

Before timing, freeze source/build/native hashes, exact cases, sample counts,
oracles, CPU settings and total wall-clock estimate. Use the maintained
two-round, ten-fresh-process-pair AB/BA contract and provenance/resource checks;
all chosen cases together receive 600 seconds including setup, warm-up,
correctness, startup and cleanup. Builds are separately estimated/reported.
Do not replace that protocol with the old withdrawn parallel-SUM experiments.

Merge requires a genuine improved end-to-end target under the maintained
verdict, no confirmed regression in selected controls, correctness and required
CI/Codacy gates, and final review. Syntax admission/task-count reduction alone
does not establish performance. Report compact activation, full emitted output
volume and all inconclusive results. If scheduling savings do not overcome
lost overlap, or runtime boundaries cannot be preserved, close/withdraw the
implementation with evidence instead of changing the output contract.

The initial specification stage performed no product code, build, test or
performance execution. Implementation decisions above record the subsequent
focused work; final validation and performance verdicts belong to its handoff. Next role: `cf-implementer` on the independent main baseline;
route only a discovered unresolved lifecycle/contract change to `cf-critic`.
