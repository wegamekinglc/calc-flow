# P1b Expression to compact SQL task fusion — critic critique

Date: 2026-11-08. Main baseline: `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.
P1a implementation analogue read-only: `873e735a` in
`target/worktrees/ta-arroyo-p1a-measured`.

## Target

- Spec: `.codex/artifacts/specs/ta-arroyo-p1b-sql-fusion.md`.
- Scope: admission, schema-free actual holdings graphs, per-node SQL lifecycle
  and the performance boundary. No public API note is required.

## Verdict

**Proceed with caveats.** No protocol incompatibility in the proposed model
requires a new executor or checkpoint design. The existing group executes the
original logical task futures and collectors. A syntax candidate may therefore
retain SQL's lazy type/plan proof and ordinary fallback inside the group.
Implementation must satisfy the concrete admission and lifecycle checks below;
this review does not prove implementation correctness, speedup or merge readiness.

## Findings

### Blocking issues

None in the specified scheduling-only model. A future implementation that
replaces the SQL callback, consumes a first batch for admission, renames its
input, or assumes compact state merely from syntax would violate that model
and require correction before acceptance.

### Significant concerns and concrete implementation requirements

1. **The actual Rust graph needs a schema-free syntax proof.**
   `examples/30_streaming_stock_positions.rs:29–55` constructs the filter/CASE
   Expression and grouped SUM SQL with default ports. In P1a,
   `ExpressionOperator::is_fusible_stream_expression` requires a known input
   schema, and `stream_required_columns` obtains it with `schema()?`
   (`operator/expression.rs:184–220` at the P1a analogue). Simply calling that
   helper in the new branch leaves the principal Rust workload unfused.
   - **Suggested implementation:** separate immutable row-local AST shape/
     function validation from schema-based dependency resolution, reusing the
     existing visitor rules where possible. With absent schemas, defer column
     existence/types to the unchanged Expression callback; do not manufacture
     a schema from names, plan early or inspect an input batch. Keep the
     declared/runtime Port correspondence, including Some/None, and preserve
     every existing exact declaration. With known schemas, reuse the existing
     stronger proof. Add a real example-shaped None/None native graph and a
     real Python lowered graph, asserting actual physical grouping.
   - **Why no extra runtime proof is needed:** this grouping does not fuse
     query plans or remove the original collector/SQL input validation. Missing
     columns, changing schemas or invalid arithmetic still fail where the
     original operator processes that batch. The group need not know compact
     state or output types to preserve those validation boundaries.

2. **SQL aliases must be used in both halves of topology admission.**
   The old `operator_fusion.rs:96–110` hardcodes `ingress == "input"`, and
   `directly_connected` separately looks up the downstream ingress named
   `input` at line 183. SQL uses its declared alias as its required table port
   (`sql.rs:412–415`); `stream_alias` validates that exact incoming name and
   the original batch (`sql.rs:1163–1188`). Changing only one constant can
   produce a passing pair predicate but no real layout, or bind the wrong edge.
   - **Suggested implementation:** obtain the sole alias/required table port
     from the actual SQL declaration. Verify the edge consumer ingress,
     runtime ingress key, operator port and producer/output endpoint all agree.
     Preserve the name through `take_node_ingresses`; do not relabel messages
     or query tables. Test a non-`input` alias, a quoted/case-sensitive table
     spelling that the existing planner accepts, and an inconsistent endpoint.

3. **Syntax candidate and actual compact activation are different facts.**
   `SqlOperator::process_data` retains ordinary retention/proof/activation
   (`sql.rs:1345–1366`), while `initialize_incremental` and its eager/grouped/
   global checks can select fallback (`sql.rs:965–1140`). A plain grouped SUM
   AST does not prove a supported native input type or compact budget admission.
   - **Suggested implementation:** admit only the parsed single-table/grouped
     built-in SUM shape chosen in the spec, with explicit function scope and
     modifiers; keep unsupported syntax on the ordinary task path. Check both
     declared UDFs and selected stream resource UDFs. Do not use the weaker
     `ValidatedQuery::is_simple_projection` predicate as an immutability proof:
     it is not the P1a dependency visitor's built-in-volatility check.
   - Test an unsupported-type/plan fallback with no compact-required budget,
     and verify it stays correct in the group, including restore. Separately
     test the existing compact-required budget rejection. A successful compact
     target can use existing internal state assertions or the maintained
     compact-required budget plus completed output witness; no public
     introspection API is needed just for a benchmark.

4. **Keep the SQL owner and recovery loan active per logical member.**
   `runner.rs:3509–3513` supplies a SQL recovery client independent of grouping.
   `prepare_operator_group` binds each member to
   `run_retained_operator_task` (`operator_task.rs:540–563`), which retains
   its actual operator around restore. Capture temporarily installs
   `CheckpointLoan` and awaits the job-owned ticket
   (`operator_task.rs:1654–1713`). Barrier acknowledgement/forwarding comes
   only after capture completes (`operator_task.rs:1629–1651`).
   - **Suggested implementation:** preserve those routes and activation of
     each client's own task ID. Do not add a group-level operator identity or
     bypass the owner because the edge became local. The spec's grouped fresh
     entry, restored entry, nonterminal barrier, capture refusal/cancellation,
     panic and launch-failure checks are necessary proof of compatibility.
     Reuse the existing seams; a happy-path final snapshot is insufficient.

### Minor / scope notes

- Preserve original error ownership and supervisor primary-versus-convergence
  ordering. Diagnostic timings and attempted-work counters need not be
  numerically identical across two schedulers; they must accurately describe
  the same logical components. Use deterministic seams for error-order tests
  instead of assuming a particular wall-clock race outcome. See
  `supervisor.rs:408–410,687–703`.
- The initial spec says to rebase on merged P1a. If P1a is withdrawn for lack
  of speedup, P1b need not inherit an unprofitable default admission change.
  Rebase on then-current main and carry only the reusable proof code necessary
  for P1b, with its own tests and evidence. Do not silently ship an unaccepted
  P1a runtime mechanism inside the P1b PR.
- State whether fan-out rejection refers to the fused internal producer or
  also the chain tail. In either admitted case the ordinary tail collector
  must keep all sender budgets and partial-delivery/error behavior.

## Why the preserved model is viable

`PreparedTaskGroup::run` polls the bound child futures with ready-pair/group
scheduling (`supervisor.rs:204–241`); it does not synchronously call SQL from
Expression's collector. The existing local edge still delegates cost and queue
admission to EdgeQueueCore (`local_edge.rs:24–73`). The original task constructs
per-node context, validates/emits output and uses its unchanged ports.

Consequently upstream output acceptance can complete and the upstream callback
can return before SQL processes that batch. SQL compact processing still emits
its complete snapshot before committing its own transaction
(`sql/compact.rs:15–61`). Preserving these two distinct boundaries requires no
new SQL state representation. Retained fallback is equally compatible when the
group makes no assumptions about compact storage. The benefit remains uncertain:
sharing a driver removes scheduling work but also removes inter-task CPU overlap.

## Focused acceptance emphasis

The spec's checklist is sufficient with the concrete points above:

- Actual Rust default-port and Python exported graphs group, including the
  non-default SQL alias, while schema/UDF/volatile/topology exclusions hold.
- Compare every full snapshot and its schema/metadata across multiple and empty
  inputs; verify compact activation separately from syntax/group eligibility.
- Preserve internal acceptance, upstream completion, then downstream processing;
  output refusal cannot retroactively roll back accepted upstream output.
- Exercise ordinary retained fallback, compact-required refusal, grouped reset/
  restore, capture and cancellation with both logical identities/acks intact.
- Keep all local/sink envelope, row and byte checks, cleanup of loans/reservations,
  and the existing P1a routes. Do not turn the lifecycle list into a full local
  regression matrix unrelated to this change.

## Performance cases and fair baseline

Prefer two actual holdings cases, with a third only if it supplies a necessary
control. A concrete modest starting protocol is 100,000 trades, 100 accounts,
100 stocks, four pinned workers/CPUs and one SQL partition:

1. Batch size 1,000: the changed Expression/SQL boundary occurs often enough to
   test scheduler savings while the output remains every full group snapshot.
2. Batch size 10,000: the same graph/data exposes loss of pipeline overlap and
   the larger-batch balance of expression versus aggregate/output costs.
3. Optional necessary control: the same real graph made ineligible by an
   upstream fan-out, with identical extra sink and delivered output on both
   versions. Do not compare that case directly with a lower-output eligible
   case as though they did identical work.

These are suggested preregistration dimensions, not authority to run them now
or expand an inadequate comparison. The performance owner must fix the exact
cases and complete budget first. Existing `grouped_sum_state_benchmark.rs` and
`scripts/benchmark_suite/sum_state.py` provide useful fixture/oracle/build
identity infrastructure, but their default fixtures/case matrix are not this
selection. Use only the minimal extension/selection needed, preserving per-
snapshot arithmetic/schema/hash witnesses and the maintained two-round,
ten-fresh-process-pair protocol. All selected acquisition work, including
fixtures, warmup, correctness, startup and cleanup, must fit 600 seconds.

The baseline is then-current main/P1a with the same held-constant harness,
query, optional schemas, full output volume, CPU/partition settings and state
budgets as the candidate. Record actual source/binary/native hashes. Neither
side may receive hand-added schemas, a rewritten SQL query, different batch
boundaries or discarded snapshots. Report actual compact activation and all
output rows, not only the final positions. Only the measured surface earns a
speedup claim. A confirmed regression in the larger-batch/control case prevents
acceptance; an inconclusive comparison does not establish improvement.

## Axis check

- **Correctness:** OK in the preserved generic-task model; concrete schema,
  alias, output/commit and lifecycle tests remain required.
- **Hidden assumptions:** unknown schemas and compact fallback are explicit;
  distinguish schema-free syntax proof from runtime type/plan validation.
- **Missing edge cases:** the spec covers empty first input, filtering, budgets,
  errors, loans and recovery. Add the endpoint consistency and required-budget
  split above; no new broad matrix is needed.
- **Backward compatibility:** project/manifest/state identities and public API
  stay unchanged; normal per-node lifecycle must actually remain in use.
- **Performance:** unknown until measured; lost CPU overlap is a real risk.
- **Surface and ergonomics:** OK; no public configuration or port renaming.
- **Test plan:** concrete and testable. Check actual native grouping and SQL
  state mode; either can be absent while final values still look correct.
- **Risk and scope:** appropriately one scheduling/admission change. A newly
  discovered lifecycle protocol change returns to design review rather than
  being hidden in eligibility code.

## Verification and handoff

Read the complete spec, governing introduction/runtime boundaries, P1a proof
analogue, original holdings graph, SQL lazy/compact flow, local edge,
ready-pair/group scheduler, per-node task ownership and capture/restore paths.
Checked cited source paths, final newline and whitespace. No build, product
test, performance acquisition or remote action ran.

Proceed to test-first implementation after the baseline/dependency decision.
Resolve the concrete admission points in code/tests and return any actual
lifecycle contract change for critique. Final specialist review, real speedup,
CI/Codacy and recovery evidence remain separate merge requirements.
