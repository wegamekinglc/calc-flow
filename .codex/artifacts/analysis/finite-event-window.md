# DAL-516 finite event window implementation evidence

Base: `142e74d0ac1a6fd6cef85d79c1a81d2de543100d` (PR #599 merged main).
Branch: `feature/dal-516-finite-event-window`. The provided checkout was reused;
the frozen API note is preserved and included in this delivery. No team definition,
dependency, project format, state version, checkpoint codec or layout was changed.

## Design and affected files

The frozen `../api-notes/finite-event-window.md` is the implementation checkpoint.
Finite execution owns fresh window scratch state, consumes all RecordBatches,
then builds the final relation using the existing membership, group-key and
aggregate kernels. Its private input context carries only identity, actual
watermark and optional finite cancellation context. It creates no stream job,
checkpoint session or synthetic watermark. Stream closure and state encoding
remain on their original paths.

- Core `operator/window.rs`, `pipeline/batch.rs`, `config.rs`: finite lifecycle,
  null snapshots, no-op reset, non-null restore rejection and project admission.
- `dataframe.rs`, `program/lower.rs`, `program/lower/window.rs`: immutable
  `TableExpr::window(WindowSpec)`, full-spec identity, schema/type/path checks,
  time passthrough and mixed graph lowering.
- `sql.rs`, `sql/window.rs`: shared Rust AST recognition of the narrow
  CF_TUMBLE/CF_HOP dialect and deterministic row-local column passthrough proof.
  SQL macros export native window nodes. SQL and DF identities remain distinct.
- Core `query_analysis.rs`: revision 5, conditional full-input EOF proof,
  unproved closed prefixes, no-window history rejection, and snapshot-to-window
  edge rejection in strict comparisons and explicit latest finite wrappers.
- Binding `config.rs` and `_native.pyi`: private data-only SQL description,
  structured native diagnostics, passthrough and authoritative window schema.
- Python `capabilities.py`, symbolic `analyzer.py`, `event_window_sql.py`,
  `program.py`, `query_analysis.py`, lower `event_windows.py`, `program.py`,
  `sql.py`: both modes, layout inventory `(1, 2)`, r5 latest/cache gates,
  preserved legacy @1/@2 identity, precise SQL window origin and shared owners.
- Focused Rust/Python tests and existing capability/revision assertions;
  `examples/finite_event_window.py`, core `examples/finite_event_window.rs`;
  batch/stream/relational, Python/Rust, symbolic and runtime documentation.

The identity brief's `.codex/references/{code-style,unit-test-style,component-boundaries}.md`
files do not exist in this repository. Repository AGENTS.md, actual code-style
rules, cf-implementer methodology, approved symbolic critique, existing event
window spec, predecessor note, relevant source and nearby tests were read instead.
There is no dal-public component here. Existing core examples were retained.
No material API/scope deviation from the frozen note was needed. The checkout
had no configured Git identity; the commit uses the explicit On Behalf Of
identity (Cheng Li, scrappedprince.li@gmail.com) through per-command Git options,
without changing shared or global configuration.

## Test-first observations

Logs are in the attached evidence bundle under the names below. Early Python RED
checks used the predecessor's baseline wheel only to demonstrate missing behavior;
all candidate GREEN Python checks use the native wheel built from this checkout.

- `red-core.log`: `cargo test --locked -p calc-flow --test finite_event_window`
  failed because `bars` was stream-only; after the finite adapter,
  `green-core.log` passed its same-plan complete/single/empty oracle.
- `red-python.log`: public DF batch construction rejected the window;
  SQL planning reported CF_TUMBLE not found. An initial incorrect collect-call
  spelling was corrected before recording the meaningful RED run.
- `red-frontends.log`: the Rust public test found no `TableExpr::window` method.
  Its unrelated `for_mode` Result handling was corrected. An early expectation
  of two owners for identical declarations was replaced by distinct output
  aliases so the test actually specifies two different complete declarations.
- `red-analysis.log`: the requested r5 report still returned `(2, 4)`.
- `red-mixed.log`: the original public domain rejected SQL before a window.
- `red-latest-batch.log`: an exact-schema low-level snapshot→window graph,
  both direct and through row-local SQL, compiled in batch despite explicit
  latest. The stream cases already rejected. The finite edge gate now rejects.
- `red-sql-diagnostic.log`: the SQL geometry field/code were flattened.
  Structured native diagnostics now retain `sql.window.geometry/invalid_literal`.
- `red-no-window-cut.log`: a graph without windows falsely proved an event-window
  cutoff. Graph-level history validation now rejects it.
- `red-sql-lineage.log` / `green-sql-lineage.log`: distinct SQL window relations
  could align columns through their shared synthetic spec identity. Origins now
  use the actual SQL declaration identity, and the same test rejects alignment.

- `red-qualified-time.log`: qualified direct fields and single FROM aliases were
  initially outside the time proof. The shared AST mapping now handles direct
  qualified columns and unmodified qualified wildcards, while still rejecting
  computed time coordinates. Both Rust and Python exercise this proof.

No production test expectation was relaxed to accept wrong results. Existing
stream-only/inventory/revision assertions were updated to the approved new
contract. A schema-failure test was corrected to expect the existing CompileError
at input validation rather than an ExecutionError after execution.

## Six acceptance groups

1. **Four public cells, C11/O02:** Rust public DF and SQL construct and execute
   tumbling and hopping in batch and real stream EOF; Python does the same.
   Independent literal fixtures cover negative floor, half-open boundaries,
   NULL group/time, non-NULL count, all five reductions, signed/unsigned/float
   sum promotion and typed empty. Geometry rejection, zero offset arguments,
   size=slide and the 1024/1025 overlap boundary are checked. Existing native
   type/group/recovery tests supplement the independent oracles.
2. **Mixed DAG, A04/A08:** both languages test DF→SQL pure time rename→window
   →DF→SQL without materializing an intermediate table, plus SQL window paths.
   Time arithmetic is rejected at the event-time field. Shared outputs retain
   one owner; independent union remains. Python also retains an independent
   ordinary aggregate. Python exports, reloads, executes and compares complete
   output rows, schemas and physical input count; distinct SQL window origins
   cannot align columns.
3. **L01–L03/C13:** one Runtime and the exact same warm compiled plan, no reset,
   consume the full fixture, multiple RecordBatches, `(1,A,8)`, NULL-time-only
   input and typed empty. Rust and Python verify overflow/schema failures and
   subsequent fresh success; Rust also tests cancellation and finite snapshot/
   restore. Existing stream tests cover cross-batch accumulation, empty data
   and new jobs with independent state.
4. **Real progress and recovery:** all 36 existing Python recovery tests pass,
   including W=end, partial hopping lateness, true EOF, changed geometry,
   corrupt layout/fingerprint/segments, shared manifest owner and durable
   unclosed-window restart with a newly constructed runner/runtime. Core
   window_tumbling/hopping/state/compile regressions pass. These are not claims
   based only on in-memory restore. Public Rust sources declare NeverEmits and
   use Disabled watermarks; EOF still produces the independent expected rows.
5. **Latest and diagnostics:** append windows are admitted in both modes;
   exact-schema low-level complete snapshots are rejected directly and through
   row-local SQL, in batch and stream, on repeated preparation. Rejection occurs
   in compilation before a source/binding/runner/state root can be acquired;
   these tests do not pretend to execute an empty snapshot past that gate.
   r4 reports remain readable but cannot authorize r5 latest preparation.
   Conditional EOF proof and unknown closed prefixes are tested. Existing
   aggregate-chain tests retain legacy 7 and admitted latest 6.
6. **Delivery surfaces:** public Rust method/rustdoc, private PyO3/stub, existing
   Python exports, both examples, docs and capability inventory are synchronized.
   Generated native project-v3 JSON schema equals the committed schema. Existing
   examples/36, windowed_streaming and example 32 were not relocated or rewritten.

## Commands and results

Builds run from this checkout, with jobs=2. Cargo's dependency/artifact cache is
`/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/pr-repair-20261112/cargo-debug`;
it is a cache, not an alternative source checkout. The interpreter is
`/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python`
(CPython 3.13.9, PyArrow 24). All candidate wheels/logs/site copies are under
this checkout's `target/finite-event-window/`; no native library is left in source.

- Core regressions: `cargo test --locked -p calc-flow --test finite_event_window
  --test finite_event_frontend --test window_tumbling --test window_hopping
  --test window_state --test window_compile --test query_history_portability`:
  42 passed. After adding Rust mixed/time-proof cases and the final graph-level
  guard, the directly affected first two targets plus query_history_portability
  pass 14 tests (overlap with the earlier 42, not an additive total).
- Candidate native: `python -m maturin build --locked --profile dev --interpreter
  <recorded-python> --out target/finite-event-window/wheels`: passed. Native/Python
  source paths and hashes are in `schema-provenance.log` and the source manifest.
- Python regressions: `python -m pytest python/tests/test_finite_event_window.py
  python/tests/test_symbolic_event_window_analysis.py
  python/tests/test_symbolic_event_window_lowering.py
  python/tests/test_symbolic_event_window_schema_planning.py
  python/tests/test_symbolic_event_window_recovery.py
  python/tests/test_update_semantics.py python/tests/test_query_history_portability.py
  python/tests/test_capabilities.py python/tests/test_finite_union.py -q -W error`:
  294 passed. After additional overflow/limit/multi-RecordBatch/reload assertions,
  the feature file has 44 passing cases (overlapping, not additive). The last
  wheel rebuild is verified by those 44 plus all 36 recovery cases: 80 passed
  under `-W error` in `python-native-final.log`.
- `pytest python/tests/test_aggregate_replacement.py -q
  -k mixed_chain_preserves_updates_and_shared_output -W error`: 6 passed,
  40 deselected; legacy 7/latest 6 retained across DF/SQL mixes.
- Core and binding `cargo clippy --locked -p <package> --lib ... -- -D warnings`:
  passed on the changed core tests/example and binding library. New SQL parsing
  was split into small functions; no complexity/lint waiver was added.
- `cargo run --locked -p calc-flow --example finite_event_window`, without
  PYO3_PYTHON: standalone core example passed. Python example passes normally
  and under `python -O` against the candidate native wheel.
- `cargo fmt --all --check`, affected-file Ruff lint/format, focused lizard,
  `scripts/verify_complexity_gates.py`, `scripts/check_spec_references.py`,
  native-generated schema comparison, core `RUSTDOCFLAGS="-D warnings" cargo doc
  --locked -p calc-flow --no-deps`, and `git diff --check`: passed.

An early recovery-module run passed 36 tests with one unraisable async-runtime
drop warning. Baseline and candidate isolated checks, baseline full recovery,
and the final candidate 294-test run under `-W error` did not reproduce it.
No unrelated runtime-lifecycle change was made; this observation is retained
for independent review rather than described as a proven root-cause fix.

## Remaining verification and handoff

Full workspace/full Python regression, coverage (including connector services),
cross-platform checks and performance measurements were not run locally.
They remain CI/scheduled responsibilities. No batch mid-query recovery,
nonzero offset, wider SQL dialect, replacement-window state or general stateful
window composition was added. Resource acquisition counters are established by
compile-time rejection before constructing those resources, not a claim of a
successful runner start with them instrumented.

Independent dal-reviewer review and required checks are still required before
merge. Hand back to the coordinator; do not merge or dispatch another specialist
from this implementation run. DAL-516 remains in_progress until its actual merge.


## PR #600 check corrections

The first pushed implementation commit is `6c32f614a2237de407ff056e6e53fe984d59cda4`.
The same PR remains the sole delivery: https://github.com/wegamekinglc/calc-flow/pull/600.
Its Codacy check `130907298406` reported 17 new findings: sixteen complexity
findings at the hosted limit of 8, plus one SQL-injection classification on a
call passing an Arrow Batch mapping to `BatchExecutionPlan.execute`. The first
local complexity check used the installed lizard 1.23.0 and limit 15; that was
not the hosted policy. The complete hosted findings and local reproduction are
preserved in `codacy-annotations.json` and `codacy-red-complexity.log`.

The correction separates AST shape/geometry/aggregate/passthrough checks,
column provenance and per-role dtype validation, example declaration/data/
validation, and repeated test setup/assertion boundaries. All original literal
oracles, error expectations, rows, schemas and lifecycle assertions are retained.
Python Batch bindings now use the explicit `inputs=` argument and a mapping
variable; they are not SQL text. No rule, threshold, exclusion, complexity
baseline, test expectation or team definition is weakened.

The bounded `cargo test --locked -p calc-flow --lib operator::window --no-run`
check also exposed six original inline tests still calling the private stream
input adapter (expected StreamOperatorContext, received WindowInputContext).
`red-window-unit-compile.log` records this actual failure. The original private
stream adapter now remains in place and delegates to the shared private kernel;
finite execution calls that kernel with its own context. Existing inline tests
are unchanged. This compiler gap justified adding only the affected window unit
module to local verification, not running the entire library test suite.

The correction is checked with pinned `lizard==1.26.0 -C 8` over every new code/
test/example file, affected Ruff checks, focused core integration and window-unit
checks, core/PyO3 clippy, a fresh candidate wheel and the feature/recovery tests.
Final head SHA, observed CI state and per-command outcomes are supplied in the
issue handoff and evidence bundle; pending checks are not described as passed.


The window unit module is GREEN: 45 passed, one pre-existing ignored M7
high-cardinality soak, 3144 unrelated library tests filtered out. The ignored
soak was not enabled because no performance/full-state-soak task was requested.
Pinned lizard 1.26.0 at limit 8 and the repository complexity ratchet both pass
without suppressions; affected Ruff passes; core clippy with warnings denied
passes after the extraction. Rebuilt-core integration, standalone example and
new wheel/Python outcomes are retained in the `codacy-*` logs.


Correction results: the affected Rust integration targets pass 14 tests; the
window unit module passes 45 (one unchanged ignored soak); the standalone core
example passes. The rebuilt wheel with this correction passes all 44 feature
and 36 recovery tests under `-W error` (80 total), and affected Python adapter
checks pass separately. Core and PyO3 clippy, pinned limit-8 complexity checks,
Ruff and the unchanged complexity ratchet are GREEN. The hosted Codacy result
for the follow-up head is separate and is reported from the final CI snapshot.

The next snapshot at `ca0c4db697498cb214c21cc424160745b1722a5c`
confirmed all sixteen complexity findings were removed. Codacy check
`130918013163` retained the SQL-injection classification at the warm-cache
test's native Batch execution call. `codacy-followup-annotations.json` preserves
that RED result. A typed `execute_table(BatchExecutionPlan, pa.Table)` helper
now isolates Arrow-to-Batch binding from the test's window declarations and
chunk matrix, and is shared by the warm-plan and exported-project checks.
It still calls the exact same plan with the exact same input objects; every
schema, row, sharing, EOF and isolation assertion remains unchanged. No SQL
text reaches this helper, and no analyzer configuration or suppression changed.
The existing candidate native binary is reused because this follow-up changes
only test organization. All 44 feature cases pass with `-W error`; affected
Ruff lint/format and pinned lizard 1.26.0 at limit 8 pass. The hosted result is
reported separately, not inferred from these local checks.

## Hosted SQL-injection follow-up after independent-review handoff

The new orchestrator handoff reports check `130919930730` at the fixed review
head `85dbad46d59bf24d4e93768484539347af9f83db`: one SQL-injection finding
at `python/tests/test_finite_event_window.py:70`. This supersedes the previous
pending snapshot. The typed helper alone did not resolve the hosted finding.
The check and its annotation are saved as `codacy-85dbad46-check.json` and
`codacy-85dbad46-annotations.json`. Codacy's public PR-issues API identifies
the tool as **Opengrep**, pattern
`Semgrep_python_sql_rule-hardcoded-sql-expression`; its response is preserved
as `codacy-public-pr-issues.json`.

The [Codacy wrapper](https://github.com/codacy/codacy-opengrep) imports GitLab
rules, and the [matching upstream rule](https://gitlab.com/gitlab-org/security-products/sast-rules/-/raw/7dffffb51fef10dd956ce1b38ef117f71963b09c/python/sql/rule-hardcoded-sql-expression.yml)
tracks formatted/concatenated SQL into generic `.execute(...)` sinks.
Local Semgrep checks did not reproduce the hosted finding and are not GREEN
evidence for this failure. Opengrep **1.33.1**, matching the wrapper's current
declared version, does reproduce the exact line with intrafile taint enabled.
This is a local reproduction, not a claim about the hosted binary's version.
The vendor release binary SHA-256 is
`46e8133cc490f9f50518061590ce2f6be5a08afaeda034471c0ceda4b3899808`,
verified against its GitHub release asset digest; it lives only under `target/`.
The unmodified upstream rule and release metadata are included in the evidence.

RED command (exit **1**, one finding at line **70**, zero scan errors):

```sh
target/finite-event-window/tools/opengrep_manylinux_x86 scan \
  --config target/finite-event-window/codacy-gitlab-sql-rule.yml \
  --no-git-ignore --include python/tests/test_finite_event_window.py \
  --taint-intrafile --error --json python/tests/test_finite_event_window.py
```

The include/no-git-ignore flags ensure this test file is actually scanned;
default local test-file exclusions are not evidence about the hosted analysis.
The dataflow trace starts at `declared`'s SELECT concatenation, flows through
`Program` and `compile_batch`, and marks the **plan object** reaching the typed
helper's execution call. That call accepts a physical-name-to-Arrow-Batch map,
not query text. Moving or annotating this sink cannot remove that source.

The only test change replaces the two fixed tumbling/hopping query assemblies
with two complete string literals selected by the existing boolean. The queries
and source bindings are byte-for-byte equal to the original two variants.
The typed execution helper, all other 25 top-level AST nodes, all 35 assertion
ASTs, parametrization, data, execution order and warm-plan/EOF checks are
unchanged. No runtime, SQL parser, public API, CI rule, suppression or reviewer
report is edited.

GREEN: the identical Opengrep command now exits **0**, scans the same test file,
and reports **zero findings and zero errors**. The module's
`pytest python/tests/test_finite_event_window.py -q -W error` run passes all
**44** cases; affected Ruff lint/format and pinned lizard 1.26.0 at limit 8 pass.
The AST/query-preservation and native-provenance check is saved in
`codacy-literal-query-preservation.log`; all 1,425 Rust/native and Python product
source files match the previous manifest, the candidate native hash remains
`d001cc88c3ad406b05b86277839b4824aa3e230e001594768699f84a858967ca`,
and installed Python sources equal this checkout. A native rebuild and repeated
unaffected Rust/full-suite checks are unnecessary for this test-only change.

The reviewer is independently reading the original fixed head `85dbad46...`;
this increment covers only the SQL-fixture organization and this evidence.
The final increment SHA and one-shot hosted status snapshot are reported in
the issue reply. Pending checks remain pending; DAL-516 remains in_progress,
and review/merge handling stays with the coordinator.
