# Bounded analytic OVER — implementation evidence

DAL-521 implements the frozen [API note](../api-notes/bounded-analytic-over.md)
on base `35ff239f7b054b640d120f02925c0113159a8cd3`, using the existing
`agent/dal-orchestrator/4d5e854a24b7` checkout. The note remains unchanged
(SHA256 `71eb1580b0d912f534a8006c22fc79db0b9f284e3fca9d1681bf9fda1f102964`).
Its opening “not implemented” statement describes its historical design checkpoint.
This report records actual product execution, not note self-validation.
The initial implementation results below apply to
`d2c567030aafd3cae911ec653995aba0a8e4e33f`; the final section records the
subsequent Codacy correction and its replacement native/source provenance.

## Result and implementation choices

Rust core and Python DataFrame/SQL declare the same explicit bounded analytic
node. A shared native Arrow evaluator computes ROW_NUMBER, LAG(1) respecting
NULLs, and checked integer SUM over cumulative or N-preceding inclusive ROWS.
SUM uses an i128 accumulator (enough for the admitted u32 row bound) and checks
each Int64/UInt64 output; ordinary DataFusion SUM remains unchanged.

The implementation reuses the bounded ordering operator's history, ownership,
CPU worker, cancellation, closure and recovery machinery through an explicit
analytic declaration. It does not duplicate that lifecycle. Analytic config,
schema and checkpoint identity are separate: `bounded_over_v1`, state/accounting
version 1, Error layout 1 / Drop layout 2. The old ordering declaration serializes
without the optional analytic member. This is an internal reuse choice, with no
public contract deviation.

New native SQL macro parsing lowers to this node; malformed explicit OVER macros
never fall back to ordinary SQL. Python owns immutable declarations and lowering,
and calls private PyO3 schema/AST seams. It has no analytic evaluator. Analysis
revision is 11; schema version and legacy sort cache/fingerprint tags remain as
before. The generated project schema adds OVER definitions and one OperatorSpec
variant; all pre-existing definitions and operator alternatives were compared
against base and remain unchanged.

## Test-first evidence

All commands below were run in this checkout with the reused Cargo cache and
the existing dev Python environment. Logs are local evidence under `target/`;
the source tests are committed. Compilation-only setup errors from ad-hoc
rustc/Python import attempts are not counted as behavior REDs.

| Focused command / case | Observed RED | Final GREEN |
|---|---|---|
| `cargo test -p calc-flow --test bounded_analytic_over test_over_strict_declarations --no-default-features` | E0432: crate-root analytic types absent | declaration/config target: 4 pass |
| `cargo test -p calc-flow --test bounded_analytic_over_execution --no-default-features` | E0599: TableExpr.over absent | compiled-plan reuse: 1 pass |
| `cargo test -p calc-flow --test bounded_analytic_over_sql` | explicit macro reached DataFusion array-literal planning instead of bounded lowering | SQL target: 3 pass |
| `cargo test -p calc-flow --test bounded_analytic_over_stream` | bounded_order_v1 identity; later sequence error carried sort path | stream target: 8 pass |
| declaration/schema diagnostic cases | row limit exceeded u32 contract, missing name minLength, public errors lost over path | strict schema, indexed roles, canonical schema test pass |
| typed budget cases | wrong node name, then worker generation returned generic InvalidArgument | StateBudgetExceeded with analytic node; prior state retained |
| `cargo test -p calc-flow --lib operator::order::over_tests -- --nocapture` | earlier partial output quote allowed the deliberately insufficient effective quota | 4 ownership/budget/transaction tests pass |
| `python -m pytest python/tests/test_bounded_analytic_over.py` | root exports/method absent; operand and explain diagnostics absent | all 15 cases pass with final wheel |
| Python `-k 'requires_sortkey or expressions_types'` | 5 failures: wrong exception category or declaration path | same 5 cases included in final 146-pass run |

Additional cases cover signed/unsigned overflow, narrow integer widening,
all-NULL frames, NULL partitions and operands, string LAG, DESC/null placement,
N=0 and u32 maximum N, negative UTC windows, typed empty final, global sequence
uniqueness, cancelled/failed emit, late Error/Drop, and open/closed recovery.
No test expectation was relaxed to accommodate implementation behavior.

The nine-row independent oracle fixes sequence order to
`[10,11,12,13,21,22,40,30,31]` and hardcodes all four analytic result columns.
It includes ts=0, excludes ts=10 and NULL timestamps, accepts pre-close members
below W=9, and emits once at W=10. Rust DataFrame and SQL both execute through
real Source progress, batch/stream compilation, export and project reload.
Python exercises both frontends, both lifecycles, shared DAGs and cached plans.
Legacy SQL OVER still emits per-batch row numbers in the mixed-output stream.

## Owned peak boundaries

The three-row fixture includes nullable LargeUtf8 LAG. These are participating
node-accounting bytes on Linux x86_64, not RSS measurements or portable constants.
Each B trial succeeds; equivalent B−1 trials fail without installing candidate
history/closure/restore state. All finite output leases are released to zero.

| Operation | B (bytes) |
|---|---:|
| Finite nonempty | 144106 |
| Finite empty | 142386 |
| Capture peak | 764852 |
| Finalization with a live capture | 898318 |
| Restore with the source capture alive | 840340 |

Stream boundary trials use fresh, equivalent jobs so first-worker generation
funding participates consistently. An earlier comparison mixed warm and fresh
jobs and was corrected before recording these values. The separate public
80KB-budget case verifies typed failure with the declared quota; effective
pool boundary trials keep the declaration/capture identity fixed.

The allocation audit added distinct provisional input/output roles, four named
schema work owners and nullable string take's source-range scratch. The
deliberately insufficient earlier quote is a regression case, not a test pass.
The [user documentation](../../../docs/bounded-analytic-over.md#budget-ledger)
lists these lifetimes and reservation roles.

## Final local verification

```sh
cargo test -p calc-flow \
  --test bounded_analytic_over --test bounded_analytic_over_execution \
  --test bounded_analytic_over_stream --test bounded_analytic_over_sql \
  --test bounded_analytic_over_values --test bounded_analytic_over_closure \
  --test bounded_global_ordering --test order_analysis --test order_closure \
  --test project_v3
cargo test -p calc-flow --lib operator::order::over_tests -- --nocapture
python -m pytest python/tests/test_bounded_analytic_over.py \
  python/tests/test_bounded_global_ordering.py python/tests/test_sort_window.py \
  python/tests/test_query_history_portability.py \
  python/tests/test_update_semantics.py python/tests/test_capabilities.py -q
```

Results: Rust integration 41 passed (21 new, 20 neighboring), focused unit 4
passed; Python 146 passed (15 new, 131 neighboring). Python ran against the
rebuilt wheel extracted into `target/installed`, not an older installed package.

The selected Rust lib, six new integration targets and new example pass
`cargo clippy ... -- -D warnings`; PyO3 lib passes the same lint gate.
`cargo fmt --all --check`, Ruff check and format on 24 changed Python/stub
files, `scripts/verify_complexity_gates.py`, `scripts/check_spec_references.py`
and `git diff --check` pass. No complexity baseline was changed.

```sh
cargo run --locked -p calc-flow --no-default-features --example bounded_analytic_over
python examples/bounded_analytic_over.py
RUSTDOCFLAGS='-D warnings' cargo doc -p calc-flow --no-deps
```

The standalone core example builds without PyO3 and asserts totals `[1,3,6]`.
The Python example asserts all columns and equality of batch/stream finals from
the same declaration. Both run successfully. Rustdoc completes without warnings.

## Native/source provenance and limits

Environment: Rust 1.94.0, CPython 3.13.9, PyArrow 24.0.0, Linux x86_64.
`maturin build --locked --profile dev --interpreter <dev-python> --out target/wheels`
completed with the committed Cargo.lock. The wheel is a local linux_x86_64
validation artifact; its platform-tag warning means it is not a release wheel.

- Build-input manifest: 1509 Rust/Python/manifests/vendor files; SHA256 of the
  sorted JSON path/hash map:
  `89c4258f982292a4aa3ccbf77975ca8878fb0645aaf163ff2c9f80905e6b0524`.
- Wheel SHA256:
  `685d787990e7d042b80788dac5b7bed56367c786a55120329fc849e947e7e66f`.
- Native library SHA256:
  `39f2bca0c96cc38a6934a9ee0a157cc432f31664bea896c69046c2211ce6c14f`.
- All 62 wheel Python/stub files match this checkout byte-for-byte; imported
  `_native` resolves under `target/installed`. Every build-input file is checked
  again against the delivery commit. No generated native library is in source.

Full regression, the combined Rust 90% line coverage gate with connector services,
release wheel checks and macOS/Windows/other Python versions are not local
passes; they belong to regular CI. No performance suite was run. Operator-level
capture/restore and native Source progress are tested; no additional durable
store/process-restart recovery scenario was executed for this node. Review and
required CI remain prerequisites to merge. The issue stays in_progress until merge.

## Changed files

- `.codex/artifacts/analysis/bounded-analytic-over.md`
- `.codex/artifacts/api-notes/bounded-analytic-over.md`
- `crates/calc-flow-python/src/config.rs`
- `crates/calc-flow/examples/bounded_analytic_over.rs`
- `crates/calc-flow/src/config.rs`
- `crates/calc-flow/src/dataframe.rs`
- `crates/calc-flow/src/lib.rs`
- `crates/calc-flow/src/operator/distinct/owned.rs`
- `crates/calc-flow/src/operator/mod.rs`
- `crates/calc-flow/src/operator/order/checkpoint.rs`
- `crates/calc-flow/src/operator/order/mod.rs`
- `crates/calc-flow/src/operator/order/over_tests.rs`
- `crates/calc-flow/src/operator/order/sequence.rs`
- `crates/calc-flow/src/operator/order/spec.rs`
- `crates/calc-flow/src/operator/order/work.rs`
- `crates/calc-flow/src/operator/over/budget.rs`
- `crates/calc-flow/src/operator/over/evaluate.rs`
- `crates/calc-flow/src/operator/over/mod.rs`
- `crates/calc-flow/src/operator/over/schema.rs`
- `crates/calc-flow/src/operator/over/spec.rs`
- `crates/calc-flow/src/pipeline/analysis.rs`
- `crates/calc-flow/src/pipeline/order.rs`
- `crates/calc-flow/src/program/lower.rs`
- `crates/calc-flow/src/query_analysis.rs`
- `crates/calc-flow/src/query_analysis/order.rs`
- `crates/calc-flow/src/query_analysis/updates.rs`
- `crates/calc-flow/src/sql.rs`
- `crates/calc-flow/src/sql/order.rs`
- `crates/calc-flow/src/sql/over.rs`
- `crates/calc-flow/tests/bounded_analytic_over.rs`
- `crates/calc-flow/tests/bounded_analytic_over/common.rs`
- `crates/calc-flow/tests/bounded_analytic_over/public_stream.rs`
- `crates/calc-flow/tests/bounded_analytic_over_closure.rs`
- `crates/calc-flow/tests/bounded_analytic_over_execution.rs`
- `crates/calc-flow/tests/bounded_analytic_over_sql.rs`
- `crates/calc-flow/tests/bounded_analytic_over_stream.rs`
- `crates/calc-flow/tests/bounded_analytic_over_values.rs`
- `crates/calc-flow/tests/distinct_analysis.rs`
- `crates/calc-flow/tests/finite_union_frontend.rs`
- `crates/calc-flow/tests/latest_relation.rs`
- `crates/calc-flow/tests/order_analysis.rs`
- `docs/bounded-analytic-over.md`
- `docs/relational-frontends.md`
- `examples/bounded_analytic_over.py`
- `python/calc_flow/__init__.py`
- `python/calc_flow/_native.pyi`
- `python/calc_flow/capabilities.py`
- `python/calc_flow/stream.py`
- `python/calc_flow/symbolic/__init__.py`
- `python/calc_flow/symbolic/analyzer.py`
- `python/calc_flow/symbolic/expr.py`
- `python/calc_flow/symbolic/lower/program.py`
- `python/calc_flow/symbolic/lower/sql.py`
- `python/calc_flow/symbolic/nodes.py`
- `python/calc_flow/symbolic/order.py`
- `python/calc_flow/symbolic/over.py`
- `python/calc_flow/symbolic/query_analysis.py`
- `python/tests/test_bounded_analytic_over.py`
- `python/tests/test_bounded_global_distinct.py`
- `python/tests/test_bounded_global_ordering.py`
- `python/tests/test_capabilities.py`
- `python/tests/test_finite_asof_join.py`
- `python/tests/test_finite_event_window.py`
- `python/tests/test_finite_inner_join.py`
- `python/tests/test_finite_union.py`
- `python/tests/test_query_history_portability.py`
- `python/tests/test_update_semantics.py`
- `schemas/project-v3.schema.json`

## Codacy correction after initial delivery

This increment responds to the actual failure on
`d2c567030aafd3cae911ec653995aba0a8e4e33f` in the same checkout/branch/PR607.
The completed Codacy check is `131859180066`, conclusion `action_required`,
title `18 new issues (0 max.) of at least minor severity.` Its annotations were
read through authenticated `gh api`, since the GitHub connector rejects the
individual check/annotations endpoints:

```sh
gh api 'repos/wegamekinglc/calc-flow/check-runs/131859180066/annotations?per_page=100'
```

The 18 findings consist of 13 Rust and two Python complexity reports (limit 8),
and three Bandit B101 reports for Python example assertions. A single diagnostic
listing found no other failed check on that HEAD. No check was polled.

### Focused RED

```sh
uvx lizard -C 8 -w crates/calc-flow/examples/bounded_analytic_over.rs \
  crates/calc-flow/src/operator/order/mod.rs crates/calc-flow/src/operator/over \
  crates/calc-flow/src/sql/over.rs python/calc_flow/symbolic/over.py
uvx bandit -q examples/bounded_analytic_over.py
python -m pytest python/tests/test_bounded_analytic_over.py -q -k optimized
```

Lizard 1.26.0 exits 1 and reproduces all 15 locations, including workspace_bound
at CCN 28, SQL analytic at 27 and plan at 21. Bandit 1.9.4 exits 1 with exactly
three B101 findings. The new optimized-Python regression exits 1: an injected
incorrect native batch total `[-1,-1,-1]` is printed successfully because all
example assertions disappear under `python -O`. These are observed failures,
not a hypothetical reason to weaken checks.

### Correction

- Split declaration installation/error mapping, input schema validation and
  analytic field construction into their existing responsibilities.
- Name the partition, provisional input/schema, expression, LAG, string gather
  and array-buffer bounds separately. All checked arithmetic and quoted amounts
  remain unchanged, as do reserve-before-allocation and commit boundaries.
- Separate ROW_NUMBER/LAG evaluation and signed/unsigned primitive reads. Move
  expired frame-row subtraction to a pure helper returning sum/count; retain
  the same i128 accumulation, NULL semantics and per-output overflow checks.
- Separate SQL shape/source/arity, shared projection order, macro arguments,
  history/budget/policy and function/frame validation. Error paths and validation
  order remain covered by the existing strict SQL tests.
- Separate Python expression validation/serialization and capability lookup
  from compatibility checks. The example uses explicit RuntimeError checks
  for expected batch rows, final count and stream/batch equality.

Only nine source/example/test files and this evidence file change in the
increment. No public API, serialized schema, frozen note, dependency, gate
configuration, suppression or complexity baseline changes. A local Clippy
finding on the extracted declaration helper was corrected to borrow its input.

### Incremental GREEN

- The original Lizard command, expanded to the changed Python test/example,
  exits 0 with no warnings. All 169 functions in the nine changed source files
  have CCN <= 8. Bandit exits 0 with no B101 reports.
- The six OVER Rust integration targets plus bounded_global_ordering pass:
  31 tests. After the constructor helper's borrow-only lint correction, only
  bounded_analytic_over and bounded_analytic_over_execution are rerun: 5 pass.
  The other unchanged integration results are reused.
- `cargo test --locked -p calc-flow --lib operator::order::over_tests -- --nocapture`:
  4 pass. All B/B−1 outcomes and measured peaks are identical: nonempty 144106,
  empty 142386, capture 764852, capture+final 898318, restore 840340 bytes.
- Rebuilt final-source native wheel:
  `python -m pytest python/tests/test_bounded_analytic_over.py -q`: 16 pass,
  including optimized-Python rejection of a wrong batch result.
- `cargo clippy --locked -p calc-flow --lib` with the six new integration
  targets and example, `-- -D warnings`: passes. Ruff check/format on the three
  changed Python files, rustfmt, complexity ratchet, spec references and
  whitespace checks pass without a baseline update.
- The standalone core command from the original report passes again; so does
  `python -O examples/bounded_analytic_over.py`, checking batch and stream values.

One example invocation was mistakenly started before wheel extraction finished
and exited with SIGBUS. It is a setup failure, not product RED or passing
evidence. After extraction completed, the same example was run serially and
passed; only that completed invocation supports the result above.

The final wheel is built with the same locked toolchain/dev environment as the
initial delivery. All 1509 build inputs are checked against the corrective
delivery commit; all 62 packaged Python/stub files match the final source.
The replacement provenance is:

- Sorted build-input path/hash map SHA256:
  `91944e0cb051d0308cf116b12881f7fe83086e9d3afdddd7cb0884b2be997c1d`.
- Wheel SHA256:
  `14457ab13610593e91cd40d64bdeba243e05c14cee480b07e51cf5e1b7906094`.
- Native library SHA256:
  `3a6819c1a784d8c984c3942deb2d1e9fd9f78dc82b46e2331c1caca73f2e62c1`.

The unaffected initial report remains accepted evidence; no unrelated full
regression, coverage, platform, release, or performance matrix is rerun locally.
Local static checks clearing these diagnostics do not establish final-head
Codacy/CI success. The handoff comment records the single post-push nonblocking
snapshot. Independent review and required checks remain outstanding; no merge
or delegation is performed, and DAL-521 remains in_progress.

## Static subprocess regression follow-up

Base `a561b38d0f6c89fabe5712763c14b40ec9189065` subsequently received one
Codacy annotation in completed/action_required check `131868119063`:
`python/tests/test_bounded_analytic_over.py:32`,
`Detected subprocess function 'run' without a static string.` The orchestrator
provided the confirmed annotation; the previous 18 corrections remain accepted.

The test now uses the existing `python/tests/test_config.py` invocation form:
literal argv[0] `python`, with `executable=sys.executable` selecting the current
interpreter explicitly. The `-O` flag, fixed injected script, example path,
default shell=False, and both rejection assertions remain unchanged. No PATH
lookup is needed to select the executable. No suppression or gate configuration
changes. Only this test and the evidence report change.

The public Semgrep rule
[dangerous-subprocess-use-audit](https://github.com/semgrep/semgrep-rules/blob/develop/python/lang/security/audit/dangerous-subprocess-use-audit.yaml)
contains the same diagnostic and explicitly recognizes a literal first argv
element. Its unchanged downloaded local copy has SHA256
`db8fb4ee4a31dafe2e286228edebc07fd1386d9f63a0f683d26af4e6ff4d455d`.
Semgrep 1.180.0 was run before and after the two-line test adjustment:

```sh
semgrep scan --config target/bounded-analytic-over-subprocess-rule.yaml \
  --error --json --metrics off --no-git-ignore \
  python/tests/test_bounded_analytic_over.py
```

RED: exit 1, exactly one finding at line 32 with the matching message.
GREEN: exit 0, exactly one file scanned, zero findings and zero scan errors.
The rule copy is local verification input under target, not a repository gate
replacement or configuration change.

```sh
PYTHONPATH=target/installed <dev-python> -m pytest \
  python/tests/test_bounded_analytic_over.py -q -k optimized
```

GREEN: 1 passed, 15 deselected. The test still runs an optimized Python child,
injects incorrect SUM values `[-1,-1,-1]` and requires a failed process with
`RuntimeError: batch analytic result differs`. Ruff check, Ruff format check,
`uvx lizard -C 8 -w python/tests/test_bounded_analytic_over.py`, and whitespace
checks pass.

Product/native/Rust, public contracts, ownership and transaction behavior are
unchanged. Their accepted evidence and the 1509-input native provenance above
are reused; no native rebuild, Rust rerun or unrelated Python/full/static matrix
is performed. This focused local rule pass is not a claim that new-head Codacy
or all CI has passed. The final reply records the single nonblocking post-push
snapshot; review remains pending, no merge/delegation is performed, and DAL-521
stays in_progress.

## Independent review F1 correction

This increment addresses only F1 from the completed independent review of
`2c1ea159f2db15a73c22b0cb1f62509499b5f63b`. The review report is preserved
byte-for-byte and left untracked, with SHA256
`c524f3aa5df760cc50f61d2432f1de61d27e0378c98d22f3abc058f742f2209f`.
It is not rewritten into an approval or included in this correction's commit.

The classifier in `symbolic/query_analysis.py::_issue_diagnostic` omitted the
OVER role from the existing definitive capability-rejection roles. Adding
`.over` to that tuple fixes both the public query diagnostics and the output
node's batch/stream support. The generic capability-mismatch fallback and all
native source/cut proof rules are unchanged. No public contract is expanded.

### Focused RED and GREEN

Three public regressions declare RANK, SUM without ROWS, and ROW_NUMBER/LAG
with mixed orders. Before the production change, this command exits 1:

```sh
PYTHONPATH=target/bounded-analytic-over-f1-python <dev-python> -m pytest \
  python/tests/test_bounded_analytic_over.py -q -k known_rejections
```

RED: 3 failed, 16 deselected. Each fails at the expected assertion:
`{'unknown'} != {'unsupported'}`, reproducing F1 through `Program.analyze().query`.
After the one-line role correction, GREEN: 3 passed, 16 deselected. Each case
checks both lifecycle diagnostic paths and capability_mismatch codes, both
output-node supports, ordinary batch/stream compile rejection, and strict
batch/stream entry rejection with input wrapping/source iteration counters zero.

A first GREEN attempt passed the analysis assertions but reached a test-harness
AttributeError: the package exports `compute` as a function. The read-spy helper
was corrected to explicitly import the module through importlib; no expectation
was weakened, and this setup error is not a second product finding.

Neighboring unknown/closure checks run through the same staged source:

```sh
PYTHONPATH=target/bounded-analytic-over-f1-python <dev-python> -m pytest -q \
  python/tests/test_bounded_analytic_over.py::test_over_unknown_source_is_rejected_before_iterator_and_root \
  python/tests/test_finite_event_window.py::test_window_analysis_proves_conditional_eof_and_leaves_prefix_unknown \
  python/tests/test_query_history_portability.py::test_unproved_selection_is_not_granted_or_given_an_invented_counterexample \
  python/tests/test_query_history_portability.py::test_unknown_upstream_updates_are_not_defaulted_to_append
```

Result: 5 passed. Missing concrete window-cut evidence and unproved SQL remain
unknown; unknown-source OVER still rejects before iterator acquisition and
temporary-root creation. These results are separate from the three F1 cases.

Ruff check and format checks pass on the changed mapping/test. A whole-module
`lizard -C 8` probe exits 1 on two pre-existing functions: _named_roles (11) and
_native_relation_kinds (16). AST source comparison proves both untouched, and
all module function CCNs are identical to the reviewed baseline. The changed
_issue_diagnostic remains CCN 8; the new public test is 8 and its helper is 2
(nested helpers 1). Focused complexity assertions and whitespace checks pass.
No baseline, gate, suppression or unrelated function is changed to clear F1.

### Tested mapping/native provenance

The test package under `target/bounded-analytic-over-f1-python` contains copies
of all 62 current source Python/stub files and a link to the accepted native
library. Every copied file was compared byte-for-byte with source. Imported
query_analysis resolves to this staged package; its file matches the corrected
source and its loaded _issue_diagnostic includes the new role.

- Corrected query_analysis SHA256:
  `26867bc79de9ae3bcdc9f4a68f116d574928e0cb1cdd1d7927cf2948b69c7a19`.
- Imported native SHA256 remains:
  `3a6819c1a784d8c984c3942deb2d1e9fd9f78dc82b46e2331c1caca73f2e62c1`.
- Of the 1509 accepted build-input entries, only this pure Python mapping
  changes; the other 1508, including Rust/PyO3/manifests, remain identical.

No native rebuild or Rust, full regression, coverage or performance rerun is
performed. The prior valid evidence is reused. The full independent review
remains complete, with only F1/direct impact requiring reviewer recheck; this
report does not grant that acceptance. The final issue reply records the single
post-push nonblocking CI snapshot. No merge/delegation; DAL-521 stays in_progress.
