# Finite ASOF join — implementation evidence

DAL-518 / S2d. Baseline `909463f0298084fc7629596215aa7d7c6b78ed07`;
branch `agent/dal-orchestrator/937ec660b6b0`. Work uses the coordinator's
requested isolated checkout. The only original untracked file, the frozen
[API note](../api-notes/finite-asof-join.md), is preserved and included.
The final delivery comment identifies the actual Git HEAD and unique main PR.
This report records source and binary correspondence independently of a
self-referential documentation commit hash.

## Contract and design

The controlling API note is the design checkpoint. The original bounded-ASOF
specification, API note, critique and Arrow-output addendum, finite-inner note,
repository AGENTS/code style, implementation methodology, relevant code and
nearby tests were read. The three `.codex/references` files named in the runtime
identity are absent in this repository; existing owning components and
`.claude/rules/code-style.md` apply. There is no dal-public component or
Machinist-generated enum in this change.

Each `BatchOperator::process` constructs a fresh ASOF scratch owner from immutable
spec/schema. It validates both envelopes, admits the complete left then right
history, and only then marks both sides complete and drains pending left rows.
The native admission, typed predecessor selection, canonical ordering, Arrow
gather, accounting and bounded prefix commit paths are reused. EOF is explicit;
no maximum watermark, StreamingRunner, StreamJobContext, checkpoint, source or
sink is manufactured for batch execution. The finite context carries the run's
cancellation/deadline and the existing byte ceiling. CPU work retains its funded
inputs and reservations on the worker. Complete inputs share the existing total
row/byte budget across sides and RecordBatches.

The compiled batch wrapper has NULL snapshots and rejects non-NULL restore;
reset is unnecessary. Failures cannot install scratch candidates, counters or
terminal state into the compiled operator. Output is one typed Batch, including
typed empty, with node source, sequence zero and empty attributes.

Rust `TableExpr::asof_join` and the two SQL frontends join Python's existing
namespace/fluent ASOF declarations at the same native node. SQL recognition and
schema authority remain Rust; the private PyO3/stub description is data-only.
`CF_ASOF_JOIN` reuses project field validation, including nested numeric/type,
unknown/missing-field and alias paths. Python adds no SQL parser or matching
implementation. Native Rust frontend schema failures retain public spec paths
and original ASOF reason codes.

Join-local SQL and pure DataFrame renames preserve proved left-derived temporal
identity. New SQL/mixed graphs use the existing memo, retaining nested/fan-out
owners, side multiplicity and unrelated union/aggregate branches. Existing pure
stream ASOF topology and hashes remain unchanged. Analysis schema stays 2 and
revision becomes 7; historical reports through r6 remain readable, while latest
and strict admission require current proof before caches or source acquisition.
Capability modes become batch/stream; semantic state 3 and native layout 10,
source replay, ordinary SQL policy, update interpretation and checkpoint codecs
are unchanged.

## Test-first observations

All candidate builds/tests use this checkout. The existing development Cargo
artifact directory is a build cache, never a second source checkout. Python RED
and the historical identity capture used the predecessor wheel only to establish
missing behavior/compatibility vectors. Candidate GREEN uses the wheel built
from this checkout plus an exact private copy of its current Python/stub files.
No `_native*.so` is placed in the source package.

- `cargo test --locked -p calc-flow --test finite_asof_join`: RED, ASOF rejected
  as stream-only. The complete-input oracle then became GREEN. A second finite
  compiler gate was exposed and removed by the Rust frontend fixture.
- `pytest python/tests/test_finite_asof_join.py`: RED, Python DF rejected batch
  and SQL reached an unknown DataFusion table function. The revision proof was
  RED at r6; a strict-cache test separately caught acceptance of a stale r6 report.
- The finality explanation test was RED because an ASOF group-final node still
  described revisable aggregate snapshots. The r7 ASOF reason now describes its
  append-final left result and closure; the final wheel passes that regression.
- The missing-alias test was RED at generic `sql.asof_join`; native parsing now
  identifies `sql.asof_join.tables.b`.
- The Rust nullable-key report was RED at unqualified `stream_asof_join.left...`
  with `schema_mismatch`; native frontend validation now reports the public
  `outputs.match.asof_join.spec.left.keys[0]` and `nullable_identity_field`.
- Mixed Python tests exposed undeclared-input issues in local schema analysis
  and lost metadata through pure renames. Both were repaired with scoped proof;
  dropping or rewriting required identity still rejects.

Fixture corrections were separate from product fixes: ProviderRegistry uses
`default`; direct ASOF runner sources require a generated watermark policy with
positive delay (Disabled is deliberately rejected); Arrow rejects actual NULL
in a non-null field during binding with TypeError, before ASOF execution. Old
stream-only test expectations were replaced by positive finite assertions while
retaining unsupported type/capability/identity checks. No test was weakened to
accept an incorrect result, and no lint baseline, limit, reader or gate was relaxed.

## Acceptance coverage

- C08/O02: literal expected predecessor values, inclusive tolerance, future and
  out-of-range unmatched rows, zero tolerance, negative/i64 endpoint times,
  numeric uint/signed/string/composite ties, nullable payloads, typed empty,
  distinct left sequences and side-scoped identities. Cross-RecordBatch
  duplicates reject even when payload differs.
- L02/A06/C13: the same compiled plan and warm Runtime execute normal/right-empty/
  left-empty/all-empty/new candidates without reset. Duplicate, total-budget and
  cancellation failures leave a fresh retry; snapshots stay NULL.
- A04–A06: Rust and Python DF/SQL actually execute both lifecycles; real stream
  outputs match independent hand-written expectations. Nested ASOF, mixed
  inner↔ASOF, pure renames, shared owners, self-ingress multiplicity and unrelated
  branches are covered. Invalid nullable right identity still rejects.
- A08: project-v3 batch reload executes with physical bindings in both languages;
  Rust stream reload also executes both output branches. The core example is an
  independently compiled executable, with no Python/PyO3 or test helper import.
- A09/C13: raw SQL numeric paths and unsupported shape rejection, r7 proof,
  historical parser/latest/strict cache gates, no-input-read latest snapshot
  rejection at either side, and legacy SQL behavior are checked.
- L01–L03: affected stream boundary/state tests cover equal frontier, partial
  closure/idle, i64::MAX EOF, next-update recovery, unchanged-segment sharing,
  repeated capture/restore and terminal EOF without re-emission. These are
  actual behavior/segment checks, not version-number assertions alone.

Historical identity vector captured from the predecessor package:

- Python ASOF @1 digest:
  `662069e2734627ef0c359c58c6c3512f8652c5a6ee9039b3539a44e6bcc5b2ca`.
- Shared-output program fingerprint:
  `cb3fdec2d27688912555f8318d542ef27d741973a19e593888d7e54dc653fc68`.
- Sorted complete exported-project JSON SHA-256:
  `3819fa0efc674ea82b4f523843465b3e3f154131c1e78bba2605672efeda8099`.

The committed compatibility regression checks all three, including native IDs,
ports, edge multiplicity, source/output bindings and configuration.

## Verification and provenance

All commands ran in the assigned checkout, with `CARGO_BUILD_JOBS=2` and the
existing development artifact cache in
`target/pr-repair-20261112/cargo-debug` of the development checkout. This changes
only Cargo's artifact location. `$CF_PY` below is the existing managed CPython
3.13 interpreter, and `PYTHONPATH=target/finite-asof-join/site` selects the exact
private candidate package.

```sh
cargo test --locked -p calc-flow --test finite_asof_frontend --test finite_asof_join
cargo test --locked -p calc-flow --test stream_asof_join_boundaries --test stream_asof_join_state --test stream_asof_join_validation --test query_history_portability
cargo test --locked -p calc-flow --test finite_asof_join --test latest_relation --test finite_union_frontend --test finite_inner_frontend
cargo test --locked -p calc-flow --lib operator::asof::tests::gather_admission::
```

Results: **8**, **42**, **22**, and **3** passed respectively. The third command
includes the same four finite-kernel tests; across these targets there are **71
distinct successful Rust tests**. Earlier unaffected stream/history results are
reused rather than labelled a fresh rerun: subsequent edits bound finite-only
output, qualify new Rust frontend diagnostics, and fix test-only imports/fixtures.
The final native kernel unit target compiles all core unit modules but executes
only its three selected tests (**3187 filtered**). It checks real output-prefix
retry, worker admission and unchanged committed state. The extra compilation
caught and fixed two test-only context import errors before handoff.

```sh
$CF_PY -m pytest python/tests/test_finite_asof_join.py python/tests/test_symbolic_stream_asof_join.py python/tests/test_asof_stream_results.py python/tests/test_stream_asof_join_builder.py -q
$CF_PY -m pytest python/tests/test_finite_asof_join.py python/tests/test_symbolic_stream_asof_join.py python/tests/test_asof_stream_results.py python/tests/test_stream_asof_join_builder.py python/tests/test_finite_inner_join.py python/tests/test_finite_event_window.py python/tests/test_finite_union.py python/tests/test_query_history_portability.py python/tests/test_update_semantics.py -q
```

The first command passes **177 tests on the final rebuilt native/package**. The
broader directly affected run passed **337** before three final identity/NULL/
mixed-direction regressions were added. Those three also pass on the final native;
its unaffected adjacent tests are reused, giving **340 distinct passing Python
tests** overall. This is a directed set, not a full-suite claim. The old legacy
SUM chain still returns 7; supported explicit latest returns 6 or rejects before
reading inputs.

```sh
cargo clippy --locked -p calc-flow --lib --test finite_asof_join --test finite_asof_frontend --example finite_asof_join -- -D warnings
cargo check --locked -p calc-flow --lib --features benchmark-internals
cargo run --locked -p calc-flow --example finite_asof_join
RUSTDOCFLAGS='-D warnings' cargo doc --locked -p calc-flow --lib --no-deps
$CF_PY examples/finite_asof_join.py
cargo fmt --all --check
```

Clippy, optional-feature compilation, core rustdoc, both standalone example
oracles, Rustfmt, changed-file Ruff check/format, complexity ratchet, changed
function CCN review, documentation links/newlines and `git diff --check` pass.
No lint allow, complexity baseline or ignore changed. Optional-feature compilation
is not a benchmark or performance result.

The final native build command was:

```sh
$CF_PY -m maturin build --locked --profile dev --interpreter "$CF_PY" --out target/finite-asof-join/wheels
```

Observed provenance at 2026-11-12 11:47 UTC:

- Native SHA-256:
  `2fe4724002b2c8a4ac7c0bbbef943c3adc25549e2cdaa3ea3cf9f5d2a394d423`.
- Wheel SHA-256:
  `98588bdd0ca2d4357163be13248146dc00c2f040fcb167e6d84d41143a31f7a6`.
- Repository source-manifest SHA-256:
  `4d7ec300c4ebfb4ea128d5e8c98ba139a7ed2ddc8ef8d0f46b8f67b25703b567`.
- **1633 Rust/Cargo repository source files**, including verification and vendored
  sources, and **59 Python/stub files** are hashed. All 59 installed private files
  match this checkout exactly. Import locations are the private
  `target/finite-asof-join/site/calc_flow/` package and its native extension.
  The interpreter is the development environment's managed CPython 3.13.

Logs, the local dev wheel, private site and full `provenance.json` are retained
under `target/finite-asof-join/` for the coordinator. The committed report and PR
are review artifacts; build outputs are not published release artifacts. Final
source hashes are checked again against the committed delivery HEAD.

## Changed files

Core changes are in `operator/asof/{finite,kernel,admission,copy,cpu,finalize,
output,retirement}` and their affected helpers, `context.rs`,
`runtime/streaming/gather_work.rs`, `pipeline/batch.rs`, `config/asof.rs`,
`dataframe.rs`, `sql/asof_join.rs`, `program/lower/asof.rs`, adjacent join/lowering
routing, `project_store/asof.rs`, and `query_analysis.rs`. PyO3 adds the private
SQL description registration in `config.rs`, with its stub and live analysis
revision assertion synchronized.

Python changes cover capability modes; symbolic ASOF declaration/analysis,
`asof_sql.py`, join-local proof, SQL memo lowering, program admission and query
analysis gates. Focused Rust/Python ASOF tests and the two public examples are
new; adjacent live revision and former stream-only assertions are synchronized.
The API note, this report, AGENTS and six relevant guides document the dual
lifecycle and supported dialect. The PR changed-file list is the exact inventory.

## Scope and remaining gates

No direction/unbounded tolerance/replacement ASOF, batch mid-query recovery,
new layout reader, general SQL ASOF, performance promise or merge is added.
Finite complete-history admission can require a larger peak budget than a
progress-evicting stream; both executions must succeed before terminal comparison.

Examples remain self-contained and formatted: Python includes the full strict
wire configuration; Rust includes typed Arrow fixtures and public fallible APIs.
They exceed the note's illustrative 20–50-line target (63/114 formatted lines)
to avoid private helpers or hidden configuration. Complex acceptance matrices
remain tests. The original stream example and its watermark/EOF purpose are intact.

Full workspace, coverage, hosted platforms and release gates belong to CI;
independent dal-reviewer acceptance and required checks remain outstanding.
DAL-518 stays in_progress and this run does not merge or delegate.
