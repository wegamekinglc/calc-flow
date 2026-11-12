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

## F1 repair — Rust DataFrame identity aliases

Repair baseline: `8135297faaea2fcb25b5cc17216a782274200641`, same branch
and [PR #602](https://github.com/wegamekinglc/calc-flow/pull/602). The reviewer
report remains untracked and byte-identical, SHA-256
`31cda92bf764ac72aacf33afd3a43ca4a4e6598c9227978f9d60da74a722a21d`.
The coordinator authorized only F1 and its direct effects; no new specification,
public API, delegation, second PR or merge is introduced.

`prepare_projection` previously discarded temporal identity when aliases removed
the old field names. The join-scoped projection helper now proves direct column
references against the input lineage and maps the complete entity/event-time/
sequence vectors together. The existing SQL mapper supplies the same complete
identity construction through a private column-mapping callback; SQL's own
column selection is unchanged. Missing identity components and derived
expressions cannot create proof. Appended fields and projections retaining an
original identity field prefer that original name over additional aliases.
Outside the established join paths, the previous projection retention and
ordinary SQL behavior remain in force. Native relation IDs, ports and operator
configuration construction are untouched.

The increment is confined to `program/lower.rs`, `program/lower/{join,projection}.rs`,
`tests/finite_asof_frontend.rs`, its `rename.rs` and `stream.rs` helpers, and this
analysis. Tests execute source → DF rename → ASOF and ASOF → DF rename → ASOF
in batch and through actual source/sink streaming, comparing literal expected
values 7 and 3. The nested test also exposes the shared first output. Controls
cover complete entity/sequence requirements, casts, sequence arithmetic,
ordinary DF/SQL proof boundaries, and additional aliases with retained originals.

### F1 RED evidence

The unchanged independent caller in the reviewer report was run first against
the pre-repair library:

```sh
CF_DEPS="$CARGO_TARGET_DIR/debug/deps"
rustdoc --test .codex/artifacts/reviews/finite-asof-join.md --edition=2024 \
  --crate-name finite_asof_rename_review -L dependency="$CF_DEPS" \
  --extern calc_flow="$CF_DEPS/libcalc_flow-0cbf0666ea3b0fc9.rlib" \
  --extern tokio="$CF_DEPS/libtokio-4a0c17d14ad9501c.rlib"
cargo test --locked -p calc-flow --test finite_asof_frontend rename::
```

Independent caller: **0 passed, 1 failed**, exit 101. DF Batch and Stream both
reported `outputs.out.asof_join.left: ordering_required`; equivalent SQL Batch
and Stream both succeeded. The added integration tests then produced **2 passed,
2 failed**, exit 101: the source and nested DF alias paths failed for the same
missing identity, while the negative and ordinary-proof controls passed.
This is the expected product RED. A later test-only refactor needed an explicit
unwrap for a known Arrow fixture field, because ArrowError does not convert to
CalcFlowError; that compile error is not counted as product RED.

### F1 independent caller GREEN

The same immutable reviewer caller passes with the repaired core artifacts:

```sh
rustdoc --test .codex/artifacts/reviews/finite-asof-join.md --edition=2024 \
  --crate-name finite_asof_rename_review -L dependency="$CF_DEPS" \
  --extern calc_flow="$CF_DEPS/libcalc_flow-c4242eaab5409a3b.rlib" \
  --extern tokio="$CF_DEPS/libtokio-b3a17c46acc34096.rlib" \
  --test-args=--nocapture
```

Result: **1 passed, 0 failed**, including all four DF/SQL × Batch/Stream
schema-collection branches. Repaired core rlib SHA-256:
`05f6164bb4d0f313b580441fbf109199f4f4d7319495bad0dc15684b07e40a28`.
The log is retained as `target/finite-asof-join/f1-caller-green.log`.

### F1 integration GREEN

`cargo test --locked -p calc-flow --test finite_asof_frontend rename::`:
**5 passed, 0 failed, 4 filtered**. Both new positive chains execute the batch
plan and an actual StreamingRunner against literal expected values, rather
than stopping at schema collection. The negative cases reject in both modes;
ordinary DF/SQL outputs have no temporal proof. Appended aliases and a select
containing both originals and aliases both execute with their original ASOF spec.

The shared Cargo cache was occupied by another task's core unit builds during
this run. Those builds were neither terminated nor counted as F1 verification.

Directly affected checks, without rerunning the five unchanged rename tests:

```sh
cargo test --locked -p calc-flow --test finite_asof_frontend \
  --test finite_inner_frontend --test query_history_portability -- --skip rename::
cargo clippy --locked -p calc-flow --lib --test finite_asof_frontend \
  --test finite_inner_frontend --test query_history_portability -- -D warnings
cargo fmt --all --check
```

Results: **4 + 3 + 9 = 16 passed**, Clippy and formatting passed. Together
with the five F1 tests, this run verifies **21 distinct Rust integration tests**
and the independent caller. Coverage includes existing DF/SQL execution and
project reload, shared/nested ASOF and inner join, incomplete identity rejection,
ordinary SQL support boundaries, snapshot/history admission and existing row-local
identity. The unchanged kernel, recovery/terminal behavior and standalone core
example results above are reused, not claimed as fresh F1 runs. No full suite,
coverage matrix or new benchmark was run. The two new production helper functions
have measured Lizard cyclomatic complexity 2 and 4; no baseline or lint was relaxed.

### F1 native/Python source correspondence

The native extension was rebuilt once after the repair and directed Rust checks:

```sh
$CF_PY -m maturin build --locked --profile dev --interpreter "$CF_PY" \
  --out target/finite-asof-join/wheels-f1
PYTHONPATH=target/finite-asof-join/site-f1 JAX_PLATFORMS=cpu \
  $CF_PY -m pytest python/tests/test_finite_asof_join.py -q \
  -k legacy_asof_declaration_and_native_topology_identity
```

Build passed in 47.35 seconds. The local dev wheel was extracted into the
separate `site-f1` directory; both imported Python and native module paths were
checked against that directory. The targeted compatibility regression passed:
**1 passed, 37 deselected**. It checks the unchanged declaration digest, shared
program fingerprint and complete project/topology JSON against the historical
vectors above. No Python or stub source changed; the earlier 340 Python results
are reused, not represented as a new broad run on this repair.

Observed provenance at 2026-11-12 12:28 UTC, in
`target/finite-asof-join/provenance-f1.json`:

- Native SHA-256:
  `7cb42529a6df7e6f4f46cb55e3f25a1d5112d161e49072d4ca56018f2eb10baa`.
- Wheel SHA-256:
  `bb3da74877e68f10337acc7101eeaf9e68f438d1c897ca8510f358f32d7116c4`.
- Source manifest SHA-256:
  `9702d0a5d86338017f664e958505cb1154bd4765e2c71d298c3a678cfac95c7f`.
- **1642 Rust/Cargo files**, including the two new F1 Rust files and seven
  unchanged fuzz/tool/vendor lock entries not listed in the earlier inventory;
  **59/59 Python/stub files** match the extracted package exactly. The manifest
  hash is SHA-256 of compact, sorted UTF-8 JSON containing `rust_sources` and
  `python_sources`. Its recorded repaired core rlib hash also matches the
  independent caller's library. The final commit is recorded in that manifest
  after verifying every source hash against Git HEAD.

The old wheel/site/provenance remain available separately. These are local dev
verification artifacts, not release wheels or release-gate evidence.

### F1 handoff and remaining gates

This repair remains on PR #602 and DAL-518 remains `in_progress`. The original
reviewer still needs to recheck F1 and its direct effects; no independent review
acceptance is claimed. Only the coordinator-supplied CI snapshot for the
pre-repair HEAD is used this turn: macOS passed, Linux/Windows/crate release
were running, and returned jobs had no failure. That does not prove all checks
green or verify the repair HEAD. There is no additional CI snapshot, watch or
poll. Full hosted, coverage and release checks remain CI responsibilities; this
run does not merge or dispatch further work.

## F1 follow-up — appended pure alias chains

Repair baseline: `318a64dfbf0e6af7fbeb737030a9899d5770fa23`, same branch
and PR #602. The reviewer's new failing caller establishes the need for this
local correction; other original review conclusions remain in force. The
updated reviewer report is preserved untracked and unchanged during this round,
SHA-256 `bb9f59cd8759e830a388ebe4506c7ae706ad4c5e35b2dc48c201ea9918321cda`.
Its caller replaces the report's prior direct-select caller; the earlier RED/
GREEN records above describe their respective historical invocations.

### Cause and bounded correction

The first F1 repair deliberately retained original identity names after
`with_columns`. A subsequent select containing only the appended aliases could
not match those names. The new comparison follows direct Project references and
transparent Filter inputs back to a `(relation id, field name)` origin. It stops
at other relation boundaries, so different native join outputs remain distinct.
Cast, arithmetic and other scalar expressions provide no column origin. The
comparison still requires an existing complete identity; it cannot manufacture
one for an unproved input. Full entity/event-time/sequence mapping, original-name
priority and the existing join-path gate remain unchanged. SQL proof, public
signatures, native node/port construction and operator state are untouched.

This increment changes only `program/lower/projection.rs`, the existing rename
test module's child registration, its new `alias_chain.rs` tests, and this report.
The new tests actually execute BatchExecutionPlan and StreamingRunner for:

- `with_columns → select only aliases → ASOF`, matching the independent caller;
- ASOF → appended aliases → Filter → another appended alias layer → select only
  aliases → ASOF, with the shared first result exposed and literal values 7/3;
- the complete two-field sequence vector through the same alias-only selection.

Controls reject dropped composite entity/sequence components, cast key/time,
sequence arithmetic and a rewritten post-ASOF sequence in both modes (12
rejection checks). An ordinary alias chain still has no temporal identity proof.
Previous direct-alias, original-name-priority, SQL and nested cases are retained.

### RED and GREEN

The updated reviewer caller was first run against its recorded pre-correction
core rlib SHA-256 `05f6164bb4d0f313b580441fbf109199f4f4d7319495bad0dc15684b07e40a28`:

```sh
CF_DEPS="$CARGO_TARGET_DIR/debug/deps"
rustdoc --test .codex/artifacts/reviews/finite-asof-join.md --edition=2024 \
  --crate-name finite_asof_f1_alias_chain_review -L dependency="$CF_DEPS" \
  --extern calc_flow="$CF_DEPS/libcalc_flow-c4242eaab5409a3b.rlib" \
  --extern tokio="$CF_DEPS/libtokio-b3a17c46acc34096.rlib"
cargo test --locked -p calc-flow --test finite_asof_frontend rename::chain::
```

Independent caller RED: **0 passed, 1 failed**, exit 101; DF Batch and Stream
both rejected with `ordering_required`, SQL controls both succeeded. Integration
RED: **2 passed, 3 failed**, exit 101; all three legitimate execution paths failed
for missing identity, while the distortion and ordinary-proof controls passed.
After the minimal projection correction, the same commands are GREEN: **1/1
independent caller and 5/5 new tests**. Repaired core rlib SHA-256:
`3a43339999c6debf09e46275e2ce7d1e5a032eef02f2aff76b02c6613433760d`.
Logs are `target/finite-asof-join/f1-chain-{caller,tests}-{red,green}.log`.

Directly affected checks, avoiding a repeat of those five tests:

```sh
cargo test --locked -p calc-flow --test finite_asof_frontend \
  --test finite_inner_frontend -- --skip rename::chain::
cargo clippy --locked -p calc-flow --lib --test finite_asof_frontend \
  --test finite_inner_frontend -- -D warnings
cargo fmt --all --check
```

Results: **9 + 3 = 12 passed**, Clippy and formatting passed. This round has
**17 distinct successful Rust integration tests**, plus the independent caller.
Production helper Lizard complexity is at most 5. The earlier nine query-history
checks, native kernel/recovery/terminal evidence, example compilation and broader
Python results remain reused evidence. No full suite, coverage or performance
run was added, and no baseline or lint rule was relaxed.

### Follow-up source correspondence and handoff

The repaired sources were used for one native build and the existing identity
compatibility check:

```sh
$CF_PY -m maturin build --locked --profile dev --interpreter "$CF_PY" \
  --out target/finite-asof-join/wheels-f1-chain
PYTHONPATH=target/finite-asof-join/site-f1-chain JAX_PLATFORMS=cpu \
  $CF_PY -m pytest python/tests/test_finite_asof_join.py -q \
  -k legacy_asof_declaration_and_native_topology_identity
```

Build passed in 30.17 seconds; the compatibility test passed **1/1** (37
deselected), including unchanged declaration, shared program and full exported
project/topology hashes. Imported package/native paths were verified under
`site-f1-chain`, extracted separately from the new local dev wheel. Python and
stub sources are unchanged and all **59/59** extracted files match exactly.

`target/finite-asof-join/provenance-f1-chain.json` records the build at
2026-11-12 12:46 UTC, using the same interpreter/cache and manifest encoding as
the preceding repair. Its inventory is **1643 Rust/Cargo + 59 Python/stub files**,
including the new alias-chain test. Hashes:

- Native: `386732408239bd09e27962299975ab2ee617919d2c4b38c739f993504db5cccd`.
- Wheel: `201a8b25b354582351b636a1ba1cea964c676a672155463d07d447d3d01a082c`.
- Source manifest:
  `f78854687c5f004cbd5781fd30d7674e1b656691f12b2854768c4dd0c3f11c8f`.

The core rlib hash matches the independent GREEN caller above. After commit,
every recorded source is verified against both the worktree and Git HEAD, and
the exact delivery commit is recorded in the manifest. Previous wheel/site/
provenance artifacts remain separate; no release verification is claimed.

The remaining gate is the original reviewer's local F1 recheck, arranged by the
coordinator. CI reuses only the supplied `318a64df` snapshot: 8 SUCCESS,
12 IN_PROGRESS, 0 failed, with four workflows unfinished. It is not evidence
that this follow-up HEAD is green. No new CI read/watch/poll, broad review chain,
delegation or merge occurred. DAL-518 remains `in_progress`.

## Codacy gate repair — tests and example only

Baseline `45bf88ecc13e2f9cb191718438969bb285f7fc11`. The independent product
review is **Approve**, with F1 closed. This is a verification/example refactor;
product sources, public APIs, native inputs, bindings and package/stubs are
unchanged, so that product approval and the valid product results remain in force.
The reviewer report is preserved untracked with SHA-256
`3b092bd1381cacf0ed341e2ea654c68d876c9d51db4ffd4429156f89492838a4`.

The actual RED is Codacy check-run `131193242438`: `action_required`, 12 new
issues against a zero-new-issues gate. The coordinator's original findings and
the known check-run annotations were read; no new CI-state poll was needed for
diagnosis. The increment is restricted to the two Rust rename/alias-chain test
files, `python/tests/test_finite_asof_join.py`, `examples/finite_asof_join.py`
and this analysis. Codacy/CI/baseline/ignore configuration is unchanged, and no
suppression is added.

### Mapping the 12 findings

Seven Rust functions are split into named fixture constructors, field selection,
identity distortion and common dual-mode rejection checks. Expected values remain
literal 7/3, and all 22 identity rejection checks remain. The original ordered
list of five tests in each file is identical after refactoring.

| Function                                                   | Codacy RED complexity | Local GREEN complexity |
|------------------------------------------------------------|-----------------------|------------------------|
| nested_alias_chain_through_filter_executes_both_lifecycles | 13                    | 6                      |
| alias_chain_preserves_complete_composite_sequence          | 9                     | 7                      |
| distorted                                                  | 10                    | 3                      |
| alias_chain_cannot_launder_derived_or_incomplete_identity  | 9                     | 7                      |
| asof_df_rename_asof_executes_in_both_lifecycles            | 9                     | 5                      |
| df_rename_does_not_prove_incomplete_or_derived_identity    | 9                     | 4                      |
| additional_aliases_retain_the_original_identity            | 12                    | 5                      |

Lizard 1.23.0 measures **all 29 functions at complexity ≤8**, including new
helpers. Codacy and local Lizard differ by one on the original `distorted`
function; the final helper is comfortably below the threshold. The raw local
before/after inventory is retained in `codacy-complexity-{red,green}.json`.

The example's two warnings are reproduced by Bandit 1.9.4 as B608 and B101.
It now spells the complete fixed CF_ASOF_JOIN JSON/SQL literal directly. Native
SQL descriptions from the original and new example are exactly equal, proving
the same aliases, quoting and full wire configuration. Its independent
`[10, None]` check now raises an explicit RuntimeError and remains active under
`python -O`, including a deliberately incorrect output control. Bandit GREEN
has **0 findings**.

The three Python diagnostics pointed to Arrow batch-port dictionaries passed
to `plan.execute(inputs=...)`. All four synchronous execution sites in the
affected test module now identify the exported `cf.BatchExecutionPlan.execute`
API explicitly and annotate the plan type. They invoke the same synchronous
method with the original instance and unchanged inputs. Normalizing those API
qualifications and local type annotations produces an AST identical to the old
test module: JSON serialization/SQL escaping, parameters, positive/negative
inputs and every assertion are preserved. All **16 test functions and their
parametrizations** are unchanged (38 collected cases).

A supplementary Semgrep 1.180.0 `p/python` scan runs 151 rules on exact copies
of both Python files, outside its default test-file ignore behavior: **0 findings,
2 files scanned**, both before and after. This local ruleset does **not** reproduce
the three hosted Codacy diagnostics, so it is not evidence that those hosted
findings are closed. The new-head hosted check remains the authority for that
gate; no bypass, method alias or catch-all wrapper is introduced.

### Directed verification

```sh
cargo test --locked -p calc-flow --test finite_asof_frontend rename::
cargo clippy --locked -p calc-flow --test finite_asof_frontend -- -D warnings
cargo fmt --all --check
PYTHONPATH=target/finite-asof-join/site-f1-chain JAX_PLATFORMS=cpu \
  $CF_PY -m pytest python/tests/test_finite_asof_join.py -q \
  -k 'same_finite_asof_plan_empty_inputs_and_failure_retry or asof_mixed_sharing_nested_export or null_identity_is_rejected_by_arrow_binding_before_execution'
PYTHONPATH=target/finite-asof-join/site-f1-chain $CF_PY examples/finite_asof_join.py
PYTHONPATH=target/finite-asof-join/site-f1-chain $CF_PY -O examples/finite_asof_join.py
$CF_PY -m ruff check python/tests/test_finite_asof_join.py examples/finite_asof_join.py
$CF_PY -m ruff format --check python/tests/test_finite_asof_join.py examples/finite_asof_join.py
bandit -q examples/finite_asof_join.py
```

Results: **10 Rust tests**, **5 Python cases** (33 deselected), both example
executions, Clippy, rustfmt, Ruff check/format and Bandit passed. Additional
author checks prove identical native SQL descriptions, an unchanged normalized
Python AST, retained Rust test names and parametrizations, and optimized-mode
rejection of incorrect example results. Tests were kept green during refactoring;
the observed hosted lint failures and local complexity/Bandit reproductions are
the RED evidence. No product behavior was implemented, no assertion or case was
weakened, and no full suite, performance or native rebuild was run.

### Reused native and updated provenance

`target/finite-asof-join/provenance-codacy.json` records the exact reused native
build from `45bf88ec` and the current verification sources. All **1643 Rust/Cargo
files** are checked against that build's manifest: only the two external Rust
test files differ. All **59 Python/package/stub sources** are unchanged and
match the imported `site-f1-chain` package exactly. The four changed verification
files receive separate hashes. The core rlib, native and wheel hashes match the
previous build; no product/native source input changed.

- Reused native SHA-256:
  `386732408239bd09e27962299975ab2ee617919d2c4b38c739f993504db5cccd`.
- Reused wheel SHA-256:
  `201a8b25b354582351b636a1ba1cea964c676a672155463d07d447d3d01a082c`.
- Current Rust/package source manifest SHA-256:
  `3ba5112cb816b0647aa87c046381d4e1b6a742679d6aa2d7d56a82821d91d48b`.

After commit, the recorded sources are checked against Git HEAD and the actual
delivery commit is recorded in the manifest. Previous product/kernel, identity,
recovery/terminal and standalone core example results are reused. Author review
finds only behavior-preserving test organization, clearer batch API calls and
the requested example guard correction; no fresh product review is requested.
The single post-push CI snapshot for the delivery HEAD is supplied with the final
comment as `codacy-ci-snapshot.json`; pending checks do not establish gate success.
DAL-518 remains `in_progress`; no delegation or merge occurs.
