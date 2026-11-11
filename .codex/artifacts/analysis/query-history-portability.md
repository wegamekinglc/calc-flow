# Query history and portability implementation evidence

DAL-511 / S0a-2, 2026-11-11. Base `2114e374441fa5099902f511bf23914c286d0c87`;
initial runtime implementation and native build commit
`93f06e2784aa8eeccaf6c74aff7cbcceddbe0e9f`. The original evidence commit
`b8d541d2a86658f28e8614e934840b4ccc7702f5` added this record, the provenance
JSON, and a stronger nullable input fixture. The original checks below describe
that source boundary. The follow-up sections below record incremental
verification; the provenance JSON identifies the latest native and retains
earlier builds under `previous_build`.

The controlling [API note](../api-notes/query-history-portability.md) is preserved
unchanged. The existing dedicated checkout and branch were reused; the branch
is now `feature/dal-511-query-history-portability`. No contract deviation.

## Delivered behavior and files

- `crates/calc-flow/src/query_analysis.rs`, `query_analysis/sql.rs`, and
  `pipeline/analysis.rs`: owned facts, revisions, stable paths/codes, conservative
  AST proofs, consumption-edge propagation, and asynchronous checked compilers.
  Crate exports, the typed error, and shared validator visibility connect these
  to the existing compilers.
- `crates/calc-flow-python/src/config.rs`, `error.rs`, and
  `python/calc_flow/_native.pyi`: the data-only analysis seam and CompileError
  report projection.
- `python/calc_flow/symbolic/query_analysis.py`, `analyzer.py`, `program.py`,
  `errors.py`, package exports, `compute.py`, and `stream.py`: immutable public
  models, logical addresses, original issue projection, and explicit gates under
  the registration lock before preparation-cache access, input wrapping,
  source binding, and job start. Async planning uses the compilation worker.
- Focused Rust/PyO3/Python regressions, `docs/query-history-portability.md`,
  documentation links, and the existing Python/Rust example 29.

The implementation distinguishes lifecycle admission from comparison proof.
No request means `portability=None`. Both unknown and unsupported comparisons
reject strict execution. No execution fingerprint, project format, preparation
key, SQL default, or checkpoint layout/version is changed. Rust's typed
DataFrame/SQL frontend remains the separate S0b work.

## RED evidence

Each production behavior started with an observed focused failure. Accepted
failures below exclude fixture/import setup errors.

- Initial Python report test: missing `AnalysisOptions`; Rust integration
  target: missing public types and checked methods.
- `cargo test -p calc-flow --test query_history_portability`: row-local case
  green, aggregate scope and SQL shape cases red (`Unknown` history/admission).
  A later mode case exposed missing stream-only classification; the snapshot
  edge case observed `Supported` where the original compiler rejects the edge.
- `pytest ... -k strict_execution`: three unexpected `portability` keyword
  failures. The sequence-source test separately observed `__getitem__(0)`
  before the gate.
- `pytest ... -k existing_schema_errors`: two incorrect `supported` reports
  despite original field/SQL errors.
- Composition cases: an unknown upstream was incorrectly labeled append;
  DISTINCT consuming an aggregate snapshot lost the complete-snapshot update.
- Selection cases: unproved per-batch LIMIT was accepted, and LIMIT 0 received
  an invented full-history counterexample. Both now remain unknown.
- Snapshot facts incorrectly changed latest-relation history to current-batch
  in rejected batch mode. Portable time schemas emitted Debug spellings instead
  of `time32[s]` / `time64[us]`.
- PyO3 error test: portability failure fell through to ExecutionError.
  Request sequence ownership and public code vocabulary tests also began red.

## GREEN verification

The original checks used this checkout's sources at the initial build boundary.
The installed development
interpreter supplies CPython 3.13.9, PyArrow 24.0.0, and JAX 0.10.2. Package and
native load paths were asserted at verification; that original metadata is
retained under `previous_build` in [provenance](query-history-portability.json).

Commands ran from the repository root with:

```bash
export CARGO_TARGET_DIR=target/query-history-portability/cargo
export CARGO_BUILD_JOBS=4
export PYO3_PYTHON=/home/wegamekinglc/anaconda3/bin/python
DEV_PY=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python
FLAGS=(--release
  --config 'profile.release.package.calc-flow.opt-level=0'
  --config 'profile.release.package.calc-flow.codegen-units=256'
  --config 'profile.release.package.calc-flow-python.opt-level=0'
  --config 'profile.release.package.calc-flow-python.codegen-units=256')
```

The package-specific optimization overrides bound local build time; dependency
optimization stays at normal release settings. No Cargo manifest/profile was
changed. These are behavior checks, with no performance claim.

```bash
cargo clippy -p calc-flow -p calc-flow-python --lib --tests "${FLAGS[@]}" -- -D warnings
cargo test -p calc-flow --test query_history_portability "${FLAGS[@]}"
PYTHONPATH="$PWD/target/query-history-portability/python" \
  LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib \
  cargo test -p calc-flow-python --lib "${FLAGS[@]}" \
  query_portability_errors_keep_the_immutable_report
```

Clippy passed; core **7 passed**; focused PyO3 test **1 passed**, 133 deselected.
The PyO3 test requires absolute PYTHONPATH because Cargo runs it in the crate
directory. The original typed compilation error for a missing UDF is retained.

```bash
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/maturin \
  build "${FLAGS[@]}" --interpreter "$PYO3_PYTHON" \
  --out target/query-history-portability/wheels
# Extract only this wheel's native module to the target overlay.
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" -m pytest python/tests/test_query_history_portability.py \
  python/tests/test_symbolic_analysis.py python/tests/test_symbolic_compile_cache.py \
  python/tests/test_program_engine.py -q \
  --basetemp=target/query-history-portability/pytest-final
```

Native build passed; Python **122 passed in 3.83s**: 27 focused cases and 95
affected analysis/cache/engine regressions. The overlay's Python files resolve
to this checkout and its extension is extracted from this build, not an
installed package from another checkout. No extension remains in source.

```bash
RUSTDOCFLAGS='-D warnings' cargo doc -p calc-flow --no-deps "${FLAGS[@]}"
RUSTDOCFLAGS='-D warnings' cargo test -p calc-flow --doc "${FLAGS[@]}" \
  pipeline::analysis::PipelineBuilder::compile_batch_checked
cargo run -p calc-flow --example 29_sql_stream_switch "${FLAGS[@]}" -- --engine both
PYTHONPATH="$PWD/target/query-history-portability/python" \
  "$DEV_PY" examples/29_sql_stream_switch.py --engine both
```

Crate rustdoc with warnings denied passed; the selected new doctest passed.
Both existing examples passed: batch 6, stream 1/6. Changed-file Ruff lint and
format, `cargo fmt --all --check`, Python 3.9 syntax parsing of changed modules
and stubs, documentation whitespace/links, and `git diff --check` passed.

## Acceptance evidence and limits

- C01: deterministic row-local reports, SQL filtering/ordering, nullable
  predicates, strict collection/streaming, schemas, immutable facts and escaped
  shared-node addresses. Unsupported requests remain distinct from unknowns.
- C02: both batch and stream preparation caches are warmed before rejected
  accepted-history/latest requests. The diagnostic locates the actual snapshot
  to append edge, including an intervening projection. Ordinary stream SUM to
  SUM still yields 1/7.
- C03: ordinary top-1 still yields `[1]` / `[2]`; DISTINCT and ROW_NUMBER remain
  per batch. Full-history counterexamples reject; unproved selections, CTEs,
  aggregate/HAVING shapes and ordering stay unknown.
- C04: one stream job yields 1/6; the same cached batch plan independently
  executes to 1/5. Simple aggregate scope/update/finality are explicit. Strict
  final relation stays unknown when zero input events leave no last snapshot.
- C13: original type/schema and lifecycle errors, missing UDF registration,
  immutable engine rejection and snapshot sink restrictions remain effective.
  Rejection probes assert **0 input wraps, 0 source iterations, 0 bindings,
  0 job starts**; async analysis runs off the event-loop thread.

The fixed baseline program fingerprint is
`20a49fa51a1c423a8bd4cadbb6b95bc142292f21f3294e4970efe9b45eddb7de`;
native batch and stream fingerprints remain respectively
`0044b2322eff73664b78a51f577af6293c86bed2e4d2cd6ba93b1fc214f733f6` and
`3bb9b534577cd7034059bcb870d041aff073e87bdc4888e49731ea06e78a39d5`.
Project bytes before/after analysis and warm cache ownership are also checked.
These were recorded before replacing the baseline native extension and are
asserted against the final matching build.

Existing DAL-510 checkpoint inventory/capture/recovery evidence is reused.
SQL checkpoint writers/readers, layouts 3/4/5, state version 2, recovery
configuration and fingerprint encoding have no diff. No new checkpoint-byte
comparison is claimed. Full regression, platform/feature/container matrices,
optimized release artifacts and the combined 90% Rust coverage gate remain CI
responsibilities. No benchmark was run. No independent review or merge is
claimed; DAL-511 remains in_progress for the coordinator's reviewer handoff.

## Dictionary schema codec review fix

Incremental base: reviewed head
`b8d541d2a86658f28e8614e934840b4ccc7702f5`; runtime fix and native build source:
`c36c0ac9461750f16f86446eb84def8101f9728d`. The review found that a valid
`Dictionary(Int32, Utf8)` schema compiled normally but failed in public Rust
analysis because the analysis encoder produced `dictionary(int32, utf8)`.
Serialization also collapsed both dictionary ordered values to that spelling.

`schema_fields` now consults the existing
`config::canonical_arrow_field_type` before the existing fallback. This
preserves the field-level `ordered=false/true` bit and canonical primitive
spellings, without changing the handling of types outside that codec. The
analysis contract and proof rules are unchanged. The production diff is one
import and the shared codec call; no SQL algorithm, execution identity,
lifecycle, cache or recovery implementation changed.

Focused regression additions are in the existing
`crates/calc-flow/tests/query_history_portability.rs` and
`python/tests/test_query_history_portability.py`. Each ordered value exercises
public Rust `analyze`, both checked compilers and JSON serialization; checked
fingerprints equal the original compilers' fingerprints. The binding cases
exercise the native data-only seam, canonical JSON and immutable Python report
projection. They do not claim new Python dictionary expression support.

Using the same setup and package-specific profile overrides as above:

```bash
cargo test -p calc-flow --test query_history_portability "${FLAGS[@]}" dictionary_schema
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" -m pytest python/tests/test_query_history_portability.py -q \
  -k native_dictionary_schema_report \
  --basetemp=target/query-history-portability/pytest-codec-red
```

RED before the production edit: Rust **2 failed / 7 filtered** with
`port.schema.data_type: unsupported Arrow type "dictionary(int32, utf8)"`;
binding **2 failed / 27 deselected**, observing the same noncanonical spelling
for both ordered values. These are the review's functional failures, not test
setup errors. Logs: `target/query-history-portability/codec-{rust,python}-red.log`.

GREEN after the codec call: the same Rust command passed **2 / 7 filtered**
in 0.01s, including field equality, serialized ordered identity, both checked
compilers and original fingerprint comparisons. Log:
`target/query-history-portability/codec-rust-green.log`.

```bash
cargo clippy -p calc-flow --lib --test query_history_portability "${FLAGS[@]}" -- -D warnings
cargo fmt --all --check
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/ruff \
  check python/tests/test_query_history_portability.py
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/ruff \
  format --check python/tests/test_query_history_portability.py
```

All passed. Clippy checks the changed core module and focused integration
target; there is no binding implementation change. The earlier PyO3 unit test
is reused rather than rerun. Clippy log:
`target/query-history-portability/codec-clippy.log`.

```bash
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/maturin \
  build "${FLAGS[@]}" --interpreter "$PYO3_PYTHON" \
  --out target/query-history-portability/wheels/codec
python target/query-history-portability/extract-codec-native.py
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" target/query-history-portability/record-codec-provenance.py
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" -m pytest python/tests/test_query_history_portability.py -q \
  -k 'native_dictionary_schema_report or portable_time_type_spellings' \
  --basetemp=target/query-history-portability/pytest-codec-green
```

Build passed in 2m47s. Only the new wheel's `_native` extension was extracted
into the existing target overlay; the original wheel is retained. Python
**4 passed / 25 deselected in 0.31s**: two dictionary report/projection cases
and two directly affected primitive time spelling cases. Logs:
`target/query-history-portability/codec-native-build.log` and
`target/query-history-portability/codec-python-green.log`.

Provenance checks assert that `calc_flow.__file__` resolves to this checkout's
`python/calc_flow/__init__.py`, the loaded native is inside this target overlay,
and its bytes match the new wheel member. No native extension exists in source.
The source digests use the existing sorted tracked path/NUL/content/NUL format;
the script also checks no runtime/package diff from build commit `c36c0ac9`.
The final evidence commit changes only this record and provenance JSON, leaving
that source match intact. Current hashes:

- Native source: `73cc2380e827f86ece1708ddef4c7115283d5c4c68db769db7d2828c87f269db`.
- Native extension: `778f595c106737b1c9f1f7859587546e44332fb2691ebbf7e46cd9b046b6efb4`.
- Wheel: `661d74cc4d2f25bfc51d5195501aaaad1cc810d0fdeacca38c166df1839e2a44`.
- Python package remains `fa14fa66a64fbb063354bbae5b4cd87638cd8523b90c01bf324da6b75cb7e463`.

That codec build's hashes and paths are retained under `previous_build` in the
latest JSON, together with the original verification metadata (earlier load
paths are historical). The same existing interpreter, target directory and local
opt-level=0 package overrides were reused. No environment was created, no
optimized-release or performance result is claimed, and no new public API or
proof-rule revision was introduced by this field codec repair.

The original review-verified lifecycle, cache, zero-execution, recovery and
independent core example evidence is reused at its recorded source boundary.
Those suites and examples were not rerun for this codec-only fix. Full matrices
and coverage remain CI responsibilities; no performance claim is added.

## CI isolation and complexity repair

Incremental base: codec-approved head
`091a86a454e02858edf0300977a8052f79b8993e`. Repair/native source commit:
`e185be269d904e10c3ec34ea52a0c4e22a926055`.

The coordinator's completed snapshot was 33 checks: 25 passed, 8 failed,
none running (Linux run `42892250346`, Windows run `42892250335`). The repair
uses those failures, not a blind rerun or an assumption that empty
`gh run view --log-failed` output means no logs.

### Root causes and corrections

- Linux parity/coverage failed before coverage measurement in the PyO3 report
  test because the standalone Rust test process could not import `calc_flow`.
  The raw Windows Rust job `130096253933` confirms the same error at `error.rs:315`
  (144 passed, one failed). The test now registers the linked crate's native
  module in memory, inserts this checkout's Python source path derived from
  `CARGO_MANIFEST_DIR`, and asserts the imported projection's exact source path.
  It checks the real report's status and node address, a frozen-assignment
  `AttributeError`, and that revision remains 1. No installed native extension
  or other checkout is required for that unit test.
- The real repository complexity ratchet grew five counters: PLR0913 in
  `stream.py` and `symbolic/program.py`, C901 twice and PLR0912 once in
  `symbolic/query_analysis.py`. The existing frozen `_StreamRequest` now carries
  internal stream arguments; `execute` retains explicit `portability=` and
  accepts the existing `config`/`watermarks` keywords through a typed options
  group, following the repository's existing `Unpack` convention. Unknown
  keywords still raise `TypeError`. Logical addressing, issue projection,
  schema projection and graph assembly are separate small functions.
- Codacy on the reviewed head was genuinely `action_required`: 20 new issues,
  with a configured maximum of zero. Its actual annotations were 17 complexity
  findings and three example assert findings (check `130096564600`). They were
  not waived. The new analysis methods are split into schema gathering,
  lifecycle/edge predicates, SQL-shape/projection/order helpers and rendering.
  Predicates, diagnostics, ordering and proof rules remain unchanged; the codec
  fallback no longer repeats primitive cases already handled by the canonical
  codec. The example uses an explicit exception for incorrect facts, retained
  under optimized Python. The cache regression separates request histories
  into parameter cases and retains all cache ownership/fingerprint assertions.
- The frozen export test omitted the 21 approved diagnostic names. Its explicit
  expected list now includes exactly those names; the export implementation is
  unchanged. Invalid runtime values were rejected by the new generic selector
  before the original compiler validator. Batch and stream compilation now
  reuse the original `_require_runtime` and its `explicit calc_flow Runtime`
  error. The invalid-runtime regression covers both entry points without
  changing its original expectation; omitted runtime behavior is retained.

No complexity baseline, Codacy exclusion, check configuration, threshold or
suppression marker changed. No new environment or PR was created. Existing
codec approval and unaffected execution/recovery evidence remain valid.

### RED and scoped GREEN

The setup, `FLAGS`, existing interpreter and target directory are the same as
above. Local reproductions before production edits:

```bash
env -u PYTHONPATH LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib \
  cargo test -p calc-flow-python --lib "${FLAGS[@]}" \
  query_portability_errors_keep_the_immutable_report
"$DEV_PY" scripts/verify_complexity_gates.py
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" -m pytest python/tests/test_symbolic_expr.py \
  python/tests/test_symbolic_lowering.py -q \
  -k 'test_public_surface_is_the_frozen_export_list or test_compile_batch_requires_a_runtime' \
  --basetemp=target/query-history-portability/pytest-ci-red
```

RED: the isolated PyO3 case failed with `ModuleNotFoundError` (the local base
interpreter found an older installed package without the new report module;
CI had no package). The complexity command reproduced all five counter growths;
the two Python cases failed exactly as CI reported. Extending the runtime case
to stream compilation reproduced the same wrong error there before correction.
Logs: `ci-import-red.log`, `ci-python-red.log`, and the reported gate output under
`target/query-history-portability/`.

```bash
cargo test -p calc-flow --test query_history_portability "${FLAGS[@]}"
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib \
  cargo test -p calc-flow-python --lib "${FLAGS[@]}" error::tests -- --test-threads=1
cargo clippy -p calc-flow --lib --test query_history_portability "${FLAGS[@]}" -- -D warnings
cargo clippy -p calc-flow-python --lib --tests "${FLAGS[@]}" -- -D warnings
"$DEV_PY" scripts/verify_complexity_gates.py
```

GREEN at the repair source commit: core **9 passed**, including both dictionary
identities and checked compilers. Isolated PyO3 error mapping group **5 passed /
129 filtered**, with user site and implicit current-directory imports disabled
and no PYTHONPATH; the report test checks this checkout's model and the linked
native crate. Both clippy commands, cargo fmt, changed-file Ruff lint/format,
example S101 check, Python 3.9 syntax and `git diff --check` passed. The actual
repository complexity gate reports `complexity ratchet holds`.

The Codacy annotation endpoint is not exposed by the GitHub plugin, so its 20
raw annotations were read through `gh api .../check-runs/130096564600/annotations`.
For a local complexity measurement, Lizard 1.26.0 was installed only under
`target/query-history-portability/tools` with `pip --target`; the interpreter
environment was not changed. Its scan covers the three new Rust analysis
files, the Python query adapter and directly affected rendering/cache-test
functions: **95 functions, maximum CCN 8, no findings above 8**. The real Codacy
annotations were reproduced before repair (including their 17 named functions);
unchanged pre-existing analyzer methods are outside this incremental scan.
This local measurement is evidence of the simplification, not a substitute for
the provider's result on the new head. JSON measurements are in
`ci-lizard-{before,after}.json`; original provider findings are in
`ci-codacy-before.jsonl` under the same target directory. An additional scan of
all 12 changed source/test files finds no new or grown methods above CCN 8
relative to the incremental base; unchanged pre-existing methods are retained.

```bash
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/maturin \
  build "${FLAGS[@]}" --interpreter "$PYO3_PYTHON" \
  --out target/query-history-portability/wheels/ci
# Extract only this wheel's native member to the existing target overlay.
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" target/query-history-portability/record-ci-provenance.py
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" -m pytest python/tests/test_query_history_portability.py \
  python/tests/test_program_engine.py python/tests/test_symbolic_analysis.py \
  python/tests/test_symbolic_compile_cache.py python/tests/test_symbolic_expr.py \
  python/tests/test_symbolic_lowering.py -q \
  --basetemp=target/query-history-portability/pytest-ci-final
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" -O examples/29_sql_stream_switch.py --engine both
```

The wheel built in 1m20s. Final matching-native Python verification: **160 passed
in 1.64s**, including stable addresses, old fingerprints/results, warmed cache
gates, zero execution, frozen exports and both invalid-runtime entry points.
The example passed under `-O`: SQL **6**, streaming **1/6**. Logs are
`ci-native-build.log`, `ci-import-final.log`, `ci-rust-green.log`,
`ci-python-final.log` and `ci-example-optimized.log` in the target directory.

The extension is extracted from this repair commit's wheel; its loaded bytes
equal the wheel member, Python modules resolve to this checkout, and source
contains no generated native extension. Current provenance hashes:

- Native source: `5cacf16ca45295918d818897e5f8bdfb9dcff9d39f0ea9b1d2d3ed050643d92a`.
- Native extension: `2f5710528ecf0706aa41649f56e390f602adaad4c4d6dea8e88ebb0519d995d5`.
- Wheel: `8fa70075e2cd0807a90f1390fbc88d8e070320c182795d4480ec7908ce9a2cef`.
- Python package: `226f0844aa44b6ab5403833f33ca50c83846032acd7372b4612aafc7e0dd7008`.

The provenance JSON preserves the preceding codec and initial builds under
`previous_build`. Runtime/package source diff from `e185be26` is empty; the
following evidence commit changes only this record and provenance JSON. The
existing interpreter, Cargo target and local opt-level=0 package overrides
are reused. The initial codec approval remains relevant, and these direct
regressions cover this repair's schema/proof projection changes.

The coverage failure was a test prerequisite failure, not evidence that the
90% floor was missed. No coverage gate was lowered or locally claimed passed.
Cross-platform/full coverage jobs remain CI responsibilities. No benchmark or
merge is claimed; DAL-511 remains in_progress pending incremental review and
required gates.
