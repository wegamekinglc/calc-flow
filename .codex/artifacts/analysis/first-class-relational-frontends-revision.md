# DAL-512 focused review fixes

Reviewed baseline: `e12fc8ba6f8d45c83cd462628fde46481fc56ced`, PR #591.
The original isolated checkout/branch is reused. The review report is preserved
byte for byte in its initial untracked task-local file and existing issue
attachment (SHA256
`f2b6751470416825b3bd2fcb1a8347ad11d02a50966e14f32f08ce2fae748776`).
The frozen note and original analysis/evidence files remain unchanged.

## Fixes and focused RED/GREEN

- **F1:** `program/lower.rs` retains exact input/output `SchemaRef`s separately
  from portable `PortSpec`s. In-memory operators use those exact schemas rather
  than reconstructing ports from the data-only export representation. Native
  schema/field metadata therefore reaches input validation. No caller Batch or
  schema is rewritten, and project-v3 still serializes the same portable fields.
  `relational_metadata` observed the expected graph-input schema-mismatch RED,
  then GREEN for plain/schema-metadata/field-metadata batches. The public
  consumer also exercises metadata through an owned stream with exact source
  schema, source open/close checks and caller-schema immutability.
- **F2:** the private schema-only expression planner uses `*` for a known empty
  identity projection. It still calls the native planner, preserving the
  executable operator validator. Two batch/stream regression cases observed
  empty-select RED, then GREEN for Program/TableExpr, mixed empty/nonempty
  outputs, and unresolved NULL rejection.
- **F3:** the public schema boundary checks analyzed output fields against the
  existing portable domain before generated SQL placeholders are constructed.
  Unsupported fields become `CompileError` at an escaped logical
  `outputs...schema[index].data_type` path with `unsupported_type`. Six
  batch/stream/simple/dotted/quoted-newline-name cases observed generated-ID
  ValueError RED, then GREEN. Decimal remains unsupported.
- **F4:** the exact frozen symbolic export expectation adds only the approved
  `CompileMode`, at its actual list position. The existing equality assertion
  observed RED and then GREEN; it was not replaced by a subset assertion.

Core/public API signatures, legacy fingerprints, runtime state ownership,
project format, dependencies and gates are unchanged. The guide now distinguishes
cross-mode/stream ownership from the existing same-mode batch cache. Its precise
same-mode `for_mode("batch")` cache identity has a preservation assertion.

## Actual Codacy gate

The 28 annotations from run `130370405171` were read with:

```bash
gh api repos/wegamekinglc/calc-flow/check-runs/130370405171/annotations
```

The existing local Lizard 1.23.0 undercounts some Rust match branches. A
task-local Lizard 1.26.0 reproduces all **25** annotated functions' original
complexities exactly, including `dtype=21`, `render=11`, and `binary_type=9`.
It was installed under `target/relational/lizard126` without changing repository
dependencies or configuration:

```bash
python -m pip install --no-deps --target target/relational/lizard126 'lizard==1.26.0'
```

Type checks, rendering, error projection, graph construction, test scenarios and
example stages were split by responsibility. All 25 annotated functions now have
CCN <=8. A scan of all changed/new functions finds no new value above 8; the
unchanged pre-existing `_validate_engine_node=9` is byte-identical to the reviewed
head. The JSON records each before/after value. No suppression or baseline update
was introduced.

The Python example replaces its three asserts with explicit runtime checks.
Its AST contains zero `Assert` nodes; validations remain enabled with `python -O`.

## Verification and reuse boundary

Final build/test commands, results, source trees, native/wheel hashes and log
SHA256 values are recorded in the companion revision JSON. Local Rust commands
use `CARGO_TARGET_DIR=target/relational/cargo`, `CARGO_BUILD_JOBS=4`, `--release`,
and package-specific `opt-level=0` / `codegen-units=256` overrides as before.
These are build-cost settings, not performance evidence.

The initial focused Python schema/export RED/GREEN used current adapters with
pre-existing native artifacts for iteration. A wider interim adapter run used
the much older baseline native, causing two missing-getter failures; the
environment was corrected to the reviewer-verified native and both passed.
Those environment failures are not counted as product REDs. Final Python results
use the newly rebuilt wheel and supersede the interim adapter runs.

Original low-level history, async entry capture/cancellation/owned cleanup,
PyO3 getter and release/architecture checks remain evidence for their unchanged
code paths and dependency/features. This revision reruns the directly modified
typed Rust targets, public consumer, examples and Python schema/export/mode/cache
checks. The new independent consumer build also compiles the current core without
Python/PyO3. Original artifacts and logs are retained; old hashes are not presented
as hashes of the new product. The metadata stream fixture declares the actual
input Batch schema, preserving exact-schema preflight rather than relaxing it.

Full workspace, connector services, coverage, cross-platform and performance
matrices are not local prerequisites for this focused repair. Required CI and an
independent reviewer recheck remain separate merge gates. DAL-512 stays
`in_progress`; this handoff does not merge the PR.

## Final results and provenance

Product fix commit: `1fda9d3b7087c59f740ba67914c4201a16adfe65`.
Its incremental product diff contains 14 source/test/doc paths, 854 insertions
and 398 deletions. Most changed lines split the 25 annotated functions while
retaining their original assertions and expected values. The evidence follow-up
does not change product trees.

- Four directly affected Rust targets: **19 passed** (4 + 1 + 13 + 1).
- Updated independent public consumer: metadata owned stream, all four relation
  shapes, nullable/empty/duplicate data, named fan-out, export/reload and state
  boundaries passed. Rust example net remains 18/27.
- Selected core/test/example Clippy with `-D warnings`: passed. Rustfmt, Ruff
  lint/format, Python 3.9 syntax and unchanged complexity ratchet: passed.
- Rebuilt wheel: **84 passed** across the relational frontend module, exact
  frozen export case, engine compatibility, compile cache, and legacy native
  identity/state regression. Both normal and optimized Python examples pass
  with `{'order_id': [1, 3], 'net': [18.0, 27.0]}`.

The wheel was built with CPython 3.13.9, PyArrow 24.0.0 and Rust/Cargo 1.94.0
using the same core/PyO3 profile overrides as the original build. All 55 packaged
Python files match this checkout byte for byte. Imports resolve to the new
`target/relational/revision-wheel-python` package, not the old native. Build
finished in 4m08s; the local Linux tag is not a release-platform certification.

Wheel SHA256:
`b5dea1ab10578c0e6d2d89f280a59efb03e4c4101a3805ba841bb90c237b7796`.
Native SHA256:
`2a436ee1eee9d78deeaa32f8e462d6f02f37d60eefca3e9f5311f8dc4c0c2ba8`.

Final Python command:

```bash
PYTHONPATH=target/relational/revision-wheel-python JAX_PLATFORMS=cpu \
  python -m pytest python/tests/test_relational_frontend.py \
  python/tests/test_symbolic_expr.py::test_public_surface_is_the_frozen_export_list \
  python/tests/test_program_engine.py python/tests/test_symbolic_compile_cache.py \
  python/tests/test_query_history_portability.py::test_legacy_results_and_native_identities_survive_analysis \
  -q --junitxml=target/relational/revision-python-final.xml
```

No full regression, coverage, connector service or performance matrix was run.
The independent recheck should focus on F1–F4 and directly affected paths.

## CI failure diagnosed and repaired

The single nonblocking snapshot on `77a4c4f8031de1702720e430a89702dbccf4fd29`
showed Documentation checks failure (job `130389253020`). Reading that completed
job's log identified the original review report's extra EOF blank line. The
report had been added unchanged with the evidence commit; a full-base whitespace
check therefore rejected it. The earlier worktree-only whitespace check did not
cover this newly tracked artifact.

To honor the instruction to preserve the report, its exact original bytes are
retained in the task-local untracked file and original Multica review attachment
`01a1c967-d08e-77a2-b329-6a1dcc25a56a`. Only its newly introduced Git tracking is
removed. No report prose, gate or product source is changed.

The actual CI RED was reproduced with
`git diff --check 775ea9738265597de492b848f8e0763962c7041b HEAD`.
The corrected candidate passes the corresponding full-base staged diff and
`python scripts/check_spec_references.py`, both steps used by this CI job.
Product source trees still match the native/wheel build commit, so those valid
results are retained. The fix is pushed to the same PR; new-head CI is left to
run without another status poll. The attached snapshot is explicitly the
pre-repair snapshot, not a claim that current CI is green.
