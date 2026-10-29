DAL-475 follow-up to PR #515, after the original standalone implementation and
177-second performance acquisition.

## Dependency repair

Commit `4cefc4a7` updates only `libz-sys` 1.1.29 to 1.1.31 and its checksum in
Cargo.lock, addressing the RUSTSEC-2026-0418 failure reported by CI. Package
inventory comparison confirmed that every other lock entry and dependency edge
is unchanged. Cargo manifests, audit ignores, deny policy and Codacy settings
are unchanged.

The exact failed-policy commands were repeated with cargo-audit 0.22.2 and
cargo-deny 0.20.2:

```bash
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177 --ignore RUSTSEC-2026-0235
cargo deny --locked check
```

Audit passed with the existing allowed `chacha20` yanked-version warning. Deny
reported `advisories ok, bans ok, licenses ok, sources ok`. No waiver was added.

## Codacy diagnosis and repair

The original check `123536653553` was `ACTION_REQUIRED` because of 19 new code
findings, not an account authorization requirement: seven Rust cyclomatic
complexity findings and twelve Bandit assert findings in the wheel smoke script.

Commit `2aa586c1` decomposes the private Arrow validation and numeric dispatch
functions. It keeps primitive casts, chunk traversal, allocation ownership,
metadata and error behavior, with no interface or dtype expansion. The smoke
script uses explicit checks that execute in optimized Python too. An AST
comparison verified that all 15 original smoke predicates are preserved; the
existing failure paths still raise AssertionError.

Local Lizard 1.24.3 reproduced exactly the original seven Rust findings. After
the changes, the same CCN limit of 8 is met by both `arrow.rs` and its extracted
`arrow/buffers.rs`. Bandit 1.9.4 reports no findings in `scripts/smoke_wheel.py`.
No analysis exclusions, forgiveness markers or thresholds were added.

The original check was on `9ca97c6a`; its 19 exact annotations are preserved in
[`data/codacy-original-annotations.json`](data/codacy-original-annotations.json).
The seven complexity locations in `crates/calc-flow-python/src/arrow.rs` were:

- Line 26, `guard_input`: CCN 10; decomposed guard, now 8.
- Line 130, `validate_time`: CCN 11; separated checks, now 7.
- Line 162, `numpy_array`: CCN 19; split dtype families into small dispatchers
  in `arrow/buffers.rs`.
- Line 223, `register`: CCN 11; split stream/schema registration, now 4.
- Line 255, `table_matrix`: CCN 37; separated dispatch and owned allocation in
  `arrow/buffers.rs`, now 4.
- Line 386, `validate_matrix_casts`: CCN 10; separated per-column validation in
  `arrow/buffers.rs`, now 5.
- Line 417, `import_table`: CCN 15; separated import and validation, now 5.

The twelve security findings all match Bandit B101 in `scripts/smoke_wheel.py`,
at original lines **12, 19, 32, 33, 34, 40, 42, 50, 72, 73, 82, 99**.
Every reported assert was replaced by an explicit check; all original
predicates remain enforced under normal and optimized Python. These locations
refer to the original commit, not the refactored file's shifted line numbers.
Codacy subsequently reported success on both `cbbf4cc5` and `ee2af8a4`; the
latter also passed the remote Rust dependency policy check. No service-side
authorization action was necessary. Later CI fixes are recorded in
[`CI-REPAIR.md`](CI-REPAIR.md).

## Focused verification

- Release Clippy for `calc-flow-python`, Cargo formatting, smoke-script Ruff
  lint/format and the corresponding Lizard/Bandit checks passed.
- Installed rebuilt wheel: 133 protocol, Batch, metadata and symbolic-matrix
  tests passed; 40 table-matrix tests passed; 12 event-time/byte-budget/producer
  tests passed. These 185 cases cover the moved code. The earlier 555-case run
  remains historical evidence for the original implementation and untouched
  surfaces, not a claim that all 555 were repeated on this binary.
- Normal resolved core-wheel installation passed ordinary and `python -O`
  standalone smoke checks. An optimized-mode negative control confirmed that a
  false predicate still fails.
- NumPy/JAX matrix attachment and input isolation passed without PyArrow; real
  PyArrow 25 input/output/schema guards passed on the rebuilt wheel.
- Source distribution inspection verified 1,181 files and the new native module
  is present in the archive.

The directed pytest commands were:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest python/tests/test_arrow_protocol.py python/tests/test_batch.py python/tests/test_compute_metadata.py python/tests/test_symbolic_matrix.py -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest python/tests/test_array.py -k 'table_matmul or table_matrix' -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest python/tests/test_stream_results.py python/tests/test_arrow_stream_inputs.py -k 'time or byte_budget or producer' -q
```

The original performance raw data, source/native identities and verdicts remain
unchanged. This follow-up made no performance acquisition and does not assert
that old timings prove equality for the refactored native binary. Performance
analysis and any additional measurement remain with the assigned specialist.
Platform Codacy and CI results require the new commit's own checks; local zero
findings are not a claim that those remote checks have completed.
