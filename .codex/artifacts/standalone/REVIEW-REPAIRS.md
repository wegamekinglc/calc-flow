DAL-475 / PR #515: focused repairs after review of `35f37e5b`.

## Single-use materialized streams

Implementation `de405c7a` fixes the P1 reported in the reviewer attachment
`review-repro.py` (attachment `01a188a6-a7e7-7596-aaff-7b8796c423f1`).
Schema validation already imported a finite Arrow producer, but ordinary
bindings and snapshot iteration subsequently imported the original object
again. RecordBatchReader input was exhausted before execution.

Both paths now retain the first Batch: ordinary binding normalizes that Batch,
and snapshot `_whole_table` yields it. Schema, row/byte budget, event-time and
metadata checks remain in their existing boundaries. Public interfaces and
project/checkpoint formats are unchanged.

The unchanged reviewer script reproduced four failures before repair and passes
all eight paths afterward, including on the final native build:

- TableExpr.stream and Program.stream return `value=[2,3]` for `a=[1,2]`.
- Their snapshot SUM paths return `total=3`.
- All four Batch controls agree; every path ends with zero native tasks.

Twelve permanent regression combinations additionally exercise a producer that
counts exports and rejects a second import. They check both entry points,
ordinary/snapshot modes, reader/producer/Batch inputs, caller schema/field and
Batch metadata, and zero native/Python tasks after cleanup.

## Arrow entry overhead

Implementation `99c0acc7` follows the performance specialist's confirmed
regression on `35f37e5b`:

- Traverse the type's MRO tuple through PyO3 and borrow module/version strings,
  replacing Python iterator creation and owned String extraction. Constant
  Python names use `intern!`.
- Return the currently validated PyArrow module to the exporter and reuse it,
  avoiding a second import lookup and an intermediate argument vector.
- Resolve the C Stream exporter once and consume its array reader directly
  into validated RecordBatches. This removes the intermediate chunked-array
  collection. Capsule validation still uses the existing pyo3-arrow importer;
  struct root/null checks, schema/metadata reconstruction, zero-column row
  counts and event-time validation remain. Array-only producers retain their
  existing extraction path, and producer failures propagate.

Every adapter boundary reads the live module/version; successful validation is
not cached. New tests reject unsafe PyArrow subclasses before their exporter is
called, observe version/module changes after successful use, reject a schema
capsule used as a stream, and preserve attribute-failure propagation. No unsafe
code, dependency or policy exception was added.

## Final-build verification

The final wheel was built with:

```bash
.venv/bin/maturin build --release --interpreter .venv/bin/python --out target/arrow-performance-repair-wheels
uv pip install --python .venv/bin/python --reinstall target/arrow-performance-repair-wheels/calc_flow_python-2026.10.27-cp313-cp313-linux_x86_64.whl
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q python/tests/test_arrow_protocol.py python/tests/test_batch.py python/tests/test_arrow_stream_inputs.py python/tests/test_snapshot_sql.py
.venv/bin/python target/review-repro/review-repro.py
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --release -p calc-flow-python --lib -- -D warnings
```

- **128 tests passed** on the installed final wheel. These cover the changed
  protocol, buffer ownership, metadata, invalid roots/capsules, budget/time
  checks, materialized-stream cleanup and snapshot paths. The preceding 75-case
  P1 run used the intermediate build; these 128 cases rerun its affected paths
  against the new native implementation.
- The reviewer script passed all eight cases with no tasks left.
- Real PyArrow 25.0.0 was rejected for Table, RecordBatch, schema import and
  PyArrow output on the new native build.
- Normal installation into a fresh CPython 3.13.9 environment resolved exactly
  one package. Ordinary and optimized-mode `scripts/smoke_wheel.py` passed,
  including standalone computation, configuration and checkpoint recovery.
- Release Clippy for the binding library, Cargo formatting, scoped Ruff
  lint/format, the unchanged complexity ratchet and `git diff --check` passed.
  Lizard's existing CCN-8 check passed for the Arrow modules; changed functions
  have CCN at most 7, including `Batch.to_pyarrow`.

The [build manifest](data/arrow-entrypoints-build.json) pins implementation
`99c0acc70d7fb70908b3eaf0e79a37ff9eb898b6`, source overrides relative to the
previous sealed manifest, wheel/native hashes and dependency identities.
All 44 Python/stub files in the wheel and installed package match current
source bytes. The native SHA256 is
`2f7d9fdabb30b7c32fba5f6eb51d464345185fd7ce39b66bd7f31df56e6051e2`;
wheel SHA256 is
`0e5568fbf643a8b418bb0e30b483d2aeddf3913d437c34b9b5d700080fde0b22`.
This is a local CPython 3.13 release wheel; ABI3/platform builds and full
coverage remain with CI. Cargo.lock and package requirements are unchanged.

## Performance acceptance

The specialist's report attachment `01a188ae-113b-79cc-a763-c50e1347b056`
confirmed a **17.29% Arrow round-trip regression** on `35f37e5b`, with both
paired interval lower bounds above +5%. NumPy remained statistically
inconclusive. The latest reported measurement budget is **481.99 / 600
seconds**, leaving approximately 118 seconds.

This repair adds no timing samples. The final native hash differs from that
measured build, and materialized stream handling also changed. Prior timing
results are evidence of the problem, not proof that the new implementation
meets the performance requirement. The coordinator will arrange fixed-plan
confirmation and review of these repairs. Native allocation equivalence,
GPU behavior and unmeasured paths remain unproven.
