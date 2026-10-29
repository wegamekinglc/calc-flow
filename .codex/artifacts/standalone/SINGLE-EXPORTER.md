DAL-475 / PR #515: focused import-boundary repair after `83d99b05`.
Implementation: `fc52433da013123beafbb5c5125ad0e8a92631ed`.

## Source-level cost analysis

The specialist's report attachment `01a18915-f038-7d95-b82a-b24132ba7027`
confirms a stable import regression on the preceding default wheel: 3.109 to
3.566 microseconds (+14.70%), with valid A/A controls. Round-trip and NumPy
acceptance remain uncertain. Its component evidence does not isolate the
exclusive contribution of live protection or reader/RecordBatch wrapping.
The preceding proxy contrasts for the compatibility precheck and method lookup
both crossed zero; they are not evidence of a quantified optimization benefit.

The actual import path is:

1. `Batch.from_pyarrow` probes `__arrow_c_stream__` with `hasattr`, discards the
   resolved attribute, and delegates to `from_arrow`.
2. `import_table` traverses the MRO, imports the current PyArrow module when
   applicable, reads/parses its current version and compares it to the fixed
   interpreter threshold. It then resolves the same stream attribute again.
3. The exporter creates a capsule. pyo3-arrow 0.18.0 checks its name/pointer and
   moves the stream into an array reader; the reader owns stream/schema
   lifetime, including release on drop. Its `PyArrayReader` has a Mutex/Option
   wrapper that is consumed immediately, but the lower-level capsule importer
   and reader are private dependency APIs.
4. Each stream array becomes a StructArray. Our bridge checks struct root and
   root nulls, preserves schema/field metadata, validates ordered input where
   applicable, and constructs a RecordBatch with checked columns/row count.
   `Batch::table` then applies its existing table consistency/accounting checks.

Only step 1 is provably redundant work within our safe API boundary. A
stateful descriptor also demonstrates the duplicate access semantically:
the old compatibility entry resolves it twice while the general entry does
so once. The new regression reproduced that exact failure on the prior wheel.

This repair resolves the compatibility stream exporter once after the same
dynamic guard, retains it through import, and shares `stream_records` with the
general entry. Missing stream still raises the compatibility TypeError, array
fallback remains unavailable there, and other descriptor/producer exceptions
propagate. Metadata conversion and core Batch construction are unchanged.

The guard, reader, root/schema/metadata/row-count validation, event-time checks
and buffer ownership code are unchanged. No module, package version or
successful check is cached. Interpreter threshold evidence and the approved
one-shot stream P1 evidence remain applicable to their unchanged code.

We did not replace the generic reader with pyo3-arrow's RecordBatchReader:
it retains the same Box/Mutex pattern and Arrow 59.2.0's reader discards the
root null mask through `StructArray::into_parts().1` before constructing the
RecordBatch, preventing our current explicit nullable-root rejection.
Direct access to the private capsule importer would require new unsafe FFI or
a dependency API change. Neither is justified by the unseparated component
timings or this repair's scope. No speculative reader rewrite was made.

## Validation and artifacts

The prior default wheel reproduced the stateful-descriptor failure at the
compatibility entry; the general entry control passed. On the new normally
installed default CPython wheel, **56 tests passed, 5 unrelated external-array
tests deselected**. These include the protocol file and affected table/metadata
Batch cases. No optional case was skipped. The new coverage checks single
attribute resolution/export, producer release, surviving shared buffers,
metadata and unchanged input. Compatibility-entry coverage also checks unsafe
subclasses, live module/version replacement, wrong capsules, descriptor errors
and rejection of array-only producers.

```bash
.venv/bin/maturin build --release --interpreter .venv/bin/python --out target/single-exporter-wheels
uv pip install --python .venv/bin/python --reinstall target/single-exporter-wheels/calc_flow_python-2026.10.27-cp313-cp313-linux_x86_64.whl
.venv/bin/python -m pytest -q python/tests/test_arrow_protocol.py python/tests/test_batch.py -k 'not external_batch and not repeated_external and not table_accessor'
uv pip install --python target/single-use-arrow-core/bin/python --reinstall target/single-exporter-wheels/calc_flow_python-2026.10.27-cp313-cp313-linux_x86_64.whl
target/single-use-arrow-core/bin/python scripts/smoke_wheel.py
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --release -p calc-flow-python --lib -- -D warnings
cargo fmt --all --check
lizard --CCN 8 crates/calc-flow-python/src/arrow.rs
.venv/bin/python scripts/verify_complexity_gates.py
.venv/bin/ruff check python/tests/test_arrow_protocol.py
.venv/bin/ruff format --check python/tests/test_arrow_protocol.py
git diff --check
```

All final checks above passed. Normal core installation resolved one package;
its package list contains only calc-flow-python. The smoke covers core
computation, configuration, owned streams and checkpoint recovery without
optional Python packages. Existing security, complexity and coverage policies
are unchanged. The default release build took 3m20s, recorded separately from
the measurement budget. It uses a local Linux tag; it is a validation artifact,
not a repaired manylinux publication artifact.

The [sealed build manifest](data/single-exporter-build.json) links to the prior
source manifest and overrides only `arrow.rs` and `batch.rs`. All 991 composed
source entries match the build tree. All 44 packaged Python/stub files match
source and installation bytes; the native loaded in both verification
environments matches the wheel. The native SHA256 is
`35bed3efd27116908ee0c372ff1943dcf3f20d1ceba92bfcef021da854aa3c19`;
wheel SHA256 is
`5f534ced7e5d2a8c4aa8e2bf286a44aab3dbc37d72e3aa63c33d6988b6e1df7d`.

The interpreter threshold and dynamic guard implementation did not change.
Reuse the prior same-abi3-wheel Python 3.9/modern threshold evidence and the P1
closure for their unchanged code. This batch does not rebuild or claim new
abi3 verification, and does not rerun the full 128-case stream/snapshot suite.
Prior review approval covers old code; the new compatibility-entry difference
still needs focused review.

## Remaining acceptance

No performance samples are added by this repair. Cumulative measurement remains
**796.31 / 900 seconds**, leaving **103.69 seconds** under coordinator control.
Removing one redundant attribute access does not establish a performance pass
or a quantified gain. The final native identity needs its own authorized safe
path confirmation and review of this increment. Full CI, platform and coverage
gates remain in force. The issue remains `in_progress`.
