DAL-475 / PR #515: interpreter threshold initialization after `48e92fd8`.
Implementation: `9adf1315551ce4f22f2b28054d726b69f54f1a0b`.

Only `crates/calc-flow-python/src/arrow.rs` changes product code. A
`PyOnceLock<u32>` lazily derives the expected PyArrow major from the running
interpreter once: 21 on Python 3.9, 24 on modern Python. It stores no Python
object, package version or successful validation result. Module import, current
version parsing, comparison, MRO routing and FFI rejection still run on every
adapter call. PyO3's initialization primitive handles interpreter attachment
while waiting; the initializer itself reads only the fixed CPython version.

The abi3 build exposed two `to_str()` calls introduced in the preceding Arrow
optimization that are unavailable under the Python 3.9 limited API. Both now
use `to_cow()`: modern builds can borrow, while the minimum ABI uses PyO3's
compatible conversion. This preserves the version and module checks and fixes
the actual abi3 compile failure. No dependencies, public interfaces, checkpoint
formats or checking policies change.

## Directed results

The existing `python/tests/test_arrow_protocol.py` is reused unchanged. It
covers live module/version replacement after success, unsafe subclasses before
FFI, invalid capsules/roots, metadata, buffer ownership, repeated exports and
owned stream cleanup.

- CPython 3.13.9 / PyArrow 24, default CPython wheel: **19 passed**.
- CPython 3.9.25 / PyArrow 21, abi3 wheel: **18 passed, 1 skipped**.
- CPython 3.13.9 / PyArrow 24, the **same abi3 wheel and native hash**:
  **18 passed, 1 skipped**. Acceptance of different supported package majors
  verifies runtime selection rather than a build-interpreter constant.
- Both skips are the optional nanoarrow interoperability case. Dynamic-version,
  module-replacement, subclass and FFI guard cases all ran.
- Real PyArrow 25.0.0 was rejected for Table, RecordBatch, schema and output on
  the new default native build. Normal standalone installation still contains
  only calc-flow-python; `scripts/smoke_wheel.py` passed, including computation,
  configuration and checkpoint recovery.
- Binding-library release Clippy, Cargo formatting, the existing CCN-8 check and
  `git diff --check` passed. Existing P1 closure and the rest of the approved
  review remain applicable to unchanged code; no full suite was repeated.

The build and test commands were:

```bash
.venv/bin/maturin build --release --interpreter .venv/bin/python --out target/expected-major-wheels
.venv/bin/maturin build --release --features legacy-python --interpreter target/zero-dependency-python39/bin/python --out target/expected-major-legacy-wheels
uv pip install --python .venv/bin/python --reinstall target/expected-major-wheels/calc_flow_python-2026.10.27-cp313-cp313-linux_x86_64.whl
.venv/bin/python -m pytest -q python/tests/test_arrow_protocol.py
uv pip install --python target/ci-repair-adapters39/bin/python --reinstall target/expected-major-legacy-wheels/calc_flow_python-2026.10.27-cp39-abi3-linux_x86_64.whl
target/ci-repair-adapters39/bin/python -m pytest -q python/tests/test_arrow_protocol.py
uv pip install --python target/ci-repair-adapters/bin/python --reinstall target/expected-major-legacy-wheels/calc_flow_python-2026.10.27-cp39-abi3-linux_x86_64.whl
target/ci-repair-adapters/bin/python -m pytest -q python/tests/test_arrow_protocol.py
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --release -p calc-flow-python --lib -- -D warnings
```

The abi3 build and its two protocol results completed before the runtime
interruption. They were preserved and matched to the loaded native hashes;
only the missing modern build/checks were completed after resuming.

## Build identity and remaining acceptance

[`data/expected-major-build.json`](data/expected-major-build.json) records both
wheel/native hashes, the source override against the previous sealed manifest,
and runtime identities. Both wheels' 44 Python/stub files match current source;
the Python files are unchanged by this increment. These local Linux wheels are
validation artifacts; full platform/release and coverage gates remain with CI.

No timing samples were added. The specialist's latest controlled localization
confirmed a stable import regression on `48e92fd8`; round-trip and NumPy
acceptance remain unresolved. Report attachment
`01a188df-705d-78be-b202-314e3f7d7d5d` records cumulative usage **640 / 900
seconds**, leaving about 260 seconds. The coordinator will arrange final safe
path measurement and review of this increment. The new native identity must
not inherit a performance pass from older builds.
