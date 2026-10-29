DAL-475 / PR #515: directed repair of the failed checks on `ee2af8a4`.
Implementation commit: `494a7539f64afaf6b762ca1ba8aa136c120f9ed6`.

## Failed checks and changes

The nine failed jobs had three causes:

- Linux Python checks failed at the complexity ratchet before pytest. New
  `output_format` and project-validation arguments introduced four additional
  PLR0913 findings. Private stream setup now groups its three related settings
  in an immutable value. Program execution forwards engine-specific keyword
  arguments through the existing typed overloads and rejects inappropriate
  options as before. Project validation accepts the compatibility `extra`
  keyword through checked options; native unknown-field rejection still applies.
  No threshold, baseline or exclusion changed.
- All six installed-wheel compatibility jobs (3.9–3.14) installed NumPy before
  the smoke that deliberately requires a clean environment. The workflow now
  normally resolves the wheel in a fresh core environment and runs that smoke
  before installing the PyArrow extra, NumPy and pytest for adapter tests.
  Previously those tests also lacked their explicit PyArrow dependency.
- Windows Python tests and Linux combined coverage reached six failing private
  schema assertions. They expected PyArrow methods on the native Arrow C Schema
  producer introduced by this PR. The tests now import that producer explicitly
  with `pa.schema`. The original metadata, field order, nullability, cache
  identity/bounds and invalidation assertions remain. Local reproduction was
  **6 failed, 35 passed** in the three affected files; the repaired cases pass.
  The Windows job reported 6 failed / 1954 passed / 7 skipped; Linux reported
  6 failed / 1956 passed / 5 skipped / 1 warning. Rust tests in the combined
  coverage job passed before its Python failures.

The failed job logs were retrieved from GitHub job IDs `123586785752`,
`123575048976`, `123575535276`, `123586786170`, `123586786038`,
`123586786882`, `123586786133`, `123586786063`, and `123586786260`.
PR reviews and inline review comments were empty when inspected.

## Directed verification

The release wheels were rebuilt from the implementation above, then installed
normally; no `--no-deps` installation or source-tree extension was used.

```bash
.venv/bin/python scripts/verify_complexity_gates.py
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m unittest scripts.test_release_config scripts.test_release_workflow scripts.test_verify_complexity_gates -q
JAX_PLATFORMS=cpu .venv/bin/python -m pytest python/tests/test_native_sql_schema.py python/tests/test_symbolic_event_window_lowering.py python/tests/test_symbolic_event_window_schema_planning.py python/tests/test_program_engine.py python/tests/test_stream_entrypoints.py python/tests/test_arrow_protocol.py python/tests/test_config.py -q
.venv/bin/python -m pytest python/tests/test_config.py::test_project_validation_keywords_keep_native_strictness -q
.venv/bin/python scripts/verify_security_gates.py
```

- Complexity ratchet and Ruff passed (492 formatted files). Release/workflow/
  complexity helper tests: **58 passed**.
- The seven-file installed-wheel group: **118 passed**. The subsequently added
  project-validation keyword regression: **1 passed**. That case also proves
  `extra="ignore"` cannot bypass native unknown-field rejection. These are 119
  distinct directed cases, not a full-suite run.
- Both clean wheel variants passed standalone smoke; real CPython 3.9.25,
  3.10.20, 3.11.15, 3.12.13 and 3.13.9 environments contain only
  `calc-flow-python`. This includes async execution, cancellation cleanup,
  checkpoint cursor/state recovery and unknown-version rejection. Python 3.13
  also passed optimized-mode smoke.
- The exact compatibility pytest selection below passed in fresh adapter
  environments: Python 3.13 / PyArrow 24 / NumPy 2.5.4: **118 passed**;
  Python 3.9 / PyArrow 21 / NumPy 2.0.2: **117 passed, 1 skipped**.
  The remaining compatibility interpreters run this selection in CI.

```bash
python -m pytest -q python/tests/test_import.py python/tests/test_python_version_contract.py python/tests/test_stream_entrypoints.py python/tests/test_execution_options.py python/tests/test_compute.py
```

`verify_security_gates.py` passed its checklist, the existing three-ignore
`cargo audit` command, and `cargo deny --locked check`. Existing allowed yanked
and duplicate-version warnings remain. The rebuilt legacy abi3 wheel passed
the existing inspector (51 files); `git diff --check` passed.

## Codacy closure and remaining acceptance

All 19 original locations and messages are retained in
[`data/codacy-original-annotations.json`](data/codacy-original-annotations.json).
The per-rule fixes are documented in
[`SECURITY-AND-CODACY.md`](SECURITY-AND-CODACY.md). Codacy and Rust dependency
policy were both **successful on `ee2af8a4`**, before this CI repair. The
original `ACTION_REQUIRED` was a code-finding result; no external account
authorization or manual service action was required. New-head remote checks
remain necessary; prior success is not a claim that new CI has completed.

[`data/final-build.json`](data/final-build.json) seals source commit `494a7539`,
the current Python/stub contents, both wheel hashes and native hashes. Both
native modules are byte-identical to their preceding tested builds; the Python
wrappers and workflow changes above have their own directed evidence. The
sdist hash is explicitly historical, not a rebuilt source archive.

No timing samples were added. The budget remains **248.69 / 600 seconds**.
The old Arrow signal is statistically inconclusive and does not establish
performance equivalence for this final build. Coordinator-arranged performance
verification, focused review, and new-head full CI/coverage remain outstanding.
No new external manual-operation blocker was found.
