DAL-475 follow-up to PR #515: remove every required third-party Python package,
including Pydantic and typing-extensions, as requested in the issue discussion.

## Package and public compatibility

- `project.dependencies` is empty. Wheel metadata has no unconditional
  `Requires-Dist`. Optional PyArrow, NumPy and JAX adapters remain opt-in extras;
  the native Rust engine is bundled in the extension.
- Project parsing, defaults, schema and canonicalization still come from the
  Rust project-v3 validator. Python retains strict JSON range/depth/cycle checks
  and duplicate-key rejection before native validation. Caller inputs and dumped
  mappings remain independent. Project documents retain value equality.
- `ProjectDocument` retains `root`, `model_validate`, `model_validate_json`,
  `model_dump`, `model_dump_json`, `model_json_schema` and `canonical_json`.
  It is a slotted data container, no longer a Pydantic `RootModel`. Inherited
  Pydantic field selection, unchecked construction and model configuration APIs
  are not supported. Catch `calc_flow.ProjectValidationError` (a `ValueError`)
  instead of `pydantic.ValidationError`. Error entries retain native issue codes,
  location tuples and messages; `errors()` returns independent records.
- Caller-installed Pydantic can use the custom validation/serialization protocol
  without Calc Flow importing it. The plain core-schema dictionary follows the
  [Pydantic core-schema protocol](https://pydantic.dev/docs/validation/dev/api/pydantic-core/pydantic_core_schema/).
  The standalone path neither installs nor imports Pydantic.
- Annotation-only use of `Never` becomes stdlib `NoReturn`; ordinary aliases
  replace `TypeAlias` markers. Python 3.12+ keeps stdlib named aliases; earlier
  Python uses conventional runtime aliases. `ParamSpec` and `Unpack` use stdlib
  where available and deferred annotations elsewhere. The Python 3.9 minimum
  remains unchanged. Dynamic evaluation of annotations using newer typing forms
  on older interpreters is not covered by these smoke tests.
- Root exports, the Python error module and the native stub are synchronized.
  This follow-up introduces no native API or checkpoint-format change. PyArrow
  unsafe-version guards, provider rules, audit ignores and CI policies remain.

## Directed validation

Normal resolution installs exactly one package into each clean environment.
Neither `--no-deps` nor a source checkout masks package dependencies.

```bash
uv pip install --python target/zero-dependency-venv/bin/python target/zero-dependency-wheels/calc_flow_python-2026.10.27-cp313-cp313-linux_x86_64.whl
target/zero-dependency-venv/bin/python scripts/smoke_wheel.py
target/zero-dependency-venv/bin/python -O scripts/smoke_wheel.py
```

The smoke checks cover package absence and metadata, project validation and
canonical round trips, declarations, Arrow capsules, finite/async computation,
Program execution, owned streaming and missing-adapter diagnostics.
They also run `smoke_wheel_recovery.py` entirely in those clean environments:
checkpoint a cumulative SQL sum of 3, cancel and await cleanup, reject a copied
checkpoint with unknown format version before opening its source, restore cursor
1 into a fresh runner, and verify the next cumulative result is 6 with no tasks
or source/sink resources left open. CPython 3.9–3.13 all passed this recovery
check; 3.13 also passed under `-O`. The recovery wrapper reports corrupted
manifests through its existing `checkpoint_mismatch` error category.

The `legacy-python` wheel was normally resolved and smoke-tested with real
CPython 3.9.25, 3.10.20, 3.11.15 and 3.12.13 interpreters. Every package module
and stub AST matches the final source. The legacy wheel was packaged before
Ruff added one blank line and collapsed one stub alias expression; those two
formatting differences do not change its AST. CPython 3.13 uses the separate
wheel built with the original default features and release configuration.

Final installed-wheel regressions: **130 passed**, including strict JSON,
duplicate keys, cycles, depth/range boundaries, structured join errors, defensive
copies, equality, stores, schema, export/reload, snapshot SQL and caller-installed
Pydantic serialization. The first run found missing document value equality;
the slotted data container now preserves it and the affected group passed.

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest python/tests/test_config.py python/tests/test_store.py python/tests/test_python_version_contract.py python/tests/test_stream_join_raw_validation.py python/tests/test_expression_api.py python/tests/test_sql_pipeline.py python/tests/test_snapshot_sql.py -q
.venv/bin/python -m unittest scripts.test_release_config scripts.test_release_workflow -q
```

The latter ran **42 tests, OK**. Scoped Ruff lint/format, `git diff --check`,
and Bandit for the touched project wrapper and wheel smoke passed. The existing
JSON traversal retains CCN 10, identical to `cbbf4cc5`; newly added functions
stay at or below 8 without new complexity suppressions.

The abi3 legacy wheel passed `inspect_wheel.py core-wheel` (51 files), and the
sdist passed its inspector (1181 files). The same core-wheel inspector rejects
the local CPython-specific 3.13 wheel because it requires an abi3 module; this
is an artifact-format limitation, not an installed-wheel failure. Release/CI
uses its existing abi3 build settings, which were not changed.
Smoke scripts are repository CI assets, not files distributed inside wheels or
sdists; the source archive contains the package and native build sources.

`cargo audit` and `cargo deny` results from
[the security/Codacy repair](SECURITY-AND-CODACY.md) remain applicable: Cargo.lock,
native source and audit policy are unchanged by dependency removal. That repair
also records the preceding 185 passing native-adapter tests and remote Codacy
success on `cbbf4cc5`; it does not substitute for checks on the newest commit.

## Performance and remaining acceptance

No timing sample was added by this follow-up. The performance specialist's
issue attachment reports 248.69/600 seconds used in total, including the original
177.23 seconds. The Arrow signal remains statistically inconclusive, including
same-version noise controls; performance equivalence has **not** passed.
Those measurements cover `5c2c3515`, not the final dependency/Codacy build.
`data/final-build.json` identifies the final wheel variants, native libraries,
source manifest and build settings for subsequent focused measurement.
Python/native allocation equality, GPU behavior, minimum optional backend
combinations, and full CI/coverage remain outside these local checks. The final
performance verification and focused review remain with the coordinator.
