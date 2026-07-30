# DAL-47 / GitHub #41: Pre-implementation Critique

## Verdict

**Approved with one implementation-blocking verification requirement.**

The proposed one-line stub change is compatible with both the native constructor
and the Python typing surface. The specification correctly keeps the concrete
runtime defaults in Rust/PyO3 and uses `...` only as the stub's opaque default
marker. No API-design stage is needed.

The implementation must not claim that an existing check provides regression
coverage for the two-ellipsis convention: no such check exists on the reviewed
baseline.

## Evidence Reviewed

- Specification:
  `.codex/artifacts/specs/dal-47-execution-options-stub-defaults.md`
- Current stub:
  `python/calc_flow/_native.pyi`
- Native constructor and text signature:
  `crates/calc-flow-python/src/execution_options.rs`
- Runtime tests:
  `python/tests/test_execution_options.py`
- Repository-layout tests:
  `python/tests/test_repository_layout.py`
- Python lint/test configuration:
  `pyproject.toml` and `.github/workflows/ci.yml`
- Introduction history:
  `7043728a5808c2e4f179a22ebc6b236c705109f1` and
  `942e5fe564aeee7f9c97052fd81aa70dd25f2065`

The baseline has the exact mixed declaration described by the specification.
The native constructor still publishes
`(settings={}, deadline=None)`, maps omitted and explicit `None` settings to an
empty map, and maps omitted and explicit `None` deadlines to no deadline.

## Blocking Finding

### B1. Acceptance criterion 4's “existing equivalent check” escape hatch is not available

The specification allows either a focused source/stub contract test or an
implementation report explaining an existing equivalent check. On the reviewed
baseline:

- `ruff check` and `ruff format --check` accept both the mixed declaration and
  the proposed two-ellipsis declaration;
- `test_native_execution_and_provider_signatures_are_canonical` inspects the
  compiled native class, not `_native.pyi`;
- no test under `python/tests` reads or parses `_native.pyi`; and
- the CI workflow has no separate stub-consistency or type-stub validation
  stage that enforces this convention.

Therefore the implementation **must add a focused source-level regression
assertion** that fails when `deadline` returns to `None` (or either constructor
default ceases to be `...`). General Ruff success or the existing runtime
signature assertion is not equivalent evidence.

This does not require expanding production scope. A small repository-contract
test is sufficient. Prefer parsing the checked-in stub and locating
`ExecutionOptions.__init__` over matching the entire formatted signature as a
raw multiline string. The test must read the repository copy of
`python/calc_flow/_native.pyi`, not an installed package copy, and must be
included explicitly in the reported test command if it is not added to
`test_execution_options.py`.

## Compatibility and Boundary Assessment

### Runtime boundary

The `.pyi` file is not executed by the native constructor. Changing only the
stub token cannot alter omission, positional and keyword calls, explicit
`settings=None`, explicit `deadline=None`, error behavior, stored values, or
`inspect.signature`. Any Rust/PyO3 edit would be both unnecessary and a scope
violation.

### Typing boundary

In a stub, `= ...` records that a parameter has a default while intentionally
leaving the concrete runtime value unspecified. The accepted annotations remain
`Mapping[str, _JSONInput] | None` and `datetime | None`; the callability of both
parameters as omitted and the acceptance of explicit `None` therefore remain
unchanged for static consumers.

The change does make the concrete `None` default less visible to tools that
render the stub declaration text, but that is the intended normalization in
GitHub #41, not a runtime or accepted-type compatibility break. The native text
signature remains the authoritative concrete-value representation.

### Testability boundary

The specification separates the two contracts correctly:

- a source/stub test must observe the two `...` defaults; and
- the existing native test must continue to observe
  `(settings={}, deadline=None)`.

Both are required because neither one proves the other.

## Non-blocking Recommendations

1. Keep the regression test narrow: assert the class, constructor parameter
   names/order, and the two ellipsis defaults. Avoid building a general-purpose
   stub parser or normalizing unrelated defaults.
2. If AST parsing is used, inspect the constructor defaults structurally rather
   than executing the `.pyi`. Execution would introduce irrelevant import and
   extension-build dependencies.
3. Report the source-contract test separately from
   `python/tests/test_execution_options.py` when they live in different files,
   so acceptance criterion 6 is reproducible.
4. Preserve the specification's no-docs/no-changelog decision. This is a
   shipped typing-file correction, but it does not change runtime behavior or
   the accepted type surface.
5. Review the final diff against baseline `f5c51f1` (excluding the upstream
   spec and critique artifacts as process artifacts) to ensure production
   changes are limited to `_native.pyi` plus the focused regression test.

## Scope Guardrails

The implementation should not:

- change the Rust constructor, PyO3 `text_signature`, or argument parsing;
- substitute `None` or `{}` for the existing `settings = ...`;
- alter annotations, parameter kinds, exports, or documentation;
- normalize other default values in `_native.pyi`; or
- add a type checker, stub generator, or broad repository-wide stub policy for
  this one-line consistency fix.

## Required Handoff Evidence

Before this stage's blocking finding is considered cleared, the implementer
must report:

- the focused source/stub regression test and evidence that it covers both
  defaults;
- the passing runtime signature/default test;
- passing Ruff lint and format checks for `_native.pyi`;
- `git diff --check`; and
- a scoped diff showing no Rust, docs, schema, generated, or unrelated stub
  changes.

## Residual Risk

After B1 is addressed, residual risk is low. The main remaining failure mode is
accidentally conflating the stub's opaque default marker with the native
constructor's concrete `None`/`{}` defaults; the paired source and runtime
assertions directly guard that boundary.
