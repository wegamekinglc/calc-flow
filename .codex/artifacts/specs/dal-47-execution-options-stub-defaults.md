# DAL-47 / GitHub #41: Unify `ExecutionOptions` Stub Defaults

## Status and Sources

- Multica issue: `DAL-47`
- Upstream issue: `wegamekinglc/calc-flow#41`
- Baseline branch at investigation time:
  `agent/cf-orchestrator/2df630d1`
- Relevant introduction commits:
  - `7043728a5808c2e4f179a22ebc6b236c705109f1` (PR #29)
  - `942e5fe564aeee7f9c97052fd81aa70dd25f2065` (PR #32)

This specification covers a type-stub consistency fix only. It does not change
the native constructor or any runtime behavior.

## Problem Statement

`python/calc_flow/_native.pyi` currently declares one constructor with two
different default-value conventions:

```python
class ExecutionOptions:
    def __init__(
        self,
        settings: Mapping[str, _JSONInput] | None = ...,
        deadline: datetime | None = None,
    ) -> None: ...
```

The mixed `...` and concrete `None` defaults are internally inconsistent in the
same public stub signature. The native implementation already owns the exact
runtime defaults, so the stub should use one convention for both optional
constructor parameters.

## Verified Facts and Root Cause

1. `crates/calc-flow-python/src/execution_options.rs` publishes the runtime text
   signature `(settings={}, deadline=None)`.
2. Omitting `settings` constructs an empty native settings map. Passing
   `settings=None` also constructs an empty map.
3. Omitting `deadline` or passing `deadline=None` means no deadline.
4. `python/tests/test_execution_options.py` already asserts the exact runtime
   signature and the empty-settings/no-deadline behavior.
5. Git history shows the mixed style originated when PR #29 introduced
   `settings = ...` and `deadline = None`; PR #32 widened the settings type to
   accept `None` but retained that representation split.
6. The baseline stub passes Ruff lint and format checks. Therefore this is not a
   parser, type, or runtime defect; it is an unguarded public-stub consistency
   defect.

Root cause: the stub combines an opaque default marker for `settings`, whose
runtime default is a mutable empty mapping, with a concrete default literal for
`deadline`. No existing check asserts a single convention for this constructor.

## Decision

Use the stub-only opaque default marker for both parameters:

```python
class ExecutionOptions:
    def __init__(
        self,
        settings: Mapping[str, _JSONInput] | None = ...,
        deadline: datetime | None = ...,
    ) -> None: ...
```

This is the smallest change and accurately communicates that both parameters
are optional without duplicating native default objects in the stub. The native
runtime remains the source of truth for the concrete `{}` and `None` values.

### Rejected Alternatives

- `settings = None` and `deadline = None`: consistent, and explicit
  `settings=None` is accepted, but it describes `None` as the nominal settings
  default while the native public text signature intentionally publishes `{}`.
- `settings = {}` and `deadline = None`: mirrors concrete runtime values but
  retains the mixed convention and places a mutable literal in a public stub.
- Change the native text signature or constructor implementation: unnecessary
  and outside a pure stub-style fix.

## Scope

### Required

- Change only the `deadline` default marker in
  `python/calc_flow/_native.pyi` from `None` to `...`.
- Preserve the existing parameter names, order, positional-or-keyword calling
  convention, annotations, and return type.
- Preserve or add a focused regression assertion that makes the chosen
  two-ellipsis stub convention observable.
- Keep the existing runtime signature assertion
  `(settings={}, deadline=None)` green.

### Out of Scope

- Rust or PyO3 constructor changes.
- Changes to `ExecutionOptions` construction, settings copying, deadline
  normalization, equality, hashing, freezing, or error handling.
- Changes to `ExecutionPlan.execute`, provider callbacks, exports, docs,
  examples, project/checkpoint schemas, Studio REST/OpenAPI, or generated
  TypeScript.
- Broader normalization of defaults elsewhere in `_native.pyi`.

## Compatibility Requirements

- Existing source calls using omission, positional arguments, keyword
  arguments, `settings=None`, or `deadline=None` must behave identically.
- `inspect.signature(calc_flow.ExecutionOptions)` must remain exactly
  `(settings={}, deadline=None)`.
- Static consumers must continue to see both parameters as optional with the
  same accepted types.
- No generated artifact or packaged native module may be introduced into the
  source tree.

The changed `.pyi` text is part of the shipped Python typing surface, but it
does not add, remove, or reinterpret a public API. No new public API design is
required.

## Acceptance Criteria

1. The `ExecutionOptions.__init__` declaration in `_native.pyi` uses `= ...`
   for both `settings` and `deadline`.
2. The annotations remain
   `Mapping[str, _JSONInput] | None` and `datetime | None`, respectively.
3. The native implementation and its published runtime signature are
   unchanged; the focused runtime signature/default tests pass.
4. A focused source/stub contract check fails on the original mixed declaration
   and passes after the change, or the implementation report explains which
   existing automated stub check provides equivalent regression coverage.
5. `ruff check python/calc_flow/_native.pyi` and
   `ruff format --check python/calc_flow/_native.pyi` pass.
6. The affected Python test surface passes, at minimum
   `python/tests/test_execution_options.py` plus any added stub contract test.
7. `git diff --check` passes, and the diff contains no unrelated source,
   documentation, schema, or generated-file changes.

## Documentation and Changelog

No user documentation or changelog entry is expected because runtime behavior
and the accepted type surface do not change. The documentation specialist
should confirm this after review.

## Risks

- Adding no regression assertion would allow the mixed convention to return.
- Changing the native signature to match the stub would turn a style-only issue
  into a compatibility change.
- Expanding the patch to normalize unrelated stub defaults would obscure the
  one-line fix and increase review risk.
