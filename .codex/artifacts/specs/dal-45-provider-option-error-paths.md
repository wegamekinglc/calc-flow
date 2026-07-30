# DAL-45 ProviderOption Error Paths - Specification

## Source

- Multica issue: DAL-45
- External issue: [GitHub #39](https://github.com/wegamekinglc/calc-flow/issues/39)
- Related docs: [`docs/python-api.md`](../../../docs/python-api.md), Runtime
  capabilities
- Existing contract:
  [`.codex/artifacts/api-notes/dal-5-studio-capabilities-contract.md`](../api-notes/dal-5-studio-capabilities-contract.md)

## Problem Statement

`ProviderOption` validates each option before a `ProviderOptionsSchema` can
know that option's tuple position. Its strict-data errors nevertheless report
the hard-coded paths `fields[0].name` and `fields[0].value_type`. In a
multi-field schema, a failure while constructing a later option is therefore
misreported as a failure in the first option.

## Goals

- Make every `ProviderOption` strict-data error identify the failing
  attribute without claiming an unavailable tuple index.
- Preserve the failing runtime type in strict-data errors without echoing the
  rejected value.
- Cover later-option failures for both `name` and `value_type` with focused
  Python regressions.
- Preserve all valid capability and provider-registration behavior.

## Non-Goals

- Changing the public fields, defaults, frozen/slot behavior, or constructor
  signatures of `ProviderOption` or `ProviderOptionsSchema`.
- Changing the supported option value types, duplicate-name rules, field
  sorting, or `additional_properties=False` contract.
- Changing capability schema version 1, Studio/OpenAPI response shapes,
  project format, Rust/PyO3 behavior, runners, or checkpoints.
- Adding position tracking to standalone `ProviderOption` values.
- Refactoring unrelated capability validation or changing exception classes.

## Functional Requirements

- **FR1** - A non-string `ProviderOption.name` must raise `TypeError` with the
  message
  `provider options_schema field name must contain strict data; found <type>`,
  where `<type>` is the rejected value's runtime type name.
- **FR2** - A non-string `ProviderOption.value_type` with a valid string
  `name` must raise `TypeError` with the message
  `provider options_schema field '<name>'.value_type must contain strict data; found <type>`.
- **FR3** - Neither strict-data error may contain `fields[0]`, another
  `fields[<index>]` segment, or any other claim about the option's position in
  a surrounding tuple.
- **FR4** - Strict-data errors must not interpolate, call `str()` on, or
  otherwise expose the rejected value; only its runtime type name may appear.
- **FR5** - Unsupported string `value_type` values must continue to raise
  `ValueError`; non-boolean `required` values must continue to raise
  `TypeError`; their existing validation meaning is unchanged.
- **FR6** - `ProviderOptionsSchema` must continue to reject invalid container
  shapes and duplicate option names and to sort valid fields by name.
- **FR7** - A failed provider option or schema construction must not mutate a
  runtime or advance its capability revision.

## Non-Functional Requirements

- **Performance** - No benchmark target is required. Validation remains
  bounded by the existing option/schema input and must add no I/O or global
  state.
- **State and checkpoints** - No runtime state, runner, snapshot, restore, or
  checkpoint behavior changes.
- **Compatibility** - Python exception classes and all valid public
  constructors/exports remain compatible. Capability schema version 1,
  project JSON Schema, Studio OpenAPI, and generated TypeScript declarations
  remain unchanged. Only the two misleading human-readable error messages
  change.
- **Security** - Rejected callable, path, sequence, or other object values
  must not be reflected in an exception.

## Inputs and Outputs

| Name                           | Type                         | Constraints / Output                                                                            |
| ------------------------------ | ---------------------------- | ----------------------------------------------------------------------------------------------- |
| `ProviderOption.name`          | `str`                        | Non-strings raise the FR1 `TypeError` without a positional path.                                |
| `ProviderOption.value_type`    | `OptionValueType`            | Accepts the four existing scalar type names; other inputs retain the specified errors.          |
| `ProviderOption.required`      | `bool`                       | Non-booleans retain the existing `TypeError`.                                                    |
| `ProviderOptionsSchema.fields` | `tuple[ProviderOption, ...]` | Valid tuples retain duplicate detection and deterministic sorting.                              |

## Acceptance Criteria

- [ ] Given a schema expression whose first option is valid and whose second
      `ProviderOption` has a non-string `name`, construction raises the exact
      FR1 `TypeError`, reports the runtime type, and contains no
      `fields[<index>]` segment.
- [ ] Given a schema expression whose first option is valid and whose second
      `ProviderOption` has a non-string `value_type`, construction raises the
      exact FR2 `TypeError`, identifies the second option by its valid name,
      reports the runtime type, and contains no `fields[<index>]` segment.
- [ ] Callable, `Path`, and container regression cases prove that neither
      strict-data error exposes the rejected value or nested secret content.
- [ ] Existing tests for unsupported string value types, invalid schema
      containers, duplicate names, valid provider registration, capability
      revision, and top-level public exports remain green.
- [ ] Focused Python verification passes with
      `JAX_PLATFORMS=cpu uv run pytest python/tests/test_capabilities.py -q`.
- [ ] Python lint and formatting checks for the touched files pass with
      `uv run ruff check python/calc_flow/capabilities.py
      python/tests/test_capabilities.py` and `uv run ruff format --check
      python/calc_flow/capabilities.py python/tests/test_capabilities.py`.
- [ ] No changes are produced in `schemas/project-v2.schema.json`,
      `web-ui/openapi.json`, or `web-ui/src/api/schema.d.ts`.
- [ ] The existing capability contract note and qualifying user-facing
      documentation are reconciled with the corrected messages.

## Open Questions

- None. The observable messages and compatibility boundary are fixed above;
  the implementation may choose any local validation structure that satisfies
  them without adding positional state.
