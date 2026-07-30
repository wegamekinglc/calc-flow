# DAL-45 ProviderOption Error Paths - API Note

Status: proposed contract for critic review; no implementation or public
schema change is included in this note.

Baseline: `f5c51f187deb0913b3605f4a389c96181ac0ea95`
(`origin/main` and branch `agent/cf-orchestrator/cee2157f` on 2026-07-30).

Upstream specification:
`.codex/artifacts/specs/dal-45-provider-option-error-paths.md`.

## Decision

Keep strict-data validation in `ProviderOption.__post_init__`. A
`ProviderOption` remains invalid at construction time even when it is never
inserted into a `ProviderOptionsSchema`.

Change only the two diagnostics that currently claim the unavailable tuple
position `fields[0]`:

| Invalid constructor input | Required exception | Required exact message |
| --- | --- | --- |
| `ProviderOption(name=<non-str>, value_type="string")` | `TypeError` | `provider options_schema field name must contain strict data; found <type>` |
| `ProviderOption(name=<valid-str>, value_type=<non-str>)` | `TypeError` | `provider options_schema field <name!r>.value_type must contain strict data; found <type>` |

`<type>` is replaced with `type(rejected_value).__name__`. `<name!r>` means the
normal Python `repr()` rendering of the already validated string field name;
for example, `ProviderOption("scale", [])` raises:

```text
provider options_schema field 'scale'.value_type must contain strict data; found list
```

The rejected `name` cannot safely identify itself, so its diagnostic identifies
only the `name` attribute. A rejected `value_type` may be associated with the
already validated string `name`, but never with a surrounding tuple index.
Neither message may contain `fields[`, an inferred index, or a schema-level
path.

The rejected object itself must not be interpolated, stringified, represented,
or reflected upon. Only `type(rejected_value).__name__` is observable. The
valid string `name` in the `value_type` diagnostic is schema metadata, not the
rejected value.

## Public Surface

The public Python declarations remain unchanged:

```python
@dataclass(frozen=True, slots=True)
class ProviderOption:
    name: str
    value_type: OptionValueType
    required: bool = False


@dataclass(frozen=True, slots=True)
class ProviderOptionsSchema:
    fields: tuple[ProviderOption, ...] = ()
    additional_properties: Literal[False] = False
```

There is no constructor parameter, field, factory, overload, exception type,
or export change. In particular:

- `ProviderOption` remains independently constructible and independently
  validated.
- `ProviderOptionsSchema` does not re-validate raw option-like inputs and does
  not receive partially constructed `ProviderOption` values.
- `ProviderOptionsSchema.fields` remains an exact tuple of `ProviderOption`
  values, with duplicate detection and deterministic sorting unchanged.
- `ProviderOption.value_type` still accepts only `string`, `integer`,
  `number`, and `boolean`.
- `ProviderOption.required` remains strict `bool`.
- Frozen and slotted dataclass behavior remains unchanged.

Python evaluates each `ProviderOption(...)` expression before calling
`ProviderOptionsSchema(...)`. A failure in the second expression of a
multi-field schema therefore still originates from standalone option
construction. The corrected message deliberately does not pretend that
`ProviderOption` knows it would have occupied index 1.

## Preserved Error Contract

All validation branches other than the two strict-data diagnostics above
retain their exception classes, validation order, and messages. This includes:

- unsupported string `value_type` values raising `ValueError`;
- non-boolean `required` values raising `TypeError`;
- non-tuple or non-`ProviderOption` schema fields raising `TypeError`;
- `additional_properties` values other than `False` raising `ValueError`;
- duplicate option names raising `ValueError`.

The strict `name` check must still run before the strict `value_type` check.
The strict `value_type` check must still run before membership in the four
supported scalar type names. Construction must not defer any of these checks
until provider registration or capability serialization.

## Compatibility

This is a narrow diagnostic compatibility change:

- **Source and runtime compatibility:** unchanged for every valid
  `ProviderOption`, `ProviderOptionsSchema`, provider registration, and
  capability snapshot.
- **Exception compatibility:** the same invalid inputs raise the same
  `TypeError` classes at the same constructor boundary.
- **Message compatibility:** tests, log processors, or callers matching the
  two old `fields[0]` strings must update to the exact messages in this note.
  Callers must not infer a tuple position from standalone option validation.
- **Wire and generated-schema compatibility:** capability schema version 1,
  project format/schema, Studio REST/OpenAPI, and generated TypeScript
  declarations do not change.
- **State compatibility:** failed construction cannot register a provider or
  advance a runtime capability revision.
- **Security compatibility:** rejected callable, path, sequence, mapping, or
  other object values remain absent from error text.

No deprecation period or capability schema-version bump is warranted because
no callable signature, accepted value, serialized field, or wire shape
changes. The exact diagnostic replacement is the bug fix.

## Alternatives Rejected

1. **Keep standalone validation and remove the unavailable index (selected).**
   This fixes the false location while preserving construction strictness and
   every public type.
2. **Move strict-data validation into `ProviderOptionsSchema`.** This could
   report a true `fields[index]` path, but a standalone invalid
   `ProviderOption` would either be constructible or require duplicate
   validation. It violates the decision not to weaken construction.
3. **Add an index, owner, or validation-context parameter to
   `ProviderOption`.** The context is not available at ordinary construction
   time, would alter the public dataclass contract, and would introduce state
   whose only purpose is formatting an error.

## Required Verification

Focused regressions in `python/tests/test_capabilities.py` must cover:

1. A schema expression with a valid first option and a second option whose
   `name` is non-string. Assert the exact first message above, its runtime type
   name, and absence of every `fields[<index>]` segment.
2. A schema expression with a valid first option and a second option whose
   `value_type` is non-string. Assert the exact second message above using the
   second option's valid name, its runtime type name, and absence of every
   `fields[<index>]` segment.
3. Callable, `pathlib.Path`, and container cases across the two strict-data
   branches. Assert that callable labels, paths, nested strings, and other
   rejected-value content do not appear in `str(error)`.
4. The existing unsupported-string `value_type`, invalid `required`, schema
   container, duplicate-name, and valid field-sorting behavior.
5. Rejected provider/schema construction leaves runtime capability revision
   unchanged; valid provider registration and top-level public exports remain
   green.

Run at minimum:

```text
JAX_PLATFORMS=cpu uv run pytest python/tests/test_capabilities.py -q
uv run ruff check python/calc_flow/capabilities.py python/tests/test_capabilities.py
uv run ruff format --check python/calc_flow/capabilities.py python/tests/test_capabilities.py
```

Verify that the implementation produces no changes in:

```text
schemas/project-v2.schema.json
web-ui/openapi.json
web-ui/src/api/schema.d.ts
```

## Documentation

Required reconciliation:

- Update the generic provider-option strict-data row and the hard-coded
  callable example in
  `.codex/artifacts/api-notes/dal-5-studio-capabilities-contract.md` so they
  no longer advertise `fields[0]`.
- Add a concise `CHANGELOG.md` fix entry because the exact human-readable
  Python diagnostic is observable.

`docs/python-api.md` and `docs/api-reference.md` describe the public values but
do not publish these error strings, so they require no content change for this
contract. The documentation pass should record that determination rather than
adding positional-validation detail to general API guidance.

## Implementation Guardrails

- Limit implementation code to the local strict-data message construction in
  `python/calc_flow/capabilities.py` and focused tests.
- Do not add schema-level index discovery, mutable state, or a second
  validation pass.
- Do not change `required`, unsupported-type, duplicate-name, sorting, or
  registration behavior while editing adjacent code.
- Do not regenerate project, OpenAPI, or TypeScript schema artifacts.
- Do not call `str()` or `repr()` on a rejected value.

## Open Questions

None. The exception classes, constructor boundary, exact replacement text,
name rendering, compatibility scope, tests, and documentation targets are
fixed.
