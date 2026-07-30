# DAL-45 ProviderOption Error Paths - Implementation-Readiness Critique

## Review target

- Specification:
  `.codex/artifacts/specs/dal-45-provider-option-error-paths.md`
- API note:
  `.codex/artifacts/api-notes/dal-45-provider-option-error-paths-contract.md`
- Branch: `agent/cf-orchestrator/cee2157f`
- Base: `origin/main` at
  `f5c51f187deb0913b3605f4a389c96181ac0ea95`
- Review date: 2026-07-30

The review compared both artifacts with the current
`ProviderOption`/`ProviderOptionsSchema` implementation, focused capability
tests, top-level exports, the existing DAL-5 capability contract, public Python
documentation, generated-schema boundaries, and the repository changelog
charter.

## Verdict

**Pass with implementation guardrails. Implementation may start.**

Remaining blocking findings: **0**.

The design identifies the real ownership boundary: Python evaluates and
validates each standalone `ProviderOption` before
`ProviderOptionsSchema.__post_init__` can observe a tuple or index. Removing
the false index while retaining constructor-time validation is therefore the
smallest compatible repair.

The exact replacement messages, exception classes, validation order, accepted
values, public declarations, registration behavior, and non-change surfaces
are sufficiently fixed. The implementation can remain local to the two
strict-data diagnostics and their focused tests. No Rust, PyO3, project,
Studio, runner, checkpoint, capability-version, OpenAPI, or generated
TypeScript change is justified.

This is a design approval, not implementation evidence. No source,
test, documentation, generated artifact, commit, PR, or merge was produced by
this review.

## Blocking findings

None.

## Required finding dispositions

| ID | Area | Severity | Disposition |
| --- | --- | --- | --- |
| N1 | Strict-data regression matrix | Recommendation | Exercise callable, `Path`, and container values against **both** `name` and `value_type`, not only one representative per branch. Exact-message assertions must also prove the runtime type name and absence of `fields[` and rejected content. |
| N2 | Validation-order preservation | Recommendation | Add a case where both `name` and `value_type` are invalid and assert that `name` wins, plus a case where `value_type` and `required` are invalid and assert that `value_type` wins. The API note promises the existing order, so it should be pinned rather than inferred from adjacent tests. |
| N3 | Rejected-object hooks | Recommendation | Include a sentinel rejected object whose `__str__` and `__repr__` raise or record access. This directly proves that message construction does not invoke value hooks. Retrieve only the runtime type name; do not format the rejected object. A custom metaclass can intercept ordinary `type(value).__name__` attribute access, so the implementation should avoid broadening the security promise to adversarial metaclass behavior unless it also bypasses and tests that hook. |
| N4 | Changelog ownership | Documentation-stage correction | The API note says a `CHANGELOG.md` entry is required, but the repository charter reserves the changelog for fundamental changes and assigns the qualifying judgment to `cf-doc-writer`. Treat the specification's wording ("qualifying user-facing documentation") as controlling: reconcile the stale DAL-5 contract row, then let `cf-doc-writer` decide and record whether this diagnostic-only fix crosses the changelog bar. This does not block source implementation. |
| N5 | Cross-platform type-name assertion | Recommendation | For the `Path` case, assert the actual runtime type name rather than hard-coding `PosixPath` unless the test is intentionally Linux-only. The public contract promises `type(rejected).__name__`, not one platform's concrete `Path` implementation. |

## Contract review

### Correctness and ambiguity

The exact diagnostics are unambiguous:

- invalid `name` identifies only the `name` attribute because no safe
  `ProviderOption` metadata exists yet;
- invalid non-string `value_type` may identify the already validated string
  `name`;
- neither branch claims a tuple position;
- unsupported string `value_type` remains a distinct `ValueError`.

Using normal `repr()` rendering for the already type-checked exact `str` name
is deterministic and cannot invoke a rejected value's formatting hooks.
Because the name is approved provider-schema metadata rather than the rejected
`value_type`, including it does not violate the redaction boundary.

The multi-field acceptance examples are intentionally expressions, not
schema-level validation events. Construction of the second option fails before
the tuple and outer schema exist. Tests should retain that expression shape so
they demonstrate the ownership boundary that caused the original false
`fields[0]` path.

### Backward compatibility

Valid constructor inputs, dataclass layout, frozen/slot behavior, imports,
serialized capability values, provider registration, and revision behavior
are unchanged. Invalid inputs retain their exception class and constructor
boundary. The only compatibility cost is intentional: callers matching either
old human-readable `fields[0]` message must update.

No schema-version bump or deprecation period is warranted because the change
does not alter accepted data, a callable signature, a public field, or a wire
shape.

### Overlooked surfaces

The affected production branch is confined to
`python/calc_flow/capabilities.py`. The relevant regression surface is fully
reachable from `python/tests/test_capabilities.py`, which already covers
provider registration, revisions, capability values, helper registrations,
and top-level exports.

The existing DAL-5 API contract contains the stale callable-name diagnostic
and must be reconciled. `docs/python-api.md` and `docs/api-reference.md` name
the public values but publish no error text, so adding path-detail prose there
would be noise. The project schema, Studio OpenAPI, and generated TypeScript
declarations are legitimate frozen non-change surfaces.

### Security and regression risk

The important security property is non-observation of the rejected value:
message construction must not call its `str`, `repr`, iteration, path
conversion, callable, or container traversal hooks. Type-name-only reporting
keeps callable labels, filesystem paths, and nested container content out of
the exception.

The principal regression risk is accidental editing of adjacent validation
branches or ordering. Exact assertions for invalid `required`, unsupported
string `value_type`, invalid schema containers, duplicate names, sorting, and
the two mixed-invalid precedence cases keep the repair narrow.

Failed standalone value construction cannot itself mutate a runtime because
it occurs before registration. Retaining the existing rejected-registration
revision test is useful integration evidence; a focused before/after revision
assertion around rejected option/schema expressions documents the promised
non-effect without implying that those expressions ever reached the runtime.

## Acceptance and verification judgment

The proposed focused pytest, Ruff lint, and Ruff formatting commands are
proportionate to the touched Python files. The focused test file already
traverses the valid provider/capability paths at risk. The independent tester
may add broader Python verification, but a full Rust, backend, or frontend run
is not an implementation-readiness requirement for this diagnostic-only
change.

Implementation evidence should include:

1. exact new messages for later-option invalid `name` and `value_type`;
2. the N1-N3 regression/precedence coverage;
3. preserved invalid `required`, unsupported string type, schema container,
   duplicate, sorting, registration, revision, and export tests;
4. the three required pytest/Ruff command results;
5. `git diff --exit-code` or equivalent evidence that the three generated
   schema/client files did not change;
6. documentation reconciliation, with the changelog decision owned and
   recorded by `cf-doc-writer`.

## Go / no-go

**GO for implementation.**

The recommendations above refine test and documentation ownership; they do
not change the approved error contract and do not require another spec or API
design round. Implementation must not open a PR, merge, regenerate schemas, or
modify unrelated surfaces as part of this delegated stage.
