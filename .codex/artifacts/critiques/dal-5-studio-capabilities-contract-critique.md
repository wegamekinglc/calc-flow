# DAL-5 5.1 Studio Capabilities Contract - Re-review Critique

## Target

- Revised API note:
  `.codex/artifacts/api-notes/dal-5-studio-capabilities-contract.md`
  (reviewed from the restored issue attachment; SHA-256
  `7c0d69c14299de5a248a3d83e63c5cc10d26064a173fca56fb75820a50a36a9a`)
- Prior critique:
  `.codex/artifacts/critiques/dal-5-studio-capabilities-contract-critique.md`
  (pre-revision SHA-256
  `b8c7f7f741da771457493a7e5aa4cfb5cd37b86f61ee305a195ed9c082a9c3e2`)
- Upstream analysis:
  `.codex/artifacts/analysis/dal-5-weekly-capability-progress-2026-07-25.md`
- Upstream API audit:
  `.codex/artifacts/api-notes/dal-5-public-surface-audit.md`
- Upstream critique:
  `.codex/artifacts/critiques/dal-5-weekly-capability-critique-2026-07-25.md`
- Public-surface baseline:
  `2413ffdb74e2a004b3f1b541e9c2917c3b8f5ef2`

The three upstream originals were restored byte-for-byte and read in full.
Their SHA-256 digests are, in the order listed above,
`52fbe05aafaa658c42c4ffe8ece2924690045b1e1d0a67091910fdbf3f494145`,
`859ea4d05690f490e899a0d4d9f4d472fd6a3d66561a6d99f12e5fc059549e35`,
and `ee204ab9f6a8d4d0370c85a62265a57f3626f523d2d2378aad23efbcb0fe3213`.

## Verdict

**Pass**

The revision resolves all three former blockers and gives an explicit,
technically consistent disposition for all 13 prior findings. There are no
remaining design blockers and no unresolved prior advisories. The acceptance
matrix is implementable against `2413ffdb`; one future-version assertion should
be tracked as a staged migration gate rather than reported as a test that 5.1
can execute before capability schema v2 exists.

This is a design pass, not an implementation pass. The capabilities route,
strict response models, decoders, and generated types are still absent from the
current code, as expected.

## Findings

### Blocking Issues

None.

### Significant Concerns

None unresolved from the prior critique.

### Minor / Style Notes

- **Separate the current v1 gate from the future v2 migration gate** - The
  compatibility row requires a future v2 client to retain an explicit v1
  decoder (revised note lines 388-395 and 943). The architecture is sound and
  implementable now by keeping a named, version-dispatched v1 decoder and
  testing that v1 rejects `schemaVersion: 2` before v1 field parsing. An actual
  “v2 client reads v1” fixture becomes executable only when a v2 schema and
  decoder exist.
  - **Suggested fix:** Split the implementation ticket into a 5.1 gate
    (closed-v1 decoding, extra-field rejection, unsupported-version
    pre-dispatch, and old-server fallback) and a retained future-v2 migration
    obligation. Do not fake a v2 parser merely to mark the current matrix row
    green.

## Former Blockers - Technical Re-verification

### 1. Current provider and `table_matmul@1` surface - resolved

The revision is now pinned to `2413ffdb` (revised note lines 5-7) and describes
both current registration modes:

- public `Runtime.register_provider(...)` records a successful single-array
  provider under one registration lock
  (`python/calc_flow/pipeline.py:84-109`);
- private `_register_mapping_provider(...)` records copied, declared input and
  output Port contracts (`python/calc_flow/pipeline.py:111-143`), and the native
  stub exposes that private seam (`python/calc_flow/_native.pyi:64-78`);
- `register_numpy()` and `register_jax()` each register `expression@1` followed
  by mapped `table_matmul@1`
  (`python/calc_flow/array.py:650-674`);
- the mapped contract is exactly
  `(("table", "table"), ("weights", "array")) ->
  (("output", "array"),)` (`python/calc_flow/array.py:24-25`);
- the public functional builder emits the same required Port order and
  `columns` list (`python/calc_flow/pipeline.py:393-445`);
- adjacent tests already pin defensive mapped records and the two identities
  per helper (`python/tests/test_runtime.py:152-182`,
  `python/tests/test_array.py:1246-1276`).

The revised capability entries reproduce these identities and Ports exactly,
give only `expression@1` the scalar `expression` schema, and correctly use
`optionsSchema=null` for `table_matmul@1` because its non-empty unique
`list[str]` cannot fit schema v1 (revised note lines 245-298). This also matches
`docs/introduction.md:129-137` and `docs/introduction.md:187-208`.

**Result: Pass.** The former stale-surface blocker is closed.

### 2. Parent Runtime versus spawned-worker scope - resolved

The current implementation proves that compilation registration and worker
transport are different boundaries:

- the parent owns successful callback-bearing registration records and returns
  defensive snapshots (`python/calc_flow/pipeline.py:145-202`);
- `RunManager.submit()` selects only project-referenced records and serializes
  them with the project payload
  (`web-ui/backend/src/calc_flow_studio/run_manager.py:662-692`);
- the worker restores legacy providers, mapped providers with exact Ports, and
  scalar UDFs before compilation
  (`web-ui/backend/src/calc_flow_studio/run_manager.py:532-601`);
- transport uses `cloudpickle` and redacts serialization failure
  (`web-ui/backend/src/calc_flow_studio/run_manager.py:567-585`);
- the current lazy selector recognizes only missing
  `numpy:expression@1` and `jax:expression@1`, not
  `table_matmul@1` (`web-ui/backend/src/calc_flow_studio/run_manager.py:460-488`);
- adjacent tests prove both the runtime-capturing callback failure and exact
  mapped restore behavior
  (`web-ui/backend/tests/test_run_manager.py:497-568`).

The revision now reports the parent compile snapshot under `runtime` and worker
transport under the closed `preview.workerRegistrations` union, including
`serialized`, `lazyBuiltin`, and redacted `unavailable` variants (revised note
lines 506-533 and 549-590). It explicitly says reconstruction is not an
execution guarantee and preserves the current absence of lazy
`table_matmul@1`.

**Result: Pass.** The former scope-conflation blocker is closed.

### 3. Closed schema bump rule - resolved

The revision now states one coherent rule:

- every capability object is closed and rejects extra fields;
- every added object field, including optional/defaulted fields, requires a
  capability schema bump;
- every added union member or discriminator value also requires a bump;
- collection entries that already fit the existing item schema do not require
  a bump;
- a version bump never relaxes the callback/source/path/environment/serialized
  callback/options-value/credential/secret prohibition;
- the root version is selected before version-specific parsing, and
  `schemaVersion: 1` plus an unknown field is malformed v1.

These rules are explicit at revised note lines 360-395 and are repeated for the
worker `registrationKind` domain at lines 531-533. They are compatible with the
proposed Pydantic `extra="forbid"` models and strict TypeScript decoders.

**Result: Pass.** The former version-contract blocker is closed.

## 13/13 Disposition Verification

| # | Prior finding | Re-review result |
| -: | --- | --- |
| 1 | Stale pre-#27 provider surface | **Verified resolved:** current baseline, both registration modes, and exact four NumPy/JAX identities are normative. |
| 2 | Parent runtime versus worker availability | **Verified resolved:** compile scope and reconstruction scope are separate, closed projections. |
| 3 | Closed schema versus optional-field rule | **Verified resolved:** every new field/member/discriminator bumps the capability schema. |
| 4 | Illegal empty revision-2 example | **Verified resolved:** revision 2 now contains NumPy `expression` and `table_matmul`; lines 347-358 define legal and partial transitions. |
| 5 | Generated-client source compatibility | **Verified resolved:** validation and run source breaks, deployment coupling, fixtures, and external regeneration are explicit at lines 787-833. |
| 6 | Missing browser runtime parser | **Verified resolved:** raw JSON must cross production decoders before React at lines 712-739. |
| 7 | Validation malformed-report callers | **Verified resolved:** validate/create/put/import share strict parsing; malformed runtime data is `500` with no write at lines 768-780. |
| 8 | Result validation after state transition | **Verified resolved:** the full result is validated before `COMPLETED`, with failed/no-result fallback at lines 703-710. |
| 9 | Coarse result edge cases | **Verified resolved:** table/array boundary, empty, null-only, Unicode, exact-limit, and malformed cases are explicit at lines 948-952. |
| 10 | Missing Python export/parity gate | **Verified resolved:** public imports, `__all__`, stub, adapter, docs, and tests are required at lines 304-312 and 939. |
| 11 | Ambiguous `arrowTypes` | **Verified resolved:** `portableArrowTypes` is the exact v2 project/UDF declaration vocabulary at lines 300-302. The listed values match `crates/calc-flow/src/config.rs:931-953`. |
| 12 | Ambiguous generated-file drift check | **Verified resolved:** clean-CI and staged-local procedures are distinguished and project schema is separately frozen at lines 1015-1034. |
| 13 | Repeated snapshot allocation | **Verified resolved:** immutable serialized responses are cached by `(sessionId, revision)` and callback classification is not retried per poll at lines 337-345. |

**Disposition result: 13/13 verified.**

## Acceptance Matrix Review

All 19 surface rows at revised note lines 932-954 have an executable owner and
observable assertion:

- Python/PyO3 rows test both registration modes, exact immutable Port order,
  success-only revisions, partial compound states, and public export parity.
- REST rows preserve `/catalog`, require a concrete closed `/capabilities`
  schema, exercise legal session/revision states, and distinguish malformed
  runtime output (`500`) from caller semantic invalidity (`200` or existing
  mutating-route `422`).
- Worker rows exercise ordinary, unserializable, lazy, mapped, UDF, and
  compile-capable/preview-unavailable records through spawned processes rather
  than trusting a metadata-only stub.
- Run rows test the state machine at the manager boundary before FastAPI, so a
  no-op or response-only validator cannot pass.
- Table/array rows traverse worker, Pydantic model, HTTP, raw TypeScript
  decoder, and `ResultsPanel`, including exact and one-above truncation
  boundaries.
- Frontend rows require production raw-response decoders and generated types;
  component casts and hand-written duplicate interfaces cannot satisfy them.
- Rust, lazy-DataFusion, project-schema, generated-file, and documentation
  gates guard explicitly out-of-scope surfaces.

The matrix is large but not aspirational: every current 5.1 claim can be
exercised by the named Rust/Python/backend/frontend gates. The future-v2 half
of the compatibility row is the one staged obligation noted above; its current
v1 precursor is fully testable without inventing schema v2.

## Axis Check

| Axis | Result |
| --- | --- |
| Correctness | **OK:** the design matches immutable registration snapshots, exact mixed Ports, table/array ownership vocabulary, and local spawned-worker behavior. Engine execution and checkpoint semantics are unchanged. |
| Hidden assumptions | **OK:** dynamic registration, partial compound registration, session replacement, parent/worker mismatch, table-only Studio input, and lazy-selector limits are explicit. |
| Missing edge cases | **OK:** empty/single/null-only/zero-length/Unicode/truncation, malformed network values, malformed worker values, duplicate and partial registrations are all gated. |
| Backwards compatibility | **OK with disclosed cost:** `/catalog`, project v2, checkpoints, and successful run JSON stay stable; generated validation/run clients intentionally require coordinated migration. |
| Performance | **OK:** no hot execution-path change; immutable discovery is cached per session/revision. |
| Surface and ergonomics | **OK:** parent compile capability and worker reconstruction answer distinct questions; provider options that cannot fit the safe scalar schema remain honestly non-declarative. |
| Test plan | **OK with staged v2 note:** current gates reach the real registry, manager, raw HTTP decoder, generated type, and UI boundaries and cannot be satisfied by a metadata-only stub. |
| Risk and scope | **OK:** the work is cross-layer but bounded to contract production/validation; it does not change Rust engine, project, checkpoint, runner, or provider execution behavior. |

## Data-Exposure Review

The two public projections have separate closed whitelists (revised note lines
592-618). Provider capability data is built only from explicit successful
registration records and may not reflect over callbacks. Worker classification
may serialize the private trusted record to classify transportability, but
emits only identity, reconstruction, and the fixed unavailable reason code.
Raw serialization exceptions, executable bytes, callback identity/repr,
source, import/path/environment data, options values/defaults, credentials,
and secrets remain forbidden across every capability schema version.

This is consistent with `docs/introduction.md:166-185`, which keeps executable
callbacks inside the trusted registration boundary and exposes only sorted
JSON-compatible metadata.

## Counter-Proposals

None. The revised shape now implements the prior critique's proposed split:
one parent runtime snapshot, one worker reconstruction projection, one explicit
registration-record model covering both provider modes, and a closed capability
schema whose collection contents may evolve without changing its field set.

## Questions for the Author

No blocking questions.

For implementation tracking only: will the capability-version acceptance row
be split into the current v1 decoder gate and the future v2 migration gate, so
5.1 does not claim to have executed a nonexistent v2 decoder?

## Actual Checks and Limits

- Confirmed `HEAD == origin/main ==
  2413ffdb74e2a004b3f1b541e9c2917c3b8f5ef2`.
- Read the revised note, prior critique, three restored upstream originals, and
  `docs/introduction.md` in full.
- Checked the public and private Runtime registration paths, native stub,
  NumPy/JAX helpers, public `table_matmul` builder, registration tests, worker
  selection/serialization/restore paths, worker regression tests, current
  Pydantic models/routes, raw TypeScript client, hand-written frontend types,
  ResultsPanel cast, checked-in OpenAPI, and portable Arrow vocabulary.
- `jq` confirmed the current baseline still has UDF-array `/catalog`, no
  `/capabilities`, an unconstrained validation object, and an unconstrained
  nullable `RunResponse.result`; this confirms the note is a proposal rather
  than already-implemented behavior.
- Static counts confirmed 13 `Accepted` disposition rows and 19 acceptance
  surface rows.
- No build, test, generator, benchmark, implementation, PR, status, or remote
  mutation was run or performed. That limit is intentional for a critic
  re-review; implementation gates remain unexecuted and must not be reported as
  passing.

## Remaining Risks

- Worker serialization classification proves transportability, not execution
  for arbitrary options or inputs.
- Lazy worker reconstruction still covers NumPy/JAX `expression@1`, not
  `table_matmul@1`.
- Dynamic registration can advance revision after a run or client snapshot;
  captured run payloads and client caches must remain revision-scoped.
- Generated-client source compatibility is intentionally tightened inside
  `/api/v2`; external consumers need explicit migration notice.
- The safe scalar provider options schema cannot describe nested/list-valued
  options such as `table_matmul.columns`; those providers remain usable but
  non-declarative.

These are disclosed implementation/product risks, not remaining design
blockers.
