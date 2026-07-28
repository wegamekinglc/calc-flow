# DAL-5 5.2 Current-Main Reconciliation and Hardening Addendum

Status: **approved after critic amendments; ready for incremental implementation**

Decision: **approved option C — retain PR #29 and patch only verified gaps**

Authoritative implementation baseline:

- current `origin/main` fetched on 2026-07-28:
  `f5282fdc4965ed687df736400d8c0717db51e992`;
- PR #29 squash commit:
  `7043728a5808c2e4f179a22ebc6b236c705109f1`;
- PR #29 title:
  `feat: align Python ExecutionOptions with Rust execution context`.

This addendum reconciles the historical DAL-5 5.2 specification, API note, and
critique with the implementation already merged by PR #29. It supersedes only
the conflicting ownership, naming, callback, mapped-provider, constructor, and
worker-field decisions identified below. The historical bodies remain useful
as test and algorithm references.

## 1. Authority and Scope

The implementation order of authority is:

1. the current-main public surface frozen by this addendum;
2. the behavioral hardening requirements in this addendum;
3. PR #29 behavior not explicitly changed here;
4. the historical 5.2 artifacts for non-conflicting details.

The preserved dirty worktree based on
`86ffb574d49e39fa275dd0ec39192a053283a234` is not an implementation base. Its
tests and strict-copy algorithms may be consulted, but its changes must not be
cherry-picked wholesale, rebased into the hardening branch, or used to replace
PR #29.

This hardening is intentionally incremental. It does not introduce a second
execution-options API, a compatibility alias, or a migration from the PR #29
object model.

## 2. Frozen Current-Main Public Surface

### 2.1 Python values

The authoritative public classes remain native PyO3 classes exported at the
package root:

```python
calc_flow.ExecutionOptions
calc_flow.ProviderContext
```

`calc_flow.ExecutionOptions is calc_flow._native.ExecutionOptions` remains
true. `ProviderContext` remains engine-created and publicly importable.

The constructor continues to accept positional or keyword arguments:

```python
ExecutionOptions(
    settings: Mapping[str, JSONValue] | None = None,
    deadline: datetime | None = None,
)
```

The implementation may keep its current native text signature
`(settings={}, deadline=None)`, provided omission and explicit `None` both
produce empty settings and typing documents the accepted `None`.

The hardening does not make the constructor keyword-only and does not move
`ExecutionOptions` to a pure-Python module. Native object identity,
`ProviderContext`, and the current repr/equality/hash behavior are not changed
by this patch. Changing those behaviors requires a separate compatibility
decision.

Both classes remain frozen. Their `settings` getters return a fresh exact
built-in JSON tree on every read. Their `deadline` getters return either
`None` or an exact base `datetime` whose `tzinfo is datetime.UTC`.

No Python cancellation token becomes public.

### 2.2 Execution plans

The public plan signatures remain:

```python
class ExecutionPlan:
    def execute(
        self,
        inputs: Mapping[str, Batch],
        *,
        options: ExecutionOptions | None = None,
    ) -> RunResult: ...

    def execute_async(
        self,
        inputs: Mapping[str, Batch],
        *,
        options: ExecutionOptions | None = None,
    ) -> Awaitable[RunResult]: ...
```

`options` remains keyword-only and exact-type checked. Omission, `None`, and
an empty `ExecutionOptions` retain equivalent engine settings/deadline
semantics.

### 2.3 Provider registration and callback shape

The authoritative opt-in remains `accepts_context`, not
`accepts_execution_options`:

```python
class Runtime:
    def register_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        options_schema: ProviderOptionsSchema | None = None,
        accepts_context: bool = False,
    ) -> None: ...

    def _register_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        input_ports: Sequence[tuple[str, str]],
        output_ports: Sequence[tuple[str, str]],
        options_schema: ProviderOptionsSchema | None = None,
        accepts_context: bool = False,
    ) -> None: ...
```

Both public single-array and private mapped providers retain this opt-in.
Missing or exact `False` means the legacy two-argument callback. Exact `True`
means a third engine-created `ProviderContext`. No arity inspection or
TypeError retry is added.

The third argument exposes the authoritative `RunContext.settings()` and
`RunContext.deadline()` through defensive Python copies. It is not a reused
caller `ExecutionOptions` instance.

The flag remains exact-bool validated. Native registration succeeds before
the private Python registration record is appended, preserving the rule that
rejected or duplicate registrations do not advance the capability revision.

### 2.4 Rust surface

The PR #29 Rust surface remains authoritative:

```rust
impl RunContext {
    pub const fn deadline(&self) -> Option<&DateTime<Utc>>;
}
```

The existing Rust `ExecutionOptions` fields and `CancellationToken` remain
unchanged. This hardening adds no second Rust execution context or deadline
accessor.

## 3. Strict Settings Hardening

### 3.1 Accepted data

Omitted `settings`, `{}`, and explicit `settings=None` all normalize to an
empty `calc_flow::JsonMap`.

The root and nested JSON objects accept `collections.abc.Mapping`. Arrays
accept exact `list` only. Scalar leaves accept exact built-in:

- `None`;
- `bool`;
- `int` in `[-2**63, 2**64 - 1]`;
- finite `float`;
- `str` containing no code point in `U+D800..U+DFFF`.

Object keys must be exact `str` and obey the same surrogate prohibition.
Scalar subclasses, tuples, sets, bytes, `Decimal`, enum members, datetime
values, classes, modules, path objects, arbitrary instances, and callables that
are not accepted Mapping containers are rejected. Mapping recognition takes
precedence over the arbitrary-value rejection: an object that satisfies
`Mapping` is copied only as a data container even if it is also callable, and
the callable object itself is never retained. A value is never retained merely
because it is serializable by a user hook.

The snapshot normalizes every accepted mapping/list to owned built-in JSON
data before it reaches the core. Mutating a caller container after
construction, or mutating a tree returned by `settings`, cannot mutate the
stored options or any later provider context.

### 3.2 Traversal invariants

The top-level mapping has depth zero. Depth 32 is accepted; a child at depth 33
is rejected.

Traversal tracks ancestors, not every object ever observed:

- direct and indirect cycles are rejected;
- a shared acyclic alias is accepted and copied independently;
- no custom mapping or list object is retained.

Each mapping's items are snapshotted in one traversal. Before inserting into
the native map, the implementation validates all keys and rejects a duplicate
key produced by a non-conforming custom mapping. It must not rely on
`dict(source)` or `BTreeMap::insert()` in a way that silently collapses a
duplicate.

Object keys are sorted after the snapshot for deterministic traversal and
native storage.

For each root or nested Mapping, "one traversal" means:

1. recognize the Mapping and obtain its `items()` iterable once;
2. consume that iterable once, without a preliminary `len()`, `keys()`,
   `dict(source)`, second iteration, or `__getitem__` pass added by Calc Flow;
3. validate and retain each yielded exact `(key, value)` pair only long enough
   to detect non-string/surrogate/duplicate keys;
4. sort the captured exact-string keys only after every key is validated; and
5. recursively copy the captured values, then release all source references.

The seen-key set is local to one object. Pair-unpacking failures, Mapping
recognition failures, `items()` acquisition failures, iterator failures, and
failures from Mapping hooks used by the one pass all use the same fixed
Mapping-copy error. They are not reclassified using a dynamic exception or
type name.

### 3.3 Redacted locations and errors

Errors may disclose only structural locations:

- `$` for the root;
- `.*` for any object child;
- `[N]` for an array element.

No error or chained cause contains a user key, value, repr, dynamic class/type
name, mapping exception, callback, path, URL, token, timestamp, timezone name,
or secret.

The stable errors are:

| Condition | Exception | Exact message |
| --- | --- | --- |
| root is neither Mapping nor None | `TypeError` | `settings must be a mapping or None` |
| unsupported value | `ValueError` | `settings at <path> contains a non-JSON value` |
| non-string object key | `ValueError` | `settings at <path> contains a non-string object key` |
| surrogate in key or string value | `ValueError` | `settings at <path> contains a non-portable Unicode string` |
| duplicate mapping key | `ValueError` | `settings at <path> contains duplicate object keys` |
| integer outside the portable range | `ValueError` | `settings at <path> contains an integer outside the portable JSON range` |
| NaN or infinity | `ValueError` | `settings at <path> contains a non-finite JSON number` |
| cycle | `ValueError` | `settings at <path> contains a cycle` |
| depth above 32 | `ValueError` | `settings exceeds the maximum JSON depth of 32 at <path>` |
| any custom Mapping read/iteration failure | `ValueError` | `settings could not be copied as strict JSON data` |
| unexpected normalized encoding failure | `ValueError` | `settings could not be encoded as strict JSON data` |

`<path>` is composed only from the redacted tokens above. A caught user
Mapping exception is replaced with the fixed error without a raw chained
cause.

Suppression with `raise ... from None` alone is not sufficient: it hides a
context from normal display but can leave the original exception reachable as
`__context__`. Every replacement error produced from a Mapping hook must have
`__cause__ is None` and `__context__ is None`; its arguments and traceback
must retain only the fixed Calc Flow error. Tests inspect the exception object
directly, not only `str(error)`.

This design is implemented inside the retained native
`PyExecutionOptions`; it does not require a public or private pure-Python
replacement class.

## 4. Deadline Hardening

`deadline=None` and omission mean no deadline.

Any `isinstance(value, datetime)` candidate with a valid non-`None` UTC offset
is accepted. The constructor:

1. obtains `utcoffset()` inside a redacting boundary;
2. rejects `None` as naive;
3. converts with `astimezone(datetime.UTC)` inside the same boundary;
4. reconstructs an exact base `datetime` from normalized components;
5. stores the corresponding `DateTime<Utc>` without dropping microseconds.

Positive and negative offsets therefore normalize to the same absolute UTC
instant. Offset zero remains source- and behavior-compatible. The input
datetime subclass and its `tzinfo` object are never retained.

The stable errors are:

| Condition | Exception | Exact message |
| --- | --- | --- |
| value is neither datetime nor None | `TypeError` | `deadline must be a datetime or None` |
| `utcoffset()` returns None | `ValueError` | `deadline must be timezone-aware` |
| offset lookup, UTC conversion, or range normalization fails | `ValueError` | `deadline must be a valid timezone-aware datetime representable in UTC` |

All exceptions raised by user `tzinfo`/datetime subclass behavior are replaced
without a raw chained cause. Errors do not include the object repr, offset,
timestamp, timezone name, or original exception.

As with Mapping failures, the exposed replacement must have both
`__cause__ is None` and `__context__ is None`; merely suppressing display with
`raise ... from None` does not satisfy the contract. Direct exception-object
tests use a sentinel-raising datetime/tzinfo and verify that neither the
sentinel exception nor its traceback remains reachable.

The public getter continues to return exact UTC datetime values with
microseconds `000000` through `999999` preserved.

## 5. Execution and Cancellation Semantics

### 5.1 Synchronous error precedence

`ExecutionPlan.execute()` checks for a running event loop before copying
inputs, validating `options`, or calling native code.

Inside an event loop it always raises:

```text
execute() cannot run inside an event loop; use execute_async()
```

This error wins even when `inputs` or `options` would otherwise be invalid.
Outside an event loop, exact options validation still occurs before input
copying and before native/state mutation.

### 5.2 Async cancellation linearization

The Python wrapper retains `_execute_async_cancellable()` and its native
cancellation handle. It removes the callback-maintained
`native_completed` flag.

The cancellation handler has one linearization check:

1. on entering `except asyncio.CancelledError`, evaluate `native.done()`;
2. if true, the native result or native exception wins via
   `native.result()`;
3. if false, call the run's native cancellation handle exactly once;
4. wait until the native future finishes cleanup, tolerating repeated
   cancellation of the Python task while waiting;
5. raise the originally accepted `asyncio.CancelledError`.

The handler calls `native.done()` exactly once for the decision and stores the
bool locally. No callback, `await`, event-loop turn, cancellation-handle call,
or second `done()` decision may occur between handler entry and that check.
Later `native.done()` calls are permitted only as the cleanup-loop condition
after cancellation has already won.

If cancellation wins at step 3, a later native success/error does not replace
the Python cancellation. The wrapper does not start a watcher task, a second
rollback, a second token, or another completion path.

Deterministic race tests must use explicit event-loop synchronization:

- complete the native future and cancel the Python task in the same scheduled
  callback, before native done callbacks can run; native result wins;
- repeat with a completed native exception; native exception wins;
- hold the native future incomplete at handler entry, cancel the task,
  observe one cancellation-handle call, then release cleanup; Python
  cancellation wins;
- cancel the waiting task more than once while cleanup is held and still
  observe exactly one native cancellation request and one completion.

Tests must not depend on sleeps to create the completion/cancellation ordering.
The wrapper-level race fixture records the order of `done()`, `result()`, and
`cancel()` calls. Result/exception-wins cases set the future terminal and
cancel the task from one scheduled callback. Cancellation-wins cases keep the
future non-terminal through the handler-entry check, expose an explicit
cleanup barrier, and issue repeated task cancellation only after that barrier
is observed.

## 6. Same-Plan Deadline and Lifecycle Contract

Every invocation owns an independent cloned settings/deadline snapshot and a
fresh cancellation token.

The existing core lifecycle remains authoritative:

1. native option conversion happens before acquiring the plan lock;
2. the absolute deadline is captured once and is not extended by lock wait;
3. after lock acquisition, existing recovery, input validation, state
   snapshot, and rollback-marker creation retain precedence over the first
   deadline check;
4. cancellation/deadline is checked before the first operator and immediately
   before and after each operator;
5. the post-operator cancellation/deadline check runs before an operator error
   is unwrapped;
6. failure uses the existing single rollback/recovery path.

Required deterministic same-plan tests:

- run A holds the plan; run B queues with different settings/deadline; after A
  releases, each callback observes only its own context;
- run B's deadline expires while waiting for the lock; after lifecycle setup
  and the first cancellation checkpoint, B returns the existing
  `calc_flow.CancelledError` with zero B provider calls;
- cancelling B while it only waits for the plan lock creates no transaction
  marker, does not cancel A, and does not affect run C;
- a provider returns an error after its deadline; the existing post-check
  returns `CancelledError`, not `ProviderError`;
- recovery/input/snapshot/marker failures that occur before the first deadline
  check keep their existing precedence;
- deadline, provider failure, and accepted task cancellation restore state
  once, release retained Python objects, and allow a subsequent execution with
  a fresh token.

Synchronization uses events/barriers/notifications, not timing-only sleeps,
except when advancing an absolute wall-clock deadline is the behavior under
test.

Queued and lifecycle tests must authenticate their claimed phase. A core or
binding test may manually poll run B once while run A owns the plan lock and
assert `Pending`, or use an equivalent existing test-only notification, before
expiring/cancelling B. Marker assertions belong in a co-located
`#[cfg(test)]` module that can inspect the existing private operation state, or
use existing state/restore counters that uniquely prove marker creation and
recovery. No production test hook, public lock accessor, sleep-based
"probably queued" assertion, or lifecycle reordering is allowed.

The priority fixtures separately inject recovery, input-validation, snapshot,
and marker failures and assert provider/restore call counts. Exactly-once
rollback is authenticated by restore counters plus a successful run C, not
only by the final output value.

## 7. Worker Reconstruction Hardening

The trusted worker field remains exactly:

```python
"accepts_context": bool
```

No `accepts_execution_options` alias or second field is introduced.

The field is valid only on provider registration records. For both public
single-array and mapped provider modes:

- missing means `False` for backward compatibility;
- exact `False` restores the two-argument callback;
- exact `True` restores the three-argument `ProviderContext` callback;
- any other value fails closed.

For a provider record, missing `provider_mode` means public single-array and
exact `"mapping"` means mapped. Any other `provider_mode` value fails closed
before mutation. A non-provider record must not carry either
`provider_mode` or `accepts_context`.

Mapped opt-in is intentionally retained by option C. The historical rule that
mapped `True` must fail is superseded.

Before the worker calls any runtime registration method, it preflights every
record in the entire selected registration tuple and builds an immutable
validated restoration plan. The validation order is closed:

1. missing or unknown `kind` is an unsupported registration kind;
2. a provider accepts only missing `provider_mode` (public single-array) or
   exact `"mapping"`;
3. on either provider mode, missing `accepts_context` means `False`, and a
   present value must be an exact bool;
4. a non-provider record must not carry `provider_mode` or
   `accepts_context`; and
5. every key later dereferenced by restoration must be present for the
   selected kind/mode before mutation begins.

Required-key presence is explicit:

- every provider has `kind`, `provider`, `name`, `version`, and `callback`;
- a mapped provider additionally has `input_ports` and `output_ports`;
- every scalar UDF has `kind`, `provider`, `name`, `version`, `input_types`,
  `return_type`, `volatility`, and `function`;
- `options_schema` remains optional.

Thus a malformed final record cannot allow earlier valid records to register
or advance a revision. Preflight does not promise that later native
registrations are transactional with one another; it guarantees that all
transport structure/mode/flag failures are found before the first
registration call.

Missing/unknown kind uses the existing fixed `RuntimeError`:

```text
worker received an unsupported registration kind
```

Missing required transport structure uses:

```text
worker received an invalid registration contract
```

Invalid mode, flag value, or provider-only field placement uses:

```text
worker received an invalid accepts_context registration contract
```

All three contain no field value, Python type name, registration identity,
callback, source, path, serialization detail, or chained raw exception.
Replacement exceptions have `__cause__ is None` and `__context__ is None`.

After successful preflight, reconstruction passes the validated exact bool to
the existing public or mapped registration method. It does not change
`serialized`, `lazyBuiltin`, `unavailable`, exact-reference selection, or the
set of available lazy built-ins.

The worker still invokes:

```python
plan.execute(batches)
```

Thus Studio executions still provide empty settings and no deadline.
Execution settings cannot originate in project JSON, REST requests,
capability responses, source bundles, import paths, or arbitrary installed
packages.

## 8. Explicit Non-Change Surfaces

The hardening must leave these byte/schema contracts unchanged relative to
the current-main baseline:

- `CapabilitySnapshot.schema_version == 1`;
- capability provider/operator/port/option closed fields;
- session UUID, revision, registration snapshot atomicity, and later-revision
  isolation;
- NumPy/JAX `expression@1` and mapped `table_matmul@1` capability metadata;
- `/api/v2/catalog`;
- `/api/v2/capabilities`;
- Validation, Run, table, and array closed unions;
- frontend v1 strict decoder behavior;
- project v2 schema;
- checkpoint format;
- Studio REST OpenAPI;
- generated TypeScript schema.

The private trusted worker registration record is the only Studio transport
surface modified.

## 9. Incremental Implementation Plan

Implementation is TDD and starts from
`f5282fdc4965ed687df736400d8c0717db51e992`.

### Task 1: Native settings and deadline normalization

Files:

- modify `crates/calc-flow-python/src/execution_options.rs`;
- modify `python/calc_flow/_native.pyi`;
- extend `python/tests/test_execution_options.py`.

Steps:

1. Add failing Rust/Python tests for `settings=None`, nested custom Mapping,
   duplicate keys, surrogates, exact numeric bounds, cycles, depth 32/33,
   shared aliases, redacted paths, and custom Mapping failure.
2. Implement one-pass mapping snapshots and exact redacted errors without
   changing native class ownership.
3. Add failing tests for positive/negative aware offsets, microsecond
   boundaries, naive values, invalid/raising tzinfo, and normalization
   overflow.
4. Normalize through Python UTC and reconstruct exact base values under a
   fixed redaction boundary.
5. Run binding unit tests and the focused Python execution-options suite.

### Task 2: Python execution race and error precedence

Files:

- modify `python/calc_flow/pipeline.py`;
- extend `python/tests/test_execution_options.py`.

Steps:

1. Add deterministic result-wins and exception-wins race tests with a
   complete-and-cancel same-callback barrier.
2. Add cancellation-wins, repeated-cancel, cleanup-wait, and exact-one-token
   tests.
3. Replace callback-state completion detection with the single
   `native.done()` handler-entry check.
4. Add the running-event-loop precedence test using invalid inputs/options.
5. Move the event-loop check ahead of options validation and input copying.

### Task 3: Same-plan lifecycle regression coverage

Files:

- extend `python/tests/test_execution_options.py`;
- extend `crates/calc-flow-python/src/pipeline.rs` where native lifecycle/GC
  observation is required;
- extend existing core execution tests only if Python integration cannot
  authenticate marker/rollback ordering.

Steps:

1. Add barrier-based same-plan option-isolation and lock-wait deadline tests.
2. Authenticate B as pending on the held plan lock, then add queued task
   cancellation with no marker and unaffected active/next-run assertions.
3. Add post-check cancellation-over-provider-error coverage.
4. Add recovery/input/snapshot/marker precedence fixtures with stage-specific
   call counters.
5. Assert exactly-once rollback/restore, object release, and subsequent plan
   reuse.

No production core lifecycle change is expected. If a test proves otherwise,
stop and return to critic review before changing core ordering.

### Task 4: Worker preflight

Files:

- modify `web-ui/backend/src/calc_flow_studio/run_manager.py`;
- extend `web-ui/backend/tests/test_run_manager.py`.

Steps:

1. Add public and mapped missing/false/true restoration tests.
2. Add missing/unknown kind, missing required field, non-bool,
   non-provider-field, and unknown-mode rejection tests whose recording
   runtime proves zero registration calls.
3. Add each late-invalid-record variant after an earlier valid record, proving
   the full tuple is preflighted before any runtime call or revision advance.
4. Implement a closed preflight helper returning validated modes/bools.
5. Assert each fixed redacted error plus `__cause__ is None` and
   `__context__ is None`.

### Task 5: Documentation and public typing

Files:

- modify `README.md`;
- modify `CHANGELOG.md`;
- modify `docs/api-reference.md`;
- modify `docs/introduction.md`;
- modify `docs/python-api.md`;
- modify `examples/05_async_execution.py` and/or `examples/README.md`;
- modify `python/calc_flow/_native.pyi`.

Documentation must:

- retain native `ExecutionOptions`, public `ProviderContext`,
  `accepts_context`, mapped opt-in, and positional constructor compatibility;
- state that `settings=None` is empty;
- document nested Mapping normalization and redacted validation;
- replace zero-offset-only wording with arbitrary aware-offset normalization;
- state the native-completion/cancellation precedence;
- distinguish absolute cooperative deadlines from Studio preview limits;
- avoid promising value equality/hash/repr changes.

`CHANGELOG.md` records this as hardening of the PR #29 contract, not a second
5.2 API.

### Task 6: Non-change and full gates

Required verification:

```text
uv sync --extra dev
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
(cd web-ui/backend && uv run pytest --cov=calc_flow_studio -q)
cargo test --workspace --all-targets --all-features
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps
cargo llvm-cov --workspace --all-features --fail-under-lines 90
uv run ruff check .
uv run ruff format --check python web-ui/backend/src web-ui/backend/tests
(cd web-ui && npm ci)
(cd web-ui && npm run sync:api)
(cd web-ui && npm run build)
(cd web-ui && npm test -- --reporter=dot)
(cd web-ui && npm run test:e2e)
(cd web-ui && npm audit --omit=dev)
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
uv build
python scripts/inspect_wheel.py core-wheel dist/calc_flow-*.whl
python scripts/inspect_wheel.py sdist dist/calc_flow-*.tar.gz
git diff --check
```

The isolated core-wheel smoke and Studio wheel build/smoke jobs from the
current CI workflow are required on the final head. The reviewer must confirm
all required GitHub checks for that exact head; a local editable install is
not a substitute for wheel coverage.

Generated and schema non-change proof:

```text
git diff --exit-code \
  f5282fdc4965ed687df736400d8c0717db51e992 -- \
  crates/calc-flow/src/checkpoint.rs \
  python/calc_flow/capabilities.py \
  schemas/project-v2.schema.json \
  web-ui/backend/src/calc_flow_studio/app.py \
  web-ui/backend/src/calc_flow_studio/models.py \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts \
  web-ui/src/api/decoders.ts
```

In addition to the full suites, run the checkpoint wire-format tests, the full
Python capability suite, the backend catalog/capabilities/OpenAPI tests, and
the selected-registration/transportability worker tests. These tests are the
behavioral proof for surfaces such as capability JSON and worker projection
that cannot be proven by a source-file diff alone.

Tasks 1 through 6 are the authoritative six serial stages. A later task does
not begin until the preceding task's focused RED/GREEN evidence is recorded.
The historical stage 0–6 plan is provenance only and does not add a seventh
implementation stage.

## 10. Reconciled Original 22-Row Acceptance Matrix

| Original row | PR #29 baseline | Incremental hardening required | Final authority |
| --- | --- | --- | --- |
| 1 Python value object | Satisfied: native top-level frozen object/defaults | Accept explicit `settings=None`; no ownership replacement | This addendum |
| 2 settings copy | Satisfied for dict/list input and getters | Extend the same isolation to nested custom Mapping | Historical behavior retained |
| 3 strict JSON | Partially satisfied | Complete nested Mapping, duplicate, surrogate, path, and exception-redaction contract | This addendum §3 |
| 4 deadline | Conflicts: zero offset only | Accept any valid aware offset and normalize to UTC with microseconds | This addendum §4 |
| 5 deadline errors | Partially satisfied | Redact all datetime/tzinfo failures and freeze errors | This addendum §4 |
| 6 omitted options | Satisfied | Regression test omission/None/empty parity | PR #29 plus test |
| 7 native plumbing | Satisfied | Regression only | PR #29 |
| 8 opted-in sync provider | Satisfied with `ProviderContext` | Retain current type/name; verify defensive copies | This addendum §2.3 |
| 9 opted-in async provider | Satisfied with `ProviderContext` | Retain current type/name; include race-safe execution | This addendum §5 |
| 10 legacy provider | Satisfied | Regression only | PR #29 |
| 11 concurrent isolation | Partial independent-plan coverage | Add same-plan queue and absolute lock-wait deadline tests | This addendum §6 |
| 12 expired deadline | Satisfied | Regression only | PR #29 |
| 13 cooperative deadline | Partially satisfied | Add deterministic post-check precedence coverage | This addendum §6 |
| 14 rollback | Partially satisfied | Add lifecycle precedence and exactly-once reuse coverage | This addendum §6 |
| 15 asyncio cancellation | Known completion race remains | Fix handler linearization; cover queued/repeated cancellation and cleanup | This addendum §5–6 |
| 16 registration | Satisfied | Preserve exact bool and revision rules | PR #29 |
| 17 worker reconstruction | Public path satisfied; historical mapped rule conflicts | Retain mapped opt-in; add closed whole-tuple preflight, zero-mutation late-invalid tests, and fixed redaction | This addendum §7 |
| 18 5.1 stability | Satisfied | Re-run golden/non-change tests | PR #29 plus regression |
| 19 schema stability | Satisfied | Prove current-main no-diff plus focused checkpoint/REST/capability behavior | This addendum §8–9 |
| 20 type surface | Historical ownership/name conflicts | PR #29 native classes, `ProviderContext`, and `accepts_context` are authoritative | This addendum §2 |
| 21 docs | Satisfied for PR #29 behavior | Update strict Mapping, any-aware deadline, and race semantics | This addendum §9 task 5 |
| 22 full gates | PR #29 CI passed | Run full local, wheel, security, and same-head CI matrices | This addendum §9 task 6 |

Rows 4, 17, and 20 are not implemented by reverting to the historical
contract. Their conflicting clauses are expressly superseded by option C and
the current-main decisions above.

## 11. Compatibility and Change Budget

Allowed additive/bug-fix changes:

- `settings=None` becomes accepted;
- nested Mapping inputs become accepted;
- non-zero aware deadlines become accepted and normalized;
- malformed Mapping/tzinfo errors become stable and redacted;
- cancellation/native completion follows the frozen linearization;
- malformed worker registrations fail before mutation.

Forbidden changes in this patch:

- replacing the native class with a pure-Python class;
- removing or privatizing `ProviderContext`;
- renaming or aliasing `accepts_context`;
- disabling mapped opt-in;
- making the constructor keyword-only;
- exposing cancellation tokens;
- changing capability v1 or Studio REST/project/checkpoint schemas;
- changing core lifecycle ordering merely to simplify tests;
- wholesale cherry-picking the preserved dirty 86ff-based implementation.

## 12. Critic Review Focus

The critic should treat these as the remaining review questions:

1. Can the native one-pass Mapping snapshot detect duplicate keys without
   calling user mapping hooks more than required or retaining custom objects?
2. Do all caught Mapping/datetime/tzinfo failures lose raw exception context
   while preserving the exact public error class/message?
3. Does the async `native.done()` check define a single observable
   linearization point for native result, native exception, and accepted task
   cancellation?
4. Do same-plan tests authenticate queued-before-marker versus
   marker-created cancellation without adding production-only test hooks?
5. Does worker preflight cover the entire tuple before any registration call,
   while retaining mapped `accepts_context=True`?
6. Do generated/schema diffs prove every 5.1 and Studio non-change claim?

No other architectural migration is required for this hardening.

## 13. Critic Review Record

Verdict: **Pass after blocking amendments.**

Blocking findings found and resolved in this review: **4**.
Remaining blockers: **0**.

### B1 — Raw exceptions were display-suppressed but still object-reachable

The pre-review text required no "chained cause" and referred to
`raise ... from None`, which can still leave a caller exception reachable via
`__context__`. Sections 3, 4, and 7 now require both cause and context to be
`None`, fixed-only arguments/traceback retention, and direct exception-object
tests.

### B2 — Queued/marker tests did not authenticate the lifecycle phase

Events around run A proved only that A was active; they did not prove run B
had polled pending on the plan lock or distinguish pre-marker from
post-marker cancellation. Section 6 and Task 3 now require an authenticated
pending poll or equivalent existing test-only notification, private
co-located marker observation/counters, and prohibit production hooks or
timing-only inference.

### B3 — Worker preflight did not close all late-invalid transport shapes

The initial rule covered the bool and provider mode but did not freeze
missing/unknown kind, missing later-dereferenced fields, validation order, or
the guarantee that a malformed final record causes zero earlier mutations.
Section 7 and Task 4 now define a whole-tuple validated restoration plan,
fixed redacted errors, and late-invalid zero-call/revision tests.

### B4 — Non-change and full gates were incomplete

The initial command list omitted wheel/sdist inspection and current
supply-chain gates, while the diff proof covered project/OpenAPI/generated
types but not checkpoint, capability definitions, REST models, or the
frontend decoder. Task 6 now names those gates and adds focused behavioral
proof for capability/worker surfaces that legitimately share a modified
source file.

Implementation guardrails after review:

1. preserve PR #29's native classes, positional constructor,
   `ProviderContext`, `accepts_context`, and mapped opt-in;
2. perform one-pass Mapping snapshots and clear raw exception objects, not
   merely their display;
3. use exactly one handler-entry `native.done()` decision before cancellation
   is accepted;
4. authenticate lock/marker phases with existing test-only observability and
   do not change production lifecycle ordering for tests;
5. preflight the full worker tuple before any runtime mutation or revision;
6. prove 5.1/project/checkpoint/REST/OpenAPI/generated-client non-change
   against the current-main baseline and focused tests; and
7. complete all six serial TDD stages and the current full CI/wheel/security
   matrix before reviewer sign-off.
