# Array Benchmark Regression Implementation Plan

> **Historical status:** Implemented and merged in PR #13. Unchecked boxes
> preserve the original execution plan; final Phase 1 and Phase 2 outcomes are
> recorded in the [Phase 1](../handoffs/2026-07-16-array-benchmark-phase1.md)
> and [Phase 2](../handoffs/2026-07-16-array-benchmark-phase2.md) evidence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NumPy/JAX benchmark comparisons trustworthy, isolate the regressed layer, and recover measured Python-provider overhead with a bounded expression cache and a one-copy immutable NumPy snapshot.

**Architecture:** Keep the public Rust-v2 execution and provider contracts unchanged. Add a versioned array-benchmark identity at the Python benchmark boundary, enforce it in a pure TypeScript comparator, split array timing into kernel/provider/plan/ownership scopes, then make two private Python optimizations under safety tests. Stop at a same-host Phase 1 evidence gate; collect Rust-core measurements only if that gate misses, and require a new approved design before changing core execution semantics or Python GC ownership.

**Tech Stack:** Python 3.13+, NumPy, JAX/JAXlib CPU, PyO3 0.28.3, Rust 1.88.0, Apache DataFusion 54.0.0, pytest, pytest-benchmark, py-cpuinfo, psutil, React, TypeScript, Vitest, Criterion, uv, and Maturin.

## Global Constraints

- Treat `docs/superpowers/specs/2026-07-16-array-benchmark-regression-design.md` as the approved source of truth.
- Use `b42b687c8021291bcb537a57fded7a40f7f8477d` as the exact performance baseline.
- Preserve the public project-v2 format, provider identity, `Batch`, `Runtime`, `ExecutionPlan`, transaction, rollback, cancellation, checkpoint, runner, metadata, and error contracts.
- Do not reintroduce Python v1, add a compatibility shim, use `eval`/`exec`, or serialize ASTs or executable callables.
- Keep the expression cache private and bounded to 256 successful exact-string keys. Do not cache invalid expressions or skip per-operation result validation.
- Keep NumPy inputs caller-independent and irrevocably non-writeable through the complete reachable array base chain.
- Keep JAX inputs and outputs as `jax.Array` values and synchronize every timed JAX result.
- Do not assert timing thresholds in unit tests or CI. CI remains informational until 20 compatible main-branch samples exist.
- Treat a provider-preparation protocol as a separate design boundary: a prepared Python callable introduces plan-specific GC-root ownership that the current runtime-root model does not expose safely.
- Treat any Rust execution-core optimization as a separate design boundary. This plan may add measurements, not change core semantics.
- Preserve unrelated user changes and stage only the files named by the active task.
- Start every behavior change with a focused failing test, record the expected failure, and end every task with focused verification and an intentional imperative commit under 72 characters.
- Keep generated build output under `target/` and remove any `python/calc_flow/_native*.so` before committing.

---

## Target File Map

### Benchmark contract and comparison

- `benchmarks/support.py` — contract-v2 identity, canonical fingerprints, machine/dependency metadata, and immutable benchmark recording.
- `tests/test_benchmark_support.py` — deterministic identity, normalization, hashing, backend metadata, and input-immutability tests.
- `pyproject.toml` — direct `py-cpuinfo` dependency in the `benchmark` and `dev` extras.
- `web-ui/src/components/benchmarkComparison.ts` — pure report parser, compatibility engine, and performance classification.
- `web-ui/src/components/benchmarkComparison.test.ts` — parser and compatibility matrix tests.
- `web-ui/src/components/BenchmarkComparison.tsx` — file loading and structured compatibility rendering only.
- `web-ui/src/components/BenchmarkComparison.test.tsx` — user-visible unverified/incompatible/compatible states.

### Layered array measurements

- `benchmarks/array_support.py` — immutable workload definitions, input factories, synchronization, dtype observation, and recording helpers.
- `benchmarks/test_array_kernel.py` — raw NumPy/JAX backend controls.
- `benchmarks/test_array_provider.py` — bounded evaluator plus Python/native batch boundary.
- `benchmarks/test_array_plan.py` — public compiled-plan end-to-end execution.
- `benchmarks/test_array_ownership.py` — `Batch.from_array` construction by scale.
- `benchmarks/test_array.py` — delete after its scenarios move to the four scoped modules.
- `benchmarks/README.md` — contract, scope, compatibility, and evidence instructions.

### Semantics-preserving optimizations

- `python/calc_flow/array.py` — bounded successful-expression cache and one-copy NumPy ownership.
- `python/tests/test_array.py` — cache, expression safety, layout, dtype, ownership, metadata, GC, rollback, and reuse tests.

### Evidence and conditional diagnostics

- `docs/superpowers/handoffs/2026-07-16-array-benchmark-phase1.md` — exact A/B commands, identities, per-case results, geometric mean, ownership results, and gate decision.
- `crates/calc-flow/benches/core.rs` — conditional Criterion measurements for DataFusion session setup and native passthrough execution; no runtime behavior changes.
- `docs/superpowers/handoffs/2026-07-16-array-benchmark-phase2.md` — conditional component timing and stop/go decision.

---

### Task 1: Add the Versioned Array Benchmark Identity

**Files:**
- Modify: `benchmarks/support.py`
- Modify: `tests/test_benchmark_support.py`
- Modify: `pyproject.toml`

**Interfaces:**

```python
ArrayBenchmarkScope = Literal[
    "backend_kernel",
    "provider_boundary",
    "plan_end_to_end",
    "batch_ownership",
]


@dataclass(frozen=True, slots=True)
class ArrayBenchmarkRecord:
    scenario: str
    scope: ArrayBenchmarkScope
    backend: Literal["numpy", "jax"]
    expression: str
    input_dtype: str
    output_dtype: str
    input_rows: int
    output_rows: int


def record_array_benchmark(
    benchmark: BenchmarkFixture,
    record: ArrayBenchmarkRecord,
) -> None: ...
```

The recorder adds contract-v2 fields without changing generic `record_benchmark` used by current DataFusion/runtime benchmarks.

- [ ] **Step 1: Add the owned dependency**

Add `"py-cpuinfo>=9"` to both `project.optional-dependencies.benchmark` and `project.optional-dependencies.dev`. Do not commit `uv.lock`; it is intentionally ignored.

- [ ] **Step 2: Write failing deterministic metadata tests**

Extend `tests/test_benchmark_support.py`. Monkeypatch private collectors rather than asserting the active host:

```python
from benchmarks import support
from benchmarks.support import ArrayBenchmarkRecord, record_array_benchmark


def test_record_array_benchmark_emits_complete_contract_v2_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support,
        "_machine_identity",
        lambda: {
            "operating_system": "linux",
            "architecture": "x86_64",
            "cpu_brand": "example cpu",
            "logical_cpu_count": 8,
            "python_implementation": "cpython",
        },
    )
    monkeypatch.setattr(
        support,
        "_dependency_identity",
        lambda backend: {
            "python_version": "3.13.9",
            "numpy_version": "2.5.1",
            **(
                {"jax_version": "0.10.2", "jaxlib_version": "0.10.2"}
                if backend == "jax"
                else {}
            ),
        },
    )
    monkeypatch.setattr(
        support,
        "_backend_configuration",
        lambda backend: (
            {"jax_platform": "cpu", "jax_enable_x64": False}
            if backend == "jax"
            else {}
        ),
    )
    benchmark = SimpleNamespace(extra_info={"retained": True})

    record_array_benchmark(
        benchmark,
        ArrayBenchmarkRecord(
            scenario="array_mean",
            scope="plan_end_to_end",
            backend="jax",
            expression="mean(x)",
            input_dtype="float32",
            output_dtype="float32",
            input_rows=1_000,
            output_rows=1,
        ),
    )

    assert benchmark.extra_info["benchmark_contract_version"] == 2
    assert benchmark.extra_info["workload_version"] == 1
    assert benchmark.extra_info["machine_identity"]["cpu_brand"] == "example cpu"
    assert benchmark.extra_info["jax_platform"] == "cpu"
    assert benchmark.extra_info["jax_enable_x64"] is False
    for name in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        assert len(benchmark.extra_info[name]) == 64
        int(benchmark.extra_info[name], 16)
```

Also prove:

- CPU brands normalize with `" ".join(value.casefold().split())`;
- machine identity excludes hostname, clock frequency, PID, RSS, and total memory;
- hashes are lower-case SHA-256 over compact, sorted-key canonical JSON;
- NumPy dependency identity excludes JAX/JAXlib while JAX includes them;
- NumPy configuration is `{}` and omits flat JAX fields;
- recording replaces `extra_info` without mutating its previous mapping;
- changing scope, dtype, expression, scale, or JAX configuration changes the workload fingerprint;
- changing RSS changes no compatibility fingerprint;
- unsupported scope/backend and empty scenario/expression fail before metadata is attached.

- [ ] **Step 3: Run RED**

```bash
uv run pytest tests/test_benchmark_support.py -q
```

Expected: collection fails because `ArrayBenchmarkRecord` and `record_array_benchmark` do not exist.

- [ ] **Step 4: Implement identities and fingerprints**

Add `hashlib`, `json`, `platform`, `cpuinfo.get_cpu_info`, `importlib.metadata.version`, and `Literal`. Keep collectors private.

```python
BENCHMARK_CONTRACT_VERSION = 2
ARRAY_WORKLOAD_VERSION = 1


def _canonical_fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_cpu_brand(value: str) -> str:
    return " ".join(value.casefold().split())


def _machine_identity() -> dict[str, object]:
    raw = get_cpu_info()
    brand = str(raw.get("brand_raw") or platform.processor() or platform.machine())
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None:
        raise RuntimeError("logical CPU count is unavailable")
    return {
        "operating_system": platform.system().casefold(),
        "architecture": platform.machine().casefold(),
        "cpu_brand": _normalize_cpu_brand(brand),
        "logical_cpu_count": logical_cpu_count,
        "python_implementation": platform.python_implementation().casefold(),
    }
```

Build dependency identity from `platform.python_version()` and `importlib.metadata.version(...)`. Build JAX configuration from `jax.default_backend()` and `bool(jax.config.jax_enable_x64)`. Do not import JAX for NumPy records.

Hash an explicit workload document so transient fields cannot leak:

```python
workload_identity = {
    "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
    "scenario": record.scenario,
    "scope": record.scope,
    "workload_version": ARRAY_WORKLOAD_VERSION,
    "backend": record.backend,
    "scale": scale.name,
    "table_rows": scale.table_rows,
    "array_elements": scale.array_elements,
    "matrix_dimension": scale.matrix_dimension,
    "input_rows": record.input_rows,
    "output_rows": record.output_rows,
    "expression": record.expression,
    "input_dtype": record.input_dtype,
    "output_dtype": record.output_dtype,
    "backend_configuration": backend_configuration,
}
```

Call `record_benchmark(...)` first, then replace `benchmark.extra_info` once more with the contract, all identity documents/hashes, and flat dependency/backend fields. Preserve prior metadata and RSS.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv sync --extra dev
uv run pytest tests/test_benchmark_support.py -q
uv run ruff check benchmarks/support.py tests/test_benchmark_support.py
uv run ruff format --check benchmarks/support.py tests/test_benchmark_support.py
git diff --check
git add pyproject.toml benchmarks/support.py tests/test_benchmark_support.py
git diff --cached --check
git commit -m "perf: add array benchmark identities"
```

---

### Task 2: Enforce Compatibility Before Classifying Performance

**Files:**
- Create: `web-ui/src/components/benchmarkComparison.ts`
- Create: `web-ui/src/components/benchmarkComparison.test.ts`
- Modify: `web-ui/src/components/BenchmarkComparison.tsx`
- Modify: `web-ui/src/components/BenchmarkComparison.test.tsx`

**Interfaces:**

```typescript
export type CompatibilityStatus = 'compatible' | 'incompatible' | 'unverified';

export interface BenchmarkCompatibilityIssue {
  code:
    | 'missing_contract_metadata'
    | 'contract_version_mismatch'
    | 'machine_mismatch'
    | 'dependency_mismatch'
    | 'scale_mismatch'
    | 'scope_mismatch'
    | 'workload_mismatch'
    | 'dtype_mismatch'
    | 'backend_configuration_mismatch';
  field: string;
  baseline: unknown;
  current: unknown;
}

export interface BenchmarkComparisonResult {
  status: CompatibilityStatus;
  rows: BenchmarkComparisonRow[];
  issues: BenchmarkCompatibilityIssue[];
}
```

- [ ] **Step 1: Write pure comparator RED tests**

Move current parser/statistics tests into `benchmarkComparison.test.ts` and use contract-v2 fixtures with fixed 64-character fingerprints. Add table-driven tests for all nine issue codes, changing exactly one field per case and requiring no rows.

Also prove:

- equal v2 identities classify stable/noisy/regression/improvement;
- reports containing only legacy entries are `unverified` with no rows;
- mixed full-suite reports compare matched v2 entries and ignore unrelated legacy entries;
- a v2 case cannot fall back to a legacy case with the same scenario;
- malformed contract-v2 types are hard parse errors;
- duplicate `(scenario, backend, scope)` identities and duplicate workload fingerprints are hard parse errors;
- non-finite/non-positive means, negative standard deviation, and invalid rounds remain errors;
- issue order follows the union declaration and rows sort by descending delta.

- [ ] **Step 2: Write component RED tests**

Keep `BenchmarkComparison.test.tsx` for DOM behavior. Upload `File` fixtures and prove legacy reports show “Unverified”, machine mismatch shows `machine_mismatch` and no table, compatible reports show the table, malformed JSON shows the parser error, and replacing a file clears stale results.

Run:

```bash
cd web-ui
npm test -- src/components/benchmarkComparison.test.ts src/components/BenchmarkComparison.test.tsx
```

Expected: FAIL because the pure module and structured result do not exist.

- [ ] **Step 3: Extract strict parsing/comparison**

Move types, `parseBenchmarkReport`, and pure comparison helpers to `benchmarkComparison.ts`. Keep legacy entries structurally parseable. Treat a missing contract version as legacy; accept a positive integer version long enough to report `contract_version_mismatch`, but apply the full field validation below to version 2. For contract version 2 require every contract field, supported scope/backend, non-empty text, identity objects, lower-case 64-hex fingerprints, and JAX platform/x64 for JAX only. Reject JAX-only fields on NumPy entries and reject duplicates during parse.

Match on `(scenario, backend, scope)`. If the exact scope is absent but a unique same-scenario/backend candidate exists, emit `scope_mismatch`. Compare:

1. contract version;
2. machine fingerprint;
3. dependency fingerprint;
4. scale and all scale dimensions;
5. scope;
6. input/output dtype;
7. JAX platform/x64;
8. workload fingerprint.

Gather deterministic issues, but return no rows if a matched v2 pair is incompatible. A pure legacy comparison is unverified. Unrelated legacy entries in mixed reports are not evidence for or against a v2 pair.

- [ ] **Step 4: Render structured results**

`BenchmarkComparison.tsx` owns only React state/file loading/rendering. Remove the global scales set.

- `compatible`: existing timing table.
- `incompatible`: invalid banner plus issue code, field, baseline, and current.
- `unverified`: informational banner stating no classification was made.
- no matching v2 work: explicit no-match message.

Never render stable/improvement/regression outside `compatible`.

- [ ] **Step 5: Run GREEN and commit**

```bash
cd web-ui
npm test -- src/components/benchmarkComparison.test.ts src/components/BenchmarkComparison.test.tsx
npm run build
cd ..
git diff --check
git add web-ui/src/components/benchmarkComparison.ts web-ui/src/components/benchmarkComparison.test.ts web-ui/src/components/BenchmarkComparison.tsx web-ui/src/components/BenchmarkComparison.test.tsx
git diff --cached --check
git commit -m "perf: reject incompatible benchmark reports"
```

---

### Task 3: Split Array Benchmarks by Measurement Scope

**Files:**
- Create: `benchmarks/array_support.py`
- Create: `benchmarks/test_array_kernel.py`
- Create: `benchmarks/test_array_provider.py`
- Create: `benchmarks/test_array_plan.py`
- Create: `benchmarks/test_array_ownership.py`
- Delete: `benchmarks/test_array.py`
- Modify: `benchmarks/README.md`
- Modify: `tests/test_benchmark_support.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class ArrayWorkload:
    scenario: str
    expression: str
    operation: Callable[[object, object], object]
    input_factory: Callable[[BenchmarkScale], np.ndarray[Any, Any]]
```

`ARRAY_WORKLOADS` contains elementwise `(x * x + 1) ** 0.5`, `mean(x)`, `x @ x`, and transpose/reshape. It never uses `eval`.

- [ ] **Step 1: Write workload RED tests**

Assert unique scenario/expression pairs, the exact four scenarios, deterministic overhead shapes, and recording of all four scopes. Run:

```bash
uv run pytest tests/test_benchmark_support.py -q
```

Expected: FAIL because `benchmarks.array_support` does not exist.

- [ ] **Step 2: Create immutable workload/timing helpers**

In `array_support.py`:

- create NumPy input once outside timing;
- convert to JAX before timing JAX kernels;
- expose `synchronize(value)` via `block_until_ready`;
- derive dtype from observed values;
- expose `runtime_for`, `batch_for`, and `plan_for`;
- warm once, synchronize, record metadata before `benchmark(...)`, and synchronize returned JAX results;
- implement raw operations with explicit callables such as `namespace.sqrt`, `namespace.mean`, `operator.matmul`, and `namespace.reshape(namespace.transpose(...))`.

- [ ] **Step 3: Add `backend_kernel` controls**

Parameterize NumPy/JAX and all workloads. Time only the upstream operation on an already-created backend value. Assert observed dtype/shape after timing.

- [ ] **Step 4: Add `provider_boundary` measurements**

Construct private `_ArrayProvider` directly with the backend namespace and a native input batch. Time:

```python
provider(input_batch, {"expression": workload.expression})
```

This intentionally includes bounded evaluation, intermediate validation, Python/native conversion, and output ownership; it excludes graph and run-scoped DataFusion setup.

- [ ] **Step 5: Add `plan_end_to_end` measurements**

Move the public `Runtime`/`PipelineBuilder` path unchanged. Compile and create input outside timing; time only:

```python
plan.execute({"input": values}).outputs["output"]
```

Elementwise, mean, and matrix multiplication for both backends form the six acceptance cases. Keep transpose/reshape diagnostic but exclude it from the geometric mean.

- [ ] **Step 6: Add `batch_ownership` measurements**

Time `Batch.from_array(source, backend=backend)` for both backends and selected scale. Source creation stays outside timing. Record `expression="Batch.from_array"` as the stable operation label and record actual dtypes. Assert final NumPy caller isolation and base-chain non-writeability outside timing.

- [ ] **Step 7: Delete ambiguity and document scopes**

Delete `benchmarks/test_array.py`. Update README with inside/outside timing boundaries, contract fields, legacy-unverified semantics, same-machine/dependency/workload requirement, JAX sync, six-case gate, standard/nightly ownership checks, and informational-CI policy.

- [ ] **Step 8: Run GREEN and inspect JSON**

```bash
CALC_FLOW_BENCHMARK_SCALE=overhead   JAX_PLATFORMS=cpu   uv run --extra benchmark pytest   benchmarks/test_array_kernel.py   benchmarks/test_array_provider.py   benchmarks/test_array_plan.py   benchmarks/test_array_ownership.py   -q --benchmark-only   --benchmark-json=target/benchmark-results/array-overhead.json
uv run python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("target/benchmark-results/array-overhead.json").read_text())
required = {
    "benchmark_contract_version",
    "scenario",
    "scope",
    "workload_version",
    "backend",
    "scale",
    "input_dtype",
    "output_dtype",
    "machine_fingerprint",
    "dependency_fingerprint",
    "workload_fingerprint",
}
assert report["benchmarks"]
for entry in report["benchmarks"]:
    missing = required - entry["extra_info"].keys()
    assert not missing, (entry["fullname"], missing)
PY
uv run pytest tests/test_benchmark_support.py -q
uv run ruff check benchmarks tests/test_benchmark_support.py
uv run ruff format --check benchmarks tests/test_benchmark_support.py
git diff --check
git add benchmarks tests/test_benchmark_support.py
git diff --cached --check
git commit -m "perf: split array benchmark layers"
```

---

### Task 4: Cache Successful Bounded Expressions

**Files:**
- Modify: `python/calc_flow/array.py`
- Modify: `python/tests/test_array.py`

- [ ] **Step 1: Write focused cache RED tests**

Import `calc_flow.array as array_module` and clear the private cache before/after each cache-specific test.

```python
def test_array_expression_cache_reuses_successful_exact_strings() -> None:
    array_module._parse_valid_expression.cache_clear()

    first = array_module._parse_expression("x + 1")
    second = array_module._parse_expression("x + 1")

    assert first is second
    assert array_module._parse_valid_expression.cache_info().hits == 1
    assert array_module._parse_valid_expression.cache_info().misses == 1


def test_array_expression_cache_is_bounded() -> None:
    array_module._parse_valid_expression.cache_clear()
    for value in range(257):
        array_module._parse_expression(f"x + {value}")
    before = array_module._parse_valid_expression.cache_info()

    array_module._parse_expression("x + 0")
    after = array_module._parse_valid_expression.cache_info()

    assert before.maxsize == 256
    assert before.currsize == 256
    assert after.misses == before.misses + 1
```

For invalid expressions, monkeypatch `ast.parse`, call the same invalid string twice, and assert two parser calls and identical existing errors. Execute a cached multi-step expression twice while making an intermediate backend function return object dtype on the second run; the second run must still fail through the existing provider category. Keep unsafe syntax/bounds tests unchanged.

- [ ] **Step 2: Run RED**

```bash
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q -k "expression_cache"
```

Expected: FAIL because `_parse_valid_expression` does not exist.

- [ ] **Step 3: Split object validation from cached parsing**

```python
def _parse_expression(expression: object) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise _array_error("expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise _array_error(
            f"expression length limit is {_MAX_EXPRESSION_LENGTH} characters"
        )
    return _parse_valid_expression(expression)


@lru_cache(maxsize=256)
def _parse_valid_expression(expression: str) -> ast.Expression:
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise _array_error("syntax is invalid") from error
    if not isinstance(parsed, ast.Expression):
        raise _array_error("syntax is invalid")
    nodes = list(ast.walk(parsed))
    if len(nodes) > _MAX_AST_NODES:
        raise _array_error(f"node limit is {_MAX_AST_NODES}")
    if _ast_depth(parsed) > _MAX_AST_DEPTH:
        raise _array_error(f"depth limit is {_MAX_AST_DEPTH}")
    _validate_node(parsed.body)
    return parsed
```

Import `lru_cache`. Do not export either parser. Do not cache options, results, exceptions, or evaluator validation.

- [ ] **Step 4: Run GREEN and midpoint measurement**

```bash
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check python/calc_flow/array.py python/tests/test_array.py
uv run ruff format --check python/calc_flow/array.py python/tests/test_array.py
CALC_FLOW_BENCHMARK_SCALE=overhead   JAX_PLATFORMS=cpu   uv run --extra benchmark pytest   benchmarks/test_array_provider.py benchmarks/test_array_plan.py   -q --benchmark-only   --benchmark-json=target/benchmark-results/array-cache-midpoint.json
git diff --check
```

- [ ] **Step 5: Apply the preparation design boundary**

Measure option serialization/conversion plus dynamic dispatch using a local diagnostic probe outside production timing tests. Below either 10 microseconds or 10% of compatible plan mean, record “retain dynamic callback boundary” and continue.

At or above both 10 microseconds and 10%, stop and write a separate design for plan-owned prepared Python roots. It must cover factory-to-plan root transfer, `PyExecutionPlan.__traverse__`/`__clear__`, async/runner ownership, compile-time errors, legacy callbacks, and cyclic-GC tests. Do not implement `PreparableProvider` in this plan.

- [ ] **Step 6: Commit**

```bash
git add python/calc_flow/array.py python/tests/test_array.py
git diff --cached --check
git commit -m "perf: cache bounded array expressions"
```

---

### Task 5: Reduce NumPy Ownership to One Payload Copy

**Files:**
- Modify: `python/calc_flow/array.py`
- Modify: `python/tests/test_array.py`

- [ ] **Step 1: Add layout and ownership RED tests**

Add immutable factories for C-contiguous, non-contiguous slice, transpose, Fortran order, negative stride, zero-dimensional scalar, shaped empty, ndarray subclass, nested view/base chain, and all 13 supported bool/integer/float/complex dtypes.

For each, capture `expected = np.asarray(source).copy(order="C")`, construct `Batch.from_array`, mutate original storage when possible, and assert exact dtype, shape, values, `num_rows`, and metadata. Walk the returned ndarray base chain; every array is non-writeable and every `setflags(write=True)` raises `ValueError`.

Add a boundary test that creates its source, monkeypatches `np.array` to raise, calls private `_owned_numpy(source)`, and proves the old intermediate `np.array(copy=True)` allocation is gone. Retain unsupported/object/non-native-endian and magic-dispatch tests.

- [ ] **Step 2: Run RED**

```bash
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q   -k "layout or ownership or reachable or intermediate_array_copy"
```

Expected: the boundary test fails because current `_owned_numpy` calls `np.array`.

- [ ] **Step 3: Implement one-copy ownership**

```python
def _owned_numpy(value: object) -> object:
    import numpy as np

    array = np.asarray(value)
    _validate_numpy_dtype(array.dtype)
    immutable_bytes = array.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=array.dtype).reshape(array.shape)
```

Do not special-case read-only, contiguous, empty, or subclass inputs. `tobytes(order="C")` is the one payload copy and `frombuffer(bytes)` gives an immutable ultimate owner.

- [ ] **Step 4: Run GREEN and candidate-only diagnostics**

```bash
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check python/calc_flow/array.py python/tests/test_array.py
uv run ruff format --check python/calc_flow/array.py python/tests/test_array.py
for scale in overhead standard nightly; do
  CALC_FLOW_BENCHMARK_SCALE="$scale"     JAX_PLATFORMS=cpu     uv run --extra benchmark pytest     benchmarks/test_array_ownership.py benchmarks/test_array_plan.py     -q --benchmark-only     --benchmark-json="target/benchmark-results/array-ownership-${scale}.json"
done
git diff --check
```

Expected: semantics/lint pass and reports have contract-v2 metadata. Do not claim acceptance from candidate-only runs.

- [ ] **Step 5: Commit**

```bash
git add python/calc_flow/array.py python/tests/test_array.py
git diff --cached --check
git commit -m "perf: remove duplicate NumPy array copy"
```

---

### Task 6: Run the Same-Host Phase 1 Evidence Gate

**Files:**
- Create: `docs/superpowers/handoffs/2026-07-16-array-benchmark-phase1.md`

This task changes no runtime source. It compares the exact baseline engine with the candidate using the candidate harness and one frozen dependency set.

- [ ] **Step 1: Create detached worktrees and shared harness**

```bash
git worktree add --detach /tmp/calc-flow-array-baseline b42b687c8021291bcb537a57fded7a40f7f8477d
git worktree add --detach /tmp/calc-flow-array-candidate HEAD
rm -rf /tmp/calc-flow-array-harness
mkdir -p /tmp/calc-flow-array-harness
cp -R /tmp/calc-flow-array-candidate/benchmarks /tmp/calc-flow-array-harness/benchmarks
```

The baseline engine stays at the exact SHA; only the external candidate benchmark harness is shared.

- [ ] **Step 2: Freeze dependencies and build isolated environments**

```bash
cd /tmp/calc-flow-array-candidate
uv lock
uv export --frozen --extra benchmark --no-dev --no-emit-project --output-file /tmp/calc-flow-array-requirements.txt
uv venv --python 3.13 /tmp/calc-flow-array-baseline/.venv
uv venv --python 3.13 /tmp/calc-flow-array-candidate/.venv
uv pip sync --python /tmp/calc-flow-array-baseline/.venv/bin/python /tmp/calc-flow-array-requirements.txt
uv pip sync --python /tmp/calc-flow-array-candidate/.venv/bin/python /tmp/calc-flow-array-requirements.txt
VIRTUAL_ENV=/tmp/calc-flow-array-baseline/.venv uvx --directory /tmp/calc-flow-array-baseline --from maturin==1.14.1 maturin develop --release
VIRTUAL_ENV=/tmp/calc-flow-array-candidate/.venv uvx --directory /tmp/calc-flow-array-candidate --from maturin==1.14.1 maturin develop --release
```

Confirm both interpreters report identical Python, NumPy, JAX, and JAXlib versions. The calc-flow source SHA is the only intended difference.

- [ ] **Step 3: Alternate five overhead runs**

Run from the external harness so neither interpreter imports checkout-local Python source:

```bash
mkdir -p /tmp/calc-flow-array-results
cd /tmp/calc-flow-array-harness
for round in 1 2 3 4 5; do
  if [ $((round % 2)) -eq 1 ]; then
    revisions="baseline candidate"
  else
    revisions="candidate baseline"
  fi
  for revision in $revisions; do
    CALC_FLOW_BENCHMARK_SCALE=overhead       JAX_PLATFORMS=cpu       "/tmp/calc-flow-array-${revision}/.venv/bin/python" -m pytest       benchmarks/test_array_kernel.py       benchmarks/test_array_provider.py       benchmarks/test_array_plan.py       -q --benchmark-only       --benchmark-json="/tmp/calc-flow-array-results/${revision}-overhead-${round}.json"
  done
done
```

Do not run unrelated work between alternating pairs. Record host power mode. Every compared pair must have equal machine, dependency, and workload fingerprints.

- [ ] **Step 4: Collect standard/nightly ownership evidence**

```bash
cd /tmp/calc-flow-array-harness
for scale in standard nightly; do
  for round in 1 2 3; do
    for revision in baseline candidate; do
      CALC_FLOW_BENCHMARK_SCALE="$scale"         JAX_PLATFORMS=cpu         "/tmp/calc-flow-array-${revision}/.venv/bin/python" -m pytest         benchmarks/test_array_ownership.py         -q --benchmark-only         --benchmark-json="/tmp/calc-flow-array-results/${revision}-${scale}-${round}.json"
    done
  done
done
```

- [ ] **Step 5: Compute the exact gate**

For each logical case, report the median of five report means (three for ownership), median CoV, absolute microseconds, ratio, and percent change. Reject differing identities. Rerun instead of aggregating a case when baseline or candidate median CoV exceeds 5%.

```python
improvement = 1.0 - math.prod(
    candidate_mean[key] / baseline_mean[key]
    for key in (
        ("array_elementwise", "numpy"),
        ("array_mean", "numpy"),
        ("array_matrix_multiplication", "numpy"),
        ("array_elementwise", "jax"),
        ("array_mean", "jax"),
        ("array_matrix_multiplication", "jax"),
    )
) ** (1.0 / 6.0)
```

Pass only if:

- no compatible scenario is more than 5% slower;
- six-case `plan_end_to_end` geometric-mean improvement is at least 20%;
- NumPy `batch_ownership` improves at least 30% at both 100,000 and 1,000,000 elements;
- `backend_kernel` controls show no material change outside noise;
- every array entry has complete contract-v2 metadata.

- [ ] **Step 6: Write and commit the evidence**

The handoff records both SHAs, commands, dependency versions, machine/dependency fingerprints, a table per scope (mean, CoV, ratio, delta), geometric mean, ownership deltas, option-conversion midpoint, every predicate, and exactly one decision: `Phase 1 accepted` or `Phase 2 measurements required`. Never claim on an incompatible/noisy case.

```bash
git add docs/superpowers/handoffs/2026-07-16-array-benchmark-phase1.md
git diff --cached --check
git commit -m "perf: record array benchmark evidence"
```

If Phase 1 passes, skip Task 7. If it misses, Task 7 is mandatory.

---

### Task 7: Measure Remaining Rust-Core Cost Without Changing Semantics

**Condition:** Run only when the Phase 1 handoff says `Phase 2 measurements required`.

**Files:**
- Modify: `crates/calc-flow/benches/core.rs`
- Create: `docs/superpowers/handoffs/2026-07-16-array-benchmark-phase2.md`

- [ ] **Step 1: Measure DataFusion setup**

Import `DataFusionConfig`/`DataFusionRuntime` and add:

```rust
fn create_datafusion_runtime(c: &mut Criterion) {
    let config = DataFusionConfig::default();
    c.bench_function("execute/datafusion_runtime_new", |b| {
        b.iter(|| black_box(DataFusionRuntime::new(black_box(config)).unwrap()));
    });
}
```

Add a second case that constructs a runtime and calls `register_udfs` with a stable empty snapshot/reference list. It measures setup without a production fast path.

- [ ] **Step 2: Measure native external passthrough**

Define a benchmark-local immutable `PassthroughOperator` with one required external input/output. `process` returns an input clone. Compile once and time public `ExecutionPlan::execute` with default options. Use a benchmark-only `ExternalPayload` with length 1,000 and case name `execute/external_passthrough_1000_rows`.

This includes transaction capture, graph routing, validation, run-scoped DataFusion setup, node timing, and result assembly, excluding Python.

- [ ] **Step 3: Run measurements**

```bash
cargo bench -p calc-flow --bench core
CALC_FLOW_BENCHMARK_SCALE=overhead   JAX_PLATFORMS=cpu   uv run --extra benchmark pytest   benchmarks/test_array_provider.py benchmarks/test_array_plan.py   -q --benchmark-only   --benchmark-json=target/benchmark-results/array-phase2.json
```

Use Criterion plus provider/plan deltas for attribution. Never subtract incompatible environments.

- [ ] **Step 4: Apply the core design gate**

The Phase 2 handoff records absolute component times and percentages of remaining compatible plan overhead.

- No component at least 10 microseconds and at least 20%: record `No core redesign justified` and stop optimization.
- Any component meeting both: record `New core design required`, name it, and stop for a separate reviewed design covering semantics, concurrency, cancellation, rollback, metrics, checkpointing, and API impact.

Do not add cached DataFusion sessions, unchecked execution, skipped transactions, or alternate callback routes.

- [ ] **Step 5: Verify and commit measurements**

```bash
cargo fmt --all --check
cargo clippy -p calc-flow --benches --all-features -- -D warnings
cargo test -p calc-flow --all-targets --all-features
git diff --check
git add crates/calc-flow/benches/core.rs   docs/superpowers/handoffs/2026-07-16-array-benchmark-phase2.md
git diff --cached --check
git commit -m "perf: measure remaining execution overhead"
```

---

### Task 8: Run Full Repository Gates and Prepare Delivery

**Files:**
- Modify only when verification exposes a real defect: files already owned by Tasks 1-7.

- [ ] **Step 1: Check source scope**

```bash
git status --short
git diff --stat b42b687c8021291bcb537a57fded7a40f7f8477d...HEAD
find python/calc_flow -maxdepth 1 -name '_native*.so' -print
git diff --check
```

Expected: intended changes only, no source-tree native module, clean whitespace.

- [ ] **Step 2: Run Rust gates**

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

- [ ] **Step 3: Run PyO3/Python gates**

```bash
uv sync --extra dev
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
```

Remove generated `python/calc_flow/_native*.so` afterward and confirm `find` prints nothing.

- [ ] **Step 4: Run Studio backend gates**

```bash
cd web-ui/backend
uv run --project . --extra dev pytest --cov=calc_flow_studio
```

- [ ] **Step 5: Run frontend/generated API gates**

```bash
cd web-ui
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev
```

- [ ] **Step 6: Run supply-chain/release gates**

```bash
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

- [ ] **Step 7: Prove generated contracts did not drift**

```bash
git diff --exit-code --   schemas/project-v2.schema.json   web-ui/openapi.json   web-ui/src/api/schema.d.ts
git diff --check
git status --short
```

Expected: no generated contract diff and no uncommitted implementation changes. A verification-driven fix requires its focused RED/GREEN test, affected full group, and separate commit.

- [ ] **Step 8: Review exact final head**

```bash
git log --oneline --decorate b42b687c8021291bcb537a57fded7a40f7f8477d..HEAD
git diff --check b42b687c8021291bcb537a57fded7a40f7f8477d...HEAD
git status --short --branch
```

Confirm the Phase 1 handoff identifies the final candidate SHA or explains any later verification-only commits. Do not push or open/update a PR unless the user explicitly requests publication.
