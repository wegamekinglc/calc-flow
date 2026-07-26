# Table-Array Matrix Multiplication Implementation Plan

> **Historical status:** Implemented and merged in PR #27. Unchecked boxes and
> commands below preserve the execution plan as written; for current behavior,
> use `docs/introduction.md`, `docs/python-api.md`, `docs/api-reference.md`, and
> `examples/README.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a copy-bounded NumPy/JAX matrix operator that multiplies selected Arrow table columns by a separate backend weight matrix.

**Architecture:** Keep Rust's existing multi-port `Operator` and scheduler unchanged. Add a focused Python builder method, an internal mapping-mode Python provider bridge, safe Rust-owned NumPy buffers for copy-free result adoption, and backend-specific table-matrix providers registered by `register_numpy` and `register_jax`.

**Tech Stack:** Rust 2024, PyO3 0.28, rust-numpy 0.28, ndarray 0.17, PyArrow 24, NumPy 2, JAX, pytest, Ruff.

## Global Constraints

- Read `AGENTS.md`, `docs/superpowers/specs/2026-07-24-table-array-matmul-design.md`, `docs/introduction.md`, and `.codex/guidance/code-style.md` before editing.
- Enter an isolated worktree with `superpowers:using-git-worktrees` before changing files. If the source checkout's `.git` is read-only, use a temporary clone and copy the final narrow diff back only when the user requests that workspace handoff.
- Preserve the existing uncommitted `examples/README.md` and `examples/07_array_and_dataframe.py` intent; Task 6 replaces their independent-branch behavior with the approved matrix example.
- Keep `unsafe_code = "forbid"` and do not add local `unsafe` blocks. rust-numpy may encapsulate its own reviewed unsafe implementation.
- Table input means an Arrow C Stream provider such as `pyarrow.Table`, not pandas or Polars.
- Both operands are rank two in version 1; sparse, batched, vector-matrix, and matrix-vector forms are out of scope.
- The output remains a NumPy or JAX array batch. Do not convert the result back to Arrow.
- The copy guarantee starts after input `Batch` construction.
- NumPy execution permits one table-to-dense allocation and one result allocation.
- JAX execution permits one contiguous host staging allocation, one host-to-device table buffer, and one device result allocation; it must not return the weight or result through NumPy.
- Do not add a public unsafe ownership-transfer constructor. Owned-result adoption is private and restricted by a one-use native token.
- Existing `Runtime.register_provider`, `PipelineBuilder.external`, array-expression providers, projects, and checkpoints must remain source- and behavior-compatible.
- Every behavior change follows red, green, refactor and records the expected failing reason before implementation.
- Never leave `python/calc_flow/_native*.so` in source.
- Commit only the task's files with imperative summaries under 72 characters. Do not push or open a PR without explicit user authority.

---

## File Structure

- `python/calc_flow/pipeline.py` owns the functional `table_matmul` builder
  step and private mapping-provider registration wrapper.
- `python/calc_flow/array.py` owns shared table-matrix validation plus NumPy
  and JAX provider implementations.
- `python/calc_flow/_native.pyi` owns the private native
  allocation/adoption and mapping-registration type surface.
- `crates/calc-flow-python/src/provider.rs` owns the legacy single-array and
  new exact-port mapping callback bridge.
- `crates/calc-flow-python/src/config.rs` owns the GC-safe native
  registration entrypoint for mapping providers.
- `crates/calc-flow-python/src/batch.rs` owns the Rust-owned NumPy allocation
  token and copy-free owned-result adoption.
- `crates/calc-flow-python/Cargo.toml` and `Cargo.lock` own the direct
  rust-numpy, ndarray, complex-number, and locked dependency declarations.
- `python/tests/test_pipeline.py` owns builder/project-shape and graph-input
  tests.
- `python/tests/test_array.py` owns NumPy/JAX behavior, validation,
  immutability, and copy-ceiling tests.
- `examples/07_array_and_dataframe.py` owns the runnable NumPy and optional
  JAX table-matrix example.
- `examples/README.md`, `docs/python-api.md`, and `docs/api-reference.md` own
  user-facing example, usage, and reference documentation.

---

### Task 1: Add the Functional Builder Surface

**Files:**

- Modify: `python/calc_flow/pipeline.py`
- Test: `python/tests/test_pipeline.py`

**Interfaces:**

- Consumes: existing immutable `PipelineBuilder._from_json` and `_updated_project`.
- Produces:

```python
def table_matmul(
    self,
    node_id: str,
    *,
    backend: Literal["numpy", "jax"],
    columns: Sequence[str],
) -> PipelineBuilder
```

- Emits provider `<backend>:table_matmul@1`, table/weights inputs, and one array output.

- [ ] **Step 1: Write the failing project-shape test**

Add `Literal` to the typing imports in `python/calc_flow/pipeline.py` only during
the implementation step. First add this test to `python/tests/test_pipeline.py`:

```python
def test_table_matmul_builder_is_functional_and_defensive() -> None:
    columns = ["quantity", "unit_price"]
    original = PipelineBuilder("matrix")
    builder = original.table_matmul(
        "multiply",
        backend="numpy",
        columns=columns,
    )
    columns[0] = "mutated"

    assert original.project["pipeline"]["nodes"] == []
    assert builder.project["data_sources"] == [
        {
            "data": [],
            "format": "inline_json",
            "id": "source_1",
            "input": "table",
        },
        {
            "data": [],
            "format": "inline_json",
            "id": "source_2",
            "input": "weights",
        },
    ]
    assert builder.project["pipeline"]["nodes"] == [
        {
            "id": "multiply",
            "input_ports": [
                {
                    "kind": "table",
                    "name": "table",
                    "required": True,
                    "schema": [],
                },
                {
                    "kind": "array",
                    "name": "weights",
                    "required": True,
                    "schema": [],
                },
            ],
            "operator": {
                "kind": "external",
                "name": "table_matmul",
                "options": {"columns": ["quantity", "unit_price"]},
                "provider": "numpy",
                "version": "1",
            },
            "output_ports": [
                {
                    "kind": "array",
                    "name": "output",
                    "required": True,
                    "schema": [],
                }
            ],
        }
    ]
```

- [ ] **Step 2: Run the focused test and record the expected failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_pipeline.py::test_table_matmul_builder_is_functional_and_defensive \
  -q
```

Expected: FAIL with `AttributeError: 'PipelineBuilder' object has no attribute 'table_matmul'`.

- [ ] **Step 3: Add failing argument-validation tests**

Add:

```python
@pytest.mark.parametrize("backend", ["", "numpy ", "pandas", "JAX"])
def test_table_matmul_rejects_unknown_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
        PipelineBuilder("matrix").table_matmul(
            "multiply",
            backend=backend,  # type: ignore[arg-type]
            columns=("a",),
        )


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ((), "at least one"),
        (("a", "a"), "unique"),
        (("",), "non-empty strings"),
        (("a", 1), "non-empty strings"),
        ("a", "sequence of column names"),
    ],
)
def test_table_matmul_rejects_invalid_columns(
    columns: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        PipelineBuilder("matrix").table_matmul(
            "multiply",
            backend="numpy",
            columns=columns,  # type: ignore[arg-type]
        )
```

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_pipeline.py -q -k table_matmul
```

Expected: FAIL because `table_matmul` is absent.

- [ ] **Step 4: Implement the minimal immutable builder method**

Update the imports and add the method next to `external`:

```python
from typing import Any, Literal
```

```python
    def table_matmul(
        self,
        node_id: str,
        *,
        backend: Literal["numpy", "jax"],
        columns: Sequence[str],
    ) -> PipelineBuilder:
        if backend not in {"numpy", "jax"}:
            raise ValueError("backend must be 'numpy' or 'jax'")
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
            raise TypeError("columns must be a sequence of column names")
        copied_columns = list(columns)
        if not copied_columns:
            raise ValueError("columns must contain at least one column name")
        if not all(isinstance(column, str) and column for column in copied_columns):
            raise TypeError("columns must contain non-empty strings")
        if len(set(copied_columns)) != len(copied_columns):
            raise ValueError("columns must be unique")

        def add(project: dict[str, Any]) -> None:
            project["pipeline"]["nodes"].append(
                {
                    "id": node_id,
                    "input_ports": [
                        {
                            "kind": "table",
                            "name": "table",
                            "required": True,
                            "schema": [],
                        },
                        {
                            "kind": "array",
                            "name": "weights",
                            "required": True,
                            "schema": [],
                        },
                    ],
                    "operator": {
                        "kind": "external",
                        "name": "table_matmul",
                        "options": {"columns": copied_columns},
                        "provider": backend,
                        "version": "1",
                    },
                    "output_ports": [
                        {
                            "kind": "array",
                            "name": "output",
                            "required": True,
                            "schema": [],
                        }
                    ],
                }
            )

        return self._from_json(_updated_project(self._project_json, add))
```

- [ ] **Step 5: Run builder and strict-project tests**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_pipeline.py -q -k 'table_matmul or project_json'
```

Expected: PASS.

Run:

```bash
git diff --exit-code -- schemas/project-v2.schema.json
```

Expected: exit 0; the existing project schema already supports explicit mixed-kind ports.

- [ ] **Step 6: Commit the builder task**

```bash
git add python/calc_flow/pipeline.py python/tests/test_pipeline.py
git commit -m "feat: add table matrix builder"
```

---

### Task 2: Add the Exact-Port Mapping Provider Bridge

**Files:**

- Modify: `crates/calc-flow-python/src/provider.rs`
- Modify: `crates/calc-flow-python/src/config.rs`
- Modify: `python/calc_flow/pipeline.py`
- Modify: `python/calc_flow/_native.pyi`
- Test: inline tests in `crates/calc-flow-python/src/provider.rs`
- Test: `python/tests/test_runtime.py`

**Interfaces:**

- Consumes: `PipelineBuilder.table_matmul` project shape from Task 1.
- Produces:

```python
Runtime._register_mapping_provider(
    provider: str,
    name: str,
    version: str,
    callback: object,
    *,
    input_ports: Sequence[tuple[str, str]],
    output_ports: Sequence[tuple[str, str]],
) -> None
```

- Mapping callback contract:

```python
callback(
    inputs: Mapping[str, Batch],
    options: dict[str, object],
) -> Mapping[str, Batch]
```

- [ ] **Step 1: Write the failing native mapping-bridge test**

In `crates/calc-flow-python/src/provider.rs`, add a `table_port` helper and this
test:

```rust
fn table_port(name: &str) -> calc_flow::Port {
    calc_flow::Port::new(name, calc_flow::BatchKind::Table, true, None).unwrap()
}

#[tokio::test]
async fn mapping_operator_round_trips_exact_named_batches() {
    Python::initialize();
    let root = Python::attach(|py| {
        let locals = PyDict::new(py);
        py.run(
            c"class Callback:\n    def __call__(self, inputs, options):\n        assert sorted(inputs) == ['table', 'weights']\n        assert options == {'columns': ['a']}\n        return {'output': inputs['weights']}\ncallback = Callback()",
            None,
            Some(&locals),
        )
        .unwrap();
        Arc::new(PythonRoot::new(
            locals.get_item("callback").unwrap().unwrap().unbind(),
        ))
    });
    let factory = PythonOperatorFactory::new_mapping(
        root,
        "python",
        "table_matmul",
        "1",
        vec![
            PortContract::new("table", calc_flow::BatchKind::Table),
            PortContract::new("weights", calc_flow::BatchKind::Array),
        ],
        vec![PortContract::new("output", calc_flow::BatchKind::Array)],
    );
    let spec = calc_flow::ExternalOperatorSpec::new(
        "python",
        "table_matmul",
        "1",
        BTreeMap::from([("columns".into(), json!(["a"]))]),
    )
    .unwrap();
    let mut operator = factory
        .create(
            &spec,
            vec![table_port("table"), array_port("weights")],
            vec![array_port("output")],
        )
        .unwrap();

    let table = calc_flow::Batch::table(
        vec![datafusion::arrow::record_batch::RecordBatch::new_empty(
            Arc::new(datafusion::arrow::datatypes::Schema::empty()),
        )],
        calc_flow::BatchMetadata::default(),
    )
    .unwrap();
    let weights = Python::attach(|py| {
        PyBatch::_from_external(
            PyDict::new(py).into_any().unbind(),
            "python".into(),
            1,
            PyDict::new(py).as_any(),
        )
        .unwrap()
        .clone_inner()
        .unwrap()
    });
    let cancellation = calc_flow::CancellationToken::new();
    let run = calc_flow::RunContext::new(BTreeMap::new(), None, cancellation).unwrap();
    let outputs = operator
        .process(
            &BTreeMap::from([
                ("table".into(), table),
                ("weights".into(), weights.clone()),
            ]),
            &calc_flow::OperatorContext { run: &run },
        )
        .await
        .unwrap();

    assert_eq!(outputs["output"].num_rows(), weights.num_rows());
}
```

- [ ] **Step 2: Run the Rust test and record the expected failure**

Run:

```bash
cargo test -p calc-flow-python mapping_operator_round_trips_exact_named_batches
```

Expected: compile FAIL because `PortContract` and
`PythonOperatorFactory::new_mapping` do not exist.

- [ ] **Step 3: Implement separate legacy and mapping modes**

In `provider.rs`, add:

```rust
#[derive(Clone)]
pub(crate) struct PortContract {
    name: String,
    kind: calc_flow::BatchKind,
}

impl PortContract {
    pub(crate) fn new(name: &str, kind: calc_flow::BatchKind) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }
}

#[derive(Clone)]
enum PythonProviderMode {
    SingleArray,
    Mapping {
        inputs: Vec<PortContract>,
        outputs: Vec<PortContract>,
    },
}
```

Store `mode: PythonProviderMode` in `PythonOperatorFactory` and
`PythonOperator`. Keep `new` as the legacy constructor and add:

```rust
pub(crate) fn new_mapping(
    callback: Arc<PythonRoot>,
    provider: &str,
    name: &str,
    version: &str,
    inputs: Vec<PortContract>,
    outputs: Vec<PortContract>,
) -> Self {
    Self {
        callback,
        provider: provider.into(),
        name: name.into(),
        version: version.into(),
        mode: PythonProviderMode::Mapping { inputs, outputs },
    }
}
```

Replace `validate_ports` with exact mode-aware validation:

```rust
fn ports_match(ports: &[calc_flow::Port], expected: &[PortContract]) -> bool {
    ports.len() == expected.len()
        && ports.iter().zip(expected).all(|(port, contract)| {
            port.name() == contract.name
                && port.kind() == contract.kind
                && port.required()
                && port.schema().is_none()
        })
}
```

The legacy error text remains unchanged. Mapping errors use:

```text
Python mapping provider ports do not match the registered contract
```

Add `call_python_operator_mapping` that constructs a `PyDict` in contract
order, calls the callback, requires a mapping with exactly the declared keys,
extracts each `PyBatch`, calls `rehome_python_payload`, and returns a
`BTreeMap<String, Batch>`. Validate every output against its declared `Port`
before returning it.

- [ ] **Step 4: Add the native private registration entrypoint**

In `config.rs`, add a kind parser:

```rust
fn mapping_port_contract(
    name: String,
    kind: &str,
) -> PyResult<crate::provider::PortContract> {
    let kind = match kind {
        "table" => calc_flow::BatchKind::Table,
        "array" => calc_flow::BatchKind::Array,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mapping provider port kind must be 'table' or 'array'",
            ));
        }
    };
    Ok(crate::provider::PortContract::new(&name, kind))
}
```

Add this private `PyRuntime` method:

```rust
#[pyo3(signature = (provider, name, version, callback, *, input_ports, output_ports))]
fn _register_mapping_provider(
    &self,
    py: Python<'_>,
    provider: &str,
    name: &str,
    version: &str,
    callback: Py<PyAny>,
    input_ports: Vec<(String, String)>,
    output_ports: Vec<(String, String)>,
) -> PyResult<()> {
    if !callback.bind(py).is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "provider callback must be callable",
        ));
    }
    let inputs = input_ports
        .into_iter()
        .map(|(port, kind)| mapping_port_contract(port, &kind))
        .collect::<PyResult<Vec<_>>>()?;
    let outputs = output_ports
        .into_iter()
        .map(|(port, kind)| mapping_port_contract(port, &kind))
        .collect::<PyResult<Vec<_>>>()?;
    let root = Arc::new(PythonRoot::new(callback));
    let factory: Arc<dyn calc_flow::ExternalOperatorFactory> = Arc::new(
        crate::provider::PythonOperatorFactory::new_mapping(
            Arc::clone(&root),
            provider,
            name,
            version,
            inputs,
            outputs,
        ),
    );
    self.register_provider_factory(provider, name, version, &factory, root)
}
```

- [ ] **Step 5: Add the Python private wrapper and stub**

In `python/calc_flow/pipeline.py`, add:

```python
    def _register_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        input_ports: Sequence[tuple[str, str]],
        output_ports: Sequence[tuple[str, str]],
    ) -> None:
        copied_inputs = tuple((port, kind) for port, kind in input_ports)
        copied_outputs = tuple((port, kind) for port, kind in output_ports)
        with self._registration_lock:
            self._inner._register_mapping_provider(
                provider,
                name,
                version,
                callback,
                input_ports=copied_inputs,
                output_ports=copied_outputs,
            )
            self._registrations.append(
                {
                    "kind": "provider",
                    "provider": provider,
                    "name": name,
                    "version": version,
                    "callback": callback,
                }
            )
```

In `_native.pyi`, add the corresponding private method to `Runtime`.

- [ ] **Step 6: Add Python integration and compatibility tests**

In `python/tests/test_runtime.py`, add:

```python
def test_mapping_provider_executes_mixed_named_inputs() -> None:
    runtime = Runtime()

    def callback(
        inputs: dict[str, Batch],
        options: dict[str, object],
    ) -> dict[str, Batch]:
        assert sorted(inputs) == ["table", "weights"]
        assert options == {"columns": ["value"]}
        return {"output": inputs["weights"]}

    runtime._register_mapping_provider(
        "test",
        "table_matmul",
        "1",
        callback,
        input_ports=(("table", "table"), ("weights", "array")),
        output_ports=(("output", "array"),),
    )
    project = (
        PipelineBuilder("mapping")
        .table_matmul("multiply", backend="numpy", columns=("value",))
        .project
    )
    project["pipeline"]["nodes"][0]["operator"]["provider"] = "test"
    plan = runtime.compile_project(json.dumps(project))
    weights = Batch.from_array(np.array([[2.0]]), backend="numpy")

    result = plan.execute({"table": _batch(3), "weights": weights})

    assert result.outputs["output"].array.tolist() == [[2.0]]
```

Add `import json` to `python/tests/test_runtime.py`. Do not weaken
`table_matmul` backend validation for this bridge test.

Also rerun the existing public provider tests unchanged:

```bash
cargo test -p calc-flow-python provider::
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_runtime.py python/tests/test_array.py -q \
  -k 'provider or registration'
```

Expected: PASS.

- [ ] **Step 7: Commit the mapping bridge**

```bash
git add \
  crates/calc-flow-python/src/provider.rs \
  crates/calc-flow-python/src/config.rs \
  python/calc_flow/pipeline.py \
  python/calc_flow/_native.pyi \
  python/tests/test_runtime.py
git commit -m "feat: support mapped Python providers"
```

---

### Task 3: Add Safe Rust-Owned NumPy Result Adoption

**Files:**

- Modify: `crates/calc-flow-python/Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `crates/calc-flow-python/src/batch.rs`
- Modify: `python/calc_flow/_native.pyi`
- Test: inline tests in `crates/calc-flow-python/src/batch.rs`
- Test: `python/tests/test_array.py`

**Interfaces:**

- Consumes: Python-hosted array payload support.
- Produces private native methods:

```python
Batch._new_owned_numpy(
    shape: Sequence[int],
    dtype: str,
) -> tuple[object, object]

Batch._from_owned_array(
    array: object,
    *,
    backend: str,
    token: object | None,
    metadata: Mapping[str, object],
) -> Batch
```

- The token is one-use and binds one exact Rust-owned NumPy object identity.

- [ ] **Step 1: Write the failing Python pointer and immutability test**

Add to `python/tests/test_array.py`:

```python
def test_owned_numpy_result_is_adopted_without_copy_and_cannot_be_reopened() -> None:
    owned, token = Batch._new_owned_numpy((2, 2), "float64")
    assert type(owned) is np.ndarray
    assert owned.flags.writeable
    assert not owned.flags.owndata
    owned[:] = [[1.0, 2.0], [3.0, 4.0]]
    pointer = owned.__array_interface__["data"][0]

    batch = Batch._from_owned_array(
        owned,
        backend="numpy",
        token=token,
        metadata={"operation": "table_matmul"},
    )
    output = batch.array

    assert output is owned
    assert output.__array_interface__["data"][0] == pointer
    assert output.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert output.flags.writeable is False
    with pytest.raises(ValueError):
        output.setflags(write=True)
    with pytest.raises(ValueError, match="already consumed"):
        Batch._from_owned_array(
            owned,
            backend="numpy",
            token=token,
            metadata={},
        )
```

- [ ] **Step 2: Run the focused test and record the expected failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_array.py::test_owned_numpy_result_is_adopted_without_copy_and_cannot_be_reopened \
  -q
```

Expected: FAIL because the two private methods do not exist.

- [ ] **Step 3: Add safe direct dependencies**

In `crates/calc-flow-python/Cargo.toml`, add exact compatible dependencies:

```toml
ndarray = "0.17.2"
num-complex = "0.4.6"
numpy = "0.28.0"
```

Run:

```bash
cargo check -p calc-flow-python
```

Expected: PASS with the lockfile retaining one `numpy 0.28.0` and one
`ndarray 0.17.2`.

- [ ] **Step 4: Add the one-use native token**

In `batch.rs`, add:

```rust
use std::sync::atomic::{AtomicBool, Ordering};

#[pyclass(frozen, module = "calc_flow._native")]
struct OwnedArrayToken {
    object_identity: usize,
    consumed: AtomicBool,
}

impl OwnedArrayToken {
    fn consume(&self, object: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.object_identity != object.as_ptr() as usize {
            return Err(PyValueError::new_err(
                "owned array token does not match the array",
            ));
        }
        self.consumed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PyValueError::new_err("owned array token was already consumed"))?;
        Ok(())
    }
}
```

The token is private, frozen, non-cloneable, and registered with the module
only so PyO3 can carry it between the allocator and adoption method.

- [ ] **Step 5: Add dtype-dispatched Rust-owned NumPy allocation**

Use safe rust-numpy APIs only:

```rust
use ndarray::{ArrayD, IxDyn};
use numpy::{Element, IntoPyArray};

fn owned_numpy<T>(
    py: Python<'_>,
    shape: &[usize],
) -> PyResult<(Py<PyAny>, Py<OwnedArrayToken>)>
where
    T: Clone + Default + Element,
{
    let array = ArrayD::<T>::default(IxDyn(shape)).into_pyarray(py);
    let object = array.into_any().unbind();
    let token = Py::new(
        py,
        OwnedArrayToken {
            object_identity: object.bind(py).as_ptr() as usize,
            consumed: AtomicBool::new(false),
        },
    )?;
    Ok((object, token))
}
```

Add `Batch._new_owned_numpy` and dispatch these exact dtype names:

```text
int8 int16 int32 int64
uint8 uint16 uint32 uint64
float32 float64
complex64 complex128
```

Use `num_complex::Complex32` and `Complex64` for complex storage. Reject bool,
float16, extended precision, object, string, structured, temporal, and
non-native-endian dtypes with:

```text
owned NumPy arrays require a supported native numeric dtype; received <dtype>
```

Reject empty shapes, zero dimensions, rank above 16, dimensions above
1,000,000, and total elements above 10,000,000 before allocation.

- [ ] **Step 6: Add copy-free owned-result adoption**

Implement `Batch._from_owned_array`:

```rust
#[staticmethod]
#[pyo3(signature = (array, *, backend, token, metadata))]
fn _from_owned_array(
    py: Python<'_>,
    array: Py<PyAny>,
    backend: String,
    token: Option<PyRef<'_, OwnedArrayToken>>,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<Self> {
    let len = match (backend.as_str(), token) {
        ("numpy", Some(token)) => {
            token.consume(array.bind(py))?;
            let flags = array.bind(py).getattr("flags")?;
            if flags.getattr("owndata")?.extract::<bool>()? {
                return Err(PyValueError::new_err(
                    "owned NumPy array must use binding-owned storage",
                ));
            }
            array.bind(py).call_method1("setflags", (false,))?;
            let shape: Vec<usize> = array.bind(py).getattr("shape")?.extract()?;
            shape.first().copied().unwrap_or(1)
        }
        ("numpy", None) => {
            return Err(PyTypeError::new_err(
                "NumPy owned arrays require an ownership token",
            ));
        }
        ("jax", None) => {
            let jax = py.import("jax")?;
            if !array.bind(py).is_instance(&jax.getattr("Array")?)? {
                return Err(PyTypeError::new_err(
                    "JAX owned arrays require a jax.Array",
                ));
            }
            let shape: Vec<usize> = array.bind(py).getattr("shape")?.extract()?;
            shape.first().copied().unwrap_or(1)
        }
        ("jax", Some(_)) => {
            return Err(PyTypeError::new_err(
                "JAX owned arrays do not accept a NumPy ownership token",
            ));
        }
        _ => {
            return Err(PyValueError::new_err(
                "owned array backend must be 'numpy' or 'jax'",
            ));
        }
    };
    let metadata = metadata_from_python(py, Some(metadata))?;
    let payload = PythonPayload {
        object: array,
        backend,
        len,
    };
    let inner = calc_flow::Batch::external(Arc::new(payload), metadata)
        .map_err(crate::error::to_py_err)?;
    Ok(Self::from_inner(inner))
}
```

After clearing the NumPy writeable flag, add an internal assertion/test that
`setflags(write=True)` raises because rust-numpy storage leaves `OWNDATA`
unset and hides its Rust `Vec` in the base owner. Do not probe by leaving the
flag writable.

- [ ] **Step 7: Add mismatch, dtype, token, and JAX tests**

Cover:

```python
@pytest.mark.parametrize("dtype", ["bool", "float16", "object"])
def test_owned_numpy_rejects_unsupported_dtypes(dtype: str) -> None:
    with pytest.raises(ValueError, match="supported native numeric dtype"):
        Batch._new_owned_numpy((1, 1), dtype)


def test_owned_numpy_token_rejects_a_different_array() -> None:
    first, token = Batch._new_owned_numpy((1, 1), "float32")
    second, _ = Batch._new_owned_numpy((1, 1), "float32")
    with pytest.raises(ValueError, match="does not match"):
        Batch._from_owned_array(
            second,
            backend="numpy",
            token=token,
            metadata={},
        )
    assert first.flags.writeable


def test_owned_jax_result_retains_identity() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    result = jnp.asarray([[1.0, 2.0]])
    batch = Batch._from_owned_array(
        result,
        backend="jax",
        token=None,
        metadata={},
    )
    assert isinstance(batch.array, jax.Array)
    assert batch.array is result
```

Run:

```bash
cargo test -p calc-flow-python batch::
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q -k owned
```

Expected: PASS.

- [ ] **Step 8: Update the native stub and commit**

Add the two private `Batch` methods to `_native.pyi`, using `Sequence[int]`,
`object` for the private token, and keyword-only adoption arguments.

Run:

```bash
cargo fmt --all --check
cargo clippy -p calc-flow-python --all-targets -- -D warnings
uv run ruff check python/tests/test_array.py
```

Expected: PASS.

Commit:

```bash
git add \
  Cargo.lock \
  crates/calc-flow-python/Cargo.toml \
  crates/calc-flow-python/src/batch.rs \
  python/calc_flow/_native.pyi \
  python/tests/test_array.py
git commit -m "feat: adopt owned array results"
```

---

### Task 4: Implement the NumPy Table-Matrix Provider

**Files:**

- Modify: `python/calc_flow/array.py`
- Test: `python/tests/test_array.py`

**Interfaces:**

- Consumes:
  - `PipelineBuilder.table_matmul`.
  - `Runtime._register_mapping_provider`.
  - `Batch._new_owned_numpy` and `Batch._from_owned_array`.
- Produces `numpy:table_matmul@1`.

- [ ] **Step 1: Write the failing NumPy end-to-end test**

Add:

```python
def test_numpy_table_matmul_multiplies_selected_arrow_columns() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    table = pa.table(
        {
            "quantity": [3.0, 1.0, 4.0],
            "unit_price": [10.0, 12.0, 10.0],
            "ignored": [99.0, 99.0, 99.0],
        }
    )
    weights = Batch.from_array(
        np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        backend="numpy",
    )
    plan = (
        PipelineBuilder("numpy-table-matmul")
        .table_matmul(
            "multiply",
            backend="numpy",
            columns=("quantity", "unit_price"),
        )
        .compile(runtime)
    )

    run = plan.execute(
        {
            "table": Batch.from_pyarrow(table, {"source": "orders"}),
            "weights": weights,
        }
    )
    output = run.outputs["output"]

    assert output.kind == "array"
    assert output.backend == "numpy"
    assert output.array.tolist() == [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
    assert output.metadata == {
        "backend": "numpy",
        "columns": ["quantity", "unit_price"],
        "operation": "table_matmul",
        "source": "orders",
    }
    assert run.datafusion_metrics == []
```

- [ ] **Step 2: Run the test and record the expected failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_array.py::test_numpy_table_matmul_multiplies_selected_arrow_columns \
  -q
```

Expected: compile FAIL containing
`provider numpy:table_matmul@1 is unavailable`.

- [ ] **Step 3: Add configuration and input validation helpers**

In `array.py`, add:

```python
_TABLE_MATMUL_INPUT_PORTS = (("table", "table"), ("weights", "array"))
_TABLE_MATMUL_OUTPUT_PORTS = (("output", "array"),)


def _table_matmul_columns(options: Mapping[str, object]) -> tuple[str, ...]:
    unknown = set(options) - {"columns"}
    if unknown:
        raise ValueError(
            "invalid table_matmul options: unsupported options: "
            + ", ".join(sorted(unknown))
        )
    columns = options.get("columns")
    if isinstance(columns, (str, bytes)) or not isinstance(columns, list):
        raise ValueError(
            "invalid table_matmul options: columns must be a JSON array"
        )
    if not columns:
        raise ValueError(
            "invalid table_matmul options: columns must contain at least one name"
        )
    if not all(isinstance(column, str) and column for column in columns):
        raise ValueError(
            "invalid table_matmul options: columns must contain non-empty strings"
        )
    if len(set(columns)) != len(columns):
        raise ValueError(
            "invalid table_matmul options: columns must be unique"
        )
    return tuple(columns)
```

Add exact helpers for:

- extracting `table` and `weights` from a copied input mapping;
- checking table kind and weight array kind/backend;
- rejecting zero rows;
- resolving every selected `pyarrow.ChunkedArray`;
- rejecting missing names, nulls, and non-integer/non-floating Arrow types;
- checking rank-two positive weight shape;
- choosing `np.result_type` and requiring `np.can_cast(..., casting="safe")`.

Every error string starts with:

```text
invalid table_matmul <field>:
```

and includes the field named in the design.

- [ ] **Step 4: Materialize Arrow columns without intermediate full columns**

Add:

```python
def _numpy_table_matrix(
    table: object,
    columns: tuple[str, ...],
    dtype: object,
) -> np.ndarray:
    matrix, _token = _native.Batch._new_owned_numpy(
        (table.num_rows, len(columns)),
        np.dtype(dtype).name,
    )
    for column_index, name in enumerate(columns):
        offset = 0
        for chunk in table[name].chunks:
            values = chunk.to_numpy(zero_copy_only=True)
            next_offset = offset + len(values)
            np.copyto(
                matrix[offset:next_offset, column_index],
                values,
                casting="safe",
            )
            offset = next_offset
    return matrix
```

The staging token is deliberately not adopted; dropping it and the staging
array releases the temporary Rust-owned buffer after multiplication. Do not
call `combine_chunks`, `to_pylist`, `np.column_stack`, `np.stack`, or
`np.asarray(table)`.

- [ ] **Step 5: Implement the NumPy provider with an owned output buffer**

Add a frozen/slot provider:

```python
@dataclass(frozen=True, slots=True)
class _TableMatmulProvider:
    backend: str
    namespace: object

    def validate(self, options: Mapping[str, object]) -> None:
        _table_matmul_columns(options)

    def __call__(
        self,
        inputs: Mapping[str, _native.Batch],
        options: Mapping[str, object],
    ) -> dict[str, _native.Batch]:
        columns = _table_matmul_columns(options)
        table_batch, weights_batch = _table_matmul_inputs(inputs, self.backend)
        table = table_batch.to_pyarrow()
        weights = weights_batch.array
        table_dtypes = _validated_table_dtypes(table, columns)
        result_dtype = _common_matrix_dtype(
            self.backend,
            self.namespace,
            table_dtypes,
            weights,
        )

        if self.backend == "numpy":
            dense = _numpy_table_matrix(table, columns, result_dtype)
            output, token = _native.Batch._new_owned_numpy(
                (table.num_rows, weights.shape[1]),
                np.dtype(result_dtype).name,
            )
            np.matmul(dense, weights, out=output)
        else:
            output, token = _jax_table_matmul(
                table,
                columns,
                result_dtype,
                weights,
            )

        metadata = table_batch.metadata
        metadata.update(
            {
                "backend": self.backend,
                "columns": list(columns),
                "operation": "table_matmul",
            }
        )
        return {
            "output": _native.Batch._from_owned_array(
                output,
                backend=self.backend,
                token=token,
                metadata=metadata,
            )
        }
```

In Task 4, define `_jax_table_matmul` as a function that raises
`AssertionError("JAX table_matmul provider is not registered")` only inside
the unreachable JAX branch. Task 4 registers only the NumPy provider, so this
guard is permanent and testable for that task's supported surface. Task 5
replaces the guard with the complete JAX helper before registering the JAX
provider.

- [ ] **Step 6: Register the NumPy matrix provider**

Extend `register_numpy`:

```python
def register_numpy(runtime: Runtime) -> None:
    import numpy as np

    runtime.register_provider("numpy", "expression", "1", _ArrayProvider("numpy", np))
    runtime._register_mapping_provider(
        "numpy",
        "table_matmul",
        "1",
        _TableMatmulProvider("numpy", np),
        input_ports=_TABLE_MATMUL_INPUT_PORTS,
        output_ports=_TABLE_MATMUL_OUTPUT_PORTS,
    )
```

The runtime catalog must contain both NumPy registrations in sorted order.

- [ ] **Step 7: Add copy-ceiling and immutability tests**

Add tests that:

- build a multi-chunk Arrow column;
- monkeypatch `pyarrow.ChunkedArray.combine_chunks` to fail if called;
- retain `weights_batch.array` and assert the provider sees the same object;
- monkeypatch `_owned_numpy` to fail during execution;
- assert the output pointer is the pointer allocated by
  `_new_owned_numpy`;
- walk every reachable NumPy base and prove no writable ndarray exists;
- assert `output.setflags(write=True)` raises;
- assert the original Arrow table and weight array values are unchanged.
- compile two independently built but identical NumPy plans and assert their
  fingerprints are equal.

Use a small spy around `_native.Batch._new_owned_numpy`:

```python
allocations: list[tuple[tuple[int, ...], str, int]] = []
original = _native.Batch._new_owned_numpy

def counted(shape: tuple[int, ...], dtype: str):
    array, token = original(shape, dtype)
    allocations.append(
        (tuple(shape), dtype, array.__array_interface__["data"][0])
    )
    return array, token
```

Expected allocation shapes are exactly `[(3, 2), (3, 2)]`: dense table and
result. Do not count the caller's earlier weight `Batch` construction.

- [ ] **Step 8: Run the NumPy suite and commit**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q -k 'numpy or table_matmul'
uv run ruff check python/calc_flow/array.py python/tests/test_array.py
uv run ruff format --check python/calc_flow/array.py python/tests/test_array.py
```

Expected: PASS.

Commit:

```bash
git add python/calc_flow/array.py python/tests/test_array.py
git commit -m "feat: multiply Arrow tables with NumPy"
```

---

### Task 5: Add JAX Table-Matrix Execution

**Files:**

- Modify: `python/calc_flow/array.py`
- Test: `python/tests/test_array.py`

**Interfaces:**

- Consumes: `_TableMatmulProvider`, validation helpers, and host staging from
  Task 4.
- Produces `jax:table_matmul@1`.

- [ ] **Step 1: Write the failing JAX end-to-end test**

Add:

```python
def test_jax_table_matmul_stays_on_jax() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    weights = Batch.from_array(
        jnp.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        backend="jax",
    )
    plan = (
        PipelineBuilder("jax-table-matmul")
        .table_matmul(
            "multiply",
            backend="jax",
            columns=("quantity", "unit_price"),
        )
        .compile(runtime)
    )
    table = Batch.from_pyarrow(
        pa.table(
            {
                "quantity": pa.array([3.0, 1.0, 4.0], type=pa.float32()),
                "unit_price": pa.array([10.0, 12.0, 10.0], type=pa.float32()),
            }
        )
    )

    output = plan.execute({"table": table, "weights": weights}).outputs["output"]

    assert isinstance(output.array, jax.Array)
    assert output.backend == "jax"
    assert output.array.tolist() == [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
    assert output.array.device == weights.array.device
```

- [ ] **Step 2: Run the test and record the expected failure**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_array.py::test_jax_table_matmul_stays_on_jax \
  -q
```

Expected: compile FAIL containing
`provider jax:table_matmul@1 is unavailable`.

- [ ] **Step 3: Implement the bounded JAX staging path**

Replace the Task 4 unregistered-backend guard with:

```python
def _jax_table_matmul(
    table: object,
    columns: tuple[str, ...],
    dtype: object,
    weights: object,
) -> tuple[object, None]:
    import jax
    import jax.numpy as jnp

    host = _numpy_table_matrix(table, columns, dtype)
    dense = jax.device_put(host, device=weights.device)
    if dense.dtype != jnp.dtype(dtype):
        raise ValueError(
            "invalid table_matmul dtype: JAX changed "
            f"{jnp.dtype(dtype)} to {dense.dtype}; enable the required dtype "
            "or choose a lossless supported dtype"
        )
    result = jnp.matmul(dense, weights)
    if not isinstance(result, jax.Array):
        raise TypeError("table_matmul JAX result must remain a jax.Array")
    return result, None
```

Use the `jax.Array.device` property, which is present in the repository's
supported JAX line. Do not call the deprecated `device()` form.

- [ ] **Step 4: Register the JAX matrix provider**

Extend `register_jax`:

```python
def register_jax(runtime: Runtime) -> None:
    import jax.numpy as jnp

    runtime.register_provider("jax", "expression", "1", _ArrayProvider("jax", jnp))
    runtime._register_mapping_provider(
        "jax",
        "table_matmul",
        "1",
        _TableMatmulProvider("jax", jnp),
        input_ports=_TABLE_MATMUL_INPUT_PORTS,
        output_ports=_TABLE_MATMUL_OUTPUT_PORTS,
    )
```

- [ ] **Step 5: Add device-transfer and no-round-trip tests**

Add:

```python
def test_jax_table_matmul_does_not_round_trip_weights_or_result_through_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    original_asarray = np.asarray

    def guarded(value: object, *args: object, **kwargs: object) -> np.ndarray:
        assert not isinstance(value, jax.Array)
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", guarded)
    runtime = Runtime()
    register_jax(runtime)
    plan = (
        PipelineBuilder("jax-no-host-round-trip")
        .table_matmul(
            "multiply",
            backend="jax",
            columns=("a", "b"),
        )
        .compile(runtime)
    )
    weights = Batch.from_array(
        jnp.asarray([[1.0], [2.0]], dtype=jnp.float32),
        backend="jax",
    )
    result = plan.execute(
        {
            "table": Batch.from_pyarrow(
                pa.table(
                    {
                        "a": pa.array([1.0], type=pa.float32()),
                        "b": pa.array([3.0], type=pa.float32()),
                    }
                )
            ),
            "weights": weights,
        }
    )

    assert isinstance(result.outputs["output"].array, jax.Array)
    assert result.outputs["output"].array.item() == 7.0
```

Also spy on `_new_owned_numpy` and assert JAX execution makes exactly one host
staging allocation with shape `(m, n)`. Verify that the returned JAX object is
the exact object passed to `_from_owned_array`.

- [ ] **Step 6: Add backend, shape, dtype, and table validation matrix**

Parameterize both backends over:

- missing and duplicate columns;
- null-containing columns;
- bool, string, temporal, decimal, dictionary, and nested Arrow types;
- empty tables;
- rank 0, rank 1, and rank 3 weights;
- zero-width and incompatible weights;
- configured/actual backend mismatch;
- float32 success;
- float64 success when JAX x64 is enabled;
- explicit failure when JAX would narrow float64 with x64 disabled.

Every failure assertion matches the exact field prefix from the design, for
example:

```python
with pytest.raises(ProviderError, match=r"weights\.shape\[0\].*expected 2"):
    plan.execute(inputs)
```

- [ ] **Step 7: Run both backend suites and commit**

Run:

```bash
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q -k table_matmul
JAX_PLATFORMS=cpu uv run pytest python/tests/test_array.py -q -k jax
uv run ruff check python/calc_flow/array.py python/tests/test_array.py
uv run ruff format --check python/calc_flow/array.py python/tests/test_array.py
```

Expected: PASS.

Commit:

```bash
git add python/calc_flow/array.py python/tests/test_array.py
git commit -m "feat: multiply Arrow tables with JAX"
```

---

### Task 6: Replace the Mixed Example and Document the API

**Files:**

- Modify: `examples/07_array_and_dataframe.py`
- Modify: `examples/README.md`
- Modify: `docs/python-api.md`
- Modify: `docs/api-reference.md`
- Preserve: `docs/superpowers/specs/2026-07-24-table-array-matmul-design.md`
- Preserve: `docs/superpowers/plans/2026-07-24-table-array-matmul.md`
- Test: `tests/test_examples.py`
- Test: `python/tests/test_array.py`

**Interfaces:**

- Consumes: completed NumPy/JAX `table_matmul` feature.
- Produces runnable example output:

```text
NumPy result: [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
JAX result: [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
```

- [ ] **Step 1: Make the example harness fail on the old independent branches**

Update the existing example-focused test or add this focused assertion in
`python/tests/test_array.py`:

```python
def test_array_and_dataframe_example_uses_table_matmul() -> None:
    project = (
        PipelineBuilder("example-shape")
        .table_matmul(
            "multiply",
            backend="numpy",
            columns=("quantity", "unit_price"),
        )
        .project
    )
    node = project["pipeline"]["nodes"][0]
    assert [port["name"] for port in node["input_ports"]] == ["table", "weights"]
    assert node["operator"]["name"] == "table_matmul"
```

Run the current script:

```bash
JAX_PLATFORMS=cpu uv run python examples/07_array_and_dataframe.py
```

Expected before editing: output shows independent table and doubled-array
branches, not matrix multiplication.

- [ ] **Step 2: Replace the example with the approved matrix flow**

Use this structure in `examples/07_array_and_dataframe.py`:

```python
"""Multiply selected Arrow table columns by NumPy and JAX weight matrices."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from calc_flow import (
    Batch,
    PipelineBuilder,
    Runtime,
    register_jax,
    register_numpy,
)

EXPECTED = [[6.0, 10.0], [2.0, 12.0], [8.0, 10.0]]
COLUMNS = ("quantity", "unit_price")


def table() -> pa.Table:
    return pa.table(
        {
            "quantity": pa.array([3.0, 1.0, 4.0], type=pa.float32()),
            "unit_price": pa.array([10.0, 12.0, 10.0], type=pa.float32()),
        }
    )


def run_numpy(source: pa.Table) -> list[list[float]]:
    runtime = Runtime()
    register_numpy(runtime)
    plan = (
        PipelineBuilder("numpy-table-matmul")
        .table_matmul("multiply", backend="numpy", columns=COLUMNS)
        .compile(runtime)
    )
    weights = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    source_before = source.to_pydict()
    weights_before = weights.copy()
    run = plan.execute(
        {
            "table": Batch.from_pyarrow(source),
            "weights": Batch.from_array(weights, backend="numpy"),
        }
    )
    output = run.outputs["output"]
    result = output.array.tolist()
    assert output.backend == "numpy"
    assert result == EXPECTED
    assert source.to_pydict() == source_before
    np.testing.assert_array_equal(weights, weights_before)
    assert run.datafusion_metrics == []
    return result


def run_jax(source: pa.Table) -> list[list[float]] | None:
    try:
        import jax.numpy as jnp
    except ImportError:
        return None
    runtime = Runtime()
    register_jax(runtime)
    plan = (
        PipelineBuilder("jax-table-matmul")
        .table_matmul("multiply", backend="jax", columns=COLUMNS)
        .compile(runtime)
    )
    weights = jnp.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    run = plan.execute(
        {
            "table": Batch.from_pyarrow(source),
            "weights": Batch.from_array(weights, backend="jax"),
        }
    )
    output = run.outputs["output"]
    result = output.array.tolist()
    assert output.backend == "jax"
    assert result == EXPECTED
    assert run.datafusion_metrics == []
    return result


def main() -> None:
    source = table()
    print("NumPy result:", run_numpy(source))
    jax_result = run_jax(source)
    if jax_result is None:
        print("JAX result: skipped; install calc-flow[jax]")
    else:
        print("JAX result:", jax_result)


if __name__ == "__main__":
    main()
```

Keep imports optional: importing `calc_flow` or the example must not import
JAX until `run_jax` is called.

- [ ] **Step 3: Update the example index**

In `examples/README.md`:

- keep “DataFrame-style data means `pyarrow.Table`”;
- describe `07_array_and_dataframe.py` as explicit table-array matrix
  multiplication;
- document the NumPy budget as one dense table allocation plus one result;
- document the JAX ceiling as one host staging buffer, one device table
  buffer, and one result;
- say input `Batch` construction is outside that operator-execution budget;
- keep NumPy/JAX optional installation commands accurate.

- [ ] **Step 4: Update Python API and reference docs**

Add the exact builder signature and this concise example to
`docs/python-api.md`. Add `table_matmul` to the builder list and the two new
catalog entries to `docs/api-reference.md`.

Do not claim strict end-to-end zero-copy. Use “no redundant execution copies”
and repeat the exact physical allocation ceilings.

- [ ] **Step 5: Run example, docs, and formatting checks**

Run:

```bash
JAX_PLATFORMS=cpu uv run python examples/07_array_and_dataframe.py
JAX_PLATFORMS=cpu uv run pytest tests/test_examples.py -q -k test_example_script_runs
JAX_PLATFORMS=cpu uv run pytest \
  python/tests/test_array.py::test_array_and_dataframe_example_uses_table_matmul \
  -q
uv run ruff check examples/07_array_and_dataframe.py
uv run ruff format --check examples/07_array_and_dataframe.py
git diff --check
```

Expected:

- exact NumPy and JAX matrices print;
- all numbered example scripts pass;
- focused shape test passes;
- lint, format, and diff checks pass.

The separate missing-notebook baseline in
`test_quickstart_notebook_is_clean_and_valid` is not part of this feature and
must not be changed unless it independently exists on the implementation
branch.

- [ ] **Step 6: Commit examples and docs**

```bash
git add \
  examples/07_array_and_dataframe.py \
  examples/README.md \
  docs/python-api.md \
  docs/api-reference.md \
  docs/superpowers/specs/2026-07-24-table-array-matmul-design.md \
  docs/superpowers/plans/2026-07-24-table-array-matmul.md \
  python/tests/test_array.py
git commit -m "docs: demonstrate table matrix multiplication"
```

---

### Task 7: Run Full Verification and Packaging Inspection

**Files:**

- Verify only; modify files only to fix failures caused by Tasks 1-6.
- Inspect: `schemas/project-v2.schema.json`
- Inspect: `web-ui/openapi.json`
- Inspect: `web-ui/src/api/schema.d.ts`
- Inspect: built core wheel and source distribution under `target/`.

**Interfaces:**

- Consumes: all completed feature tasks.
- Produces: exact command evidence, clean generated contracts, inspected
  artifacts, and no source-tree native module.

- [ ] **Step 1: Run Rust formatting, lint, tests, coverage, and rustdoc**

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

Expected: every command exits 0; line coverage is at least 90%.

- [ ] **Step 2: Rebuild the binding and run all Python checks**

Keep build outputs under `target/`:

```bash
UV_CACHE_DIR=target/uv-cache uv sync --extra dev
CARGO_TARGET_DIR=target/maturin UV_CACHE_DIR=target/uv-cache \
  uv run maturin develop
JAX_PLATFORMS=cpu UV_CACHE_DIR=target/uv-cache \
  uv run pytest python/tests -q
UV_CACHE_DIR=target/uv-cache uv run ruff check .
UV_CACHE_DIR=target/uv-cache uv run ruff format --check .
```

Expected: 0 failures and no formatting drift.

- [ ] **Step 3: Run Studio backend and frontend matrices**

```bash
cd web-ui/backend
UV_CACHE_DIR=../../target/uv-cache \
  uv run --project . --extra dev pytest --cov=calc_flow_studio
cd ..
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev
cd ..
```

Expected: every command exits 0 and backend coverage remains at least 85%.

- [ ] **Step 4: Run supply-chain and release helper checks**

```bash
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

Expected: exit 0.

- [ ] **Step 5: Prove generated contracts are unchanged**

```bash
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 6: Build and inspect the core wheel and source distribution**

Use repository release helpers and keep artifacts under `target/`:

```bash
CARGO_TARGET_DIR=target/release-build \
  UV_CACHE_DIR=target/uv-cache \
  uv run maturin build --release --out target/dist
UV_CACHE_DIR=target/uv-cache \
  uv build --sdist --out-dir target/dist
python scripts/inspect_wheel.py target/dist/*.whl
python scripts/inspect_wheel.py target/dist/*.tar.gz
```

Expected:

- wheel inspection passes;
- source-distribution inspection passes;
- NumPy and JAX remain optional Python dependencies;
- no repository-only guidance, fixtures, Studio package, or frozen source tree
  leaks into the core artifacts.

- [ ] **Step 7: Prove NumPy and JAX remain optional at import time**

```bash
uv venv --python 3.13 target/core-wheel-smoke
UV_CACHE_DIR=target/uv-cache \
  uv pip install --python target/core-wheel-smoke/bin/python target/dist/*.whl
target/core-wheel-smoke/bin/python -c \
  'import importlib.util; assert importlib.util.find_spec("numpy") is None; import calc_flow; print(calc_flow.__version__)'
```

Expected: the installed core imports and prints `2.0.0` without NumPy or JAX
installed. If the platform virtual-environment executable lives under
`Scripts/python.exe`, use that exact executable for the same command.

- [ ] **Step 8: Verify source hygiene and exact scope**

```bash
find python/calc_flow -maxdepth 1 -name '_native*.so' -print
git status --short
git diff --stat main...HEAD
git diff --check main...HEAD
```

Expected:

- `find` prints nothing generated by this work;
- status is clean after task commits;
- the diff contains only the planned binding, Python, test, example, docs,
  dependency, design, and plan files;
- diff check exits 0.

- [ ] **Step 9: Create a verification-only commit only if generated lock or formatting changes were necessary**

If and only if the verification steps legitimately changed `Cargo.lock` or
formatted planned files:

```bash
git add \
  Cargo.lock \
  crates/calc-flow-python/Cargo.toml \
  crates/calc-flow-python/src/batch.rs \
  crates/calc-flow-python/src/config.rs \
  crates/calc-flow-python/src/provider.rs \
  python/calc_flow/_native.pyi \
  python/calc_flow/array.py \
  python/calc_flow/pipeline.py \
  python/tests/test_array.py \
  python/tests/test_pipeline.py \
  python/tests/test_runtime.py \
  examples/07_array_and_dataframe.py \
  examples/README.md \
  docs/python-api.md \
  docs/api-reference.md \
  docs/superpowers/specs/2026-07-24-table-array-matmul-design.md \
  docs/superpowers/plans/2026-07-24-table-array-matmul.md
git diff --cached --check
git commit -m "chore: finalize table matrix verification"
```

Otherwise do not create an empty commit.

---

## Final Review Checklist

- [ ] `PipelineBuilder.table_matmul` is functional, defensive, deterministic,
  and limited to NumPy/JAX.
- [ ] The project contains exact `table`, `weights`, and `output` mixed-kind
  ports and remains strict v2 data.
- [ ] Legacy single-array provider callbacks are unchanged.
- [ ] Mapping callbacks receive and return exact declared names and kinds.
- [ ] NumPy execution allocates only the dense table and result buffers after
  input batch construction.
- [ ] NumPy result adoption preserves the pointer and cannot be made writable.
- [ ] JAX uses one bounded host staging buffer, one device table buffer, and
  one result without a result-to-host round trip.
- [ ] Missing/invalid columns, nulls, ranks, shapes, backends, and dtypes fail
  with field-specific messages.
- [ ] The example prints the expected NumPy and JAX matrices and does not teach
  pandas/Polars or implicit batch-kind conversion.
- [ ] All commands in Task 7 have fresh successful evidence.
- [ ] No native module remains in `python/calc_flow/`.
- [ ] No push, PR, review-thread mutation, or merge occurs without explicit
  user authorization.
