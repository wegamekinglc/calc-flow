# Table-Array Matrix Multiplication Design

## Status

Approved on 2026-07-24; implemented and merged in PR #27. This document is a
point-in-time design record, not current API guidance.

## Problem

Calc Flow can execute Arrow table nodes through DataFusion and array nodes
through registered NumPy or JAX providers, but the Python provider bridge
currently exposes only one required array input and one required array output.
The existing mixed-data example therefore runs independent branches; it cannot
multiply numeric columns from a table by a separate backend weight matrix.

The new operation must:

- accept an Arrow table batch and a NumPy or JAX weight batch;
- select table columns in an explicit order and treat them as a rank-two
  feature matrix;
- calculate `table_matrix @ weights`;
- preserve immutable `Batch` semantics;
- avoid redundant copies after the input batches have been constructed; and
- demonstrate the behavior in a runnable Python example.

## Goals

- Add one functional Python builder step for table-array matrix
  multiplication.
- Support NumPy and JAX through the same project shape and backend-specific
  registered providers.
- Use the Rust engine's existing multi-port operator model rather than
  introducing a second execution path.
- Make the backend boundary, physical allocations, and device transfers
  explicit and testable.
- Preserve existing provider registrations, projects, callbacks, and
  single-input external nodes unchanged.

## Non-Goals

- Implicit conversion between arbitrary table and array nodes.
- A general DataFrame abstraction. The table input is an Arrow C Stream
  provider such as `pyarrow.Table`; pandas and Polars objects remain outside
  the public contract.
- Returning an Arrow table from the matrix operator. The output remains a
  backend array so the operation does not incur a backend-to-Arrow conversion.
- Sparse, batched, vector-matrix, or matrix-vector multiplication in version
  1. Both operands are rank two.
- Cross-backend multiplication or automatic NumPy-to-JAX/JAX-to-NumPy
  conversion.
- Weakening the rule that caller-owned inputs are never mutated.

## Public Python API

Add this immutable `PipelineBuilder` method:

```python
def table_matmul(
    self,
    node_id: str,
    *,
    backend: Literal["numpy", "jax"],
    columns: Sequence[str],
) -> PipelineBuilder: ...
```

It emits one external operator node with this exact project shape:

```json
{
  "id": "multiply",
  "input_ports": [
    {
      "kind": "table",
      "name": "table",
      "required": true,
      "schema": []
    },
    {
      "kind": "array",
      "name": "weights",
      "required": true,
      "schema": []
    }
  ],
  "operator": {
    "kind": "external",
    "name": "table_matmul",
    "options": {
      "columns": ["quantity", "unit_price"]
    },
    "provider": "numpy",
    "version": "1"
  },
  "output_ports": [
    {
      "kind": "array",
      "name": "output",
      "required": true,
      "schema": []
    }
  ]
}
```

`backend` selects the provider identity and is not duplicated in options.
`columns` is defensively copied, must contain at least one unique non-empty
column name, and is serialized in caller-specified order.

For a graph containing only this node, execution uses the unqualified input
names `table` and `weights` and output name `output`. Normal graph
qualification rules continue to apply when names collide.

`register_numpy(runtime)` registers
`numpy:table_matmul@1` in addition to the existing expression provider.
`register_jax(runtime)` does the same for `jax:table_matmul@1`. A plan fails
compilation if the matching registration is absent.

## Operator and Provider Architecture

Rust already represents operator inputs and outputs as deterministic
`BTreeMap<String, Batch>` values and supports arbitrary validated port lists.
No change is required to the core `Operator` trait or pipeline scheduler.

The Python binding gains an internal mapping-mode provider factory alongside
the existing single-array provider factory:

- single-array mode retains the callback contract
  `callback(batch, options) -> Batch`;
- mapping mode uses
  `callback(inputs, options) -> Mapping[str, Batch]`;
- the factory captures and enforces its exact input and output port
  declarations;
- input and output mappings are defensively copied and deterministically
  ordered;
- every returned batch must have been created by the Python host and must
  satisfy its configured port kind.

Mapping mode is an internal binding facility used by the built-in matrix
providers in version 1. It is not exposed as a new arbitrary callback
registration API. This keeps the public addition focused while leaving a
future general provider API possible without committing to its shape here.

The NumPy and JAX matrix providers share validation and data-flow rules while
using their backend namespace for dtype selection, dense materialization, and
matrix multiplication.

## Data Flow

Given an Arrow table `A` with `m` rows, `n` selected columns, and a backend
weight matrix `W` with shape `(n, p)`:

1. The engine passes borrowed clones of the immutable `table` and `weights`
   batches to the provider bridge. Arrow buffers and the Python array payload
   remain shared.
2. The provider validates table columns, weight backend, ranks, shapes, nulls,
   and dtypes before allocating the dense table matrix.
3. The selected Arrow columns are materialized in the requested order as one
   dense backend matrix with shape `(m, n)`.
4. The provider computes `A @ W`, producing one backend-native result with
   shape `(m, p)`.
5. The binding adopts the fresh result into an array batch without running the
   defensive caller-input copy path.
6. The output preserves the table batch metadata and adds only
   JSON-compatible attributes for selected columns, backend, and operation.

The weight batch is not modified or converted. The operator rejects a
different weight backend instead of silently crossing providers.

## Copy and Ownership Contract

The guarantee starts after the caller has constructed the two input batches.
`Batch.from_array` retains its existing defensive ownership behavior for
caller-supplied NumPy arrays; this design does not add an unsafe public
ownership-transfer constructor.

During operator execution:

- the Arrow batch crosses the Rust/Python provider boundary without copying
  its Arrow buffers;
- the weight batch crosses the boundary without copying its backend payload;
- NumPy performs one table-to-dense allocation and one matrix-result
  allocation;
- JAX may require one contiguous host staging allocation plus one
  host-to-device table transfer, followed by one device result allocation;
- JAX performs no result-to-host round trip;
- neither backend performs an additional provider-entry copy or
  output-wrapping copy.

The JAX staging allocation may be omitted when the installed Arrow/JAX
versions provide a directly usable dense transfer path, but the API does not
promise that optimization. The guaranteed ceiling is one host staging buffer,
one device table buffer, and one device result buffer.

The binding adds an internal owned-result constructor available only to
trusted provider code. For NumPy, it must:

- keep the result data pointer unchanged;
- hide the original writable owner behind an opaque binding-owned anchor;
- expose only a non-owning, read-only NumPy view;
- reject attempts to restore write access through `setflags(write=True)`; and
- release the allocation exactly once when the batch and all exported views
  are gone.

For JAX, the constructor retains the immutable `jax.Array` object without
conversion. If a safe implementation cannot prove both pointer preservation
and non-reenableable NumPy write protection, the implementation is blocked;
it must not weaken `Batch` immutability or silently add a freeze copy.

## Dtype and Shape Rules

- The table must contain at least one row and at least one selected column.
- Selected names must be unique and present in the Arrow schema.
- Selected columns must be primitive, non-null numeric columns. Boolean,
  string, binary, temporal, decimal, dictionary, nested, and extension types
  are rejected in version 1.
- Chunked columns are accepted and copied directly into their slice of the
  single dense table allocation; combining chunks must not allocate an
  intermediate full column.
- Weights must be a rank-two array with shape `(n, p)`, where `n` equals the
  selected column count and `p` is positive.
- The weights backend must equal the configured provider.
- The provider chooses a supported common backend dtype before allocating the
  dense table matrix. Unsupported or lossy promotions fail with an error that
  names the involved Arrow and weight dtypes.
- The result has shape `(table.num_rows, weights.shape[1])` and retains the
  configured backend.

## Errors

Errors identify the node/provider and the failing field:

- `columns`: empty, duplicate, missing, unsupported, or null-containing
  selected column;
- `weights.backend`: configured and actual backend differ;
- `weights.rank`: expected rank two;
- `weights.shape[0]`: expected the selected column count;
- `weights.shape[1]`: expected a positive output width;
- `dtype`: unsupported or lossy common dtype;
- `provider.output`: wrong output name, kind, backend, shape, or host
  ownership.

Validation that depends only on configuration runs during compilation.
Schema-, null-, backend-, shape-, and dtype-dependent validation runs before
the first allocation during execution.

## Example

Revise `examples/07_array_and_dataframe.py` to run both NumPy and JAX when the
corresponding optional dependency is installed. The example uses:

```text
Arrow columns       weights         result
[[3, 10],          [[2, 0],        [[6, 10],
 [1, 12],     @     [0, 1]]   =     [2, 12],
 [4, 10]]                            [8, 10]]
```

The NumPy path is always shown when the NumPy extra is installed. The JAX path
is a separate function using the same table and expected values. Missing JAX
is reported as an optional skipped demonstration rather than making the
NumPy example fail.

The example asserts output name, kind, backend, shape, values, input
immutability, and the absence of DataFusion metrics because calculation occurs
in the external provider. `examples/README.md` explains the explicit
table-to-backend materialization and its copy budget.

## Testing

### Rust core and project compilation

- Existing multi-port external operators continue to compile and execute.
- The generated v2 project schema remains unchanged because it already
  supports explicit mixed-kind port lists.
- Fingerprints remain deterministic for ordered column lists and port maps.
- Connections reject table/array kind mismatches at compile time.

### Python binding

- Legacy single-array callbacks retain their exact call and return contracts.
- Mapping-mode callbacks receive only the declared named inputs and must
  return exactly the declared named outputs.
- Input Arrow buffer addresses and weight object identities are unchanged
  across the provider boundary.
- NumPy owned-result adoption preserves the result pointer, exposes no
  reachable writable array owner, rejects `setflags(write=True)`, and releases
  once.
- JAX owned-result adoption preserves `jax.Array` identity and device without
  converting through NumPy.
- Invalid callback mappings and foreign-host payloads fail with provider
  identity and output name.

### NumPy and JAX providers

- Both backends produce `[[6, 10], [2, 12], [8, 10]]`.
- Multi-chunk Arrow inputs produce the same result without an intermediate
  full-column allocation.
- Missing, duplicate, null, unsupported, empty, and incompatible inputs
  produce the specified errors before matrix allocation.
- Instrumented allocation tests enforce the backend-specific copy ceilings.
- NumPy tests prove that provider entry and output adoption do not call the
  defensive `_owned_numpy` path.
- JAX tests reject any attempted conversion of the weight or result through
  `numpy.asarray` and confirm that the result remains on its selected device.
- Caller-owned Arrow and weight inputs remain unchanged.

### Example and documentation

- The numbered-example harness runs the NumPy path.
- A focused JAX test runs the JAX example path on CPU.
- Ruff, Python tests, Rust tests, coverage, rustdoc, generated-contract drift
  checks, and `git diff --check` pass according to `AGENTS.md`.

## Compatibility and Release Impact

The builder method and provider catalog entries are additive. Existing project
documents, checkpoints, provider callbacks, and expression nodes retain their
behavior. The strict project schema and Studio OpenAPI contract do not change.

The Python API documentation, API reference, example index, and native stub
must describe any new internal native method that is visible to type checking.
Because the change affects the binding and optional array providers, core
wheel and source-distribution inspection must confirm that NumPy and JAX
remain optional and no generated native module is left in `python/calc_flow/`.

## Risks and Mitigations

- **NumPy ownership adoption is unsafe or writable:** block implementation
  unless pointer and write-protection tests pass without local `unsafe` code.
- **JAX staging creates hidden transfers:** enforce the stated ceiling with
  backend-specific instrumentation and document device placement.
- **The convenience method drifts from project v2:** assert its canonical
  project JSON and round-trip it through Rust validation.
- **Mapping mode breaks legacy providers:** keep separate internal factory
  modes and run the full existing provider suite unchanged.
- **Dtype promotion differs by backend:** test an explicit supported matrix and
  reject promotions that cannot be made consistent without loss.
