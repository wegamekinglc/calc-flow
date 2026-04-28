# Tech Spec: Array Engine API Surface

## Motivation

`docs/introduction.md` specifies two distinct requirements for the calculation engines:

> 1. dataframe engine — "should support an expression engine"
> 2. array engine — "should define a **set of apis** to do the array and matrix calculation"

Requirement (1) is covered: `PandasEngine.evaluate()`, `PolarsEngine.evaluate()` + `sql()`, `DataFusionEngine.evaluate()` + `sql()` all support expression evaluation.

Requirement (2) is **not** covered. The array engines (`NumpyEngine`, `JaxEngine`) currently expose only `evaluate(expression, batch)` — a single string-based expression evaluator, identical in shape to the dataframe approach. The Python Array API standard defines a rich namespace of functions (`add`, `matmul`, `sum`, `transpose`, etc.), but the engine does not surface them as first-class methods.

## Design

### `xp` property

Add an `xp` property to `ArrayEngine` that returns the Array API namespace module. Each subclass provides its own:

```python
class ArrayEngine(Engine):
    @property
    def xp(self) -> Any:
        raise NotImplementedError

class NumpyEngine(ArrayEngine):
    @property
    def xp(self):
        import numpy as np
        return np

class JaxEngine(ArrayEngine):
    @property
    def xp(self):
        import jax.numpy as jnp
        return jnp
```

This allows all operation methods to be defined once on `ArrayEngine` and shared across backends.

### Operation methods

Operation methods return raw Array API arrays backed by the engine's namespace.
All operands are coerced with `self.xp.asarray(...)`, keeping backend ownership
explicit: `NumpyEngine` operations return NumPy arrays, and `JaxEngine`
operations return JAX arrays, even when callers pass mixed inputs.

Results are wrapped with `self.xp.asarray(...)` before returning to keep the
contract consistent when backends return scalar or 0-D values (e.g.
`sum(axis=None)`).

| Category | Method | Array API function |
|---|---|---|
| Element-wise | `add(left, right)` | `xp.add` |
| | `subtract(left, right)` | `xp.subtract` |
| | `multiply(left, right)` | `xp.multiply` |
| | `divide(left, right)` | `xp.divide` |
| Linear algebra | `matmul(left, right)` | `xp.matmul` |
| Reductions | `sum(data, *, axis)` | `xp.sum` |
| | `mean(data, *, axis)` | `xp.mean` |
| | `max(data, *, axis)` | `xp.max` |
| | `min(data, *, axis)` | `xp.min` |
| Shape | `transpose(data, *, axes)` | `xp.permute_dims` |
| | `reshape(data, shape)` | `xp.reshape` |

All methods operate on raw arrays. Operands are coerced with `self.xp.asarray(...)`;
results are wrapped with `self.xp.asarray(...)`.

### `evaluate()` simplified

`evaluate(expression, data)` evaluates expressions against Array API arrays.
The scope contains `{"x": arr, "xp": namespace}`. Assignment expressions
(`"c = ..."`) are parsed but only the value expression is evaluated — arrays
don't have named columns.

## Files modified

| File | Change |
|---|---|
| `src/calc_flow/engine/array.py` | Added `xp` property + operation methods. Simplified `_evaluate_expression` (arrays only). |
| `tests/calc_flow/engine/test_array.py` | Tests for each operation method on both engines. |

## Verification

```bash
uv run pytest tests/calc_flow/engine/test_array.py -v
uv run ruff check .
uv run ruff format --check .
```

Test coverage includes:

- every operation method on `NumpyEngine` and `JaxEngine`;
- binary methods with two arrays, array + scalar, and array + raw array API array;
- reduction methods with `axis=None` and a specific axis, including 0-D scalar results;
- backend coercion, e.g. raw NumPy input to `JaxEngine` returns a JAX array;
- `evaluate()` expression evaluation on raw arrays.
