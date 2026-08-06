# Canonical docs examples - API Note

This note is the direct input for `cf-doc-writer`. It defines the ONE canonical
example set spanning the Rust and Python surfaces, the consistency decisions
that bind them, and an exact per-doc mapping of which file to embed or
reference. Doc snippets should be **verbatim excerpts** of the checked example
files (or the canonical Tier-1 snippet below), never freshly invented variants.

Goal: every user-facing doc that describes an API or feature embeds or links a
runnable example, and the same example is reused across docs so snippets stop
diverging.

Status: the documentation reconciliation described by this note has landed.
The per-doc divergence column records the pre-reconciliation state and remains
here as historical evidence, not as a claim about the current docs.

## Audiences

- Rust users: the `calc-flow` crate examples under
  `crates/calc-flow/examples/`.
- Python users: the `examples/` scripts and the inline Tier-1 snippet.
- Studio clients: n/a (no REST examples in scope for this pass).

## Canonical example inventory

### Rust (`crates/calc-flow/examples/`)

| File                      | Demonstrates                                                    | API surface        | Status    |
| ------------------------- | --------------------------------------------------------------- | ------------------ | --------- |
| `expression_pipeline.rs`  | One-node `ExpressionOperator`; `total = a + b` to `[3, 7]`      | Rust core          | Aligned   |
| `sql_join.rs`             | Multi-input `SqlOperator`; orders + fees to net `[70, 108, 36]` | Rust core          | NEW       |
| `micro_batch_recovery.rs` | Replay `Source`, recording `Sink`, checkpoint commit + recovery | Rust core runners  | Unchanged |
| `export_schema.rs`        | Print the canonical v2 project JSON Schema                      | Rust core projects | Unchanged |
| `README.md`               | Index of the four examples with run commands                    | Rust core          | NEW       |

### Python (`examples/`)

| File                         | Demonstrates                                                | API surface    | Status    |
| ---------------------------- | ----------------------------------------------------------- | -------------- | --------- |
| `01_datafusion_pipeline.py`  | Expression node + projection + filter over an orders table  | Python binding | Unchanged |
| `02_sql_join.py`             | `SqlOperator` join of orders + fees to net                  | Python binding | Unchanged |
| `03_registered_udf.py`       | Trusted Python scalar UDF `double_amount`                   | Python binding | Unchanged |
| `04_micro_batch_recovery.py` | Replay source, checkpoint commit, cursor recovery           | Python runners | Unchanged |
| `05_async_execution.py`      | `execute_async()` inside asyncio                            | Python binding | Unchanged |
| `06_numpy_array.py`          | NumPy provider + bounded array expression `x - mean(x)`     | Python arrays  | Unchanged |
| `README.md`                  | Index of the six scripts + Rust counterpart cross-reference | Python binding | Aligned   |

The six Python scripts are unchanged: they already cover the documented
features one-to-one and pass `ruff check` and `ruff format --check`. The work on
the Python side is doc-side (aligning the inline doc snippets to these files).

## Consistency decisions

### 1. One canonical "first example" (Tier-1), identical on both surfaces

Every quickstart, introduction, smoke test, and API-reference "minimal" snippet
embeds the SAME example. Pipeline name `"totals"`, expression `total = a + b`,
inputs `a = [1, 3]`, `b = [2, 4]`, output `[3, 7]`.

Python (matches `docs/python-api.md` "Table batches and builder" verbatim):

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
plan = (
    PipelineBuilder("totals")
    .expression("calculate", "total = a + b")
    .compile()
)
result = plan.execute({"input": batch})

assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3, 7]
```

Rust (matches `crates/calc-flow/examples/expression_pipeline.rs` verbatim; the
file is the canonical Rust Tier-1 and should be embedded whole where a full
program fits, or excerpted):

```rust
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, ExecutionOptions, ExpressionOperator, PipelineBuilder, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = PipelineBuilder::new("totals")?
        .add_node(
            "calculate",
            Box::new(ExpressionOperator::new(
                "calculate",
                "total = a + b",
                Vec::new(),
                None,
                Vec::new(),
            )?),
        )?
        .compile(&UdfRegistry::new().snapshot())?;
    let input = RecordBatch::try_from_iter(vec![
        (
            "a",
            Arc::new(Int64Array::from(vec![1, 3])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("b", Arc::new(Int64Array::from(vec![2, 4])) as _),
    ])?;
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                Batch::table(vec![input], BatchMetadata::default())?,
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = result.outputs["output"].table_payload()?;
    let totals = output.batches()[0]
        .column_by_name("total")
        .expect("expression output contains total")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("total is an Int64 column");

    assert_eq!(totals.values(), &[3, 7]);
    println!("calculated totals: {totals:?}");
    Ok(())
}
```

### 2. Pipeline names are stable per scenario

| Scenario        | Pipeline name (both surfaces) |
| --------------- | ----------------------------- |
| First example   | `totals`                      |
| SQL join        | `orders-and-fees`             |
| Recovery        | `recovery-example`            |
| Async (Python)  | `async-example`               |
| NumPy (Python)  | `numpy-array`                 |

### 3. The SQL example shares one dataset on both surfaces

orders `(order_id = [1, 2, 3], amount = [75, 120, 40])` joined to fees
`(order_id = [1, 2, 3], fee = [5, 12, 4])` producing `net = amount - fee` to
`[70, 108, 36]`, pipeline `orders-and-fees`. Python: `examples/02_sql_join.py`.
Rust: `crates/calc-flow/examples/sql_join.rs`.

### 4. The doc is the follower, the file is the source of truth

Where a doc section illustrates the same concept as a checked example file, the
doc embeds the file verbatim (or a verbatim excerpt) and links the file. When a
doc snippet currently diverges, the doc is changed to match the file, never the
reverse.

### 5. Recovery examples are deliberately not expression-aligned

`04_micro_batch_recovery.py` computes `result = value + 1` over `[1, 2, 3]`;
`micro_batch_recovery.rs` computes `total = a + b` over items `(1, 2)` and
`(3, 4)`. Recovery demonstrates the checkpoint lifecycle, not the calculation,
so the differing expressions are acceptable and should NOT be force-aligned.

## Per-doc mapping (what each doc should embed or reference)

| Doc                       | Section                    | Use                                                                                                               | Pre-reconciliation divergence                                                                                    |
| ------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `README.md`               | Python quickstart          | Embed Tier-1 Python                                                                                               | Now `quantity * unit_price` `[2,3]/[10,4]` to `[20,12]`; change to `a + b`                                       |
| `README.md`               | Rust quickstart            | Embed `expression_pipeline.rs` verbatim                                                                           | Was byte-copy with pipeline `expression-example`; re-copy (now `totals`)                                         |
| `README.md`               | "Run the checked examples" | Keep the two `cargo run` lines; optionally add `sql_join`                                                         | None (optionally add the SQL example command)                                                                    |
| `docs/introduction.md`    | First example              | Embed Tier-1 Python AND Tier-1 Rust, side by side (now true twins)                                                | Python now `quantity * unit_price`; Rust pipeline now `expression-example`; align both to Tier-1                 |
| `docs/getting-started.md` | Smoke-test the engine      | Embed Tier-1 Python                                                                                               | Now `gross = quantity * unit_price` over orders; change to `a + b`                                               |
| `docs/getting-started.md` | Rust smoke pointer         | Reference `expression_pipeline.rs` + `cargo run` (already correct)                                                | None                                                                                                             |
| `docs/python-api.md`      | Table batches and builder  | Embed Tier-1 Python (already canonical; keep)                                                                     | None                                                                                                             |
| `docs/python-api.md`      | Multi-input SQL            | Align to `02_sql_join.py` dataset (orders/fees/net)                                                               | Now `left.a + right.b` cross join; switch to the file's dataset                                                  |
| `docs/python-api.md`      | Trusted Python scalar UDFs | Align to `03_registered_udf.py` (`double_amount`, column `amount`, `[100,250,400]`)                               | Now `double_value`/`value`; switch to the file's names                                                           |
| `docs/python-api.md`      | Async execution            | Align to `05_async_execution.py` (`total = a + b`, `[1,3]/[2,4]`)                                                 | Now `b = a + 1` over `[1,2]`; switch to the file's expression                                                    |
| `docs/python-api.md`      | NumPy and JAX              | Align to `06_numpy_array.py` data (`[1.0, 2.0, 4.0, 6.0]`)                                                        | Now `[1.0, 2.0, 3.0]`; switch to the file's data                                                                 |
| `docs/python-api.md`      | More examples              | Keep pointer to `examples/README.md`                                                                              | None                                                                                                             |
| `docs/rust-api.md`        | Expression pipeline        | Embed `expression_pipeline.rs` verbatim (already; re-copy for `totals`)                                           | Pipeline name now `expression-example` in file? no longer; re-copy                                               |
| `docs/rust-api.md`        | SQL operators              | Embed `sql_join.rs` verbatim (NEW)                                                                                | No runnable example today; add one                                                                               |
| `docs/rust-api.md`        | Micro-batch recovery       | Keep the `micro_batch_recovery.rs` excerpt (already)                                                              | None                                                                                                             |
| `docs/rust-api.md`        | Projects and stores        | Reference `export_schema.rs`                                                                                      | None (optionally add a pointer)                                                                                  |
| `docs/rust-api.md`        | "Run both paired examples" | Add the `sql_join` command to the existing two                                                                    | Add `cargo run -p calc-flow --example sql_join`                                                                  |
| `docs/api-reference.md`   | Examples section           | Reference `examples/README.md` and the NEW `crates/calc-flow/examples/README.md`; minimal snippet = Tier-1 Python | Rust list omits `sql_join.rs`; minimal snippet now `quantity * unit_price`; align to Tier-1 and add the SQL file |
| `docs/README.md`          | Project orientation        | Add a pointer to the Rust examples index (optional)                                                               | None                                                                                                             |

## What changed in this pass (example files only)

- `crates/calc-flow/examples/expression_pipeline.rs` pipeline renamed
  `expression-example` to `totals` so it matches `docs/rust-api.md` (already
  `totals`) and the Python Tier-1 snippet. This makes the Rust file and the
  Python Tier-1 true twins.
- `crates/calc-flow/examples/sql_join.rs` added; closes the Rust SQL example gap
  and mirrors `examples/02_sql_join.py`.
- `crates/calc-flow/examples/README.md` added; the Rust examples index, paralleling
  `examples/README.md`.
- `examples/README.md` gained a "Rust counterparts" cross-reference.

No engine, source, or runner behavior changed. The `unsafe_code = "forbid"`
lint posture is untouched.

## Verification evidence (this pass)

```
cargo build -p calc-flow --examples           # Finished, all examples compile
cargo run -p calc-flow --example expression_pipeline   # Arrow Int64 totals: 3 and 7
cargo run -p calc-flow --example sql_join              # net amounts: [70, 108, 36]
cargo run -p calc-flow --example micro_batch_recovery  # recovered deliveries: [1, 2]
cargo fmt --all --check                       # clean
cargo clippy -p calc-flow --all-targets --all-features -- -D warnings   # clean
uv run ruff check examples/                   # All checks passed!
uv run ruff format --check examples/          # 6 files already formatted
```

## Hand-off outcomes

1. **Rust native UDF example — scoped separately.** `docs/rust-api.md`
   accurately describes `UdfRegistry`. A new runnable example would require
   Rust example code, not a Markdown correction, so this documentation-only
   reconciliation does not add one.
2. **getting-started smoke narrative — resolved.** The smoke test and closing
   output now use Tier-1 (`a + b`, `[3, 7]`).
3. **api-reference Rust examples list — resolved.** The list now includes
   `expression_pipeline`, `sql_join`, `micro_batch_recovery`, and
   `export_schema`, with a link to `crates/calc-flow/examples/README.md`.
4. **SQL snippet in python-api.md — resolved.** The guide now uses the
   orders/fees dataset and links `examples/02_sql_join.py`.
