# Engine Boundary Isolation Evidence

## Outcome

External-only plans are now structurally independent of the table engine.
They compile to `table = None`, retain no DataFusion configuration, UDF
snapshot, or selected UDF list, construct no `DataFusionRuntime`, and expose
no DataFusion metrics. Table-only and mixed plans retain the existing one
run-scoped DataFusion session and selected-UDF behavior.

The native external passthrough measured `2.216533 us` with a 95% confidence
interval of `[2.205106, 2.229365] us`. The previous same-host lazy-runtime
evidence reported `2.365372 us`; the unpaired reference difference is
`-0.148840 us` (`-6.292%`). This is a reference comparison, not a paired
Criterion change test. The architectural classification and focused tests are
the isolation proof.

The contract-v2 Python report passed all 16 NumPy/JAX cases. All identity
fingerprints match the prior report, but every new case has a CoV above the
existing 5% noise ceiling. The report is therefore diagnostic and does not
support a directional Python performance claim.

## Revisions

- Prior published branch head: `5be0e5a6b84a01162d180ce737a6868de1113890`
- Measured implementation head: `89b03b937ad50931bda159f360d86e9dc5a03cd6`
- Review-fixed verification head: `618127ec15539a33657b544cc3263787a60b379d`
- Earlier same-host lazy-runtime evidence: `69403f1a32a3c85d17bb0b790e881329e4fd0c82`

All final code, test, and release verification below ran at the review-fixed
verification head. The refreshed benchmark table in this handoff is the only
uncommitted documentation change on top of that head; runtime behavior was
fully committed before measurement.

## Environment

| Item                   | Value                                                              |
| ---------------------- | ------------------------------------------------------------------ |
| Operating system       | Linux `5.15.167.4-microsoft-standard-WSL2`                         |
| Architecture           | `x86_64`                                                           |
| CPU                     | 13th Gen Intel Core i9-13900HX                                     |
| Logical CPUs           | `32`                                                               |
| Rust                   | `rustc 1.88.0 (6b00bc388 2025-06-23)`                              |
| Cargo                  | `cargo 1.88.0 (873a06493 2025-05-10)`                              |
| Python                 | `3.13.13`                                                          |
| NumPy                  | `2.5.1`                                                            |
| JAX / JAXlib           | `0.11.0` / `0.11.0`                                                |
| JAX execution          | CPU; x64 disabled                                                  |
| Machine fingerprint    | `fe554238f5c55b49c8d1961066a86debe87eb538f7fbb6fc19a72f926e264a56` |
| NumPy dependency hash  | `dc347e7721846de1ab42e877047ff66a9ca37f6e95855ef46e85274559dbf302` |
| JAX dependency hash    | `ef27ba28ef2b53bf88ad5c1fb510e368a95f4ebbc0cd08537da625bbd0d200a4` |

## Commands

Focused native correctness and binding tests:

```bash
CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow \
  --test config --test pipeline_compile --test pipeline_execute --test operator
LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib \
  CARGO_TARGET_DIR=target/cargo cargo test -p calc-flow-python --lib
```

Release binding and focused Python correctness:

```bash
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  .venv/bin/maturin develop --release
JAX_PLATFORMS=cpu VIRTUAL_ENV="$PWD/.venv" \
  .venv/bin/pytest python/tests/test_array.py python/tests/test_pipeline.py -q
```

Native controls and contract-v2 Python benchmarks:

```bash
CARGO_TARGET_DIR=target/cargo CARGO_BUILD_JOBS=1 \
  cargo bench -p calc-flow --bench core -- 'execute/'
CALC_FLOW_BENCHMARK_SCALE=overhead JAX_PLATFORMS=cpu \
  VIRTUAL_ENV="$PWD/.venv" .venv/bin/pytest \
  benchmarks/test_array_provider.py benchmarks/test_array_plan.py \
  -q --benchmark-only \
  --benchmark-json=target/benchmark-results/engine-isolation.json
```

## Final verification

- `cargo fmt --all --check` passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  passed.
- Bounded final Rust tests passed: `234/234` core tests and `49/49` binding
  tests. The monolithic all-target workspace command was attempted twice, but
  its statically linked test binaries exceeded the constrained 16 GiB
  filesystem; the same tests were therefore run as package groups, while all
  targets were compiled by Clippy.
- `cargo llvm-cov --workspace --all-features --fail-under-lines 90` passed at
  `90.68%` line coverage.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`
  passed.
- `223/223` Python adapter tests passed; Ruff check and format verification
  passed.
- Studio verification passed: `90/90` backend tests at `91.53%` coverage,
  `106/106` frontend unit tests, `1/1` Playwright workflow, production build,
  generated API synchronization, and production dependency audit.
- Supply-chain checks passed: Cargo audit with the two repository-approved
  RustSec ignores, Cargo deny, and `26/26` release-helper tests.
- The project schema, OpenAPI document, and generated TypeScript API remained
  unchanged; `git diff --check` passed.

## Native evidence

All values are Criterion slope estimates. The prior column is the candidate
from the earlier same-host lazy-runtime handoff. Because the saved Criterion
baseline was removed with generated build artifacts after the filesystem filled,
the changes below are unpaired references rather than a new statistical
comparison.

| Case                                           | Prior us   | Isolated us | Reference change | Isolated 95% CI                    |
| ---------------------------------------------- | ---------- | ----------- | ---------------- | ---------------------------------- |
| `execute/datafusion_runtime_new`               | 0.019192   | 0.018509    | -3.559%          | `[0.018363, 0.018663] us`          |
| `execute/datafusion_runtime_new_register_udfs` | 0.056642   | 0.047853    | -15.517%         | `[0.047616, 0.048115] us`          |
| `execute/expression_1024_rows`                 | 412.286453 | 374.763071  | -9.101%          | `[372.829110, 376.894901] us`      |
| `execute/external_passthrough_1000_rows`       | 2.365372   | 2.216533    | -6.292%          | `[2.205106, 2.229365] us`          |
| `execute/external_plan_table_requirement`      | n/a        | 0.000239    | n/a              | `[0.000236, 0.000243] us`; `false` |

The table-runtime and expression cases are independent controls. They are not
summed with external passthrough. The requirement accessor is an optimized
control for the compiled `false` classification, not useful work.

The modest remaining native external cost is graph machinery shared by both
engines: input/output validation, cancellation checks, the run transaction,
operator locking, rollback snapshots, node timing, and result construction.
It is not DataFusion session setup.

## Contract-v2 Python evidence

The report `target/benchmark-results/engine-isolation.json` has SHA-256
`83f72e833eab5cad86b84780106adc48b0dbd5042745f39dc1845e1ceb6e90b9`.
It contains 16 passing cases: eight provider-boundary cases and eight
plan-end-to-end cases. All `16/16` machine, dependency, and workload
fingerprint triplets exactly match `lazy-datafusion.json`.

| Scope    | Backend | Scenario                  | Prior mean us | New mean us | Change   | New CoV  |
| -------- | ------- | ------------------------- | ------------- | ----------- | -------- | -------- |
| Provider | NumPy   | Elementwise               | 28.334        | 23.323      | -17.687% | 47.453%  |
| Provider | NumPy   | Mean                      | 15.248        | 14.640      | -3.987%  | 43.282%  |
| Provider | NumPy   | Matrix multiplication     | 16.202        | 15.684      | -3.199%  | 37.365%  |
| Provider | NumPy   | Transpose / reshape       | 17.857        | 18.190      | +1.865%  | 39.547%  |
| Provider | JAX     | Elementwise               | 120.707       | 111.903     | -7.293%  | 70.498%  |
| Provider | JAX     | Mean                      | 70.557        | 68.464      | -2.967%  | 86.971%  |
| Provider | JAX     | Matrix multiplication     | 67.075        | 66.715      | -0.537%  | 51.901%  |
| Provider | JAX     | Transpose / reshape       | 73.242        | 93.139      | +27.166% | 79.971%  |
| Plan     | NumPy   | Elementwise               | 31.301        | 29.248      | -6.559%  | 37.994%  |
| Plan     | NumPy   | Mean                      | 22.333        | 22.805      | +2.111%  | 58.818%  |
| Plan     | NumPy   | Matrix multiplication     | 23.438        | 21.303      | -9.109%  | 40.404%  |
| Plan     | NumPy   | Transpose / reshape       | 25.705        | 24.293      | -5.493%  | 40.168%  |
| Plan     | JAX     | Elementwise               | 133.981       | 129.947     | -3.011%  | 39.264%  |
| Plan     | JAX     | Mean                      | 86.484        | 83.410      | -3.555%  | 53.572%  |
| Plan     | JAX     | Matrix multiplication     | 88.253        | 86.719      | -1.738%  | 45.078%  |
| Plan     | JAX     | Transpose / reshape       | 95.941        | 89.031      | -7.202%  | 35.837%  |

Every prior and new CoV exceeds the 5% noise rule. Changes in either direction
must therefore be treated as noise. The Python results confirm compatibility
and bound the observed path, but they are not the isolation proof.

## Correctness and ownership evidence

- `234/234` final core tests passed after optional table resources and the
  connected mixed-engine regression tests were introduced.
- `49/49` Rust binding unit tests passed.
- `223/223` Python binding and adapter tests passed.
- `16/16` NumPy/JAX benchmark cases passed.
- External-only project validation accepts inactive DataFusion settings,
  excludes them from the fingerprint, and reports `datafusion_config() == None`.
- A mixed graph still reports `requires_datafusion() == true` and retains its
  DataFusion configuration and fingerprint projection.
- External operators receive only `OperatorContext { run }`; table operators
  receive the DataFusion runtime through classified internal dispatch.

An attempted removal of output payload rehoming failed the binding GC test:
all 100 Python cycles remained alive. Sharing one Rust `Arc<PythonPayload>`
between `RunResult` and a returned `Batch` prevents the two GC containers from
clearing their Python roots independently. The shortcut was reverted. Provider
options are still serialized once at operator creation, while every callback
receives a fresh decoded Python value, preserving input immutability.

## Decision

Retain the engine classification and optional table resources. The change
eliminates the architectural coupling the investigation identified without
weakening mixed graphs, rollback, cancellation, UDF selection, metrics, or
Python GC safety.

The original large array-versus-direct gap is not a table-engine effect after
this change. Stable native external execution is about `2.2 us`; the remaining
Python gap may include PyO3 object conversion and rooting, defensive option
decoding, provider dispatch, graph lifecycle, and JAX dispatch/synchronization.
The noisy report does not establish which contributor dominates. Further
optimization should profile those components independently and keep the Python
payload rehome required by GC ownership.
