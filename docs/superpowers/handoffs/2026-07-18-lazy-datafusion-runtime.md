# Lazy DataFusion Runtime Evidence

## Outcome

The external-passthrough usefulness gate passed both thresholds. Its Criterion
slope point estimate fell from `41.53283506723807 us` to
`2.3653721841732276 us`, a reduction of `39.16746288306484 us` and
`94.30481405773556%`. The reduction exceeds both the required `10 us` and
`20%` gates, so the lazy run-scoped DataFusion implementation is retained.

The table-expression control fell from `453.9091107305059 us` to
`412.2864534328358 us`. Criterion reported no statistically detected change
for that control (`p = 0.06`), so the target improvement was not accompanied by
a material expression-path regression.

## Revisions

- Baseline: `b333121b282861ea03e006db7a2a232f6a6566c2`
- Implementation: `69403f1a32a3c85d17bb0b790e881329e4fd0c82`

## Environment

| Item                          | Value                                                              |
| ----------------------------- | ------------------------------------------------------------------ |
| Operating system              | Linux `5.15.167.4-microsoft-standard-WSL2`                          |
| Architecture                  | `x86_64`                                                           |
| Normalized CPU brand          | `13th gen intel(r) core(tm) i9-13900hx`                             |
| Logical CPU count             | `32`                                                               |
| Rust                          | `rustc 1.88.0 (6b00bc388 2025-06-23)`                              |
| Cargo                         | `cargo 1.88.0 (873a06493 2025-05-10)`                              |
| Python                        | `3.13.13`                                                          |
| NumPy                         | `2.5.1`                                                            |
| JAX                           | `0.11.0`                                                           |
| JAXlib                        | `0.11.0`                                                           |
| JAX execution                 | CPU; x64 disabled                                                  |
| Machine fingerprint           | `fe554238f5c55b49c8d1961066a86debe87eb538f7fbb6fc19a72f926e264a56` |
| CPU scaling governor          | Unavailable in the WSL2 guest                                      |
| CPU scaling driver            | Unavailable in the WSL2 guest                                      |
| Energy-performance preference | Unavailable in the WSL2 guest                                      |
| ACPI platform profile         | Unavailable in the WSL2 guest                                      |
| Intel P-state status          | Unavailable in the WSL2 guest                                      |
| AMD P-state status            | Unavailable in the WSL2 guest                                      |
| `powerprofilesctl` profile    | Command unavailable                                                |
| `cpupower` policy             | Command unavailable                                                |

The host power mode was therefore not observable or fixed from the WSL2 guest.

## Commands

The saved Criterion baseline was compared with the candidate using:

```bash
CARGO_TARGET_DIR="$PWD/target/cargo" CARGO_BUILD_JOBS=1 \
  cargo bench -p calc-flow --bench core -- \
  --baseline lazy-datafusion-before
```

The current Python environment and release binding were built using:

```bash
env -u CONDA_PREFIX \
  VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv sync --extra dev --extra benchmark
env -u CONDA_PREFIX \
  VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run maturin develop --release
```

The 16 provider/plan cases were collected into one report using:

```bash
env -u CONDA_PREFIX \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest \
  benchmarks/test_array_provider.py benchmarks/test_array_plan.py \
  -q --benchmark-only \
  --benchmark-json=target/benchmark-results/lazy-datafusion.json
```

## Criterion comparison

All values below are Criterion slope estimates converted from nanoseconds to
microseconds. `Change us` is candidate minus baseline, and `Change %` is
`100 * change_us / baseline_us`.

| Case                                           | Baseline us   | Candidate us  | Change us      | Change %        | Candidate 95% CI                    |
| ---------------------------------------------- | ------------- | ------------- | -------------- | --------------- | ----------------------------------- |
| `execute/datafusion_runtime_new`               | 35.209424017  | 0.019192482   | -35.190231535  | -99.945490496%  | `[0.019043409, 0.019366532] us`     |
| `execute/datafusion_runtime_new_register_udfs` | 34.364637840  | 0.056642317   | -34.307995523  | -99.835172665%  | `[0.055742208, 0.057661568] us`     |
| `execute/expression_1024_rows`                 | 453.909110731 | 412.286453433 | -41.622657298  | -9.169821956%   | `[402.819405257, 424.901921892] us` |
| `execute/external_passthrough_1000_rows`       | 41.532835067  | 2.365372184   | -39.167462883  | -94.304814058%  | `[2.323962549, 2.408507666] us`     |

The external-passthrough reduction is the gate measurement. The runtime cases
explain the removed setup, while the expression case is the independent table
path control; these overlapping measurements are not summed.

## Contract-v2 Python evidence

The report at `target/benchmark-results/lazy-datafusion.json` has SHA-256
`5682f8b549f9c3b7ea91a0c6a34fa156ba0bb9705c4c7c3e6d14d9400c9d6179`.
The command passed all `16` cases in `34.02 s`: eight `provider_boundary`
entries and eight `plan_end_to_end` entries.

All 16 entries use benchmark contract version 2 and contain complete machine,
dependency, and workload identity documents plus lower-case SHA-256
fingerprints. Recomputing all three fingerprints for every entry checked `48`
fingerprints with `0` mismatches. Each of the eight scenario/backend
provider/plan pairs has one matching machine identity and fingerprint, one
matching dependency identity and fingerprint, and one matching workload
identity after removing only the expected `scope` field. The NumPy dependency
fingerprint is
`dc347e7721846de1ab42e877047ff66a9ca37f6e95855ef46e85274559dbf302`;
the JAX dependency fingerprint is
`ef27ba28ef2b53bf88ad5c1fb510e368a95f4ebbc0cd08537da625bbd0d200a4`.
The same-report provider/plan entries are therefore machine- and
dependency-compatible.

Provider means range from `15.248368702256155 us` to
`120.70715113321795 us`, with CoVs from `30.581409722771305%` to
`185.15612926885177%`. Plan means range from `22.333070540000907 us` to
`133.9812458009066 us`, with CoVs from `28.628368270383056%` to
`103.56260925132808%`.

The existing array-benchmark noise rule rejects a timing classification when a
report CoV exceeds `5%`. Every provider and plan case exceeds that ceiling, so
this one-report Python result is diagnostic only and does not permit a
performance claim. No cross-report comparison or subtraction is used for the
retention decision; the same-host Criterion comparison is the primary gate.

## Correctness and API evidence

The focused command covering the Calc Flow library plus DataFusion, UDF, and
pipeline execution tests passed `59/59` tests: `10` library tests, `9`
DataFusion integration tests, `19` UDF integration tests, and `21` pipeline
execution tests.

The public `DataFusionRuntime` method signatures are unchanged from the
baseline: `new(DataFusionConfig) -> Result<Self>`,
`register_udfs(&mut self, &UdfRegistrySnapshot, &[UdfReference]) -> Result<()>`,
`evaluate(&self, &str, &Batch, Option<&str>) -> Result<Batch>`,
`sql(&self, &str, &BTreeMap<String, Batch>, Option<&str>) -> Result<Batch>`,
`metrics(&self) -> Vec<DataFusionQueryMetric>`, and `close(&self)`. The change
adds only private lazy-session state and a private `context()` initializer;
the public `OperatorContext`, `Operator`, `ExecutionPlan`, and `RunResult`
interfaces are untouched.

The external NumPy plan diagnostic returned `1,000` output rows and
`datafusion_metrics=0`, preserving external-only observability. The focused
test `runtime_registers_and_executes_only_selected_native_references` proves a
selected native UDF is installed for the first lazy query and an unselected UDF
remains unavailable. `runtime_registers_a_native_udf_after_the_lazy_session_exists`
proves late registration after an initial query. The two-worker
`overlapping_evaluations_isolate_the_input_alias` test starts with a fresh
runtime, completes both concurrent first-query calls with distinct row counts,
and records two metrics, covering one shared lazy initialization plus alias
isolation. The private tests also prove preparation and close do not initialize
a session and repeated `context()` calls reuse one session.

## Decision

Retain commit `69403f1a32a3c85d17bb0b790e881329e4fd0c82`. The external-only
path saves `39.16746288306484 us` (`94.30481405773556%`) against the exact
same-host baseline, passing both usefulness thresholds, while the expression
control has no detected regression and all `59` focused correctness tests pass.
The compatible contract-v2 Python report is retained as diagnostic evidence
only because every CoV exceeds the existing `5%` noise ceiling.
