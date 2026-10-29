DAL-475 delivers a core wheel that calculates tables without Python PyArrow,
NumPy or JAX. Existing PyArrow defaults and UDF contracts remain explicit optional
adapters. This measurement does **not** establish universal performance equality.

## Revisions and environment

- Research: `eb30d22523ae62315eb87a9d8685bf1c5b8ac20c`.
- Baseline/current main: `d9e2759fb31bfd0c915d31f0cc534665ece72ee9`. Its additions after research
  are #514 benchmark evidence and #512 compilation/recovery cleanup; both are
  retained by the candidate. Remote main was checked again before delivery.
- Measured candidate code: `5c2c35154100fb256c6d87844eb27df17697e823`. Later evidence commits
  change reports only. Loaded native and Python source hashes, dependencies,
  harness hashes and build settings are in `data/summary.json` and
  `data/provenance.json`.
- WSL2 Linux x86-64, Intel i9-13900HX, CPython 3.13.9, NumPy 2.5.4,
  PyArrow 24.0.0, JAX/jaxlib 0.12.0.1. One pinned logical CPU; BLAS/OpenMP,
  Tokio workers and XLA CPU Eigen threading configured as recorded in the data.
- Baseline execution uses an archived Python tree plus its separately saved
  release native module. Its editable wheel is retained by hash, but is not a
  standalone comparison install. Candidate execution uses the unpacked normal
  wheel. Both use the same interpreter and installed dependencies. The clean
  standalone installation check is separate from those timed adapter workloads.
- Local wheels are CPython 3.13 Linux wheels. ABI3, other Python/platform versions
  and the full Rust/coverage gates remain CI responsibilities.

## Performance evidence

The five predeclared cases completed in **177.23 seconds**
of the 600-second budget, including worker startup, fixtures, preflight, first
execution, correctness, memory diagnostics and cleanup. No timing retries or
sample expansion were performed. Builds are separate: baseline 17m04s, initial
candidate 9m12s and the final binding rebuild 5m38s; lint preparation is not
included in those build durations.

Each case has two rounds of ten adjacent AB/BA pairs, with a fresh process per
invocation and the median of seven raw warm calls per invocation. All 200 worker
PIDs are distinct; all 1,400 raw call times are positive. Values, buffer sharing,
input snapshots, loaded native/source identity, dependencies and thread settings
were checked. The NumPy case reuses the maintained ownership/provider/plan
helpers. The maintained paired-median interval and +5% rules supply the verdicts.

| Case                         | Base ms | Candidate ms | Median delta | Paired verdict          |
|------------------------------|---------|--------------|--------------|-------------------------|
| Arrow round trip             | 0.155   | 0.177        | +14.1%       | inconclusive            |
| Short table compute          | 2.066   | 1.990        | -3.7%        | inconclusive            |
| NumPy expression + snapshot  | 0.461   | 0.416        | -9.8%        | inconclusive            |
| CPU table matrix             | 0.562   | 0.492        | -12.5%       | no-confirmed-regression |
| JAX CPU matrix + host return | 0.918   | 0.781        | -14.9%       | no-confirmed-regression |

Median delta above is the ratio of aggregate medians, for context. The verdicts
use the per-pair changes below; brackets are the maintained order-statistic
intervals (97.85% actual coverage for ten pairs).

| Case                         | Round 1: median [interval] | Round 2: median [interval] |
|------------------------------|----------------------------|----------------------------|
| Arrow round trip             | +16.4% [+2.9, +33.6]       | +9.6% [-0.3, +27.3]        |
| Short table compute          | -6.1% [-26.1, +0.1]        | -3.2% [-24.0, +7.0]        |
| NumPy expression + snapshot  | -1.9% [-15.3, +8.6]        | -4.2% [-22.5, +2.9]        |
| CPU table matrix             | -17.1% [-28.3, -6.8]       | -16.0% [-18.1, +1.6]       |
| JAX CPU matrix + host return | -10.4% [-15.7, +3.0]       | -11.2% [-13.7, -8.2]       |

Arrow round-trip median increased about 14%; its intervals remain inconclusive.
Short compute and the NumPy expression are also inconclusive. CPU and JAX matrix
cases show no confirmed regression at the +5% threshold. These are scoped
results, not proof of zero overhead, equivalence or an overall speedup.

JAX waits for completion and converts its result to a NumPy host view inside the
timed call. First execution includes compilation and is recorded separately in
every raw observation: JAX first-call medians were 13.60 ms baseline and 13.45 ms
candidate. These are not isolated compiler timings. No GPU or cross-device path
was measured. Large tables, alternative dtypes, supported minimum backend
versions and checkpoint throughput were not performance cases.

## Copies and memory

| Case                         | Base peak RSS MiB | Candidate peak RSS MiB | Base Python peak B | Candidate Python peak B |
|------------------------------|-------------------|------------------------|--------------------|-------------------------|
| Arrow round trip             | 123.0             | 123.0                  | 352                | 614                     |
| Short table compute          | 139.2             | 131.4                  | 38082              | 38534                   |
| NumPy expression + snapshot  | 123.0             | 123.0                  | 27158              | 27158                   |
| CPU table matrix             | 123.0             | 123.0                  | 3621               | 3874                    |
| JAX CPU matrix + host return | 270.1             | 263.1                  | 7566               | 7867                    |

RSS is whole-process `ru_maxrss`, including startup/preflight and inherited
process accounting; the repeated 123 MiB floor is not evidence of equal workload
memory. Tracemalloc covers Python-visible allocations only. Native allocation
counts, native allocation peaks and GPU memory were not instrumented; these
numbers cannot establish memory equivalence.

The directed tests retain the two-allocation NumPy matrix ceiling, one host
matrix/device placement for JAX, read-only result ownership, input snapshot
isolation and chunk handling. Compatible unsliced Arrow buffers remain shared.
For a non-byte-aligned sliced null bitmap, both baseline and candidate realign
validity on export; the logical data address remains shared. No claim is made
that every buffer/layout is copy-free.

## Functional and compatibility evidence

- Normal wheel resolution in a clean venv installed six distributions (the core
  plus Pydantic/typing dependencies), with no PyArrow/NumPy/JAX. `scripts/smoke_wheel.py`
  passed root import, Field declarations, compute/async, collection/async,
  Program execution, native stream output and missing-extra diagnostics.
- A separate environment with NumPy/JAX and **no PyArrow** passed native table
  matrix assembly, attached columns and caller snapshot checks through
  `arrays_no_pyarrow.py`. SQL casts preserve the declared float32/float64 rules.
- Real PyArrow 25.0.0 in an isolated environment was rejected at input, output and
  schema adapter entrypoints before exchange. The unsafe FFI lifecycle was not
  deliberately exercised. PyArrow 24 compatibility and nanoarrow 0.9.0 protocol
  production passed. Python 3.9/PyArrow 21 and backend minimum combinations were
  not run locally; their existing release matrix remains in place.
- Directed installed-wheel pytest: **555 passed**, including schema/metadata,
  chunk/offset/null/empty/zero-column behavior, producer errors, input identity,
  stream budget/order/cancellation cleanup, arrays, UDFs and compile caches.
- One unraisable Tokio runtime-drop warning occurred during a UDF test. The same
  warning at the same location reproduced against unmodified main (541 passed,
  one excluded implementation-specific helper test). It is an existing risk,
  not a passing claim about runtime destruction in every context.
- Candidate restored the baseline's v3 checkpoint at cursor 1 and emitted score
  12 after baseline score 8. Source/expression lineage, node digests, Program and
  stream fingerprints, and static digest version/hash match the saved identities.
  Changed weights are rejected before the source opens. State layout/version and
  provider identities were not changed.
- `cargo clippy -p calc-flow-python --lib --release -- -D warnings` passed;
  `cargo fmt --all --check`, scoped Ruff lint/format and 39 release-config tests
  passed. Source distribution inspection verified 1,180 files. Full Rust tests,
  cross-platform release checks and coverage were not run locally.

## Reproduction and evidence

`PLAN.md` fixes the cases and budget. `worker.py` and `measure.py` are specific to
this comparison. Prepare `target/baseline/python/calc_flow` from baseline Python
sources and its release native extension, and unpack the candidate wheel into
`target/candidate`. Use one dependency environment for both sides, then run:

```bash
PYTHONPATH=. .venv/bin/python .codex/artifacts/standalone/measure.py
```

The executed build commands were `uv sync --extra dev` at the baseline and
`.venv/bin/maturin build --release --interpreter .venv/bin/python --out target/candidate-wheels`
for the candidate. Core wheel installation used `uv pip install --python <clean-venv>/bin/python <wheel>`
with normal dependency resolution. The clean checks were:

```bash
target/standalone-venv/bin/python scripts/smoke_wheel.py
JAX_PLATFORMS=cpu target/array-venv/bin/python .codex/artifacts/standalone/arrays_no_pyarrow.py
PYTHONPATH=target/candidate:python/tests:. JAX_PLATFORMS=cpu .venv/bin/python .codex/artifacts/standalone/identity.py candidate
```

Identity recovery uses fresh target directories and the saved epoch-1 baseline
manifest. The comparison driver enforces the 600-second budget and settles each
worker synchronously. Raw observations and the original baseline manifest are
in `data/raw.tar.gz`; summaries, identity results and artifact hashes are beside
it. `data/SHA256SUMS` seals those files.
