# Array Benchmark Phase 1 Evidence

> **Historical status:** Intermediate evidence. Its decision to require Phase 2
> was followed by the
> [Phase 2 evidence](2026-07-16-array-benchmark-phase2.md).

## Evidence outcome

The exact baseline engine at `b42b687c8021291bcb537a57fded7a40f7f8477d`
was compared with candidate `42af5c6583ba1a09ee8e3995295e647096208c4d`
using the candidate benchmark harness copied to
`/tmp/calc-flow-array-harness`. Both revisions used isolated release builds and
one frozen dependency set. No runtime source was changed by this task.

All 22 prescribed first-pass reports completed successfully: five alternating
overhead rounds per revision and three baseline-then-candidate ownership rounds
at each of the standard and nightly scales. All 264 contract-v2 entries were
complete, canonical fingerprints recomputed correctly, and every compared
machine, dependency, and workload fingerprint matched.

The timing gate is not classifiable. Every logical case/revision had a median
per-report CoV above the 5% ceiling. Fresh same-host baseline/candidate reruns
for all affected cases also remained noisy; their lowest CoV was 13.937%.
Accordingly, no noisy first-pass timing was aggregated into an acceptance claim.

## Revisions and environment

| Item                 | Value                                                              |
| -------------------- | ------------------------------------------------------------------ |
| Baseline SHA         | `b42b687c8021291bcb537a57fded7a40f7f8477d`                         |
| Candidate SHA        | `42af5c6583ba1a09ee8e3995295e647096208c4d`                         |
| Python               | `3.13.13`                                                          |
| NumPy                | `2.5.1`                                                            |
| JAX                  | `0.11.0`                                                           |
| JAXlib               | `0.11.0`                                                           |
| PyArrow              | `24.0.0`                                                           |
| pytest               | `9.1.1`                                                            |
| pytest-benchmark     | `5.2.3`                                                            |
| Maturin              | `1.14.1`                                                           |
| Requirements SHA-256 | `22703db3c2e8ac88c6cc1ab5c3491008f84063b2d230ee4006def89a9822377e` |

The host was Linux `5.15.167.4-microsoft-standard-WSL2` on a 13th Gen Intel
Core i9-13900HX with 32 logical CPUs. WSL2 exposed no CPU scaling-governor
entries, ACPI platform profile, Intel P-state status, or `powerprofilesctl`
profile. The host power mode was therefore unavailable to the guest and could
not be fixed or verified.

| Identity               | Fingerprint                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| Machine                | `fe554238f5c55b49c8d1961066a86debe87eb538f7fbb6fc19a72f926e264a56` |
| NumPy dependencies     | `dc347e7721846de1ab42e877047ff66a9ca37f6e95855ef46e85274559dbf302` |
| JAX dependencies       | `ef27ba28ef2b53bf88ad5c1fb510e368a95f4ebbc0cd08537da625bbd0d200a4` |

Each logical case also had one exact workload fingerprint across every
baseline/candidate pair. There were 132 compatible first-pass pairs and 28
compatible rerun pairs; no identity was rejected.

## Commands

Dependencies and release builds were isolated under each detached checkout's
`target/` tree:

```bash
cd /tmp/calc-flow-array-candidate
UV_CACHE_DIR=target/uv-cache uv lock
env -u VIRTUAL_ENV -u CONDA_PREFIX UV_CACHE_DIR=target/uv-cache \
  uv export --frozen --extra benchmark --no-dev --no-emit-project \
  --output-file /tmp/calc-flow-array-requirements.txt

uv venv --python 3.13 /tmp/calc-flow-array-baseline/.venv
uv venv --python 3.13 /tmp/calc-flow-array-candidate/.venv
UV_CACHE_DIR=/tmp/calc-flow-array-baseline/target/uv-cache \
  uv pip sync --python /tmp/calc-flow-array-baseline/.venv/bin/python \
  /tmp/calc-flow-array-requirements.txt
UV_CACHE_DIR=/tmp/calc-flow-array-candidate/target/uv-cache \
  uv pip sync --python /tmp/calc-flow-array-candidate/.venv/bin/python \
  /tmp/calc-flow-array-requirements.txt

cd /tmp/calc-flow-array-REVISION
env -u CONDA_PREFIX \
  VIRTUAL_ENV=/tmp/calc-flow-array-REVISION/.venv \
  UV_CACHE_DIR=/tmp/calc-flow-array-REVISION/target/uv-cache \
  UV_TOOL_DIR=/tmp/calc-flow-array-REVISION/target/uv-tools \
  UV_TOOL_BIN_DIR=/tmp/calc-flow-array-REVISION/target/uv-bin \
  CARGO_TARGET_DIR=/tmp/calc-flow-array-REVISION/target/cargo \
  uvx --from maturin==1.14.1 maturin develop --release
```

The five overhead pairs ran in this exact order:

```text
round 1: baseline, candidate
round 2: candidate, baseline
round 3: baseline, candidate
round 4: candidate, baseline
round 5: baseline, candidate
```

Each overhead leg used:

```bash
CALC_FLOW_BENCHMARK_SCALE=overhead JAX_PLATFORMS=cpu \
  /tmp/calc-flow-array-REVISION/.venv/bin/python -m pytest \
  benchmarks/test_array_kernel.py benchmarks/test_array_provider.py \
  benchmarks/test_array_plan.py -q --benchmark-only \
  --benchmark-json=/tmp/calc-flow-array-results/overhead-REVISION-rROUND.json
```

Each ownership leg used the same external harness and ran baseline then
candidate for three rounds at both `standard` and `nightly`:

```bash
CALC_FLOW_BENCHMARK_SCALE=SCALE JAX_PLATFORMS=cpu \
  /tmp/calc-flow-array-REVISION/.venv/bin/python -m pytest \
  benchmarks/test_array_ownership.py -q --benchmark-only \
  --benchmark-json=/tmp/calc-flow-array-results/ownership-SCALE-REVISION-rROUND.json
```

After the first-pass CoV rejection, the same commands were rerun once per
affected revision/case into separate `rerun-*` reports. Those reports were not
combined with the rejected first pass.

## First-pass noise rejection

The ranges below are the minimum and maximum median per-report CoV among the
logical cases in each scope. Every value exceeds the 5% ceiling.

| Scope               | Cases per revision | Baseline median CoV range | Candidate median CoV range |
| ------------------- | ------------------ | ------------------------- | -------------------------- |
| `backend_kernel`    | 8                  | 75.035%-264.673%          | 90.071%-189.703%           |
| `provider_boundary` | 8                  | 32.154%-78.226%           | 39.486%-106.040%           |
| `plan_end_to_end`   | 8                  | 26.402%-48.726%           | 27.362%-58.735%            |
| `batch_ownership`   | 4                  | 17.498%-85.914%           | 27.313%-79.476%            |

## Fresh rerun diagnostics

These tables report one fresh paired rerun per affected case. Means, ratios,
and deltas are retained only to make the rerun auditable; every row is rejected
for timing classification because at least one CoV exceeds 5%.

### Backend kernel

| Scenario                    | Backend | Scale    | Baseline µs | Baseline CoV | Candidate µs | Candidate CoV | Ratio    | Delta    | Status |
| --------------------------- | ------- | -------- | ----------- | ------------ | ------------ | ------------- | -------- | -------- | ------ |
| array_elementwise           | jax     | overhead | 47.572730   | 101.751%     | 51.286514    | 132.701%      | 1.078065 | +7.807%  | noisy  |
| array_elementwise           | numpy   | overhead | 2.340016    | 94.981%      | 2.429817     | 89.268%       | 1.038377 | +3.838%  | noisy  |
| array_matrix_multiplication | jax     | overhead | 11.265336   | 265.893%     | 12.547477    | 248.390%      | 1.113813 | +11.381% | noisy  |
| array_matrix_multiplication | numpy   | overhead | 1.179340    | 57.545%      | 1.193886     | 220.143%      | 1.012334 | +1.233%  | noisy  |
| array_mean                  | jax     | overhead | 16.166660   | 121.316%     | 17.593782    | 118.469%      | 1.088276 | +8.828%  | noisy  |
| array_mean                  | numpy   | overhead | 2.054679    | 29.992%      | 2.156718     | 195.396%      | 1.049662 | +4.966%  | noisy  |
| array_transpose_reshape     | jax     | overhead | 28.102056   | 167.017%     | 28.452149    | 188.310%      | 1.012458 | +1.246%  | noisy  |
| array_transpose_reshape     | numpy   | overhead | 1.517235    | 122.452%     | 1.508129     | 164.933%      | 0.993998 | -0.600%  | noisy  |

### Provider boundary

| Scenario                    | Backend | Scale    | Baseline µs | Baseline CoV | Candidate µs | Candidate CoV | Ratio    | Delta    | Status |
| --------------------------- | ------- | -------- | ----------- | ------------ | ------------ | ------------- | -------- | -------- | ------ |
| array_elementwise           | jax     | overhead | 152.061451  | 111.433%     | 106.692960   | 83.535%       | 0.701644 | -29.836% | noisy  |
| array_elementwise           | numpy   | overhead | 47.590147   | 29.509%      | 22.790335    | 158.486%      | 0.478888 | -52.111% | noisy  |
| array_matrix_multiplication | jax     | overhead | 79.990494   | 45.382%      | 68.775128    | 59.004%       | 0.859791 | -14.021% | noisy  |
| array_matrix_multiplication | numpy   | overhead | 26.014962   | 52.775%      | 15.572485    | 50.633%       | 0.598597 | -40.140% | noisy  |
| array_mean                  | jax     | overhead | 88.466927   | 57.461%      | 68.635991    | 108.672%      | 0.775838 | -22.416% | noisy  |
| array_mean                  | numpy   | overhead | 29.222126   | 43.478%      | 14.448162    | 38.355%       | 0.494425 | -50.557% | noisy  |
| array_transpose_reshape     | jax     | overhead | 100.777999  | 30.902%      | 66.788311    | 31.133%       | 0.662727 | -33.727% | noisy  |
| array_transpose_reshape     | numpy   | overhead | 46.145427   | 30.226%      | 16.951301    | 51.819%       | 0.367345 | -63.265% | noisy  |

### Plan end to end

| Scenario                    | Backend | Scale    | Baseline µs | Baseline CoV | Candidate µs | Candidate CoV | Ratio    | Delta    | Status |
| --------------------------- | ------- | -------- | ----------- | ------------ | ------------ | ------------- | -------- | -------- | ------ |
| array_elementwise           | jax     | overhead | 272.123594  | 68.677%      | 218.226399   | 59.792%       | 0.801939 | -19.806% | noisy  |
| array_elementwise           | numpy   | overhead | 112.175942  | 25.479%      | 78.762490    | 35.331%       | 0.702134 | -29.787% | noisy  |
| array_matrix_multiplication | jax     | overhead | 178.530924  | 59.737%      | 158.025991   | 62.446%       | 0.885146 | -11.485% | noisy  |
| array_matrix_multiplication | numpy   | overhead | 88.005954   | 23.177%      | 74.565177    | 45.171%       | 0.847274 | -15.273% | noisy  |
| array_mean                  | jax     | overhead | 178.019586  | 28.734%      | 158.794708   | 43.793%       | 0.892007 | -10.799% | noisy  |
| array_mean                  | numpy   | overhead | 88.649522   | 28.972%      | 73.736231    | 144.009%      | 0.831772 | -16.823% | noisy  |
| array_transpose_reshape     | jax     | overhead | 203.896136  | 23.126%      | 153.622025   | 23.190%       | 0.753433 | -24.657% | noisy  |
| array_transpose_reshape     | numpy   | overhead | 108.938740  | 25.538%      | 72.474875    | 35.232%       | 0.665281 | -33.472% | noisy  |

The prescribed six-case geometric-mean expression evaluates to a diagnostic
ratio of `0.824129` and a diagnostic improvement of `17.587%` on the fresh
rerun. It is not admissible gate evidence because all six pairs are noisy.

### Batch ownership

| Scenario              | Backend | Scale    | Baseline µs | Baseline CoV | Candidate µs | Candidate CoV | Ratio    | Delta    | Status |
| --------------------- | ------- | -------- | ----------- | ------------ | ------------ | ------------- | -------- | -------- | ------ |
| array_batch_ownership | jax     | nightly  | 8.128540    | 131.463%     | 8.462746     | 91.782%       | 1.041115 | +4.112%  | noisy  |
| array_batch_ownership | jax     | standard | 7.943368    | 92.926%      | 7.886473     | 95.179%       | 0.992837 | -0.716%  | noisy  |
| array_batch_ownership | numpy   | nightly  | 1819.854643 | 13.937%      | 309.237522   | 22.670%       | 0.169924 | -83.008% | noisy  |
| array_batch_ownership | numpy   | standard | 334.280224  | 29.277%      | 18.011060    | 41.781%       | 0.053880 | -94.612% | noisy  |

The fresh NumPy ownership rerun has diagnostic deltas of `-94.612%` at 100,000
elements and `-83.008%` at 1,000,000 elements. Neither is admissible gate
evidence because the paired CoVs exceed 5%.

## Option-conversion midpoint

Task 4's frozen midpoint is carried forward unchanged:

| Item                           | Value                              |
| ------------------------------ | ---------------------------------- |
| Option-conversion probe median | `2.569848 µs`                      |
| Compatible plan geometric mean | `482.611860 µs`                    |
| Share of plan time             | `0.532488%`                        |
| Decision                       | `retain dynamic callback boundary` |

## Gate predicates

| Predicate                                                            | Result        | Evidence                                                         |
| -------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------- |
| No compatible scenario is more than 5% slower                        | Not evaluated | Every timing case was noisy before and after the required rerun. |
| Six-case plan geometric-mean improvement is at least 20%             | Not evaluated | All six contributing pairs exceeded the 5% CoV ceiling.          |
| NumPy ownership improves at least 30% at standard and nightly scales | Not evaluated | Both ownership pairs exceeded the 5% CoV ceiling.                |
| Backend kernels show no material change outside noise                | Not evaluated | Every backend-kernel control exceeded the 5% CoV ceiling.        |
| Every array entry has complete and compatible contract-v2 metadata   | Pass          | 264 first-pass and 56 rerun entries validated with matching IDs. |

## Decision

Phase 2 measurements required

Task 7 was not started by this evidence task.
