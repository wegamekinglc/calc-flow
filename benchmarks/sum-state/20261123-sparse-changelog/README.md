# Sparse grouped SUM changelog

The native release worker started with 10,000 groups and applied twenty
100-row batches. Each update changed 80 groups. The first output contained
10,000 upserts and each later output contained 80 retract/upsert pairs.
Each worker retained all outputs through natural end and validated the
reconstructed positions at every boundary.

| Surface           | Baseline median | Candidate median | Paired observations |
| ----------------- | --------------- | ---------------- | ------------------- |
| Rust, one CPU     | 352.51 ms       | 22.07 ms         | 20                  |
| Public Python API | See baseline    | Unmeasured       | 0                   |

The two Rust rounds had median paired changes of −93.76% and −93.31%.
Their respective 97.85% sign intervals were [−93.91%, −93.70%] and
[−93.87%, −90.09%]. The maintained five-percent rule reports an improvement.
The Python collection stopped during startup when system-wide major page
faults exceeded its preregistered resource limit. Its earlier baseline used
the public Python stream lifecycle and remains a single-arm observation.
[Native observations](rust-native.json) and [Python observations](python-public.json)
include separate clocks, build identities, output oracles and resource evidence.

## Reproducing the native case

1. Check out the baseline and candidate implementations recorded in
   `rust-native.json`. In each copy, set the last argument to `sample`
   inside `run_samples` in `crates/calc-flow/examples/grouped_sum_state_benchmark.rs`
   to `true` to select changelog output for the single-fixture entry point.
   The JSON records the expected SHA-256 of the resulting benchmark source.
2. Build each worker with
   `cargo build --release --locked -p calc-flow --features benchmark-internals --example grouped_sum_state_benchmark`.
   Run the baseline worker with `--prepare <fixture-directory>` or use its
   previously generated `sparse.arrow` with the recorded SHA-256.
3. Run the workers as `taskset -c 0 <worker> <fixture-directory>/sparse.arrow stream 1 1`
   with the thread environment in `rust-native.json`. Use the maintained
   `scripts.benchmark_suite.sum_state` preflight, build comparison, resource
   guard and statistics. Select `sparse-stream-1cpu`, supply the changelog row
   counts and schema digest from the JSON, and keep the original fixture's
   reconstructed snapshot digests.
4. For each of two rounds, run one warmup per arm. Alternate baseline-first
   and candidate-first order across ten measured pairs in each round. The
   26.04-second acquisition included preflight, fixture copy, all warmups,
   worker launches, validation, and cleanup. Release builds took 1,267 and
   1,178 seconds separately.
