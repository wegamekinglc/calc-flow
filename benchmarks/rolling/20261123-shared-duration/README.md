# Shared event-time Duration samples

The release Rust runner processed 100,000 rows across 1,000 entities, in
100 batches of 1,000 rows. Three SUMs read the same non-null Float64 column
over 30, 60 and 90 seconds. The input clock advanced one millisecond per row
with deterministic sub-millisecond jitter. Checkpointing was disabled.

| Native measurement                       | Independent queues | Shared queue |
| ---------------------------------------- | ------------------ | ------------ |
| Whole-runner median                      | 34.98 ms           | 28.34 ms     |
| Whole-process CPU median                 | 127.85 ms          | 114.20 ms    |
| Sampled whole-process peak RSS median    | 82.91 MiB          | 83.69 MiB    |
| Physical samples in the one-entity test  | 180                | 90           |

The median runner latency fell 18.98%. Two rounds of ten alternating pairs
gave median paired changes of −17.69% and −19.12%, with respective 97.85%
sign intervals of [−28.65%, −13.11%] and [−22.03%, −17.47%]. The maintained
five-percent rule classified the native case as improved. CPU and RSS include
fixture construction, an initial validating job, the timed job, post-clock
checks and cleanup. Peak RSS includes the input, expected result and all
three output columns.

The physical sample counts came from a separate deterministic production
test with one entity, 100 once-per-second observations and 46/54-row
batches. The shared representation finished with a 90-row charge and
admitted the final batch under a 144-row budget that includes its 54
pending rows. Public Python has a separate timing protocol and nullable
fixture. [Native samples and build records](rust-native.json)

## Reproduction

1. Use the two recorded source revisions. Copy the candidate's
   `crates/calc-flow/benches/core.rs` into the baseline checkout. Check
   that both benchmark files match the SHA-256 in the build records.
2. In each checkout, run
   `cargo build --release --locked -p calc-flow --features benchmark-internals --bench core --message-format=json-render-diagnostics`.
   Use the emitted `core` benchmark executables, Rust 1.94.0,
   `CARGO_BUILD_JOBS=2` and `CARGO_INCREMENTAL=0`.
3. Set `CRITERION_HOME` to a fresh output directory per invocation.
   Run `taskset -c 0,2,4,6 <worker> --test --exact rolling_state_history/duration_sum_30s_60s_90s_100k_keys1000_batch1000`.
   The native runtime uses four Tokio workers. Each invocation must write
   one `rolling-state-proof.json` with two verified jobs, 100,000 output
   rows, three output columns and the recorded result digest.
4. Run one warmup per arm per round and ten alternating pairs. Use the
   proof's `last_primary_elapsed` for the native clock. Validate the full
   proof after every invocation and check build identity and native process
   competition with the maintained sum-state guard. Across both rounds
   the 40 measured invocations and four warmups took 23.85 seconds
   including preflight, fixture creation, all validations and cleanup.
   The baseline and final candidate release builds took 1,204 and 761
   seconds separately.
