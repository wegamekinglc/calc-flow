# Join probe stop polling (#429 / DAL-343)

Baseline: `main` `aaf793a3259b3e3d2e18e30eb5e13e06b1141665`.

Count and fill workers poll their attempt's existing shared abandonment atomic
once per 4,096 work items. A position consumes one item; fill consumes one more
for each match after that position's first match. The budget carries across
positions, so misses, long opposite runs and repeated short runs all have bounded
polling. One-to-one fill does not add a separate first-match poll.

`GatherStop::check_abandoned` only reads that atomic. Full cancellation-token and
UTC deadline checks remain at worker entry/exit, admission and result settlement.
`finish_probe_ticket` drives cancellation/deadline waiting while the ticket is
pending; its stop branch drops the ticket and publishes abandonment. Dropping
the parent future or closing the owner uses the same existing signal. This
retains the attempt/result leases and refund paths without another observer
task, timer or job-context allocation. The atomic-only method is restricted to
loops whose owner drives those stop events.

No dispatch threshold, public API, checkpoint format, state installation,
output ordering or funding policy changes. No timing collection was performed;
ungated probe CPU <= 1.3 times wall remains unmeasured. Group counts establish
bounded polling work, not a performance acceptance verdict.

## Local verification

The original worker with the new regression fixture made 24,582 full checks
for 24,579 positions across three units and failed all three regressions.
The updated count worker makes three full checks and nine atomic polls for
that case. Fill makes six full checks and nine atomic polls. These counts cover
the worker methods; the dispatch adapter and owner retain their boundary checks.

Passed for this change (behavior results reused after a local test-binding rename):

```bash
cargo test -p calc-flow --lib probe_stop_tests -j 2
# 3 passed: grouping, bounded position/match cancellation, owned cancel/deadline drain.
cargo test -p calc-flow --lib probe -j 2 -- --skip probe_stop_tests
# 25 passed.
cargo test -p calc-flow --lib test_empty_native_result_funds_key_vector_until_actual_drop -j 2
# 1 passed, including actual owned-probe cancellation/abandonment and refunds.
cargo test -p calc-flow --lib test_native_index_updates_out_of_order_append_and_dense_eviction -j 2
# 1 passed, including reverse unit completion order and global match-limit rejection.
target/debug/deps/calc_flow-b7e69e6a365d09b0 runtime::streaming::gather_work::tests --test-threads=1
# 28 passed, using the same compiled test binary as the Cargo commands above.
cargo fmt --all --check
cargo clippy -p calc-flow --lib --tests -j 2 -- -D warnings
```

The initial gather-test command through Cargo without `--test-threads=1`
exhausted the existing 32 MiB process-infrastructure budget and aborted under
concurrent test services. Serial execution of the same binary passed all 28
tests. No infrastructure limit or assertion was changed. Full regression,
cross-platform and workspace coverage gates remain CI work.

## Codacy follow-up

The cancellation-boundary test now enumerates its four executed cases directly
instead of using a nested product and skipping two count cases that cannot reach
a second position group. Case order, fixture operations, cancellation bounds,
poll/visit counts and refund assertions are unchanged. Production code and gate
configuration are unchanged.

Lizard 1.23.0 reproduced the original CCN 9 warning and now reports CCN 6 for
`test_probe_stops_at_the_next_position_or_match_group`; the entire file has zero
warnings at the existing threshold 8. The earlier 58-test evidence at
`ee1cf390ec3da38fbcc4b650d2c24daca23557a6` remains valid for the unchanged behavior.
The affected test alone was rebuilt and rerun: 1 passed.
Scoped Clippy and formatting checks passed.

```bash
cargo test --locked -p calc-flow --lib operator::join::data_work::process::probe_stop_tests::test_probe_stops_at_the_next_position_or_match_group -j 2 -- --exact --test-threads=1
lizard -C 8 -w crates/calc-flow/src/operator/join/tests/probe_stop_tests.rs
cargo fmt --all --check
cargo clippy --locked -p calc-flow --lib --tests -j 2 -- -D warnings
```
