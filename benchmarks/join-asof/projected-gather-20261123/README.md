# Fused Join projection

The Rust Equi Join used three Arrow gather kernels per output batch for the
left sequence, left price and right factor. Band Join used four kernels for
the two sequences and prices. Each previously used eight. The counters in
`SideGather::column` record actual Arrow `take` and `interleave` calls.
Run `cargo test --locked -p calc-flow --lib projected_gather::` to exercise
both shapes, cancellation, retry and restoration.

Two matched input rows with 2,048-byte symbols produce 24-byte Band output
rows when sequence and price columns are non-null. The former implementation
rejected these rows at a 24-byte output budget. The candidate emits them at
24 bytes and rejects a limit of 23 bytes. Four rows with a nullable product
require 97 bytes, including their Arrow validity bitmap. The tests also join
separate retained owners and verify both sequence columns and Float64 bits.

## Rust release measurement

The registered latency comparison stopped during the first baseline warmup.
Its first post-preflight host interval recorded 365 major page faults against
a limit of fewer than 256. The sibling CPU was idle and swap movement was zero
in that interval. The shared fixture was ready. There were zero completed
warmups and zero measured pairs, so the latency effect remains unmeasured.
The attempt took 6.35 seconds of its 180-second inclusive budget. The separate
release builds took 20.5 and 17.1 minutes.

The selected `join_sealed` workload has 1,000,000 quote rows, a preloaded
64-row dimension and 977 data handlers. Its primary statistic is each
process's median of the 976 full 1,024-row handler durations. The final
576-row handler is a separate control. A completed worker must verify every
output row, the physical projection, assembly and cleanup using
`scripts.benchmark_suite.native_components.validate_small_batch`. Two rounds
of ten alternating pairs and one excluded warmup per arm per round were
registered. Both binaries used one worker on logical CPU 8.

The two builds used the same temporary integration of main at `3a63b112`,
the bucket-range PR #639 and the deferred-left PR #636. The candidate adds
this projection change. For reproduction, merge both recorded heads into
that version of main in that order and apply the product changes for the
candidate. Generate a shared fixture using
`benchmarks.join_asof_acceptance.prepare_acceptance_fixture` with
`workload="join_sealed"`, `rows=1_000_000`, `batch_rows=1024` and
`mode="low"`. Build each checkout with
`cargo build --locked --release -p calc-flow --features benchmark-internals --bench stream_join_perf`.
Invoke the resulting executable with `--matched-fixture /path/to/fixture.json`.
This entrypoint runs only the selected Rust replay. The registration, source
and binary identities, Arrow fixture hashes and both resource intervals are
in [result.json](result.json).
