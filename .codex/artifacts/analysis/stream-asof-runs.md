# Issue 363: ASOF ordered chunk runs

This slice implements A1 on top of the benchmark foundation at
`9b1535bcf4a9cdf394477da03c9718412a1092b1`. A2 owner aggregation and later
admission/output changes follow independently.

`ChunkIter` emits a maximal ordered range from the minimum chunk head. When
the whole remaining chunk precedes the next heap head, it emits that suffix;
otherwise binary search finds the exclusive canonical `(time, key, sequence)`
boundary. Existing row and output iterators flatten these borrowed ranges.
The heap advances once per run, with unchanged capacity and no new heap-owned
workspace. Existing physical positions, consumed heads, typed sequence owners
and checkpoint layout/accounting remain intact.

## Verification

- Before implementation, the focused nonoverlapping-chunk test failed with
  1,024 heap visits instead of four. Its full canonical output comparison passed.
- After implementation, all 291 focused ASOF module tests passed serially,
  including new nonoverlapping/interleaved order tests, existing sparse-prefix,
  cancellation, budget, owner and checkpoint recovery tests.
- An existing worker-funding assertion varied during the earlier parallel
  affected-test run (parked bytes zero rather than 16,384). Its isolated run
  and the complete serial ASOF module run passed; no funding assertion or
  implementation was changed to suppress that observation.
- The ready-prefix complexity assertion now expects one run while still
  asserting three delivered rows.
- Scoped `cargo clippy --locked -p calc-flow --lib -- -D warnings`, formatting,
  whitespace and generated-contract checks passed. Final specialist review
  approved the source and evidence artifact.

The 1,024-to-four visit reduction is a complexity result, not a measured
throughput improvement. Interleaved chunks can require a binary search per
short run. Sealed paired throughput measurements must cover that case before
claiming a general speedup. Full CI and workspace coverage remain unverified.
