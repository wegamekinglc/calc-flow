# P5 probe-cache and output-coalescing evidence

## Outcome

Cached opposite runs and bounded 64,000-row output chunks further reduce interval time, but **both absolute gates remain unmet**.

| Case               | Previous P50 | Candidate P50 | Candidate P95 | Target  | Oracle          |
|---------------------|-------------:|--------------:|--------------:|---------|-----------------|
| Join 1M, 1 CPU     | 459.643 ms   | 450.038 ms    | 492.304 ms    | ≤30 ms  | 10/10 passed    |
| Interval 1M, 1 CPU | 1,397.678 ms | 1,167.624 ms  | 1,250.154 ms  | ≤350 ms | 10/10 passed    |

The observed interval median is about 16.5% lower than the previous acquisition; Join is about 2.1% lower. These are unpaired absolute observations, not a significance claim.

The focused three-morsel regression now produces three output batches instead of 27 and resolves seven unique opposite runs instead of performing 108,080 per-match BTreeMap lookups. The complete Join module's **585 tests** and test-target Clippy passed. Output row counts, values and lifecycle are unchanged.

## Candidate and acquisition

- Runtime candidate: `12d7759c`; previous candidate: `e737fb0f`.
- Release build: 568.891 seconds, separate from acquisition.
- Acquisition: 43.949 seconds, within the 600-second budget.
- Two warmups and ten formal observations per case; every output oracle and lifecycle check passed.
- Wheel SHA-256: `7fe3a4e88a07e94f56c2cb36f337052e15a3fe03de1e9528efd312aa9489bc38`.
- Loaded native SHA-256: `8c49f7a5a4f489c578f0c5acd149b2094ce5fccf99b3feccf3f5f7f6be9a5352`.

`plan.json` fixes the cases, samples, target gates and deadline. `result.json` preserves all observations and native identities. The maintained wrapper is `benchmarks/join-asof/p5-morsel-state-20261120/measure.py`.

The next experiment targets repeated Arrow gathers for dense admitted rows. Sparse, noncontiguous and schema-converting inputs retain the existing gather fallback.
