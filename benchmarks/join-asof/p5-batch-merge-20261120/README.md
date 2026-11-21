# P5 batch-merge performance evidence

## Outcome

Constructing immutable morsel runs before one batch-level bucket/expiration merge substantially reduces interval time, but **neither absolute performance gate is met**.

| Case               | Previous P50 | Candidate P50 | Candidate P95 | Target  | Oracle          |
|---------------------|-------------:|--------------:|--------------:|---------|-----------------|
| Join 1M, 1 CPU     | 449.057 ms   | 459.643 ms    | 472.791 ms    | ≤30 ms  | 10/10 passed    |
| Interval 1M, 1 CPU | 3,574.794 ms | 1,397.678 ms  | 1,572.976 ms  | ≤350 ms | 10/10 passed    |

Interval's observed median is 60.9% lower than the prior acquisition. These are separate absolute acquisitions, not paired samples; no confidence or general speedup claim is made. Join's small descriptive difference is not a confirmed regression.

The 10,000-row regression reduced merge visits from 40,480 to exactly 20,000, one bucket visit and one expiration visit per locator. Thirteen RunState tests, seven production tests, 16 native tests and the focused Join module's **585 tests** passed. Out-of-order morsel runs retain sequential ordering and snapshot identity; all state limits and output sequences are checked before one atomic state installation. Test-target Clippy passed.

## Candidate and acquisition

- Runtime candidate: `e737fb0f`; previous candidate: `6f15e8fc`.
- Release build: 569.390 seconds, separate from acquisition.
- Acquisition: 47.989 seconds, within the registered 600-second budget.
- Both cases completed two warmups and ten formal observations with complete output oracles and completed lifecycles.
- Wheel SHA-256: `afebccaac198101466539205edf07fcb3e63e43f1dec4a64fb29df9da76636ff`.
- Loaded native SHA-256: `9e2304403392cd6872d77c2092cb4b2c0fa304f616284ef30c49cb14cc60da78`.

Exit status 2 indicates unmet absolute thresholds, not a runtime or correctness failure. `plan.json` records the fixed cases, sample counts, deadline and baseline; `result.json` contains all observations and native evidence. The maintained wrapper is `benchmarks/join-asof/p5-morsel-state-20261120/measure.py`.

The next measured costs are per-match opposite-run lookup and small output-batch fragmentation. The interval improvement is not presented as P5 acceptance.
