# P5 dense-buffer performance evidence

## Outcome

Dense admitted morsels and fully retained runs now share Arrow buffers instead of repeating gathers. This improves the observed medians, but **both absolute performance gates remain unmet**.

| Case               | Previous P50 | Candidate P50 | Candidate P95 | Target  | Oracle          |
|---------------------|-------------:|--------------:|--------------:|---------|-----------------|
| Join 1M, 1 CPU     | 450.038 ms   | 422.432 ms    | 431.512 ms    | ≤30 ms  | 10/10 passed    |
| Interval 1M, 1 CPU | 1,167.624 ms | 1,147.609 ms  | 1,184.216 ms  | ≤350 ms | 10/10 passed    |

The observed median reductions are about 6.1% for Join and 1.7% for interval. These are unpaired absolute observations, not a significance claim. Join remains far from its 30 ms target, so the next work uses phase diagnostics rather than further unlocated tuning.

Two buffer-sharing regressions verify that contiguous morsels and fully retained runs reuse their source allocations. Sparse, noncontiguous and schema-converting inputs retain the existing gather path. The complete Join module's **587 tests** and test-target Clippy passed.

## Candidate and acquisition

- Runtime candidate: `fcec2dc3`; previous candidate: `12d7759c`.
- Release build: 565.216 seconds, separate from acquisition.
- Acquisition: 43.173 seconds, within the registered 600-second budget.
- Two warmups and ten formal observations per case; every output oracle and lifecycle check passed.
- Wheel SHA-256: `6d2d1fe9bf3017c7b8ea595db9cade52783f4751de0771cfa58b0aa14819a1f3`.
- Loaded native SHA-256: `b45035b4cb68e68a9522c32aa3db31773208ebec44f19bce44c805f36d30ec8b`.

`plan.json` records the fixed cases, baseline, sample counts and deadline. `result.json` contains all observations and native evidence. The maintained wrapper is `benchmarks/join-asof/p5-morsel-state-20261120/measure.py`.
