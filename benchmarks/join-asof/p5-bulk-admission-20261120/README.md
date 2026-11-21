# P5 bulk-admission performance evidence

## Outcome

Synchronous bulk admission removes the run core's per-64-row Tokio budget boundaries, but **does not meet either absolute performance gate**. The large production regression recorded 15,847 boundaries before the change and zero afterward; output remained complete.

| Case               | Previous P50 | Candidate P50 | Candidate P95 | Target  | Oracle          |
|---------------------|-------------:|--------------:|--------------:|---------|-----------------|
| Join 1M, 1 CPU     | 461.093 ms   | 449.057 ms    | 463.458 ms    | ≤30 ms  | 10/10 passed    |
| Interval 1M, 1 CPU | 3,699.085 ms | 3,574.794 ms  | 4,189.297 ms  | ≤350 ms | 10/10 passed    |

The observed median differences are about −2.6% for Join and −3.4% for interval. These are separate absolute acquisitions, not paired measurements; no statistically confirmed improvement is claimed. Both cases produced every expected row and completed normally.

The runtime candidate is `6f15e8fc`; the prior candidate is `dfc077eb`. The candidate release build took 572.894 seconds separately. Acquisition took 84.064 seconds, within the registered 600-second budget, with two warmups and ten measured samples per case. Exit status 2 records the unmet gates, not an execution failure.

## Evidence

- `plan.json`: fixed cases, sample counts, baseline, thresholds and acquisition deadline.
- `result.json`: all samples, output oracles, lifecycle results and native identities.
- `benchmarks/join-asof/p5-morsel-state-20261120/measure.py`: unchanged maintained deadline-bounded harness used for this acquisition.
- Wheel SHA-256: `03baa9ed86a3d8aefffe835b9df8e9061eeefd4f9b59e546529c413d070acef0`.
- Loaded native SHA-256: `7b84a77e4ec7617f22f3deb557df26df7ded2f3ba49aab2ef392f8e06fa0fd0a`.

Reproduce using the same wrapper and a new empty root:

```bash
CALC_FLOW_TASKSET=cpu0 PYTHONPATH=. TOKIO_WORKER_THREADS=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --kill-after=30s 600s taskset -c 0 /path/to/venv/bin/python \
  benchmarks/join-asof/p5-morsel-state-20261120/measure.py \
  --plan benchmarks/join-asof/p5-bulk-admission-20261120/plan.json \
  --wheel /path/to/candidate.whl \
  --root /absolute/path/to/new-measurement \
  --output /absolute/path/to/new-result.json
```

The next measured target is repeated bucket/expiration merging across morsels. Scheduling removal alone is not presented as a successful performance upgrade.
