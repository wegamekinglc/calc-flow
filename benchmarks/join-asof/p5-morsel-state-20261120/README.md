# P5 multi-morsel repair and performance evidence

## Outcome

The multi-morsel correctness defect is fixed, and every formal observation now completes its full output oracle. **Both absolute performance gates remain unmet.**

| Case               | Candidate P50 | Candidate P95 | Target  | Oracle / lifecycle      |
|---------------------|----------------|----------------|---------|-------------------------|
| Join 1M, 1 CPU     | 461.093 ms     | 483.953 ms     | ≤30 ms  | 10/10 passed; completed |
| Interval 1M, 1 CPU | 3,699.085 ms   | 4,120.403 ms   | ≤350 ms | 10/10 passed; completed |

The previous interval candidate emitted only 5,376 / 10,998,080 expected rows at timeout. Candidate `dfc077eb` emits all 10,998,080 rows. This is a correctness repair, not a demonstrated speedup. The Join and interval measurements are separate from the older archived references and are not paired speedup comparisons.

Formal acquisition completed in **84.957 seconds**, within the registered 600-second budget. Both cases completed two warmups and ten measured observations. Exit status 2 records failed absolute thresholds, not a runtime or oracle failure.

## Runtime repair

`dfc077eb` assigns distinct run identities to prepared morsels and carries each prepared side update into the next morsel. The opposite-side probe still uses the batch-start state. Empty morsels retain the latest nonempty update. The previous implementation reused a run ID and prepared later morsels against the unmodified batch-start side, dropping earlier index entries.

The focused regression initially produced 18,928 instead of 108,080 matches. After the repair, all 12 RunState tests and seven production Join tests passed, including complete output, empty-morsel continuation, match-limit rollback, watermark eviction and checkpoint continuation. Scoped core Clippy passed.

A real sealed-wheel diagnostic with 10,000 rows per side emitted 108,080 rows with a complete oracle and completed lifecycle. Its single observation took 15.987 ms; this is a diagnostic, not a P50 or an additional formal sample.

## Candidate and harness

- Runtime source: `dfc077ebae272da6be13f16d3e60b3ca8df16a62`.
- Wheel SHA-256: `3ea9ac5ed31fa35907e008f65fe51e0881745728d82da693c4c79c5ac360e838`.
- Loaded native SHA-256: `1845e3fdf2b03ebf9b41f02a2c15156fb8a35fdcfa9513a9dbedb5d3621f2686`.
- Release build: 9m14s (557.219 seconds wall time), separate from acquisition.
- `result.json`: all samples, P50/P95, complete oracle and lifecycle evidence.
- `plan.json`: registered cases, samples, target gates, deadline and independent-root policy.

`measure.py` preserves the maintained `EngineCase` ready-source-to-Arrow timing boundary. It calls the underlying `run_stream` coroutine with the remaining deadline rather than passing its synchronous tuple result to `asyncio.wait_for`. Each observation uses a new checkpoint root and receives a deadline; no new observation starts after the acquisition deadline. The outer command also has a 600-second timeout with a 30-second kill-after backstop.

Four protocol tests cover deadline forwarding, maintained coroutine arguments, expired-observation refusal and deadline-exhausted case execution. Ruff checks and formatting passed.

Reproduce with a newly built wheel and new empty roots:

```bash
CALC_FLOW_TASKSET=cpu0 PYTHONPATH=. TOKIO_WORKER_THREADS=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
timeout --kill-after=30s 600s taskset -c 0 /path/to/venv/bin/python \
  benchmarks/join-asof/p5-morsel-state-20261120/measure.py \
  --plan benchmarks/join-asof/p5-morsel-state-20261120/plan.json \
  --wheel /path/to/candidate.whl \
  --root /absolute/path/to/new-measurement \
  --output /absolute/path/to/new-result.json
```

The raw observation roots remain local under `target/p5-morsel-state-dfc077eb/`. Only compact evidence is committed.
