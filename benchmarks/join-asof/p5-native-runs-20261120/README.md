# P5 native Join/interval acceptance evidence

## Outcome

The P5 absolute performance gates are **not met**. Join has valid measured samples; interval has no valid timing distribution.

| Case | Candidate P50 | Target | Correctness / lifecycle | Status |
|------|--------------:|-------:|-------------------------|--------|
| Join 1M, one CPU | 424.412 ms | ≤30 ms | 10/10 full-output oracles passed | not met |
| Interval 1M, one CPU | no valid P50 | ≤350 ms | 5,376 / 10,998,080 expected rows at timeout | not met |

The archived descriptive Join baseline is 78.51 ms; the candidate is slower in this measurement, not an upgrade. The archived baseline and candidate were not paired in the same process, so no paired speedup ratio is claimed.

The sealed candidate revision is `212ca55b8b8a7deabd37a33f105122c6458e31a8`. Its wheel SHA-256 is `aa22455ef8bfad91c5866119b470f3930bbdadab6eaaba7a1d8b961dcbabff57`; the loaded native SHA-256 is `4a1e8090e54b1c06f6fea180555b2ab205013773514846d556b5ca9cfcdd244e`. The release wheel build took 1084.473 seconds separately from acquisition.

## Harness validity and budget

Each warmup and measured observation uses a distinct checkpoint root. This prevents a new `EngineCase` from restoring the preceding observation's terminal `sample-1` checkpoint. Four consecutive 100-row diagnostic observations passed their complete oracle before this acquisition.

The formal Join case completed two warmups and ten measured samples. Interval then timed out after 600 seconds. The operator had consumed both 1M-row inputs but emitted only 5,376 rows; the sink expected 10,998,080 rows. `interval-failure.json` preserves the job, source, sink and edge status.

Total acquisition took **610.174 seconds**, exceeding the pre-registered 600-second budget by 10.174 seconds. The wrapper did not enforce `acquisition_stop_seconds=570` before starting interval's 600-second wait. This is a harness budget-enforcement defect, not a passing acquisition. No further timing retries or additional samples were used.

## Evidence

- `plan.json`: registered candidate, cases, sample counts, budget and root-isolation contract.
- `result.json`: every valid Join sample, the P50/P95 values, native identities and final error.
- `interval-failure.json`: interval's terminal observation at timeout.
- `measure.py`: the observation wrapper with per-observation root isolation used for this acquisition.

Reproduce with a newly built candidate wheel and a new empty output root:

```bash
CALC_FLOW_TASKSET=cpu0 PYTHONPATH=. TOKIO_WORKER_THREADS=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
taskset -c 0 /path/to/venv/bin/python \
  benchmarks/join-asof/p5-native-runs-20261120/measure.py \
  --plan benchmarks/join-asof/p5-native-runs-20261120/plan.json \
  --wheel /path/to/candidate.whl \
  --root /absolute/path/to/new-p5-measurement \
  --output /absolute/path/to/new-p5-result.json
```

The raw per-observation roots remain local under `target/p5-join-measurement-v2-212ca55b/`; only compact evidence is committed.
