# Public Python steady single-row session push

Returning a public PyArrow result for a one-row `Program.session` push took
**63.285 → 62.334 µs p50** and **84.384 → 82.315 µs p95** on one hosted
CPU. Both numbers include Python input binding, PyO3, native execution and
conversion of the result to Arrow. Each session received one 4,096-row
history batch followed by 64 individual pushes under ordinary Python GC.

| Independent statistic              | Baseline  | Candidate | Steady target |
| ---------------------------------- | --------- | --------- | ------------- |
| Pooled single-push p50, µs         | 63.285    | 62.334    | ≤20           |
| Pooled single-push p95, µs         | 84.384    | 82.315    | ≤40           |

Each arm contributed 1,280 formal push clocks. The paired unit was the
median of 64 successive pushes in one fresh process. The respective paired
changes and exact 97.85% intervals were −1.26% [−2.54%, −0.62%] and
−1.76% [−3.49%, −0.15%]. The maintained 5% rule returned
`no-confirmed-regression`. Both arms missed the absolute steady targets.
The candidate remained 42.334 µs above the p50 target and 42.315 µs above
the p95 target. [Protocol](protocol.md) · [Original forced-GC study](../p1-20261115.md)

The baseline was main `4c3cc98e`. The candidate combined the four published
Rust source and test files in #674 and #675 on that same base, reproducing
the native study's complete candidate tree `48717222`. Both arms used
identical Python and PyO3 sources. Explicit watermarks exercise the cached
session endpoint and primitive batch charging. The source-ordered timestamp
validation path has its own workload. The native Rust experiment uses its own
clock and host. [Native public results](../20261123-native-push/README.md)

The 44 fresh workers included four excluded warmups. They validated 183,040
output rows across five columns, the null bitmap, row order, the original
caller input and an empty final flush. This includes the history checks and
all 64 appended rows in each process. Both release artifacts used Ubuntu
26.04 image `20261116.202.1`, CPython 3.13 ABI3, Rust 1.94.0, LLD 21.1.8
and GCC 15.2.0. The collector read each installed ELF's compiler information
and verified every package file against its sealed wheel. The host's 44
resource intervals passed, CPU 2 had a maximum SMT sibling utilization of
2%, and no pages were swapped. The hosted job took 73 seconds including
wheel retrieval, setup and cleanup. The candidate wheel job took 18 minutes
and 8 seconds separately. [Baseline wheel job](https://github.com/wegamekinglc/calc-flow/actions/runs/45150768447/job/137611646674) · [Candidate wheel job](https://github.com/wegamekinglc/calc-flow/actions/runs/45175796563/job/137696585919)

The result archive includes all process clocks, host resource intervals,
source and fixture hashes, wheel native hashes and the original artifact
digest. [Raw worker records](workers.jsonl) · [Summary and provenance](result.json) · [Hosted acquisition](https://github.com/wegamekinglc/calc-flow/actions/runs/45179077556/job/137707864535)

The measured worker converted each prepared Arrow UTC timestamp with
`as_py()`. The portable worker reconstructs the same instant from its integer
microseconds and a fixed UTC epoch. This preparation occurs before every push
clock. All 65 watermarks match the observed worker on the original fixture.
[Measured worker source](https://github.com/wegamekinglc/calc-flow/blob/1a00d83ca9ec42c8d777072c0f03a1f41a25cd8b/benchmarks/steady_session.py)

## Reproduce one worker

Use a Python environment containing the selected release wheel and benchmark
dependencies. From the repository root, generate the shared fixture and run
one worker with the following commands. The recorded two-round acquisition
additionally pins the workers, verifies both installed packages, enforces
host resource limits and calculates paired intervals. The archived collector
contains the exact compressed fixture files and their digests. [Registered collector](https://github.com/wegamekinglc/calc-flow/commit/1a00d83ca9ec42c8d777072c0f03a1f41a25cd8b) · [Public worker](../../steady_session.py)

```bash
python - <<'PY'
from pathlib import Path
from runpy import run_path

fixture = Path("target/public-steady-fixture")
fixture.mkdir(parents=True, exist_ok=True)
study = Path("benchmarks/stream-session/20261123-native-push/fixture.py")
run_path(str(study))["generate"](fixture)
PY
TOKIO_WORKER_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.steady_session target/public-steady-fixture \
  target/public-steady-worker.json
```
