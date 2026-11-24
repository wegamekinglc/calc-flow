# Public Python sparse grouped SUM buffer sizing

Pre-sizing the Arrow operation-tag and version buffers for grouped SUM gave a
pooled P50 of **18.615 ms**, compared with **18.754 ms** on the baseline. The
candidate was 3.615 ms above the registered 15 ms goal. Both paired intervals
included zero. The repository verdict was `no-confirmed-regression`.

| Complete public stream | Baseline ba5471bb | Candidate 7d23e187 | Pooled change |
|------------------------|------------------:|-------------------:|--------------:|
| P50                    |         18.754 ms |          18.615 ms |        −0.74% |
| P95                    |         19.362 ms |          18.983 ms |        −1.96% |

| Round | Pairs | Median per-pair change | Exact median interval  |
|------:|------:|-----------------------:|-----------------------:|
|     1 |    10 |                 +0.22% |       [−2.36%, +1.10%] |
|     2 |    10 |                 −1.26% |       [−3.81%, +2.31%] |

Each round used one worker process per wheel. Each worker ran one excluded
warmup followed by ten formal observations. The collector alternated AB and
BA for adjacent pairs. Each observation constructed a fresh public stream.
The intervals use the second and ninth sorted changes in a round of ten,
with 97.8515625% coverage under the repository's paired-median assumptions.
Pooled P95 uses linear interpolation across 20 formal observations per wheel.

## Workload and correctness

The timer began before declaring `table_input`, the trade filter, the signed
quantity projection and the grouped SUM SQL. It stopped after the last Arrow
output and the asynchronous `TableExpr.stream` context closed. This clock
includes declaration, compilation, execution, Python delivery and cleanup.
Workers loaded the frozen Arrow fixture before timing. Output validation
followed each timed observation.

The first input seeded 10,000 account/stock groups. Each of 20 later batches
contained 100 rows. Every stream returned 10,000 initial changes followed
by 160 changes per update, or 13,200 rows in total. The oracle checked the
exact Arrow schema, ordered retract/upsert keys and signed values, and
reconstructed the entire position map after each batch. The two wheels passed
all **924 output-prefix checks** across 44 streams. This clock covers the full
21-batch Python graph. Rust component timings and individual update latency
require their own measurements.

## Releases and collection

The baseline came from official Linux distribution artifact `14897014496`,
and the candidate came from `14921540444`. For each wheel, the requested
commit and the actual PR checkout had the same Git tree. The baseline
checkout differs from the product base `85af0b53` in one plan document.
Both wheel build jobs used the optimized CPython 3.13 ABI3 Linux profile and
Ubuntu 26.04 image version `20261116.202.1`. Independent verification
matched the official artifact ZIP, inner wheel and native module digests.
[Baseline build](https://github.com/wegamekinglc/calc-flow/actions/runs/45342562023/job/138283593607) · [Candidate build](https://github.com/wegamekinglc/calc-flow/actions/runs/45392732141/job/138440879736)

The observation ran with CPython 3.13.18, PyArrow 24.0.0, NumPy 2.5.2, one
Tokio worker and affinity to logical CPU 2. The three-second preflight and ten
monitoring intervals passed. Peak whole-host CPU use was 25.5%, peak SMT
sibling use was 2.9%, and no pages were swapped. The measurement phase took
15.748 s and the full hosted job took 43 s.
[Measurement job](https://github.com/wegamekinglc/calc-flow/actions/runs/45396066297/job/138452059786)

## Archived evidence

The [original ZIP](hosted-raw.zip) contains all 171 GitHub artifact members,
including worker warmups, request/response pairs, release identities and
resource observations. Its SHA-256 is
`7082f2b9622515b4e04dda01612cf92b92f63c9f6f1147038b173b14e0ad3b8b`.
Each extracted member matched the official archive byte for byte. The
[verification receipt](receipt.json) records exact times, commits, trees and
digests. The [registered protocol](protocol.md) records the workload, clock,
budget and target.

From this directory, these commands check the archived digest and recompute
the pooled percentiles and paired interval endpoints without starting a stream.

```bash
sha256sum hosted-raw.zip
python3 - <<'PY'
import json
import statistics
import zipfile

with zipfile.ZipFile("hosted-raw.zip") as archive:
    data = json.loads(archive.read("samples.json"))

for arm in ("baseline", "candidate"):
    values = [1000 * t for round_ in data[arm] for t in round_]
    p95 = statistics.quantiles(values, n=20, method="inclusive")[-1]
    print(arm, "p50/p95 ms", statistics.median(values), p95)
for round_, (base, head) in enumerate(zip(data["baseline"], data["candidate"]), 1):
    changes = sorted(100 * (b / a - 1) for a, b in zip(base, head))
    print(round_, "change and interval (%)", statistics.median(changes), changes[1], changes[-2])
PY
```
