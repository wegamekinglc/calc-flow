# Public static Join with contiguous input routing

The public managed Python stream preloaded and sealed a 64-row dimension. The clock started with one million ready fact rows after the observed right watermark and stopped after reception and concatenation of the final Arrow table. The oracle checked every sequence and price product in canonical sequence order.

With 20 paired observations spread across two rounds, the maintained verdict was `improved`. The candidate public p50 was 87.164 ms, 57.164 ms above the 30 ms target.

| Measure | Baseline | Candidate |
| ------- | -------- | --------- |
| P50, ms | 95.550   | 87.164    |
| P95, ms | 96.628   | 88.812    |

| Round | Paired median | Exact 97.85% interval |
| ----- | ------------- | --------------------- |
| 1     | -8.42%        | [-10.21%, -7.27%]     |
| 2     | -8.69%        | [-9.85%, -8.42%]      |

Both rounds completed ten alternating AB/BA pairs. The four excluded warmups and 40 measured observations all passed, with a maximum absolute error of 0. Each arm and round owned one worker pinned to one CPU with one Tokio worker. All 16 recorded resource intervals passed.

## Join handler counters from the same measured observations

The Join's cumulative handler time had baseline/candidate p50 72.5360/63.9315 ms and p95 73.977/64.969 ms. The median within-pair changes in milliseconds were -7.9190 and -8.8935. The candidate spent less time in this handler in all 20 pairs. Each arm consumed 1,000,064 rows across two batches and emitted 245 batches containing one million rows and 16,000,000 bytes. Neither arm retained or evicted any fact rows. This runtime counter includes the dimension preload, watermark handlers and time awaiting emissions. The outer clock begins after the preload. The available counters expose no separate timings for Arrow conversion, ingestion, concatenation or backpressure. [Per-pair counters and traffic](handler-diagnostic.json)

## Sources and budget

The baseline official checkout f0bb1c7b has exactly the same Git tree d44249d5 as the parent dc1fd149. The candidate's official checkout f1aed608 has the reviewed tree 8b7810b3 for #722 at 8abc1d00. Intermediate candidates 10be1395 and 1c1b25a4 were retired before any dispatch, warmup or timed observation. Both observers and all registered hashes were frozen before acquiring this comparison. [Immutable registration](registration.json) · [Trees and release qualification](source-qualification.json)

The official releases used ubuntu-26.04 image version 20261116.202.1. The observer verifies archive, installed package files and native binary hashes before starting the worker. The public job took 49 seconds of its 180-second allowance. Wheel compilation runs separately. [Hosted observation](https://github.com/wegamekinglc/calc-flow/actions/runs/45405217201/job/138483957148)

The complete evidence archive preserves the byte contents of all 523 original artifact members, including the worker requests, resource samples and release manifests. The receipt supplies original archive and repacked file digests. [Receipt and statistics](receipt.json) · [Raw evidence](raw-evidence.tar.gz)

The preceding candidate cd879969 was measured on its own baseline. Its static comparison met the regression criterion in both rounds and its interval results were inconclusive. Those observations remain separate. [Original static result](https://github.com/wegamekinglc/calc-flow/blob/f95aa84882fc33e64bd9fb8a439bb2724a35b4f8/benchmarks/join-asof/20261124-compact-static-public/README.md) · [Original interval result](https://github.com/wegamekinglc/calc-flow/blob/589feabc7199f1e677a939b5c592afc6f74b9e1b/benchmarks/join-asof/20261124-compact-interval-public/README.md)
