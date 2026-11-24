# Public interval Join with contiguous input routing

The public managed Python stream joined two one-million-row dynamic inputs across a ±5-second time range. The clock included ready enqueue, execution, delivery and final Arrow concatenation. The independent oracle checked every pair identity in canonical sequence order and all price products across 10,998,080 matches.

With 20 paired observations spread across two rounds, the maintained verdict was `improved`. The candidate public p50 was 483.826 ms, 133.826 ms above the 350 ms target.

| Measure | Baseline | Candidate |
| ------- | -------- | --------- |
| P50, ms | 535.166  | 483.826   |
| P95, ms | 569.815  | 499.651   |

| Round | Paired median | Exact 97.85% interval |
| ----- | ------------- | --------------------- |
| 1     | -10.83%       | [-11.99%, -8.16%]     |
| 2     | -12.51%       | [-14.94%, -6.87%]     |

Both rounds completed ten alternating AB/BA pairs. The four excluded warmups and 40 measured observations all passed, with a maximum absolute error of 0. Each arm and round owned one worker pinned to one CPU with one Tokio worker. All 101 recorded resource intervals passed.

Across all 44 validated invocations, each side evicted 999,680 rows and retained 320 rows occupying 55,016 bytes. [Per-observation state counters](retained-state.json)

## Sources and budget

The baseline official checkout f0bb1c7b has exactly the same Git tree d44249d5 as the parent dc1fd149. The candidate's official checkout f1aed608 has the reviewed tree 8b7810b3 for #722 at 8abc1d00. Intermediate candidates 10be1395 and 1c1b25a4 were retired before any dispatch, warmup or timed observation. Both observers and all registered hashes were frozen before acquiring this comparison. [Immutable registration](registration.json) · [Trees and release qualification](source-qualification.json)

The official releases used ubuntu-26.04 image version 20261116.202.1. The observer verifies archive, installed package files and native binary hashes before starting the worker. The public job took 138 seconds of its 300-second allowance. The independent static job took 49 seconds. Charging both hosted jobs in full and all 60 seconds of shared external qualification and cleanup gives a conservative total of 247 seconds against the 600-second stage budget. Official wheel builds took 22 minutes 45 seconds for the baseline and 22 minutes 28 seconds for the candidate and were budgeted separately. [Hosted observation](https://github.com/wegamekinglc/calc-flow/actions/runs/45405226329/job/138483990031)

The complete evidence archive preserves the byte contents of all 523 original artifact members, including the worker requests, resource samples and release manifests. The receipt supplies original archive and repacked file digests. [Receipt and statistics](receipt.json) · [Raw evidence](raw-evidence.tar.gz)

The preceding candidate cd879969 was measured on its own baseline. Its static comparison met the regression criterion in both rounds and its interval results were inconclusive. Those observations remain separate. [Original static result](https://github.com/wegamekinglc/calc-flow/blob/f95aa84882fc33e64bd9fb8a439bb2724a35b4f8/benchmarks/join-asof/20261124-compact-static-public/README.md) · [Original interval result](https://github.com/wegamekinglc/calc-flow/blob/589feabc7199f1e677a939b5c592afc6f74b9e1b/benchmarks/join-asof/20261124-compact-interval-public/README.md)
