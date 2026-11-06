# sum-case-retest-20261106

Status: **completed**.
Performance acceptance: **False**.

| Case                     | Baseline P50 ms | Candidate P50 ms | Change | Verdict      |
|--------------------------|-----------------|------------------|--------|--------------|
| stock-stream-1cpu        | 171.881         | 167.045          | -2.81% | inconclusive |
| stock-stream-4cpu        | 155.178         | 164.850          | +6.23% | inconclusive |
| sparse-stream-1cpu       | 20.382          | 20.528           | +0.72% | inconclusive |
| uniform-buy-stream-1cpu  | 88.653          | 89.242           | +0.66% | inconclusive |
| uniform-sell-stream-1cpu | 86.366          | 86.868           | +0.58% | inconclusive |

| Case                     | Round | Paired median | Median interval    | Coverage  |
|--------------------------|-------|---------------|--------------------|-----------|
| stock-stream-1cpu        | 1     | -3.34%        | [-8.28%, +1.78%]   | 97.85156% |
| stock-stream-1cpu        | 2     | -3.55%        | [-6.59%, +10.67%]  | 97.85156% |
| stock-stream-4cpu        | 1     | +7.87%        | [-0.42%, +13.61%]  | 97.85156% |
| stock-stream-4cpu        | 2     | +1.41%        | [-10.22%, +22.23%] | 97.85156% |
| sparse-stream-1cpu       | 1     | -0.13%        | [-4.79%, +2.72%]   | 97.85156% |
| sparse-stream-1cpu       | 2     | +3.04%        | [+0.34%, +5.11%]   | 97.85156% |
| uniform-buy-stream-1cpu  | 1     | +0.55%        | [-1.06%, +5.61%]   | 97.85156% |
| uniform-buy-stream-1cpu  | 2     | -0.80%        | [-7.88%, +3.40%]   | 97.85156% |
| uniform-sell-stream-1cpu | 1     | -0.94%        | [-5.97%, +5.68%]   | 97.85156% |
| uniform-sell-stream-1cpu | 2     | -1.19%        | [-4.27%, +3.08%]   | 97.85156% |

| Case                     | Side      | P95 ms  | CPU P50 ms | Cores P50 | Peak RSS P50 MiB |
|--------------------------|-----------|---------|------------|-----------|------------------|
| stock-stream-1cpu        | baseline  | 185.675 | 171.875    | 1.000     | 225.45           |
| stock-stream-1cpu        | candidate | 188.638 | 166.081    | 1.000     | 224.06           |
| stock-stream-4cpu        | baseline  | 170.882 | 207.480    | 1.325     | 237.31           |
| stock-stream-4cpu        | candidate | 190.177 | 202.096    | 1.252     | 236.36           |
| sparse-stream-1cpu       | baseline  | 21.516  | 20.383     | 1.000     | 66.59            |
| sparse-stream-1cpu       | candidate | 21.917  | 20.529     | 1.000     | 66.71            |
| uniform-buy-stream-1cpu  | baseline  | 98.282  | 88.651     | 1.000     | 82.64            |
| uniform-buy-stream-1cpu  | candidate | 93.758  | 88.798     | 1.000     | 83.23            |
| uniform-sell-stream-1cpu | baseline  | 95.015  | 86.330     | 1.000     | 83.32            |
| uniform-sell-stream-1cpu | candidate | 92.246  | 86.866     | 1.000     | 83.32            |

Accepted profiles: 2.

Five fixed cases use200formal observations and20warmups in two rounds
of ten alternating pairs per case. Both dense1 profiles use six jobs
per product, excluding the first; five validated processing windows
are required. No case, sample, favorable retry or threshold is added.
One600-second budget includes controller setup and live global quiet
guard, fixtures, preflight, warmups, startup, full oracles, resources,
profiles/capture/decoding, statistics, publication, postseals and cleanup.
Builds are separate. At least one dense target needs the exact maintained
improved verdict, with complete evidence and no confirmed regression on
any case. Inconclusive is not equivalence, speedup or confirmed regression.
The fresh baseline is public5a66b6d5. Old1fc and experimental rankedP
products are incompatible; compatible dependency caches do not replace
fresh core/example compilation or public source and binary proof.
Both products use identical common657 instrumentation from public841.
Dense has10k zero seed rows plus20x100k source rows, selecting80k trades
per business batch. Sparse has10k seed groups plus20x100 source rows,
selecting80 trades/batch; those business batches alternate uniform sides.
Uniform-buy and uniform-sell reuse the maintained one-cycle all-trade
recipe:10k seeds plus20x10k business rows. Each forces side on the seed
AND all business records, preserving quantity, other column values/types,
exact side Field, schema metadata and caller-owned input buffers.
The two pure masks are separately paired, so one branch cannot hide the
other's regression in an alternating workload. Both retain21 complete
independent arithmetic/schema/input/snapshot checks per job.
DF target/input partitions remain1, min_rows_per_partition1, batch8192,
checkpoints off. Dense4 uses CPUs0/2/4/6 withTokio4; others CPU0/Tokio1.
The controller uses CPU31 and external helper pools stay1. Host load
and power outside the WSL guest remain unobservable.
The timer starts after IPC preload and covers graph planning, execution,
retained full outputs, natural end and cleanup. Oracles are outside that
timer but inside the total budget. Whole-process peak RSS includes input
preload and retained outputs, not operator-only heap or reservation fees.
Full CPU stacks and exact clock/loss evidence are retained. Inclusive
shares overlap and are not phase wall times; missing/inlined frames or
zero matched samples do not establish zero runtime cost or a route hit.
Mixed CASE with an all-trade filter, finite, nullable/UInt64/generic/UDF,
checkpoint and tight-budget latency remain unmeasured. Focused source
semantics and allocation tests are separate from these timings.
[Summary](summary.json), [formal samples](samples.csv),
[complete raw evidence](complete-evidence.json.gz),
[profiles](profile-summary.json), [targets](profile-targets.json) and
[manifest](evidence-manifest.json) retain receipts, intervals and seals.

Stopped reason: None
