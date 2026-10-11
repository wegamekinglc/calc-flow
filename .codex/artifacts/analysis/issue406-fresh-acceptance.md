# Issue 406: fresh acceptance attempt at a681e7da

The fixed full-inventory attempt stopped on the current candidate's 4M retained
Join with checkpoint ON during its first warmup. No later group was executed,
no sample was retried, and no result below establishes performance acceptance.
All eight requested rows remain visible. Earlier revisions' measurements are
preserved in [the historical diagnosis](issue406-eight-workload-diagnosis.md)
and must not fill the current runtime's unmeasured cells.

Measured product identities are main `588726c8bce0f1c57b90077064834c4a472ee5f2`
and candidate `a681e7daf84672fa254c87b757df5618ebaf9fb2`. The supplementary
retained-OFF control uses a fresh run of `cffcf57be9c5cada667c6109156bcbfe5b084f27`.
Build helper `121386713378459a4f16f63ee4848a9db7a5d466` and execution helper
`0262a28cf0410848b3409aaf04276d4e55af006b` are distinct from those product revisions.

## Eight required targets

Values are p50. Paired groups use two rounds of ten observations and one excluded
warmup per round. Parallel entries use 32 configured CPUs; actual utilization is
not established by affinity. Main has no true checkpoint-OFF mode. Its LOW/ON
values marked † are processing diagnostics with failed EOF completion, not
complete-lifecycle speedup denominators. Absolute stream targets use true OFF.
The historical approximate 70 ms harness-overhead target does not replace a
fresh matched projection floor. ASOF remains 1M rows per input. Capture/restore
counts both sides' actual retained rows and includes snapshot preparation.

| Workload                                 | Fresh main 588726c8        | Current runtime a681e7da   | Single / parallel target         |
|------------------------------------------|----------------------------|----------------------------|----------------------------------|
| Stream Join, sealed dimension, 1M        | LOW 128.25 / 317.96 ms †   | OFF 98.94 / 98.07 ms       | ≤60 / ≤15 ms; above              |
| Stream Join, 10M                         | LOW 3758.60 / 5897.75 ms † | OFF 1154.53 / 1107.42 ms   | ≤600 / ≤150 ms; above            |
| Retained Join, 4M ÷ 1M                   | LOW 5.992× †; ON failed    | OFF 4.011×; 4M ON failed   | ≤4.4× / —; OFF observed within   |
| Join capture / restore per retained row  | Unmeasured in this attempt | Unmeasured in this attempt | ≤0.1 / ≤0.2 µs; parallel —       |
| Join handler during compaction ÷ steady  | Unmeasured in this attempt | Unmeasured in this attempt | ≤2× / —                          |
| Stream ASOF, 1M rows per input           | Unmeasured in this attempt | Unmeasured in this attempt | ≤100 / ≤30 ms                    |
| ASOF harness overhead, 1,024-row batches | Unmeasured in this attempt | Unmeasured in this attempt | ≤2× matched projection floor / — |
| Full Join callback, 1,024 rows           | Unmeasured in this attempt | Unmeasured in this attempt | ≤50 µs / —                       |

## Separate checkpoint policies

LOW uses a 24-hour interval and retains terminal checkpoint completion; ON uses
100 ms. Relative processing-throughput goals remain LOW ≥90% of OFF and ON ≥75%
of OFF. Native checkpoint-enabled components have no periodic frequency.

| Case                    | Main LOW †, ms | Current LOW, ms | Main ON †, ms | Current ON, ms | Current OFF, ms |
|-------------------------|----------------|-----------------|---------------|----------------|-----------------|
| Join 1M, 1 CPU          | 128.25         | 98.81           | 211.04        | 107.23         | 98.94           |
| Join 1M, 32 CPU         | 317.96         | 101.39          | 398.52        | 103.80         | 98.07           |
| Join 10M, 1 CPU         | 3758.60        | 1195.46         | 4129.75       | 1380.38        | 1154.53         |
| Join 10M, 32 CPU        | 5897.75        | 1089.26         | 6152.63       | 1233.98        | 1107.42         |
| Retained Join 1M, 1 CPU | 416.61         | 328.27          | Failed warmup | 907.75         | 207.80          |
| Retained Join 4M, 1 CPU | 2496.16        | 1780.37         | Failed warmup | Failed warmup  | 833.56          |

Retained LOW observes only 63.30% / 46.82% of OFF throughput at 1M / 4M.
Retained 1M ON observes 22.89%. Those configuration goals are missed.
Completed checkpoint windows fully inside processing and qualifying retained
manifest evidence are different measurements. Current sealed Join 1M has no
completed processing window at either CPU setting. Current sealed Join 10M has
11–14 such windows per sample at 1 CPU and 10–13 at 32 CPUs; every sample includes
periodic work, but only 7/20 per setting meet the retained-manifest proof. Current
retained Join 1M has 2–3 completed processing windows in every sample, while 0/20
meet that proof. Qualification requires a retained nonterminal manifest with a
positive left-source position. Zero qualifying observations therefore does not
mean zero periodic work. The collector's stricter periodic-impact classification
remains unmeasured or mixed; retained 4M ON has no valid observation.

## Fresh retained-OFF temporal control

These are cff/current pairs, not main comparisons. Terminal includes the final
watermark; full-cycle is measured directly through owned cleanup. Separate phase
medians need not sum. The intervals below have 97.8516% per-round coverage under
the maintained method. Both sizes have all 20 pairs. Processing p50 is higher,
but the maintained paired regression gate is inconclusive. Descriptive terminal
and full-cycle reductions do not override the final inconclusive assessment.

| Phase         | Fresh cff, ms | Current, ms | Paired round changes [interval]                                      | Verdict      |
|---------------|---------------|-------------|----------------------------------------------------------------------|--------------|
| 1M processing | 193.612       | 207.803     | r1: +11.288% [+4.370%, +17.055%]; r2: +5.803% [-2.023%, +9.051%]     | inconclusive |
| 1M terminal   | 309.433       | 16.001      | r1: -94.815% [-95.031%, -94.483%]; r2: -94.861% [-94.990%, -94.421%] | inconclusive |
| 1M full_cycle | 502.745       | 224.794     | r1: -55.288% [-56.254%, -53.479%]; r2: -55.631% [-57.214%, -55.079%] | inconclusive |
| 4M processing | 759.942       | 833.559     | r1: +10.519% [+4.085%, +15.097%]; r2: +4.524% [-4.597%, +14.476%]    | inconclusive |
| 4M terminal   | 1373.001      | 60.388      | r1: -95.604% [-95.928%, -95.134%]; r2: -95.572% [-96.286%, -95.111%] | inconclusive |
| 4M full_cycle | 2129.662      | 890.984     | r1: -57.935% [-59.065%, -56.755%]; r2: -58.550% [-63.848%, -56.339%] | inconclusive |

## Failure, coverage and budget

The failed candidate warmup had completed checkpoint epoch 10 and failed at
epoch 11 in `sources_cut`, with two source acknowledgements and no operator
acknowledgements. Join had admitted 3,520,064 rows, retained 3,520,000 left rows
plus 64 right rows, and emitted 3,520,000 matches. The safe public errors identify
the Join operator and left source, but do not expose the underlying exception.
The original cause remains unresolved; neither memory exhaustion nor a specific
writer branch is proved by this status. Cleanup completed with zero tasks.

Both main retained-ON warmups (1M and 4M) matched their exact preregistered
failure signature and retired naturally. The candidate failure was not eligible
for that exception and stopped the entire collection. There were 641 physical
attempts: 580 completed timed calls, 58 completed warmups and three failed
warmups. Of 64 primary metric arms, 27 completed and 37 have no timed sample;
two additional cff arms completed. The latter reference the already collected
current OFF observations rather than duplicating them.

Inclusive time through evidence sealing was 2,162.130 seconds; adding the
five-second final seal reserve accounts for 2,167.130 / 5,400 seconds. This
includes the failed shell launch, failed compiler-provenance preflight, fixes,
review, preparation, warmups, measurements, correctness checks and cleanup.
Builds are excluded: candidate artifacts took 1,292.266 seconds, the additional
main traced wheel 575.383 seconds and main ordinary wheel 377.757 seconds
(37.42 minutes total). A trailing compiler-output newline caused the preflight
failure; the reviewed helper correction preserves the compiler identity.

All 60 sampled child PIDs retired. There were no forced kills, surviving groups,
pending controller tasks, competitors or resource-guard breaches. The collector
retains the natural worker-exit error in its cleanup diagnostics. The independent
review verified all 1,410 sealed files and all 638 successful raw observations.
The final assessment is `environment_suitable=false` after the stopped run;
WSL2 host isolation is also unverified. Every supplementary phase verdict remains
inconclusive even where descriptive paired statistics suggest an improvement.

`budget.json` reports `evidence_complete=true` for completed sealing;
`summary.json` reports `evidence_complete=false` for incomplete acceptance.
The retained live marker in `progress.json` is stale. The final result, summary
and budget files are authoritative.

Evidence lives in the parent checkout at
`target/issue406-next-acceptance-execution-retry/measurement/`.
The evidence seal SHA-256 is
`d8e3f0d5c7a2c917fe2b68c5e0f3b8700789442f7ffd2f3b2becec7357927652`;
the launch plan SHA-256 is
`cc424c24c2d98c18f8ad011d923907cea2eaa16dc29b39f089a36de3519066d6`.
Original monotonic start `312812.11854143895` and launch precharge
`655.5343466270715` remain in the sealed accounting.

## Next development work

Reproduce the candidate failure with the smallest focused test and retain its
underlying error and ownership/admission evidence before selecting a correction.
The clean-history 4M diagnostic did not cover every dirty periodic replacement.
Preserve checkpoint cuts, obsolete-row validation, cancellation, memory limits
and held-history ownership. Then preregister the remaining/final acceptance
measurements on the resulting runtime; do not extend this sealed attempt or
reinterpret unmeasured rows as passing. The Clippy generated-libtest diagnostic
also remains unresolved; this report does not establish merge readiness.


## Detailed frozen collector report

Incomplete observations; missing controls and target rows remain explicit.

Run state: stopped; budget satisfied: True; environment suitable: False; compatibility verified: True; machine: WSL2. Observed target values are descriptive; an unsuitable environment or incomplete evidence cannot establish acceptance.

Only same-configuration baseline/candidate AB/BA observations use paired confidence intervals. Absolute median intervals use the maintained exact order-statistic construction and assume iid observations. Cross-mode, cross-size and cross-component metrics use group medians without pseudo-pairing.

| Row | Workload                                                            | Fresh main (explicit policy)                                                                                                                                                                                                                                                                           | Candidate target observation                                                                                                                                                                                                     |
|-----|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | Sealed-dimension Join, 1M quotes                                    | low/1CPU: p50 0.128253, p95 0.137715 [0.126213, 0.132836] s (diagnostic); on/1CPU: p50 0.211039, p95 0.225744 [0.208709, 0.215311] s (diagnostic); low/32CPU: p50 0.317963, p95 0.352075 [0.313202, 0.346167] s (diagnostic); on/32CPU: p50 0.398524, p95 0.427595 [0.391673, 0.416952] s (diagnostic) | off/1CPU join_sealed: p50 0.0989435, p95 0.106869 [0.0974924, 0.103926] seconds (≤0.06: observed-above-target); off/32CPU join_sealed: p50 0.0980728, p95 0.109963 [0.0960866, 0.100191] seconds (≤0.015: observed-above-target) |
| 2   | Sealed-dimension Join, 10M quotes                                   | low/1CPU: p50 3.7586, p95 3.90643 [3.73076, 3.7986] s (diagnostic); on/1CPU: p50 4.12975, p95 4.57722 [4.0696, 4.15466] s (diagnostic); low/32CPU: p50 5.89775, p95 6.2771 [5.81016, 6.08394] s (diagnostic); on/32CPU: p50 6.15263, p95 6.55896 [5.97033, 6.32344] s (diagnostic)                     | off/1CPU join_sealed: p50 1.15453, p95 1.29619 [1.14833, 1.1737] seconds (≤0.6: observed-above-target); off/32CPU join_sealed: p50 1.10742, p95 1.17282 [1.07421, 1.1336] seconds (≤0.15: observed-above-target)                 |
| 3   | Fully retained Join, 1M to 4M scaling                               | low: ratio of p50 5.99162 (diagnostic); on: ratio of p50 — (unmeasured)                                                                                                                                                                                                                                | OFF ratio of p50 4.01129 (≤4.4: observed-within-target)                                                                                                                                                                          |
| 4   | Join snapshot and restore, 1M resident rows                         | enabled/join_capture: unmeasured microseconds/row; enabled/join_restore: unmeasured microseconds/row                                                                                                                                                                                                   | enabled/1CPU join_capture: unmeasured microseconds/row (≤0.1: unmeasured); enabled/1CPU join_restore: unmeasured microseconds/row (≤0.2: unmeasured)                                                                             |
| 5   | 500-row Join handler after compaction, 1M resident rows             | Enabled ratio of p50 — (unmeasured); armed unmeasured s; steady unmeasured s                                                                                                                                                                                                                           | Enabled ratio of p50 — (≤2: unmeasured)                                                                                                                                                                                          |
| 6   | ASOF, 1M rows per side                                              | low/1CPU: unmeasured s (unmeasured); on/1CPU: unmeasured s (unmeasured); low/32CPU: unmeasured s (unmeasured); on/32CPU: unmeasured s (unmeasured)                                                                                                                                                     | off/1CPU asof: unmeasured seconds (≤0.1: unmeasured); off/32CPU asof: unmeasured seconds (≤0.03: unmeasured)                                                                                                                     |
| 7   | ASOF instrumented harness overhead, 1M per side / 1024-row messages | low: H=— ms, 2×floor=— ms (unmeasured)                                                                                                                                                                                                                                                                 | candidate/off: H=— ms, 2×floor=— ms (unmeasured); candidate/low: H=— ms, 2×floor=— ms (unmeasured)                                                                                                                               |
| 8   | Join callback, 1M quotes / 1024-row messages                        | enabled/join_small_batch: unmeasured microseconds/full-batch callback                                                                                                                                                                                                                                  | off/1CPU join_small_batch: unmeasured microseconds/full-batch callback (≤50.0: unmeasured); enabled/1CPU join_small_batch: unmeasured microseconds/full-batch callback (≤50.0: unmeasured)                                       |

Main has no checkpoint-off arm. Its low/on values have their policies labeled and are not revision pairs with candidate off. Checkpoint mode comparisons use ratios of independently collected p50 values. The frozen collector labels periodic impact using retained advancing-epoch proof. Completed processing windows are reported separately above; a missing retained manifest does not establish that no periodic checkpoint ran.

Row 7 reports an instrumented matched-component residual: end-to-end processing extends through final output projection watermark completion, and native replay follows each recorded ASOF dispatch chain. Recorder hook costs remain in both clocks. The original ready-to-Arrow clock, recorder setup and trace capture costs remain separate. Replay links establish semantic matching; the residual uses independent group medians and does not claim identical global task scheduling or causal CPU attribution.

| Row | Mode                 | OFF-relative throughput | Goal  | Observed configuration | Periodic impact                    | EOF p50/p95 [interval], ms                  | Full cycle p50/p95 [interval], ms           | Checkpoint bytes p50/p95 [interval]                         |
|-----|----------------------|-------------------------|-------|------------------------|------------------------------------|---------------------------------------------|---------------------------------------------|-------------------------------------------------------------|
| 1   | N=1000000/1CPU/low   | 1.00139                 | ≥0.9  | observed-within-target | not-applicable (0/20 observations) | p50 62.8155, p95 149.601 [59.4585, 64.5001] | p50 161.87, p95 254.433 [160.48, 167.604]   | p50 3403, p95 3403 [3403, 3403]                             |
| 1   | N=1000000/1CPU/on    | 0.922716                | ≥0.75 | observed-within-target | unmeasured (0/20 observations)     | p50 100.17, p95 107.078 [96.0586, 102.74]   | p50 207.989, p95 215.979 [205.811, 213.421] | p50 19506, p95 19506 [19506, 19506]                         |
| 1   | N=1000000/32CPU/low  | 0.967237                | ≥0.9  | observed-within-target | not-applicable (0/20 observations) | p50 96.5815, p95 186.627 [91.9222, 105.516] | p50 199.662, p95 285.534 [191.722, 210.831] | p50 3403, p95 3403 [3403, 3403]                             |
| 1   | N=1000000/32CPU/on   | 0.944852                | ≥0.75 | observed-within-target | unmeasured (0/20 observations)     | p50 147.781, p95 155.505 [142.237, 151.419] | p50 252.762, p95 264.035 [244.28, 255.439]  | p50 19506, p95 19506 [19506, 19506]                         |
| 2   | N=10000000/1CPU/low  | 0.965755                | ≥0.9  | observed-within-target | not-applicable (0/20 observations) | p50 153.374, p95 162.938 [149.765, 157.344] | p50 1346.06, p95 1577.62 [1310.49, 1447.45] | p50 3410, p95 3410 [3410, 3410]                             |
| 2   | N=10000000/1CPU/on   | 0.836381                | ≥0.75 | observed-within-target | mixed-coverage (7/20 observations) | p50 111.674, p95 155.591 [107.624, 126.251] | p50 1493.95, p95 1573.81 [1471.77, 1549.02] | p50 19525, p95 19528 [19525, 19528]                         |
| 2   | N=10000000/32CPU/low | 1.01667                 | ≥0.9  | observed-within-target | not-applicable (0/20 observations) | p50 195.973, p95 211.341 [189.692, 200.623] | p50 1284.31, p95 1394.54 [1221.81, 1296.54] | p50 3410, p95 3410 [3410, 3410]                             |
| 2   | N=10000000/32CPU/on  | 0.897432                | ≥0.75 | observed-within-target | mixed-coverage (7/20 observations) | p50 161.142, p95 199.85 [97.3238, 167.794]  | p50 1404.82, p95 1519.18 [1288.2, 1437.54]  | p50 19525, p95 19528 [19525, 19525]                         |
| 3   | N=1000000/1CPU/low   | 0.633016                | ≥0.9  | observed-below-target  | not-applicable (0/20 observations) | p50 575.957, p95 652.316 [547.812, 600.038] | p50 906.789, p95 1068.9 [868.099, 932.657]  | p50 3411, p95 3411 [3411, 3411]                             |
| 3   | N=4000000/1CPU/low   | 0.468194                | ≥0.9  | observed-below-target  | not-applicable (0/20 observations) | p50 2971.64, p95 3240.01 [2836.45, 3078.07] | p50 4793.66, p95 5392.88 [4473.84, 4935.92] | p50 3412, p95 3412 [3412, 3412]                             |
| 3   | N=1000000/1CPU/on    | 0.228922                | ≥0.75 | observed-below-target  | unmeasured (0/20 observations)     | p50 914.366, p95 1276.88 [892.211, 1056.9]  | p50 1867.82, p95 2324.65 [1784.81, 2101.37] | p50 1.62527e+08, p95 1.62528e+08 [1.62527e+08, 1.62527e+08] |
| 3   | N=4000000/1CPU/on    | —                       | ≥0.75 | unmeasured             | unmeasured (0/20 observations)     | unmeasured                                  | unmeasured                                  | unmeasured                                                  |
| 6   | N=1000000/1CPU/low   | —                       | ≥0.9  | unmeasured             | not-applicable (0/20 observations) | unmeasured                                  | unmeasured                                  | unmeasured                                                  |
| 6   | N=1000000/1CPU/on    | —                       | ≥0.75 | unmeasured             | unmeasured (0/20 observations)     | unmeasured                                  | unmeasured                                  | unmeasured                                                  |
| 6   | N=1000000/32CPU/low  | —                       | ≥0.9  | unmeasured             | not-applicable (0/20 observations) | unmeasured                                  | unmeasured                                  | unmeasured                                                  |
| 6   | N=1000000/32CPU/on   | —                       | ≥0.75 | unmeasured             | unmeasured (0/20 observations)     | unmeasured                                  | unmeasured                                  | unmeasured                                                  |
| 7   | N=1000000/1CPU/low   | —                       | ≥0.9  | unmeasured             | not-applicable (0/20 observations) | unmeasured                                  | unmeasured                                  | unmeasured                                                  |
| 7   | N=1000000/1CPU/on    | —                       | ≥0.75 | unmeasured             | unmeasured (0/20 observations)     | unmeasured                                  | unmeasured                                  | unmeasured                                                  |

EOF includes terminal completion; full cycle runs from processing start through owned cleanup. Machine-readable phase costs (`summary.json` in the evidence directory) retain main/off/native setup, oracle, preparation, processing, terminal, cleanup and CPU clocks where collected. Raw observations (`result.json` in the evidence directory), process resource samples (`resources.json` in the evidence directory) and final budget ledger (`budget.json` in the evidence directory) retain their separate evidence.

All timing statistics below are seconds; native checkpoint metrics retain their complete measured boundary. Invalid baseline lifecycle results are diagnostic.

| Case                                                          | Main p50/p95 [median interval]                  | Main state  | Candidate p50/p95 [median interval]               | Candidate state | Revision paired change [CI]                              | Verdict               |
|---------------------------------------------------------------|-------------------------------------------------|-------------|---------------------------------------------------|-----------------|----------------------------------------------------------|-----------------------|
| row-1/join_sealed/1000000/batch-64000/off/cpu-1               | unmeasured                                      | unsupported | p50 0.0989435, p95 0.106869 [0.0974924, 0.103926] | ok              | —                                                        | candidate-only        |
| row-1/join_sealed/1000000/batch-64000/low/cpu-1               | p50 0.128253, p95 0.137715 [0.126213, 0.132836] | diagnostic  | p50 0.098806, p95 0.110953 [0.0967176, 0.103439]  | ok              | -23.517% [-25.558, -18.933]; -22.840% [-27.833, -18.246] | diagnostic            |
| row-1/join_sealed/1000000/batch-64000/on/cpu-1                | p50 0.211039, p95 0.225744 [0.208709, 0.215311] | diagnostic  | p50 0.107231, p95 0.117722 [0.10308, 0.112585]    | ok              | -50.294% [-54.455, -47.757]; -48.731% [-50.877, -43.837] | diagnostic            |
| row-1/join_sealed/1000000/batch-64000/off/cpu-32              | unmeasured                                      | unsupported | p50 0.0980728, p95 0.109963 [0.0960866, 0.100191] | ok              | —                                                        | candidate-only        |
| row-1/join_sealed/1000000/batch-64000/low/cpu-32              | p50 0.317963, p95 0.352075 [0.313202, 0.346167] | diagnostic  | p50 0.101395, p95 0.109324 [0.0995201, 0.104934]  | ok              | -68.670% [-71.720, -65.116]; -68.495% [-69.950, -66.828] | diagnostic            |
| row-1/join_sealed/1000000/batch-64000/on/cpu-32               | p50 0.398524, p95 0.427595 [0.391673, 0.416952] | diagnostic  | p50 0.103797, p95 0.111932 [0.102044, 0.106121]   | ok              | -72.962% [-76.697, -71.603]; -74.002% [-75.614, -71.955] | diagnostic            |
| row-2/join_sealed/10000000/batch-64000/off/cpu-1              | unmeasured                                      | unsupported | p50 1.15453, p95 1.29619 [1.14833, 1.1737]        | ok              | —                                                        | candidate-only        |
| row-2/join_sealed/10000000/batch-64000/low/cpu-1              | p50 3.7586, p95 3.90643 [3.73076, 3.7986]       | diagnostic  | p50 1.19546, p95 1.42712 [1.15399, 1.30048]       | ok              | -67.736% [-70.531, -63.431]; -68.499% [-70.295, -61.964] | diagnostic            |
| row-2/join_sealed/10000000/batch-64000/on/cpu-1               | p50 4.12975, p95 4.57722 [4.0696, 4.15466]      | diagnostic  | p50 1.38038, p95 1.44877 [1.37709, 1.40362]       | ok              | -66.314% [-67.220, -64.900]; -67.279% [-69.742, -65.267] | diagnostic            |
| row-2/join_sealed/10000000/batch-64000/off/cpu-32             | unmeasured                                      | unsupported | p50 1.10742, p95 1.17282 [1.07421, 1.1336]        | ok              | —                                                        | candidate-only        |
| row-2/join_sealed/10000000/batch-64000/low/cpu-32             | p50 5.89775, p95 6.2771 [5.81016, 6.08394]      | diagnostic  | p50 1.08926, p95 1.19248 [1.03105, 1.11234]       | ok              | -81.119% [-82.776, -79.625]; -81.605% [-82.754, -80.915] | diagnostic            |
| row-2/join_sealed/10000000/batch-64000/on/cpu-32              | p50 6.15263, p95 6.55896 [5.97033, 6.32344]     | diagnostic  | p50 1.23398, p95 1.34739 [1.19507, 1.28038]       | ok              | -80.556% [-81.726, -78.465]; -79.622% [-80.528, -77.970] | diagnostic            |
| row-3/join_retained/1000000/batch-64000/off/cpu-1             | unmeasured                                      | unsupported | p50 0.207803, p95 0.271522 [0.205023, 0.218044]   | ok              | —                                                        | candidate-only        |
| row-3/join_retained/1000000/batch-64000/low/cpu-1             | p50 0.416609, p95 0.486811 [0.412616, 0.428862] | diagnostic  | p50 0.328274, p95 0.40314 [0.320064, 0.346266]    | ok              | -22.030% [-28.308, -20.476]; -19.533% [-24.400, -12.929] | diagnostic            |
| row-3/join_retained/1000000/batch-64000/on/cpu-1              | unmeasured                                      | unmeasured  | p50 0.907747, p95 1.21785 [0.88052, 0.977618]     | ok              | —                                                        | unmeasured-or-invalid |
| row-3/join_retained/4000000/batch-64000/off/cpu-1             | unmeasured                                      | unsupported | p50 0.833559, p95 0.880581 [0.815592, 0.854627]   | ok              | —                                                        | candidate-only        |
| row-3/join_retained/4000000/batch-64000/low/cpu-1             | p50 2.49616, p95 3.10141 [2.3568, 2.54091]      | diagnostic  | p50 1.78037, p95 2.01323 [1.61044, 1.84184]       | ok              | -34.384% [-41.345, -25.816]; -27.701% [-34.421, -19.989] | diagnostic            |
| row-3/join_retained/4000000/batch-64000/on/cpu-1              | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-6/asof/1000000/batch-64000/off/cpu-1                      | unmeasured                                      | unsupported | unmeasured                                        | unmeasured      | —                                                        | candidate-only        |
| row-6/asof/1000000/batch-64000/low/cpu-1                      | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-6/asof/1000000/batch-64000/on/cpu-1                       | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-6/asof/1000000/batch-64000/off/cpu-32                     | unmeasured                                      | unsupported | unmeasured                                        | unmeasured      | —                                                        | candidate-only        |
| row-6/asof/1000000/batch-64000/low/cpu-32                     | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-6/asof/1000000/batch-64000/on/cpu-32                      | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-4/join_capture/1000000/batch-64000/enabled/cpu-1          | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-4/join_restore/1000000/batch-64000/enabled/cpu-1          | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-5/join_handler_compacting/1000000/batch-500/enabled/cpu-1 | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-5/join_handler_steady/1000000/batch-500/enabled/cpu-1     | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-7/asof/1000000/batch-1024/off/cpu-1                       | unmeasured                                      | unsupported | unmeasured                                        | unmeasured      | —                                                        | candidate-only        |
| row-7/asof/1000000/batch-1024/low/cpu-1                       | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-7/asof/1000000/batch-1024/on/cpu-1                        | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-7/projection/1000000/batch-1024/off/cpu-1                 | unmeasured                                      | unsupported | unmeasured                                        | unmeasured      | —                                                        | candidate-only        |
| row-7/projection/1000000/batch-1024/low/cpu-1                 | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-7/projection/1000000/batch-1024/on/cpu-1                  | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-7/asof_operator/1000000/batch-1024/off/cpu-1              | unmeasured                                      | unsupported | unmeasured                                        | unmeasured      | —                                                        | candidate-only        |
| row-7/asof_operator/1000000/batch-1024/low/cpu-1              | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |
| row-8/join_small_batch/1000000/batch-1024/off/cpu-1           | unmeasured                                      | unsupported | unmeasured                                        | unmeasured      | —                                                        | candidate-only        |
| row-8/join_small_batch/1000000/batch-1024/enabled/cpu-1       | unmeasured                                      | unmeasured  | unmeasured                                        | unmeasured      | —                                                        | unmeasured-or-invalid |

### Supplementary retained-OFF comparison

cff control `cffcf57be9c5cada667c6109156bcbfe5b084f27` versus current `a681e7daf84672fa254c87b757df5618ebaf9fb2`.

Fresh cff/current temporal pairs. Candidate observations reference the primary ledger; they are not additional independent samples or a main comparison.

| Rows    | Phase              | Evidence | cff p50/p95 [median interval], ms           | Current p50/p95 [median interval], ms       | Paired verdict | Round paired median changes [interval]                                                                     |
|---------|--------------------|----------|---------------------------------------------|---------------------------------------------|----------------|------------------------------------------------------------------------------------------------------------|
| 1000000 | processing_seconds | complete | p50 193.612, p95 246.768 [185.391, 209.901] | p50 207.803, p95 271.522 [205.023, 218.044] | inconclusive   | r1: 11.288% [4.370%, 17.055%], coverage 0.978516; r2: 5.803% [-2.023%, 9.051%], coverage 0.978516          |
| 1000000 | terminal_seconds   | complete | p50 309.433, p95 375.079 [302.708, 322.308] | p50 16.0012, p95 20.7263 [15.7014, 16.6715] | inconclusive   | r1: -94.815% [-95.031%, -94.483%], coverage 0.978516; r2: -94.861% [-94.990%, -94.421%], coverage 0.978516 |
| 1000000 | full_cycle_seconds | complete | p50 502.745, p95 622.205 [492.503, 538.036] | p50 224.794, p95 292.548 [221.705, 233.604] | inconclusive   | r1: -55.288% [-56.254%, -53.479%], coverage 0.978516; r2: -55.631% [-57.214%, -55.079%], coverage 0.978516 |
| 4000000 | processing_seconds | complete | p50 759.942, p95 887.08 [735.124, 820.069]  | p50 833.559, p95 880.581 [815.592, 854.627] | inconclusive   | r1: 10.519% [4.085%, 15.097%], coverage 0.978516; r2: 4.524% [-4.597%, 14.476%], coverage 0.978516         |
| 4000000 | terminal_seconds   | complete | p50 1373, p95 1585.87 [1344.6, 1414.89]     | p50 60.3882, p95 70.4279 [58.5825, 64.1907] | inconclusive   | r1: -95.604% [-95.928%, -95.134%], coverage 0.978516; r2: -95.572% [-96.286%, -95.111%], coverage 0.978516 |
| 4000000 | full_cycle_seconds | complete | p50 2129.66, p95 2473.24 [2081.4, 2228.76]  | p50 890.984, p95 958.791 [877.748, 918.424] | inconclusive   | r1: -57.935% [-59.065%, -56.755%], coverage 0.978516; r2: -58.550% [-63.848%, -56.339%], coverage 0.978516 |

Warmups are excluded from statistics. Terminal includes the final watermark before EOF; the full-cycle clock is reported directly. Partial or invalid controls have no acceptance verdict; incompatible or unsuitable comparisons remain inconclusive.
