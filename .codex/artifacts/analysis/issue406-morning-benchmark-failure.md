# October 12 morning benchmark investigation

## Confirmed failure

[Scheduled run 38178694560](https://github.com/wegamekinglc/calc-flow/actions/runs/38178694560)
started at 06:00:49 Asia/Shanghai and completed at 08:11:15. The complete report
contains 809 result rows and exactly one error:
`engines/10000000/calc-flow-stream/group_by: repeated regression exceeds +5%`.
The engine shard failed its regression check; the aggregate gate propagated that
failure. This was neither a missing-result failure nor a failed correctness oracle.

| CI observation               | Value                                        |
|------------------------------|----------------------------------------------|
| Baseline                     | `89e511416d5793e11cefd74b1a7c09aae96f81bd`   |
| Candidate                    | `8ee5b71b7b862d4e44bd0be390d182cf950d17f7`   |
| Baseline / candidate p50     | 853.563 / 953.112 ms                         |
| Combined p50 change          | +11.663%                                     |
| Round 1 paired median / CI   | +11.740% [+11.251%, +12.360%]                |
| Round 2 paired median / CI   | +11.583% [+11.330%, +11.790%]                |

Each round contains ten adjacent AB/BA pairs with successful oracles. Both lower
confidence bounds exceed the maintained +5% threshold, so the original run is a
valid regression verdict under that rule. The exact order-statistic interval is
nominally 95%, conservatively 97.85% at ten pairs under its iid assumption.
No threshold, sample filtering or verdict was changed.

This historical group_by case uses checkpoints enabled with a 24-hour interval;
its ready-to-output timer ends before EOF and the terminal checkpoint, while the
worker still requires successful job completion. It is a low-frequency case,
not a checkpoint-OFF measurement. OFF/LOW/ON acceptance remains separate.

The original native SHA256 values are:

- Baseline: `48e63a99052bece0cf59309b9057e3e950a8673383a0fbf3b6ca5e1cdc5454a6`.
- Candidate: `ef24e1d8fefb075909f602a054b03a621200abe4230823fd8259ad2b3a9defa5`.

Both release manifests record Rust 1.88.0, Python 3.13.16, equal Cargo lock hashes
and clean exact source revisions. The runner exposed four logical CPUs; both
arms used Tokio/Polars pools of 32, NumPy 2.5.2 and PyArrow 24.0.0. These thread
settings are existing catalog settings, not a newly imposed runtime restriction.

## Narrowing the cause

The source delta changes ASOF key admission and tests. The window/group aggregation
implementation, benchmark inputs and group_by timing adapter are unchanged.
The same morning run's 1M group_by pair was -0.698%, with both round intervals
below +5%. Other 10M paths did not trigger regression. The preceding manual full
run 38164813243 reported 10M group_by 770.869 / 771.040 ms (+0.022%); that run has
different artifacts and host history and cannot invalidate the morning result.

Downloaded the original wheels and verified both wheel/native seals using the
maintained release loader. A fixed local diagnostic reused the maintained
`measure_case` collector, oracle, alternating pairs and verdict, with A/A, B/B,
then A/B. Each comparison had two rounds of ten pairs and one warmup per side per
round. The recorded total budget was 600 seconds, including creation, startup,
oracles and cleanup; no builds were needed. Collection completed in 119.327 seconds
with no observed native-build overlap.

| Local diagnostic   | First arm p50   | Second arm p50   | Change    | Verdict                   |
|--------------------|-----------------|------------------|-----------|---------------------------|
| A/A                | 489.122 ms      | 489.322 ms       | +0.041%   | no-confirmed-regression   |
| B/B                | 494.282 ms      | 494.593 ms       | +0.063%   | no-confirmed-regression   |
| A/B                | 493.931 ms      | 498.305 ms       | +0.885%   | no-confirmed-regression   |

A/B paired intervals were +0.635% [-1.219%, +2.336%] and +0.509%
[-0.456%, +2.745%]. This local WSL host (Python 3.13.9, affinity 0–3 on a 32-CPU
host) is not the original GitHub environment. It did not reproduce +11.66%; it
does not establish equivalence or explain the original cause. Artifact seals
match the morning wheels; host/interpreter differences remain explicit.

## Follow-up and limits

Triggered exactly one existing `suite-recovery` run,
[38193911010](https://github.com/wegamekinglc/calc-flow/actions/runs/38193911010),
for the failed `engines-10000000` shard. It reuses the original run's sealed wheels
and exact source refs, preserves the complete shard and unchanged gate, and keeps
resource diagnostics. It does not rebuild or substitute the current ASOF feature.
It completed with all 51 cases and no missing/oracle errors. Only the original
10M stream group_by case regressed again:

| Recovery observation         | Value                                  |
|------------------------------|----------------------------------------|
| Baseline / candidate p50     | 860.439 / 947.314 ms                   |
| Combined p50 change          | +10.097%                               |
| Round 1 paired median / CI   | +9.951% [+9.255%, +10.574%]            |
| Round 2 paired median / CI   | +10.146% [+8.905%, +10.633%]           |

Both original native seals and common harness hash match. All 197 host resource
snapshots report zero swap usage; minimum available host memory was 8,543MiB.
The group_by snapshot had 14,196MiB available and a worker RSS of 827,580KiB.
These snapshots show no memory pressure explanation for the repeated regression;
they are not continuous per-worker peak measurements. The systemd unit's reported
3MiB peak does not describe the workers and is not used as their memory evidence.
No additional full-shard retries are planned.

The dispatched workflow definition comes from release main `8ce5380b`, advanced
after the morning run. The recovery job explicitly checks out candidate
`8ee5b71b` and baseline `89e51141`; its product inputs remain the original refs.
The workflow run head must not be mislabeled as the measured product.

## Fixed GitHub diagnostic and remaining uncertainty

The single five-case [diagnostic 38209692076](https://github.com/wegamekinglc/calc-flow/actions/runs/38209692076)
completed at harness commit `82306c6f` in 179.518 seconds, within its 600-second
budget, with no observed native-build overlap. It reused the original wheels and
common harness. All 120 paired observations passed their oracles, all twelve
paired worker completions succeeded, all fourteen worker input IPC hashes matched,
and all fourteen exit records were zero. Profile samples remain separate.

| GitHub diagnostic   | First arm p50   | Second arm p50   | Change    | Maintained verdict         |
|---------------------|-----------------|------------------|-----------|----------------------------|
| A/A                 | 476.592 ms      | 475.786 ms       | -0.169%   | no-confirmed-regression    |
| B/B                 | 487.065 ms      | 491.649 ms       | +0.941%   | inconclusive               |
| A/B                 | 470.508 ms      | 468.189 ms       | -0.493%   | inconclusive               |

A/B paired intervals are +0.145% [-4.010%, +9.737%] and -0.784%
[-2.793%, +0.445%]. B/B round one's upper bound is +7.252%. These intervals do
not establish equivalence or erase either earlier regression verdict. The
collector's success means valid collection, not that performance passed.

The new runner identifies an AMD EPYC 9V45, family 26/model 2, with two physical
cores/four threads and 32MiB L3. Kernel, Python and dependency fingerprints match
the failed runs, but the original/recovery reports did not retain CPU model/cache
information. Their baseline was about 854–860ms; this baseline was 471ms. A CPU
model difference is plausible but cannot be proven retroactively from missing
metadata. Scheduled shards now retain `host-cpu.json` via `lscpu --json` before
measurement; this does not change the common measuring harness or its gate.

All 692 diagnostic resource observations show zero swap and at least
14,273,576,960 bytes available. Sampling produced 1,228 baseline and 1,414
candidate CPU samples at 49Hz, with no recorded loss/throttling. The two 30-second
windows completed 44/52 oracle-checked workloads. The original stripped native
modules and their process maps remain the address source.

Offline grouping by the original ELF unwind-table function boundaries gives:

| Original native range start, A / B  | Self samples, A / B   | Share of all CPU samples, A / B  |
|-------------------------------------|-----------------------|----------------------------------|
| `0x15d14c0` / `0x15e0e60`           | 392 / 391             | 31.92% / 27.65%                  |
| `0x1364920` / `0x1905280`           | 168 / 205             | 13.68% / 14.50%                  |
| `0x1513570` / `0x1522f10`           | 115 / 133             | 9.36% / 9.41%                    |
| `0x1394130` / `0x1935770`           | 96 / 107              | 7.82% / 7.57%                    |
| `0x15140a0` / `0x1523a40`           | 64 / 74               | 5.21% / 5.23%                    |
| `0x13672d0` / `0x1907ac0`           | 48 / 61               | 3.91% / 4.31%                    |

The first range has 1,142/1,306 inclusive samples (93.00%/92.36%). Its most
sampled instruction is at the same function offset `+0x21e5` in both products,
with the mapping described below still an inference. Under that inferred mapping,
the offset lies in string-key lookup code and makes per-row hash work a lead for
a future reproduction. Samples include work outside the paired timer; percentages
alone cannot assign the earlier extra 10% to a particular function or exclude
other causes.

As an explicitly inferred mapping, the first range matches the ordinary local
symbolized `WindowAggregateOperator::process_data` future's 3,746 instructions at
the same offsets after normalizing external calls and RIP-relative relocations.
This is not a matching executable layout or proven original symbol table. Short
prefix matches for other ranges suggest hash/SipHash, window assignment and key
equality helpers, but prefixes alone are insufficient to resolve names. No
profile-build timing or false claim of resolved original Rust frames is used.

This faster-host A/B comparison is inconclusive. Stop the fixed collection here;
do not expand/retry until it produces a favorable result. A useful next causal
experiment requires reproducing the slowdown while retaining host identity and
sampling on that same host. No production optimization is selected from the
current non-reproducing profile alone.

Root cause attribution is still open. A host-dependent binary/code-layout effect
or runner behavior is a hypothesis, not an established finding. No production
change or benchmark-gate relaxation is justified by these results alone.

Raw logs, manifests, original reports, failed-case samples, local fixed plan,
per-comparison evidence and resource trace are preserved under the main checkout's
`target/issue406-morning-benchmark-failure/`.
The fixed GitHub diagnostic is artifact `11728218142`, preserved under
`github-diagnostic/` there; its outcome, raw perf streams, maps, IPC and seals
remain available. The analysis does not mark the original failure resolved.
