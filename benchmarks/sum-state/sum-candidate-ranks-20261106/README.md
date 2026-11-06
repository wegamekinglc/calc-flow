# sum-candidate-ranks-20261106

Status: **completed**.
Performance acceptance: **False**.

| Case                 | Baseline P50 ms | Candidate P50 ms | Change | Verdict                 |
|----------------------|-----------------|------------------|--------|-------------------------|
| stock-stream-1cpu    | 171.230         | 171.482          | +0.15% | no-confirmed-regression |
| stock-stream-4cpu    | 152.138         | 147.029          | -3.36% | no-confirmed-regression |
| sparse-stream-1cpu   | 21.353          | 21.074           | -1.31% | inconclusive            |
| distinct-stream-1cpu | 88.291          | 83.124           | -5.85% | no-confirmed-regression |
| repeat2-stream-1cpu  | 96.154          | 91.760           | -4.57% | no-confirmed-regression |

| Case                 | Round | Paired median | Median interval   | Coverage  |
|----------------------|-------|---------------|-------------------|-----------|
| stock-stream-1cpu    | 1     | -0.33%        | [-2.79%, +3.57%]  | 97.85156% |
| stock-stream-1cpu    | 2     | -1.42%        | [-7.27%, +3.82%]  | 97.85156% |
| stock-stream-4cpu    | 1     | -5.51%        | [-13.15%, +2.99%] | 97.85156% |
| stock-stream-4cpu    | 2     | -0.90%        | [-6.11%, +1.65%]  | 97.85156% |
| sparse-stream-1cpu   | 1     | -5.69%        | [-7.50%, +1.59%]  | 97.85156% |
| sparse-stream-1cpu   | 2     | +1.73%        | [-4.69%, +18.25%] | 97.85156% |
| distinct-stream-1cpu | 1     | -4.91%        | [-6.99%, -1.74%]  | 97.85156% |
| distinct-stream-1cpu | 2     | -6.36%        | [-9.77%, -1.72%]  | 97.85156% |
| repeat2-stream-1cpu  | 1     | -3.29%        | [-12.02%, -0.54%] | 97.85156% |
| repeat2-stream-1cpu  | 2     | -5.21%        | [-12.16%, -2.77%] | 97.85156% |

| Case                 | Side      | P95 ms  | CPU P50 ms | Cores P50 | Peak RSS P50 MiB |
|----------------------|-----------|---------|------------|-----------|------------------|
| stock-stream-1cpu    | baseline  | 184.346 | 171.222    | 1.000     | 225.87           |
| stock-stream-1cpu    | candidate | 187.955 | 170.066    | 1.000     | 226.12           |
| stock-stream-4cpu    | baseline  | 165.568 | 199.532    | 1.325     | 239.72           |
| stock-stream-4cpu    | candidate | 160.926 | 195.952    | 1.347     | 240.09           |
| sparse-stream-1cpu   | baseline  | 22.977  | 21.354     | 1.000     | 66.51            |
| sparse-stream-1cpu   | candidate | 25.630  | 21.072     | 1.000     | 66.64            |
| distinct-stream-1cpu | baseline  | 92.428  | 88.247     | 1.000     | 83.43            |
| distinct-stream-1cpu | candidate | 91.896  | 83.121     | 1.000     | 82.02            |
| repeat2-stream-1cpu  | baseline  | 104.934 | 96.147     | 1.000     | 94.88            |
| repeat2-stream-1cpu  | candidate | 96.274  | 91.751     | 1.000     | 95.14            |

Accepted profiles: 2.

Five fixed candidate/rank/slot cases use200formal observations and20warmups
under one600-second budget including preflight, fixture/oracle setup,
profiles/decoding, resources/statistics, publication/postseals and cleanup.
Builds are separate. No cases, samples or favorable retries are added.
At least one dense target needs the exact maintained improved verdict,
with no confirmed regression in any case and all evidence complete.
Inconclusive is uncertainty, not equivalence or a confirmed speedup.
The freshly compiled baseline is public1fcafde1 including Arrow59.3.2;
older22f/P74/columnar experiment products are not compatible baselines.
Both frozen build trees use common657 instrumentation from public841.
DF target/input partitions remain1, min_rows_per_partition1, checkpoints off.
Dense1/controls use CPU0/Tokio1; dense4 uses CPUs0/2/4/6 andTokio4.
The controller uses CPU31; external helper pools stay1. WSL host load
and power are unobservable; resource fingerprints and guards are retained.
Dense has10k zero seed rows plus20x100k source rows, filtered to80k trades
per business batch. Sparse has10k seeds plus20x100 rows/80trades each.
Distinct preserves10k seeds and20x10k all-trade business rows (one/key).
Repeat2 preserves10k seeds and20x20k all-trade rows (two/key,N2G).
The old edge IPC recipe is reused as low-repetition control, with no
columnar route or new enablement threshold implied by these cases.
Every native sample retains21 complete value/schema/hash-checked snapshots.
Peak process RSS includes inputs and retained outputs, not operator heap.
Common657 inputs are nonnull. UInt64/nullable/generic, finite, checkpoint
and tight-budget latency remain unmeasured; focused tests are separate proof.
Both dense1 profiles require five processing windows after excluding each
first job. All timestamps and loss/unwind diagnostics are preserved.
Full stacks and maintained candidate/hash/drop targets describe CPU samples.
Inclusive shares overlap and are not wall-time phases. Missing/inlined
frames do not establish zero cost or a separately verified runtime path hit.
[Summary](summary.json), [all formal samples](samples.csv),
[complete raw evidence](complete-evidence.json.gz),
[profiles](profile-summary.json), [targets](profile-targets.json) and
[manifest](evidence-manifest.json) retain all receipts, intervals and hashes.

Stopped reason: None
