# sum-columnar-keys-20261105

Status: **completed**.
Performance acceptance: **False**.

| Case                 | Baseline P50 ms | Candidate P50 ms | Change  | Verdict      |
|----------------------|-----------------|------------------|---------|--------------|
| stock-stream-1cpu    | 185.134         | 189.210          | +2.20%  | inconclusive |
| stock-stream-4cpu    | 172.947         | 180.635          | +4.45%  | inconclusive |
| sparse-stream-1cpu   | 22.838          | 22.725           | -0.49%  | inconclusive |
| distinct-stream-1cpu | 99.598          | 99.617           | +0.02%  | inconclusive |
| edge-stream-1cpu     | 108.790         | 121.376          | +11.57% | inconclusive |

| Case                 | Round | Paired median | Median interval   | Coverage  |
|----------------------|-------|---------------|-------------------|-----------|
| stock-stream-1cpu    | 1     | +0.66%        | [-0.34%, +8.52%]  | 97.85156% |
| stock-stream-1cpu    | 2     | +3.24%        | [-0.38%, +4.78%]  | 97.85156% |
| stock-stream-4cpu    | 1     | +5.97%        | [-3.40%, +19.74%] | 97.85156% |
| stock-stream-4cpu    | 2     | +2.12%        | [-4.46%, +9.19%]  | 97.85156% |
| sparse-stream-1cpu   | 1     | +0.71%        | [-5.92%, +9.32%]  | 97.85156% |
| sparse-stream-1cpu   | 2     | +0.24%        | [-4.22%, +4.29%]  | 97.85156% |
| distinct-stream-1cpu | 1     | +0.07%        | [-2.96%, +6.48%]  | 97.85156% |
| distinct-stream-1cpu | 2     | +0.31%        | [-1.77%, +5.57%]  | 97.85156% |
| edge-stream-1cpu     | 1     | +12.76%       | [+4.88%, +19.55%] | 97.85156% |
| edge-stream-1cpu     | 2     | +9.51%        | [+5.57%, +14.08%] | 97.85156% |

| Case                 | Side      | P95 ms  | CPU P50 ms | Cores P50 | Peak RSS P50 MiB |
|----------------------|-----------|---------|------------|-----------|------------------|
| stock-stream-1cpu    | baseline  | 198.615 | 185.128    | 1.000     | 226.99           |
| stock-stream-1cpu    | candidate | 215.003 | 189.204    | 1.000     | 227.37           |
| stock-stream-4cpu    | baseline  | 188.924 | 221.300    | 1.280     | 240.67           |
| stock-stream-4cpu    | candidate | 204.245 | 226.445    | 1.272     | 243.57           |
| sparse-stream-1cpu   | baseline  | 24.732  | 22.838     | 1.000     | 66.12            |
| sparse-stream-1cpu   | candidate | 24.834  | 22.726     | 1.000     | 66.51            |
| distinct-stream-1cpu | baseline  | 103.515 | 99.592     | 1.000     | 83.17            |
| distinct-stream-1cpu | candidate | 113.691 | 99.611     | 1.000     | 83.24            |
| edge-stream-1cpu     | baseline  | 114.974 | 108.783    | 1.000     | 94.68            |
| edge-stream-1cpu     | candidate | 133.779 | 121.369    | 1.000     | 96.80            |

Five fixed cases use two rounds of ten alternating pairs (200 formal
observations and20warmups) and one
600-second budget including preflight, fixtures, warmups, independent
full-snapshot oracles, capture/decoding, analysis, publication and
cleanup. Builds are separate; no cases, samples or retries are added.

A dense1/dense4 target must have the exact maintained improved verdict,
with no regression on any case, after complete evidence validation.
Inconclusive is uncertainty, not equivalence or a confirmed speedup.

The baseline is freshly built public22f4b9a. The older P74 binary
is incompatible with current Batch/DataFusion trusted metadata source.
Both products use common657 native instrumentation from public841 in
permanently frozen build trees; development instrumentation is unchanged.
Input partitions remain at their default one. All streams use one
DataFusion target partition, min_rows_per_partition1 and no checkpoints.
The 4T case uses CPUs0/2/4/6, four reported physical cores, Tokio4;
single-thread cases use CPU0/Tokio1. The controller uses CPU31 on WSL2.
External helper pools stay at one; host load/power is unobservable.

Dense input is 10k zero seed trades plus20x100k source rows, filtering
to80k trades per business batch. Sparse input retains10k groups and
adds20x100 source rows (80trades/batch,100updatedgroups overall).
Distinct control retains10k seed keys, adding20x10k all-trade rows,
one occurrence per key per business batch (N=G); the selector remains
legacy. Edge control adds20x20k rows with two occurrences per key
(N=2G), the first eligible edge. Both cross the8192-row chunk boundary
and keep at most10k groups; cycles follow business_index%10, with
the edge also using the next cycle modulo10. These are independent
new fixtures, not old workload digests. Sparse is also a legacy control.
Dense80k-row SQL batches contain10k keys; within-chunk dedup could
save only1.89% canonical encodings versus87.5% across the whole batch.
These counts motivate the hypothesis and are not measured speedups.
Every stream retains21 complete snapshots, independently checked after
the native timer. Whole-process peak RSS includes inputs and retained
outputs; it is not operator-only allocation or an engine memory cap.
Common657 requires non-null inputs. UInt64, nullable/generic aggregates,
checkpoint/finite/below-edge and tight-byte latency are unmeasured.
Focused allocation/size/budget and correctness tests separately cover
semantics and budgets; they do not establish end-to-end latency.

Accepted profiles: 2.
Both serial profiles require five verified processing windows after
excluding the first of six jobs; all per-thread timestamps must align.
Self/inclusive counts and full stacks cover the row-encoding/hash path.
Inclusive CPU shares overlap and are not phase wall-time measurements.
[Exact route evidence](profile-route.json) intersects sealed nm anchors
with complete canonical symbols in the existing profiles. It is serial
path attribution, not a worker gate. Dispatcher-only samples do not prove
interner work. Missing/inlined/unsampled active-work anchors
remain unresolved, not zero cost or proof that the route did not run.
Same-shape focused path-hit tests remain separate evidence. Unresolved
profile leaves and attribution limits are retained.

[Summary](summary.json) and [all formal rows](samples.csv) retain exact
verdicts, every interval, CPU/RSS and source/build/fixture identities.
[Complete raw evidence](complete-evidence.json.gz) includes every warmup,
profile and formal observation, snapshot/schema digest, command, resource
receipt and clock timestamp text. Large perf files stay at recorded local
paths with hashes. [Profiles](profile-summary.json),
[targets](profile-targets.json) and full stacks preserve attribution.
[Manifest](evidence-manifest.json) seals actual budget debit and status.

Stopped reason: None
