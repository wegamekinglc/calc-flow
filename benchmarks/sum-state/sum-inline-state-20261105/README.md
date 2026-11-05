# sum-inline-state-20261105

Status: **completed**.
Performance acceptance: **True**.

| Case               | Baseline P50 ms | Candidate P50 ms | Change  | Verdict  |
|--------------------|-----------------|------------------|---------|----------|
| stock-stream-1cpu  | 263.467         | 189.067          | -28.24% | improved |
| stock-stream-4cpu  | 250.388         | 172.311          | -31.18% | improved |
| sparse-stream-1cpu | 27.236          | 23.442           | -13.93% | improved |

| Case               | Round | Paired median | Median interval    | Coverage  |
|--------------------|-------|---------------|--------------------|-----------|
| stock-stream-1cpu  | 1     | -29.14%       | [-35.35%, -26.35%] | 97.85156% |
| stock-stream-1cpu  | 2     | -28.82%       | [-31.59%, -25.66%] | 97.85156% |
| stock-stream-4cpu  | 1     | -32.22%       | [-34.01%, -29.00%] | 97.85156% |
| stock-stream-4cpu  | 2     | -31.38%       | [-34.14%, -27.10%] | 97.85156% |
| sparse-stream-1cpu | 1     | -13.49%       | [-20.95%, -7.31%]  | 97.85156% |
| sparse-stream-1cpu | 2     | -16.71%       | [-19.13%, -9.26%]  | 97.85156% |

| Case               | Side      | P95 ms  | CPU P50 ms | Cores P50 | Peak RSS P50 MiB |
|--------------------|-----------|---------|------------|-----------|------------------|
| stock-stream-1cpu  | baseline  | 304.830 | 263.453    | 1.000     | 231.06           |
| stock-stream-1cpu  | candidate | 208.604 | 186.243    | 1.000     | 227.19           |
| stock-stream-4cpu  | baseline  | 270.820 | 297.202    | 1.189     | 243.88           |
| stock-stream-4cpu  | candidate | 186.937 | 219.231    | 1.280     | 239.95           |
| sparse-stream-1cpu | baseline  | 29.024  | 27.236     | 1.000     | 70.52            |
| sparse-stream-1cpu | candidate | 25.435  | 23.443     | 1.000     | 66.50            |

Three fixed cases use two rounds of ten alternating pairs and one
600-second budget including preflight, fixtures, warmups, independent
full-snapshot oracles, capture/decoding, analysis, publication and
cleanup. Builds are separate; no cases, samples or retries are added.

A dense1/dense4 target must have the exact maintained improved verdict,
with no regression on any case, after complete evidence validation.
Inconclusive is uncertainty, not equivalence or a confirmed speedup.

The baseline is freshly built public222f4e2d. The older af400 binary
is incompatible with current DataFusion/input-registration source.
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
Every stream retains21 complete snapshots, independently checked after
the native timer. Whole-process peak RSS includes inputs and retained
outputs; it is not operator-only allocation or an engine memory cap.
Common657 requires non-null inputs. UInt64, nullable/generic aggregates
and checkpoint latency are unmeasured; focused allocation/size/budget
and correctness tests separately cover their changed representation.

Accepted profiles: 2.
Both serial profiles require five verified processing windows after
excluding the first of six jobs; all per-thread timestamps must align.
Self/inclusive counts and full stacks cover the allocation/drop path.
Inclusive CPU shares overlap and are not phase wall-time measurements.
Missing/inlined frames do not imply zero cost; unresolved leaves remain.

[Summary](summary.json) and [all formal rows](samples.csv) retain exact
verdicts, every interval, CPU/RSS and source/build/fixture identities.
[Complete raw evidence](complete-evidence.json.gz) includes every warmup,
profile and formal observation, snapshot/schema digest, command, resource
receipt and clock timestamp text. Large perf files stay at recorded local
paths with hashes. [Profiles](profile-summary.json),
[targets](profile-targets.json) and full stacks preserve attribution.
[Manifest](evidence-manifest.json) seals actual budget debit and status.

Stopped reason: None
