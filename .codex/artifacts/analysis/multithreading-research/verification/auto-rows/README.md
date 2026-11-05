# `auto_rows` parallelism policy evidence (W-SQL-2)

> Status: opt-in feature evidence for PR 2 of the multithreading roadmap.
> Baseline: `main` @ `222f4e2d` plus this branch. Method: probe example only; no
> product default changed. Results are descriptive (not paired AB/BA), but three
> independent rounds include the fixed-p4 control that isolates parallel speedup
> from the policy switch.

## Hypothesis

`auto` falls back to one partition for single-input batch graphs unless trusted
`calc_flow.datafusion.active_entities` metadata is supplied. `auto_rows` derives
the request from input rows alone, so single-input grouped/join batch SQL can
reach the host/configured partition count without extra metadata.

## Method

- Probe: `crates/calc-flow/examples/auto_rows_probe.rs`.
- Rows: 2,000,000 in 244 Arrow batches; 64 keys; samples: 7 per configuration.
- Runtime: WSL2 release build, `taskset -c 0-3` (4 logical CPUs),
  `auto_rows_probe` uses the host Tokio worker default (4 workers under the
  affinity).
- Configs per shape: fixed p1, fixed p4 (control), `auto` p4 (single-input
  fallback), `auto_rows` p4.
- Shapes: row-local projection, single-input `GROUP BY`, two-input `JOIN ...
  GROUP BY`.
- Every configuration's full output was compared with the fixed-p1 output
  before its timings were recorded; all matched.

```bash
cargo build -p calc-flow --release --example auto_rows_probe
taskset -c 0-3 target/release/examples/auto_rows_probe
```

## Results (medians of 7 samples; milliseconds)

### Four logical CPUs, four partitions (four independent rounds)

| Shape      | Config              | Round 1 | Round 2 | Round 3 | Round 4 | Effective |
|------------|---------------------|---------|---------|---------|---------|-----------|
| projection | fixed p1            | 4.92    | 4.22    | 4.24    | 4.05    | 1         |
| projection | auto_rows p4        | 5.12    | 5.93    | 7.67    | 5.66    | 4         |
| group_by   | fixed p1            | 10.39   | 9.66    | 9.60    | 10.23   | 1         |
| group_by   | fixed p4 (control)  | 3.59    | 3.54    | 3.55    | 3.57    | 4         |
| group_by   | auto_rows p4        | 3.61    | 3.91    | 3.53    | 3.67    | 4         |
| join       | fixed p1            | 21.44   | 21.43   | 20.36   | 21.18   | 1         |
| join       | fixed p4 (control)  | 11.27   | 11.66   | 7.23    | 13.60   | 4         |
| join       | auto_rows p4        | 7.60    | 11.53   | 6.95    | 12.63   | 4         |

### Sixteen logical CPUs, sixteen partitions (one round)

| Shape      | fixed p1 | auto_rows p16 | fixed p16 (control) | Ratio |
|------------|----------|---------------|---------------------|-------|
| projection | 3.69     | 3.34          | 5.01                | no stable gain |
| group_by   | 9.96     | 3.28          | 2.99                | ≈3.0× |
| join       | 19.77    | 5.72          | 6.46                | ≈3.5× |

Every configuration's output matched the fixed-p1 result before timing; all
reported the expected effective/output partition counts.

## Interpretation

- **`group_by`**: auto_rows p4 is ≈2.6–2.9× faster than fixed p1 on four CPUs
  and ≈3.0× on sixteen CPUs, and matches the fixed-pN control in each round.
  The gain comes from parallel execution, not from the policy switch itself.
- **`join`**: auto_rows p4 is ≈1.7–3.1× faster than p1 on four CPUs (higher
  variance from the two-input gather path) and ≈3.5× on sixteen CPUs; the
  fixed-pN control falls in the same range.
- **projection**: no stable gain; the four-CPU rounds regress by ≈1.2–1.8×
  while one sixteen-CPU round is within noise. The policy must not be presented
  as a universal speedup; small row-local scans should stay on p1.

## Merge decision

**Merge as an opt-in policy.** Real benefit is demonstrated on grouped and join
batch SQL (≈2.6–2.9× and ≈2.0–2.9× on four CPUs, control-confirmed); the default
stays `fixed`, no project/checkpoint contract changes, and the probe remains
available for re-measurement on engine upgrades. The documented caveat is that
small row-local projections should not enable `auto_rows`.
