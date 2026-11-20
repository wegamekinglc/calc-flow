# P6 batched partition leases evidence

## Outcome

Batching old-group KeyStore leases did not establish a speedup and was withdrawn. The P6 four-core 3x gate remains open.

| Case | Baseline P50 | Candidate P50 | Time change | Verdict |
|------|-------------:|--------------:|------------:|---------|
| stock-stream-1cpu | 204.639 ms | 204.535 ms | −0.05% | no confirmed regression |
| stock-stream-8cpu | 136.959 ms | 148.248 ms | +8.24% | inconclusive |
| sparse-stream-1cpu | 20.737 ms | 21.525 ms | +3.80% | inconclusive |
| stock-batch-8cpu | 55.882 ms | 57.340 ms | +2.61% | inconclusive |

The baseline is `aefd17dc006270457c2e8ccc585111d6f4f1b5ea`; the candidate is `39ef601adacce6d3cf170f1c112b3174a0b069b4`. Each case completed two rounds of ten AB/BA pairs with complete snapshot and lifecycle oracles. Acquisition took 120.230 seconds; release build took 621.022 seconds separately. The candidate binary SHA-256 is `e5feb98695e415a017397ea988088280e054b6adb91a70c95e3fee4bfa6747be`.

The eight-CPU candidate was slower than both the serial baseline and the preceding dense-marks candidate (136.283 ms in its own comparison). Pre-touching every old group also added partition scans without addressing the dominant routing work. The change was reverted in `c75fe501`; the reverted source is identical to `d80a334a` for `keyed_sum.rs`, so no redundant performance rerun was performed.

This negative result rejects the hypothesis that simply reducing lock acquisitions is sufficient for the P6 target. Subsequent work must measure the serial global key-routing cost and the actual partition update cost rather than adding more lock-oriented changes by assumption.

## Evidence

- `plan.json`: cases, budgets, source seals and build receipts.
- `environment.json`: host, affinity and harness identity.
- `samples.csv`: all 160 formal observations.
- `summary.json`: paired statistics and resource summaries.

No retries or additional cases were used. The raw acquisition remains local under `target/p6-keyed-measurement-batched-leases-39ef601a/`; compact evidence is committed here.
