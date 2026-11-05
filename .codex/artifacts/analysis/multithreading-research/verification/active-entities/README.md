# Trusted active-entity metadata evidence (W-SQL-3)

> Status: opt-in API evidence for PR 3 of the multithreading roadmap. Baseline:
> `main` @ `e90ddc1d` plus this branch. Method: probe example only; no product
> default changed. Two independent 4-CPU rounds plus one 16-CPU round include
> fixed-pN controls that separate parallel speedup from the API switch.

## Hypothesis

`auto` needs trusted `calc_flow.datafusion.active_entities` metadata to select
more than one partition for single-input batch graphs. Exposing the metadata as
a first-class `Batch`/`BatchMetadata` API lets callers enable entity-aware
automatic parallelism without hand-writing JSON attributes; without the API the
same graph stays at p1 and loses the measured multi-core gain.

## Method

- Probe: `crates/calc-flow/examples/auto_rows_probe.rs`, now including an
  `Auto + entities=64` configuration next to `Auto` without metadata,
  `auto_rows`, fixed p1, and fixed pN controls.
- Rows: 2,000,000 in 244 Arrow batches, 64 keys; samples: 7 per configuration.
- Runtime: WSL2 release builds; `taskset -c 0-3` for four CPUs and
  `taskset -c 0-15` for sixteen; Tokio workers follow host affinity.
- Every configuration's full output was compared with fixed p1 before timing.

## Results (medians of 7 samples; milliseconds)

Four logical CPUs, four partitions (two rounds):

| Shape      | fixed p1 | auto (no metadata) | auto + entities | fixed p4 control | effective |
|------------|----------|--------------------|-----------------|------------------|-----------|
| projection | 3.72 / 3.80 | 4.25 / 4.37 | 5.59 / 6.82 | 6.58 / 6.87 | 1 → 4 |
| group_by   | 8.60 / 8.97 | 8.99 / 9.19 | **3.40 / 3.65** | 3.32 / 3.62 | 1 → 4 |
| join       | 18.72 / 18.81 | 7.37 / 11.68 | 7.39 / 9.40 | 6.47 / 7.48 | 1 → 4 |

Sixteen logical CPUs, sixteen partitions (one round):

| Shape      | fixed p1 | auto (no metadata) | auto + entities | fixed p16 control |
|------------|----------|--------------------|-----------------|-------------------|
| projection | 3.91     | 4.38               | 3.22            | 5.28 |
| group_by   | 9.71     | 9.01               | **2.73**        | 2.92 |
| join       | 18.75    | 4.67               | 4.70            | 5.15 |

## Interpretation

- Adding trusted metadata turns `auto` from p1 into p4/p16 for single-input
  graphs: single-input `GROUP BY` improves **2.5–2.7× on four CPUs** and
  **≈3.3× on sixteen CPUs**, matching the fixed-pN control in both cases.
- Multi-input join already uses the whole-run external-row policy under `auto`;
  the entity field is documented as ignored when multiple inputs exist and the
  measurements confirm the same pN performance.
- Trivial row-local projections do not benefit and can regress; the docs keep
  `fixed`/p1 as the default and state that this metadata is for entity-aware
  workloads, not a universal speedup.

## Merge decision

**Merge.** The API is additive, the default path and formats are unchanged, the
metadata value is validated by both the Python binding and the planner, and the
measurement demonstrates a real 2.5–3.3× benefit for the single-input grouped
workloads the metadata is designed for, with fixed-pN controls. The no-benefit
projection caveat is documented.
