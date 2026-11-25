# Contiguous ASOF admission in the public Rust streaming job

Two separately paired comparisons use 10M UTF-8-keyed rows per input, ten 1M chunks and one pinned CPU. Each worker checks every output sequence and price, both inputs at EOF, matched and emitted counts, zero retained state and zero budget failures. The production clock runs from ready sources through Arrow output assembly. The separate installed Python job uses four workers.

## release 7012b5b409 compared with 25e9595379

- Baseline p50 954.640 ms and p95 979.222 ms.
- Candidate p50 1072.434 ms and p95 1133.545 ms.

Maintained verdict `regression`. Both rounds and intervals are recorded below.

```json
[
  {
    "median": 14.056090957738066,
    "low": 10.948141229397779,
    "high": 14.81254417904232,
    "coverage": 0.978515625
  },
  {
    "median": 12.16592707830273,
    "low": 8.10434213168012,
    "high": 15.197523623359421,
    "coverage": 0.978515625
  }
]
```

Formal acquisition took 58.94 seconds. Both independent full-size qualifications passed.

## previous 84b731859d compared with 25e9595379

- Baseline p50 1178.410 ms and p95 1226.540 ms.
- Candidate p50 1063.282 ms and p95 1086.052 ms.

Maintained verdict `improved`. Both rounds and intervals are recorded below.

```json
[
  {
    "median": -9.490099481164343,
    "low": -11.586776246180452,
    "high": -7.842492680312619,
    "coverage": 0.978515625
  },
  {
    "median": -9.70499737016825,
    "low": -12.462049418280996,
    "high": -7.873273914057421,
    "coverage": 0.978515625
  }
]
```

Formal acquisition took 64.99 seconds. Both independent full-size qualifications passed.

[Sources, manifests, sealed build receipts, per-worker oracles and resource logs](asof-compaction-native-10m.tar.gz) (SHA256 `e3da7e9fa953fb060e56843248fdf4379459c040c632939cd0a70ea5846d06f6`). Extract the archive into a fresh directory, set PYTHONPATH to the benchmark repository, and run `python verify.py release/obs-job-* previous/obs-job-*`.
