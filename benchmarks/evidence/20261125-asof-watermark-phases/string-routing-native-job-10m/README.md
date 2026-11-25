# ASOF string key routing in the public Rust streaming job

Two separately paired comparisons use 10M UTF-8-keyed rows per input, ten 1M chunks and one pinned CPU. Each worker checks every output sequence and price, both inputs at EOF, matched and emitted counts, zero retained state and zero budget failures. The production clock runs from ready sources through Arrow output assembly. The separate installed Python job uses four workers.

## release 7012b5b409 compared with 84b731859d

- Baseline p50 1001.308 ms and p95 1032.902 ms.
- Candidate p50 1224.933 ms and p95 1244.275 ms.

Maintained verdict `regression`. Both rounds and intervals are recorded below.

```json
[
  {
    "median": 22.118362491243495,
    "low": 18.681447052793267,
    "high": 24.333339086451943,
    "coverage": 0.978515625
  },
  {
    "median": 23.533713363592756,
    "low": 20.783391258394303,
    "high": 25.70622741919868,
    "coverage": 0.978515625
  }
]
```

Formal acquisition took 62.58 seconds. Both independent full-size qualifications passed.

## ordered f878077127 compared with 84b731859d

- Baseline p50 1485.210 ms and p95 1544.393 ms.
- Candidate p50 1208.127 ms and p95 1274.491 ms.

Maintained verdict `improved`. Both rounds and intervals are recorded below.

```json
[
  {
    "median": -18.90527705250466,
    "low": -20.39220342507262,
    "high": -16.25163165834168,
    "coverage": 0.978515625
  },
  {
    "median": -18.346593141697745,
    "low": -19.51918703161908,
    "high": -17.025683977471385,
    "coverage": 0.978515625
  }
]
```

Formal acquisition took 73.83 seconds. Both independent full-size qualifications passed.

[Sources, manifests, sealed build receipts, per-worker oracles and resource logs](asof-routing-native-10m.tar.gz) (SHA256 `aa0b50d74cd93376253301c79fe0046cf83b01ea1faa81961357eebbb7d438f2`). Extract the archive into a fresh directory, set PYTHONPATH to the benchmark repository, and run `python verify.py release/obs-job-* ordered/obs-job-*`.
