# Managed ASOF admission at ten million rows per side

The fixed four-thread case matched 64 keys at zero tolerance, with ten batches of one million rows per side. The ordinary reference and the diagnostic used the same public entry point, machine, reserved physical CPU set, interpreter, fixture and source-to-Arrow clock. Diagnostic wheels added admission counters to 11.14 (7012b5b4) and the left-retirement revision (dce77f04).

Stopped with 4 accepted warmups and 33 accepted formal measurements. The raw request logs and resource data preserve the stop condition. The fixed comparison is inconclusive.

## Incomplete diagnostic observations

The first round supplied 10 complete pairs and the second supplied 6. The arm medians below cover 17 release and 16 candidate executions accepted before the stop. These are exploratory stage observations.

| Elapsed            | 11.14 P50 ms | Candidate P50 ms |
|--------------------|--------------|------------------|
| Total admission    | 444.55       | 1246.34          |
| Ingress validation | 51.83        | 8.63             |

| Candidate phase    | Combined P50 ms | Combined P95 ms |
|--------------------|-----------------|-----------------|
| install_indexes    | 402.19          | 450.32          |
| route_keys         | 298.57          | 324.16          |
| compact_payload    | 193.37          | 203.49          |
| validate_locators  | 105.89          | 121.34          |
| project_charge     | 104.80          | 116.68          |
| encode_sequences   | 68.77           | 72.44           |
| budget_preflight   | 31.62           | 32.77           |
| build_run          | 24.12           | 27.04           |
| late_mask          | 11.90           | 12.37           |
| ingress_validation | 8.63            | 10.50           |
| residual           | 0.08            | 0.10            |
| envelope_record    | 0.00            | 0.00            |

Release's composite preparation and installation measured 261.66 and 130.78 ms P50. They cover different code boundaries from the candidate's phases. Compared with the complete ordinary run, the incomplete release and candidate end-to-end P50 changes were +0.93% and +4.69%. The published results JSON retains per-arm P95, per-side breakdowns and the descriptive differences from complete pairs in each round.

The controller took 91.60 seconds, including interpreter and worker startup, fixture creation, warmups, correctness verification and cleanup. The archive preserves native callback sequences and elapsed times for each request, per-side row counts, terminal gauges, exact source patches and build identities. The trace timestamps bracket synchronous admission inside the Rust operator. Source transport, watermark finalization, Arrow delivery and post-clock tracing have different boundaries. [Summary](result.json) · [Original evidence](raw-evidence.tar.gz) · [Receipt](receipt.json)
