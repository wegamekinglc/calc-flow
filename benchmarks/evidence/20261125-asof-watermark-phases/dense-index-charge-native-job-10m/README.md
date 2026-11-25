# Exact dense ASOF index projection in the public Rust streaming job

The protocol required two sequential paired comparisons with 10M UTF-8-keyed rows per input, ten 1M chunks and one pinned CPU. Each worker checks every output sequence and price, both inputs at EOF, matched and emitted counts, zero retained state and zero budget failures. The production clock runs from ready sources through Arrow output assembly. The separate installed Python job uses four workers.

## release 7012b5b409 compared with ade0e864e7

The controller stopped acquisition after 18 observations (including warm-ups). The comparison has no paired verdict. Both native comparisons were launched concurrently by the session wrapper. Their sampled intervals overlapped; neither group qualifies under the sequential protocol.

Formal acquisition took 25.58 seconds. The full-size output qualifications passed on both builds.

## previous 25e9595379 compared with ade0e864e7

The controller stopped acquisition after 16 observations (including warm-ups). The comparison has no paired verdict. Both native comparisons were launched concurrently by the session wrapper. Their sampled intervals overlapped; neither group qualifies under the sequential protocol.

Formal acquisition took 24.49 seconds. The full-size output qualifications passed on both builds.

[Sources, manifests, sealed build receipts, per-worker oracles and resource logs](asof-index-charge-native-10m.tar.gz) (SHA256 `44e90548a66c25edc90fc1759cfb12d547881310a24895fd8d3a632ce2406af8`). Extract the archive into a fresh directory, set PYTHONPATH to the benchmark repository, and run `python verify.py release/obs-job-* previous/obs-job-*`.
