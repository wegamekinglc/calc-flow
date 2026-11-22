# Interval join projection control

This control times Arrow materialization of the three output columns used
by the million-row interval benchmark. It uses the same two identities and
product as the streaming graph. Prepare the exact match positions and
contiguous input columns before timing. Sort pairs by right and left physical
position and group by 4,096 right input rows. One such group produces at most
45,056 pairs in this fixture.

The clock covers gathering both sequences, gathering and multiplying prices,
and assembling the output table. The control precomputes all matches and
does not run the stream or Rust join. Its time cannot be subtracted from the
runner clock to infer Rust stage costs.

Run in an environment with the installed release wheel and benchmark
dependencies. This diagnostic uses Python 3.11 or newer. Keep all samples on
one logical CPU. Run two excluded warmups and ten observations. Every sample
must pass the maintained [full pair oracle](../interval_acceptance.py).
At one million rows per side the expected cardinality is 10,998,080 rows.

```bash
TOKIO_WORKER_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 python -m benchmarks.interval_materialization --output target/interval-projection.json
```

For the initial main baseline (`ec402c5962d9137ad2321d5dea0146a1e2b5efe5`),
this is one of five preselected workloads sharing a ten-minute collection
budget. Report the median and inclusive interpolated P95. Builds run separately.
