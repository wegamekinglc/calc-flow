# Static Join post-quote status proof

Sink delivery can complete before the Join task commits and publishes its new
status. Reading the status once after delivery can therefore inspect only the
preloaded dimension snapshot and incorrectly accept zero left-state counters.

The measurement still stops after Arrow output collection. Outside that timed
interval, the runner now waits for `emitted_match_rows >= expected_rows` and
checks retained and evicted left rows in that same causally newer snapshot.
The wait retains the existing 600-second lifecycle bound and cleanup path.
Dimension progress continues to require zero left state before timing begins.

## Validation

- RED: both delayed retained/evicted snapshots were incorrectly accepted; the
  successful case inspected only two snapshots instead of the causal third.
- GREEN: the three causal snapshot tests and the existing delayed-dimension
  native functional test passed (4 tests).
- The fixture checks that both post-delivery status reads occur after the timer
  stops; its measured duration remains exactly 1 ms.
- Scoped Ruff check and format check passed. No native build or performance
  report revision was performed.
