# Issue 406: route parallel ASOF probe preparation by chunk keys

This increment builds on the reviewed serial routing and checkpoint-evidence
work at `edb73649`. The full eight-target objective remains active. Prior
OFF/32 timing remains inconclusive, and both stopped ON acquisitions retain
their original qualification status.

## Evidence and scope

The exact ordinary `6739fa2b` release, native SHA-256 `a131c569...340011`,
is profiled for ASOF 1M rows/input, 64k batches, true OFF and 32 CPUs. One
warmup and five fixed processing windows finish within 10.614 / 180 seconds,
including symbolization, oracles, cleanup and seals. All six invocations pass
the oracle, lifecycle, cleanup and zero-checkpoint-file checks. All five windows
contain CPU samples: 659/697/669/669/660, totaling 3,354. Sixteen lost lifetime
events have unknown window membership; WSL2 host isolation remains unverified.

`match_output_prefix` has 33.81% inclusive / 18.78% self samples;
`ChunkData::output_view` has 5.28% self samples. These shares are diagnostic,
overlap, and do not establish a speedup. Parallel capture currently builds a key
view and probes its encoded-key route map for every selected row, despite the
chunk already holding dictionary IDs. Serial output matching already has a
bounded chunk-key path. Reuse that eligibility boundary for parallel capture.

## Requirements

1. Resolve each eligible chunk dictionary key once per selected output run, then
   prepare worker rows through integer key IDs. Use at most 64 stack slots and
   retain the existing four-selected-rows-per-slot threshold. Cache absent
   buckets locally; preserve generic matching for legacy, larger dictionaries
   and short or interleaved runs.
2. Preserve selected positions, time, shard assignment, right-bucket identity,
   tolerance, sequence dominance and canonical output order. No new public API,
   retained state, manifest format or caller mutation.
3. Preserve existing input/output reservations, global route-map capacity and
   worker count. Referenced buckets stay funded until owned work exits. Keep
   cancellation checks and the global per-1,024-row yield cadence, including at
   run boundaries. Budget failures and rejected collectors retain existing
   fallback/refund/continuation behavior.
4. Record the focused hash-counter regression failing before implementation.
   Compare captured results with binary-search matches, including missing keys,
   partial consumption, 63/64/65 slots, short fallback and legacy state. Reuse
   existing registered tests to respect the documented libtest array boundary.
   Run the directly affected probe/lifecycle tests, necessary format/lint and
   complexity checks, then final specialist review.
5. Keep profiling separate from paired timing. A later comparison freezes exact
   ordinary parent/candidate releases, fixed case counts, two rounds of ten
   adjacent AB/BA pairs and a 600-second inclusive budget before acquisition.
   Select only directly affected parallel OFF and necessary single-CPU control
   paths; preserve all inconclusive/stopped results and report unmet targets.
   Never infer complete eight-target acceptance from this subset.
