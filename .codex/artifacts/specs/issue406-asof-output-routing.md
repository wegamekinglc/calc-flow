# Issue 406: route ASOF output matching by chunk-local keys

The complete eight-target objective remains active. PR #420 is independently
awaiting required CI and its complete benchmark; this work starts in a separate
branch and preserves that measured product and evidence.

## Evidence and scope

The ordinary-release `b77cd8bf` ASOF OFF profile uses 1M rows per input, 64k
batches and one / 32 CPUs. Fixed acquisition is one warmup and five processing
windows per case, within 180 seconds; actual 11.765 seconds includes sealing.
All 12 oracles and lifecycles pass, with zero checkpoint files and no surviving
process group. Lost lifetime events are 473 / 2, of unknown window membership;
WSL2 host isolation is unverified. These are diagnostics, not speedup evidence.

Admission accounts for 51.69% / 40.76% inclusive samples; output matching for
27.77% / 32.17%. Shares overlap. Source inspection shows monotonic matching
hashes an encoded key for every row, although left chunks already store local
dictionary IDs. Parallel matching uses at most three workers for a 64k prefix
and retains serial capture and output-plan construction. Do not change worker
thresholds based only on these samples.

## Requirements

1. Reuse chunk-local key IDs for small dictionaries, resolving right buckets
   and monotonic cursors once per key per output run. Preserve a generic path
   for legacy rows, larger dictionaries and short runs. Initialize a dictionary
   route only when the selected run contains at least four rows per slot; this
   avoids scanning every key for each row of interleaved chunks.
2. Preserve exact match selection, tolerance, sequence dominance, payload
   positions and canonical left order, including overlapping chunks, partial
   consumption, retired dictionary slots and missing right keys.
3. Keep stack scratch bounded. Add no retained heap state or unsafe code;
   preserve existing workspace funding, refusals, checkpoint journals,
   cancellation/yield boundaries and output ownership. Caller buffers remain
   read-only and oversized parent buffers retain their current accounting.
4. First record a focused failing complexity check using the existing encoded
   key hash counter. Check real output plans and rowwise prefix ownership, then
   run affected ASOF differential/lifecycle tests and scoped lint/format checks.
5. Do not claim a timing gain from the hash counter or profiles. Any release
   comparison must use exact main/candidate identities, fixed paired counts and
   the inclusive acquisition budget. Preserve OFF/LOW/ON distinctions and all
   eight original targets; ASOF OFF remains ≤100 / 30 ms. Green required CI and
   final specialist review remain merge gates.
