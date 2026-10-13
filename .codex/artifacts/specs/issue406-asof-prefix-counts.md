# Issue 406: count ASOF prefix dictionary keys without row sorting

The complete eight-target performance objective remains active. This increment
addresses ASOF output matching after the independently reviewed admission work
in PR #413; it does not alter that PR or certify the remaining target inventory.
Its initial source base was exact main `8f890f28`. PR #413 merged as `5fef1a31`;
this increment is rebased onto that exact main before a new comparison. Main
also includes independent Runtime API, dependency-security and Studio-split
changes. Build both comparison arms from their exact current sources.

## Evidence and selection

The historical ordinary-release profile of then-main `8f890f28` in
[admission analysis](https://github.com/wegamekinglc/calc-flow/blob/0401d4b7849bcf8af203b77ca42a9e8e51795bab/.codex/artifacts/analysis/issue406-asof-admission-bulk.md) identifies
output-prefix matching at 29.34% / 33.35% inclusive at one / 32 CPUs. The
single-CPU profile attributes 5.16% self samples to sorting. These samples are
diagnostic and overlap other regions; they do not establish a timing gain.
The one-CPU profiler lost 243 lifetime events of unknown window membership.

Source inspection confirms `ChunkRun::visit_keys` copies every selected row's
already interned key ID and sorts the copy, solely to aggregate counts per
dictionary entry. The resulting key visits are in increasing dictionary ID.
Replace this repeated sorting with counting when its scratch is bounded by
the existing temporary allocation. Preserve the sort fallback for large sparse
dictionaries and small selected prefixes.

## Requirements

1. Count keys of the selected run range, including a nonzero start, partial
   consumption, independent dictionaries, interleaved runs and empty cuts.
   Preserve exact key/batch/owner counts, canonical visit order, projected
   inventory, checkpoint journals and state removal.
2. Use fixed stack scratch for dictionaries of at most 64 slots. This is a
   scratch selection size, not a supported-input or runtime key-count limit.
3. For larger dictionaries, use a dense histogram only when its byte length
   is no larger than the old selected-row `u32` copy. Otherwise preserve the
   existing sort path. Skip zero counts before resolving possibly retired slots.
4. Preserve prefix funding and explicit refusal decisions. Add no unsafe code,
   heap storage in retained state, fallible commit step or await; never mutate
   caller columns or the stored key dictionary. No default memory cap is added.
5. Record a focused failing allocation regression before implementation. Extend
   the existing test registration: prepopulate prefix maps, then measure only
   repeated run counting. Check zero heap scratch for small dictionaries and
   bounded histogram/sort storage for larger domains. Keep differential rowwise
   owner, journal, partial-cut and workspace controls.

## Verification and performance evidence

Run the focused RED/GREEN test, serial ASOF tests, scoped core/test Clippy and
format/contracts/whitespace checks. A new performance comparison must wait for
a suitable environment and use exact current main and candidate ordinary
releases. Freeze at most four affected ASOF cases, counts and the inclusive
ten-minute budget before execution; preserve OFF/LOW/ON distinctions and the
maintained paired regression rule. Interrupted evidence remains incomplete.
Keep every original target visible; ASOF's OFF targets remain 100 / 30 ms.
Final specialist review and green required CI remain merge gates.
