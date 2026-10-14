# Issue 406: Join persistent key dictionary implementation

Status: implementation in progress; no phase or performance acceptance verdict.
Isolated base: `798a97ee44b701b00bb89196a6483b01c62e16bf`.

The implementation follows the approved
[specification](../specs/issue406-join-key-dictionary.md) and
[critic](../critiques/issue406-join-key-dictionary.md). All FR1–15 and AC1–13
remain obligations. The prior batch and codec sources remain frozen and reviewed.

## Initial focused RED

Evidence directory: repository `target/issue406-join-key-dictionary-20261015/`.
`persistent-keys-red.log` records two genuine failures through `process_data`:

- Known keys crossing batches/sides encoded four canonical owners against two.
- Equal retained values across both sides did not share a canonical Arc owner.

Both controls first passed five-row sorted output, physical IDs and full V1
row-fee assertions. Fee oracles were computed before the work-counter reset;
the counter covers complete production admission, matching and retention.
Compilation took 7 minutes 59 seconds; execution took 0.01 seconds. No production
source changed before these failures; all owned handles were settled afterward.

## Internal design boundary

One crate-private identity component will own namespace, opaque address and
immutable entry leases. Join retains its V1 codec and time-qualified runs; ASOF
and its wire/rank semantics remain unchanged. A copied history address is never
trusted for current lookup without namespace, live-entry and canonical-owner
validation. History bytes hold their own credit without leasing a live slot.

Prepared births and retained-reference changes remain private until successful
output completion. Native indexes will map identities to their existing runs,
with no independent canonical-content dictionaries. Canonical allocations have
per-owner funding separate from directory/handle capacity and batch scratch.

The genuine optional-grant refusal route preserves batched SQL and existing
legacy canonical allocation sites; its exact final helper/counter and maintained
workload inventory are pending. No new unfunded map/lease owner is permitted.

## Outstanding evidence

Typed column binding, all 17 tags, leases/recycling, refusal/retention harness,
publication failure, long-value/collision cooperation, restore distinctions,
local checks, complexity seal and final specialist review remain pending.
Phase 1.1 offset-derived fees, retained chunks, pairs, journals and all eight
original performance targets remain unfinished. No performance acquisition,
native package build, full matrix or remote mutation is authorized here.
