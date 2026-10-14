# Issue 406: Join batch admission

Status: clarified proposal, before implementation. Base: `bb142320062995aeb318ad406c901288537fd766`.

This increment addresses phase 1.1. It supplies the batch ownership and selection
boundary needed by phase 1.2, but does not claim that persistent dense key IDs,
chunked retained state, chunk journals, or any of the eight performance targets
are complete. The reviewed delivery branch remains frozen while this work lives
on `feature/issue406-join-batch-admission`.

## Problem and representation

`AdmissionMasks` already classifies 64 positions at a time. Admission nevertheless
allocates `Vec<AdmittedRow>` and a second source-position vector, cloning a payload
owner for each admitted row. Copy admission additionally stages `SelectedRow`
records. Probe, SQL scratch ownership, projection, and materialization all depend
on this row container.

Replace that container throughout the incoming-batch pipeline with an owned
`AdmittedBatch`. It owns an ordered collection of record segments. Each segment
owns one payload parent, normalized admitted event times, retained-position bits,
and an original physical-position selection. Dense selections use a range;
non-dense selections use ascending indices or a bitmap with rank information.
The segment records its physical row-ID base and source offset. An admitted row's
durable ID is the base plus its original physical position, including gaps from
dropped rows. Projection may replace a segment's payload with a densely copied
parent without replacing original positions or durable IDs.

Keep three coordinates distinct: the dense admitted probe ordinal used by matched
pairs, the original physical position used for IDs and full-fee cache selection,
and the current payload position used after a selected copy. Rebinding the latter
must not change either of the former.

Borrowed row access must not allocate or clone a payload owner. An owning
`RowPayload` is created only at the existing retained-state boundary. No
production `Vec<AdmittedRow>`, equivalent record-per-row adapter, or per-row
`RecordBatch::slice` is permitted in incoming admission, capture, declined-probe
reuse, SQL probe scratch construction, projection, or output routing. Existing
`StoredRow`, `MatchedPair`, and checkpoint operations remain explicitly separate
phase 1.3–1.5 work. They are not counted as eliminated by this increment.

The copy-selection API must accept this batch metadata without recreating a
`Vec<SelectedRow>` for admission or projected admission. Existing funded payload
inventory and retained-row state may remain until their own phase. A borrowed
selection accessor may produce a temporary scalar value for an individual
copier operation; it must not allocate a scalar container or own an extra parent.

## Semantic requirements

- Keep physical record and row order. Every physical row consumes an ID,
  including null-time, null-key, and late rows. Retention depends on the opposite
  ingress frontier; lateness depends on the row's own ingress frontier.
- Preserve first-failure ordering. Reserve/check the physical row ID before
  classifying that row; then null time, time conversion, null key, lateness, and
  admission. An aggregate overflow check must not hide an earlier conversion or
  metric failure. For mask blocks with exceptional conditions, traverse only the
  relevant ordered positions while preserving the original error precedence;
  scalar exception handling must not recreate a row container.
- Preserve exact timestamp normalization for all four units, inclusive bounds,
  drop counters, maximum lateness, and one late-affected-batch increment.
- Preserve logical input/output schemas, consumer projection mappings, project
  fingerprint, full V1 admission fees and state/match limit decisions. Narrow
  payloads still use carried full fees. No checkpoint layout or writer-identity
  change belongs to this increment; authentic layout 1/2 migration and strict
  layout 3 restore remain valid.
- Keep state commit after complete output emission. Retain existing credit,
  cancellation, abandonment, retry, and worker-release ownership. All new owned
  metadata allocations must be covered by a checked workspace reservation made
  before allocation. Account vector capacity and temporary conversion overlap,
  not only logical length; transfer the reservation with owned worker inputs.
- Full input arrays remain read-only. Replacement must release obsolete parents
  at the same lifecycle boundary as the reviewed projection implementation.
  Caller-held input buffers may naturally remain alive. Sparse selected copies
  must not acquire a hidden reference to unselected payload or dictionary values.

## Workspace and parent ownership

Use fixed inline metadata for at most 64 physical rows and four record segments;
those arrays do not create a heap allocation. Larger metadata must reserve its
checked capacity before heap allocation. Existing optional-copy and optional-owned
execution refusal still declines to a vectorized serial route. Mandatory metadata
refusal is a workspace error before output and state commit; it is not a state or
match limit error and does not change their counters or V1 fees. Do not promise
identical workspace-refusal thresholds for larger batches under artificial finite
pools. Ordinary lazy runtime initialization keeps the existing default/unbounded
pool policy. Never invent a finite default to fund this representation.

Workspace preflight is a separate stage after port validation and before admission
classification. If it refuses, that workspace error wins; row-level semantic-error
ordering applies once this preflight succeeds. Avoid adding a late reserve that
reorders two semantic failures within the already admitted batch.

`TableBatch` already owns its record array through `Arc<[RecordBatch]>`; cloning
that table owner creates no new heap metadata. It may serve as a temporary owner
during the outer admission/serial call. It must not escape into retained state or
an owned background input while pinning dropped records or already replaced
parents. Before owned transfer, detach only needed borrowed record parents under
checked credit, one parent per selected record, including column-vector, Arc and
control capacity. On detachment refusal decline the transfer to serial execution.
Do not retain the whole mixed table in a background owner as a fallback. Release
the temporary table immediately when all segments have replacement owners.
The existing `StoredRow` boundary must receive an independently owned/funded or
existing legacy record parent, never a temporary whole-table owner.

Preserve the existing one-row zero-headroom SQL acceptance controls and add
mandatory metadata-refusal controls. Include inline-owner size in owned worker
control funding. Test mixed multi-record dropped/copied/borrowed inputs with
gates at worker capture and result release; after cancellation or abandonment,
unselected parents must not stay pinned in the background owner. Distinguish
caller-held table buffers from unintended worker-held references.

## Consumers and vectorized fallback

Migrate native key sizing/framing, serial and owned probe, retained-row creation,
full-fee cache selection, SQL owners/retry, projection, and materialization to the
same batch container. Parent grouping is explicit segment grouping, rather than
pointer comparisons across row descriptors. Random access returns a borrowed
payload position; sequential access visits segments in declaration order.

Generic SQL probe tables gather each key column per segment using Arrow selection
and combine selected arrays. They must not build a vector of one-row views.
Unsupported native key types, nullable/mixed selections, nested payloads, multiple
records, zero admitted rows, and workspace refusal must not silently recreate the
old admission container. Any remaining per-row algorithm must have an explicit
complexity counter and focused control; a scalar exception check is not permission
to materialize per-row owner records.

## Acceptance and verification

Before implementation, record a focused failure demonstrating allocated incoming
row descriptors or per-row parent clones for a multi-record dense batch. Repeat
the structural assertion on sparse null/late input and a generic SQL key path.
Then verify output, schema, ordering, row IDs, full fees and metrics against the
existing public/native behavior controls, including empty input, all timestamp
units, mixed retention, and an incoming right side.

Add focused first-error controls near ID/metric overflow and time conversion;
state/match limit refusal emits nothing and commits no admitted state. Exercise
owned probe refusal, cancellation and dropped futures, projected sparse copy
replacement, escaped output ownership, and pool cleanup. Reuse existing layout
1/2/3 recovery controls, with a post-restore batch containing dropped ID gaps.
Test-only fixtures may construct reference rows; they cannot select a production
adapter that recreates row ownership.

Run only affected tests, core lib/test Clippy, formatting, whitespace and generated
schema no-drift. Review changed/new functions for CCN at most 8. Final specialist
review is required. Performance measurements need a separate preregistration and
sealed paired runs; structural counters alone establish no timing, allocation/RSS
verdict, or issue acceptance. No stopped timing acquisition is resumed here.
