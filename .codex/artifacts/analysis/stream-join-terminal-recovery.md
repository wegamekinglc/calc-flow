# Streaming Join terminal managed recovery

Selected terminal manifests now restore Join nodes after the existing SQL and
ASOF validation and before sink recovery. The same configured runtime, managed
Local reader, owned load service and existing managed V1 restore consumer apply.
Supported profiles retain their funded native work; unsupported histories and
optional funding refusals retain the original decoder and diagnostic order.
No Work layout, fee, public API, checkpoint encoding or nonterminal path changes.

Native ended state, empty retained collections and zero retained row/byte
metrics are checked before manifest ingress progress can overlay public status.
Join statuses remain local until every Join validates. Rejection discards the
exclusively owned, unstarted plan; it does not claim the private operator state
was unchanged after authoritative restore. Existing driver settlement drains
load and gather ownership before reporting completion. Sources remain unopened
on this terminal path, and no synthetic EOF, eviction or counter rewrite occurs.

Launch cancellation is observed across the complete Terminal Join stage. A
borrowed wrapper forwards StartObserver/shutdown cancellation to the existing
job token, which both paid Local loading and native gather already observe,
and awaits the same recovery future. Fresh launch/job checks reject completion
before publishing Join statuses or recovering sinks. No watcher task, token,
Work field or fee is added. Awaiting the helper is not a refund claim; existing
driver settlement still awaits actual LoadOwner and gather drain. A cancelled
unstarted plan can be discarded after private restore has already run.

A new control drops the real StartObserver during paid Local loading and during
the first accepted native reader. Its original run failed because launch
cancellation was not forwarded to the job token. The unchanged control now
passes: Local cancellation reaches no reader; native cancellation reaches one
paid reader. Both stages observe pending drain before release, then actual
shutdown and zero pool accounting, with no Join status or sink recovery.
These are two positions in one focused test, not two restart test cases.

The single Managed consumer keeps one literal matched row, captures the dirty
cut, then releases real WM/EOF. Its actual terminal checkpoint is epoch 2 with
frontier 110, both ingress watermarks 120 and four delta segments: two historical
upserts followed by two removes. Same-root fresh-job recovery calls no source
open, poll or close, and preserves sink open/recover/close order and the literal
Arrow output after shutdown. This is same-process durable recovery.

Baseline RED ran one test and failed at zero reader entries versus two real
historical rows. The semantic, cursor, lifecycle and natural-history assertions
preceded that failure; later funding and native-status assertions were not
executed. A zero-test cache hit and a compile-only qualification failure are
excluded from behavior results. This consumer has no direct pool-zero assertion.
The unchanged consumer GREEN passed with two native reader entries sharing one
Work workspace and one metadata parse completed before sink recovery. Two
direct controls also passed: native/ingress/frontier coherence and diagnostic
priority, plus real attempt refusal, first-reader cancellation and drained
accounting. These are three unique focused checks; unchanged checks were not
repeated. Public status alone is not independent evidence of native ended state.
Observed reservation sizes and pointers are not allocation peaks or credit nonces.

The Windows AC14 recovery regressions exposed a real V1 writer/reader mismatch.
When all uncaptured upserts are evicted before the first checkpoint, the existing
writer produces metadata-only state with no segments. Restore now accepts that
state only after metadata validation and only when both sides' retained row and
byte gauges are zero, then performs the original complete installation checks.
Next row IDs, output sequence, cumulative metrics and native ended state retain
their original values. Missing segments with any nonzero retained gauge still
produce the original empty-inventory error; nonempty histories and V2 framing
are not relaxed. The original CI failures are the RED evidence. The new natural writer/restore
control and both unchanged AC14 recovery regressions each passed locally.
Together with the launch-cancellation control, these are four additional unique
checks. The original three focused checks were not repeated.

Final connected source review gates publication. Required CI, coverage, Codacy
and review resolution gate merge for every published head. This change makes
no performance claim and does not complete the remaining generic/V2 migration
work.
