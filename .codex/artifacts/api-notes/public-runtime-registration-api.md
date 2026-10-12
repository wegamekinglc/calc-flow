# Public Runtime registration API - API Note

Date: 2026-10-12. Source baseline: `main` at `8f890f28`, branch
`feature/public-runtime-registration-api`. Line numbers are baseline lines.
Status: surface approved. The critique at
`.codex/artifacts/critiques/public-runtime-registration-api.md` raised no
blocking finding; its wording findings W1 (narrow form), W2, W3, and its
caveats are adopted below. No open questions remain.

## Audiences

- Python users: embedders that register trusted mapping providers, or rebuild
  one runtime's registrations in another process. They need stable names, a
  stated record shape, and a clear statement that records are not data.
- Studio backend (`calc_flow_studio.run_manager`): the one known consumer. After
  the repository split it reaches this surface only through a released
  `calc-flow-python`.
- Rust users and REST clients: n/a. No Rust export, PyO3 method name, or
  `web-ui/openapi.json` change.

## Surface Today

Three underscore methods on `calc_flow.Runtime` in
`python/calc_flow/pipeline.py`: `_register_mapping_provider` (line 281),
`_registration_snapshot` (line 530), and `_capability_registration_snapshot`
(line 542).

- `_register_mapping_provider` is already named in published docs
  (`docs/api-reference.md:258-262`, `docs/python-api.md:743-744`) and has been
  since PR #29, across five tagged releases. It is in no example and no
  `CHANGELOG.md` entry. The two snapshot methods are undocumented.
- Callers outside the package: Studio (`run_manager.py:835`, `1289`, `1313`,
  `1348`) and `benchmarks/symbolic_support.py:322`. Inside it:
  `python/calc_flow/array.py:1258`, `1267`, `1310`, `1319`.
- These three are the only private `calc_flow` members Studio source touches.

## Proposed Surface

```python
class Runtime:
    def register_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        input_ports: Sequence[tuple[str, str]],
        output_ports: Sequence[tuple[str, str]],
        options_schema: ProviderOptionsSchema | None = None,
        accepts_context: bool = False,
    ) -> None:
        """Register a trusted provider mapping named input Ports to output Ports.

        Each port is a ``(port_name, batch_kind)`` pair; ``batch_kind`` is
        ``"table"`` or ``"array"``. The callback receives ``(inputs, options)``,
        or ``(inputs, options, context)`` when ``accepts_context`` is true, and
        returns a mapping from output port name to ``Batch``. Both port
        sequences are copied; the caller's sequences are never mutated.
        """

    def registration_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return this runtime's successful trusted registrations, oldest first.

        Each record is a new ``dict`` holding the arguments of one registration,
        including its live callable. Records are process-local trusted values,
        not data: never place them in a project, checkpoint, or API payload.
        """

    def capability_registration_snapshot(
        self,
    ) -> tuple[RuntimeCapabilities, tuple[dict[str, Any], ...]]:
        """Return ``capabilities()`` and ``registration_snapshot()`` as one
        consistent pair.

        The records are the registrations that ``capabilities`` describes: each
        ``provider`` and ``scalar_udf`` record has one capability entry with the
        same identity.
        """
```

Behavior is identical to today's private methods; only the names and
docstrings change. `capabilities()` is unchanged and equals the first element
of the pair. The pair docstring's first sentence wraps because it is 92
columns on one line, over the lint limit of 88.

## Why This Shape

- **Names drop the underscore and nothing else.** They compose nouns the docs
  already use (mapping provider, capability snapshot, registration), and
  `register_mapping_provider` sits beside `register_provider` with the same
  first four arguments.
- **Rename and migrate; keep no aliases.** An alias would leave the private
  spelling callable from the split-out repository, so a missed call site would
  keep passing and the cross-repository contract would not be enforced by any
  test. The underscore names carry no compatibility promise, every in-repo
  caller moves in the same change, the release workflow does not upload Studio
  to PyPI, and code-style prefers explicit modules over compatibility layers.
  This includes `_register_mapping_provider`, which published docs name; its
  removal is announced by the `CHANGELOG.md` migration line (decision 1).
- **The native method stays `_inner._register_mapping_provider`.** It is a
  different contract: it takes no `options_schema` and writes no record, so a
  direct call would desynchronize capabilities and revision. Keeping it private
  also means no Rust diff, no stub change, and no native rebuild.
- **Records stay plain `dict`.** Frozen values would break Studio's
  `type(record) is dict` preflight and the dict-literal equality in existing
  tests. `TypedDict` is runtime-identical but adds several exported names, cannot
  describe `stream_lifecycle` without publishing a private type, and no type
  checker is configured in this repository to enforce it. It remains available
  later as an additive step.
- **No new exports.** The methods live on the already exported `Runtime`.
  `python/calc_flow/__init__.py`, `python/calc_flow/_native.pyi`, and `crates/`
  are unchanged.
- **The stateless stream registration methods stay private.** Docs already say
  callers cannot supply that proof.

## Registration Record Contract

Records are returned in registration order. The table describes the keys core
produces today: a record carries every listed key with the listed type and
presence rule. It is not a closed schema, so a record may carry other keys (see
unlisted keys below). Key order inside a record is not guaranteed.

| Kind            | Key                           | Value type                        | Present                          |
|-----------------|-------------------------------|-----------------------------------|----------------------------------|
| `provider`      | `kind`                        | `str`, equal to `"provider"`      | always                           |
| `provider`      | `provider`, `name`, `version` | `str`                             | always                           |
| `provider`      | `callback`                    | callable                          | always                           |
| `provider`      | `options_schema`              | `ProviderOptionsSchema` or `None` | always                           |
| `provider`      | `accepts_context`             | `bool`, equal to `True`           | only when registered with `True` |
| `provider`      | `provider_mode`               | `str`, equal to `"mapping"`       | only for mapping providers       |
| `provider`      | `input_ports`, `output_ports` | `tuple[tuple[str, str], ...]`     | only for mapping providers       |
| `scalar_udf`    | `kind`                        | `str`, equal to `"scalar_udf"`    | always                           |
| `scalar_udf`    | `provider`, `name`, `version` | `str`                             | always                           |
| `scalar_udf`    | `input_types`                 | `tuple[str, ...]`                 | always                           |
| `scalar_udf`    | `return_type`, `volatility`   | `str`                             | always                           |
| `scalar_udf`    | `function`                    | callable                          | always                           |
| `kafka_decoder` | `kind`                        | `str`, equal to `"kafka_decoder"` | always                           |
| `kafka_decoder` | `name`, `version`             | non-empty `str`                   | always                           |
| `kafka_decoder` | `function`                    | callable                          | always                           |

- A `provider` record without `provider_mode` came from `register_provider`.
  A missing `accepts_context` means `False`.
- Port pairs are `(port_name, batch_kind)` in declaration order.
- A `kafka_decoder` record has **no `provider` key** and exists only on wheels
  built with `connector-kafka`.
- The listed keys of a record are the arguments of the one successful call that
  produced it: `register_provider`, `register_mapping_provider`,
  `register_scalar_udf`, or `register_kafka_decoder`.

**Compatibility rule (tolerant reader).** Adding a record kind or adding a key
is not a breaking change. Removing or retyping a listed key, changing a
presence rule, or changing record order is breaking and needs a `CHANGELOG.md`
entry with the migration. Consumers read `kind` first and use only the listed
keys. A consumer that lists registrations may skip kinds it does not handle. A
consumer that replays them must treat a skipped kind as a registration missing
from the target runtime, and fail or report it when the workload references it.

**Replay fidelity.** Replaying a record through the public registration
methods reproduces its identity, callable, ports, options schema, and context
opt-in. For scalar UDFs and batch-only providers that is the whole
registration. It is not for the `expression@1` and `symbolic_matrix@1`
providers installed by `register_numpy` and `register_jax`: their stream
capability comes from a process-local proof that no public registration method
accepts, so a replayed copy is batch-only. To get them with stream capability
on another runtime, call the same helper there instead of replaying their
records. Do not present a replayed runtime as equivalent to its source unless
the `ProviderCapability` entries of the replayed providers compare equal on
both.

Records can carry unlisted keys holding private process-local values; today
`stream_lifecycle` on those two helper records is the only one. Unlisted keys
carry no compatibility promise. Build each registration call, and anything you
transport, from the listed keys only.

## Copy And Trust Guarantees

- **Fresh on every call:** the outer `tuple` and each record `dict`. Adding,
  replacing, or deleting keys in a returned record never affects the runtime, a
  later snapshot, or another caller's snapshot.
- **Shared by reference:** every value. `callback` and `function` are the
  registered objects themselves, with their closures and state.
  `options_schema` is the frozen instance passed at registration. Port and
  input-type tuples are the runtime's own immutable copies, taken from the
  caller's sequences at registration.
- All shared values other than the callables are immutable, so sharing is
  observable only through a callable.
- Neither snapshot method calls, inspects, or serializes a registered callable.
- Records are **not data-only**. They are not JSON-serializable and must not
  enter a project, checkpoint, catalog, REST payload, or log. Configurations
  still name registrations only as `UdfReference(provider, name, version)`.
  `capabilities()` remains the data-only view.
- Core makes no transport promise. Moving a record to another process means
  serializing its callable under the caller's own trust decision;
  deserializing it there executes code.

## Paired Snapshot Consistency

`capability_registration_snapshot()` derives both values from one copy of the
record list, taken under one registration-lock acquisition. Deriving both from
that one copy is what makes the pair consistent. Each registration method
holds the same lock across its native registration and its record update; a
helper such as `register_numpy` is several such calls and is not atomic. For
one returned pair:

- `records` are the registrations that `capabilities` describes at
  `capabilities.scope.revision`.
- `capabilities.providers` has exactly one entry per `provider` record and
  `capabilities.udfs` one per `scalar_udf` record, with the same identities.
  `kafka_decoder` records have no capability entry.
- With no registration call, including a `register_numpy` or `register_jax`
  call, running or completing in between,
  `capabilities == runtime.capabilities()` and
  `records == runtime.registration_snapshot()`.

Across calls on one `Runtime`, `scope.session_id` is constant and
`scope.revision` never decreases; a successful registration advances it as
described under Runtime capabilities. Compare revisions for equality only: do
not derive a record count or a number of registrations from one.

The contract states no relation between a revision and a number of records,
and no prefix or append-only relation between snapshots. A helper's stream
proof step adds a key to an existing record without appending one.

**Limit (existing behavior, not changed here).** Today, equal
`(session_id, revision)` means the same registrations in the same order with
the same callables. It does not mean equal provider lifecycle fields: the
helpers' stream-proof step updates an existing record without advancing the
revision. Measured on the baseline: `capabilities()` before and after that
step share revision `1` and differ in `modes`. Through public API the
difference is observable only when another thread snapshots while
`register_numpy` or `register_jax` is running. No fix route is chosen, and
this contract commits to neither (decision 3).

## Error Cases

Messages for `register_mapping_provider` are today's and are not changed. No
failing call leaves a record or advances the revision.

| Input violation                       | Exception and message                                                             |
|---------------------------------------|-----------------------------------------------------------------------------------|
| `callback` not callable               | `TypeError`: `provider callback must be callable`                                 |
| port kind not `table` or `array`      | `ValueError`: `mapping provider port kind must be 'table' or 'array'`             |
| port entry not a two-item pair        | `ValueError`: `not enough values to unpack (expected 2, got 1)`                   |
| port name or kind not `str`           | `TypeError`: `argument 'input_ports': 'int' object is not an instance of 'str'`   |
| `accepts_context` not an exact `bool` | `TypeError`: `accepts_context must be an exact bool; found int`                   |
| `options_schema` of another type      | `TypeError`: `options_schema must be a ProviderOptionsSchema or None; found dict` |
| duplicate `(provider, name, version)` | `ConfigError`: `invalid provider: duplicate provider/name/version`                |
| ports passed positionally             | `TypeError` (keyword-only arguments)                                              |

Not fixed here, recommended as a follow-up now that the method is public: the
port-kind error omits the port name and the received value, a malformed pair
leaks a raw unpack error, the duplicate error omits the identity, and empty
port sequences and duplicate port names are accepted at registration. Two of
those texts live in Rust.

## Example

```python
from calc_flow import Runtime


def scale(inputs, options):
    return {"output": inputs["weights"]}


parent = Runtime()
parent.register_mapping_provider(
    "acme",
    "scale",
    "1",
    scale,
    input_ports=(("table", "table"), ("weights", "array")),
    output_ports=(("output", "array"),),
)

capabilities, records = parent.capability_registration_snapshot()
assert capabilities.providers[0].name == records[0]["name"] == "scale"

worker = Runtime()
for record in records:
    if record["kind"] != "provider":  # read kind before any other key
        raise ValueError(f"cannot replay a {record['kind']} registration")
    identity = (record["provider"], record["name"], record["version"])
    options = {
        "options_schema": record["options_schema"],
        "accepts_context": record.get("accepts_context", False),
    }
    if record.get("provider_mode") == "mapping":
        worker.register_mapping_provider(
            *identity,
            record["callback"],
            input_ports=record["input_ports"],
            output_ports=record["output_ports"],
            **options,
        )
    else:
        worker.register_provider(*identity, record["callback"], **options)

# A batch-only provider replays completely; a helper-installed one does not.
assert worker.capabilities().providers == capabilities.providers
```

## Acceptance Criteria

Each item is a failing test first; today the public names raise
`AttributeError`. Core tests go in `python/tests/test_runtime.py` unless noted.
Item numbers are stable across revisions of this note; item 7 is dropped.

Core:

1. `Runtime.register_mapping_provider` has four positional-or-keyword
   parameters `provider, name, version, callback`, then keyword-only
   `input_ports, output_ports` (required), `options_schema=None`,
   `accepts_context=False`, and returns `None`. Each of the three public
   methods has a docstring. `Runtime` no longer has the three underscore
   attributes.
2. Records carry every listed key with the listed type and presence rule. As a
   regression pin, not as contract, today's records carry no other key (item 8
   covers `stream_lifecycle`): a default single provider equals the six-key
   dict; `accepts_context=True` adds that key with value `True`; a mapping
   provider adds `provider_mode`, `input_ports`, and `output_ports`, with ports
   as tuples of two-`str` tuples even when registered from lists; a scalar UDF
   carries its eight keys with `input_types` a `tuple`. On a wheel with the
   `kafka` connector, a decoder record carries `kind, name, version, function`
   and no `provider` key.
3. Two `registration_snapshot()` calls return distinct tuples and distinct
   dicts whose `callback`, `function`, and `options_schema` values are the same
   objects that were registered. Setting, adding, and deleting keys on a
   returned record leaves the next snapshot unchanged. Clearing the caller's
   port lists after registration leaves the record's ports unchanged.
4. Each failure in the error table leaves `registration_snapshot()` equal to
   its prior value and `capabilities().scope.revision` unchanged.
5. Replay: for a runtime holding a single provider with `accepts_context=True`
   and an options schema, a mapping provider, and a scalar UDF, re-registering
   every record on a fresh `Runtime` through public methods only yields a
   snapshot equal to the source records, and `capabilities().providers` and
   `capabilities().udfs` equal to the source's.
6. For `capabilities, records = runtime.capability_registration_snapshot()`:
   the identities in `capabilities.providers` equal those of the `provider`
   records and the identities in `capabilities.udfs` equal those of the
   `scalar_udf` records; with nothing registering in between, the pair equals
   `(runtime.capabilities(), runtime.registration_snapshot())`; a new runtime
   returns `()` with no providers and no UDFs. After one further successful
   registration, a new pair has the same `scope.session_id` and a
   `scope.revision` that differs from the earlier one and is not lower, and the
   earlier pair is unchanged. No assertion relates a revision to a record
   count.
7. Dropped. A threaded snapshot test cannot fail for a locking regression: each
   pair is computed from one local copy, so its checks hold with the lock
   removed. Lost or corrupted concurrent registrations are already covered by
   `test_concurrent_registrations_keep_their_context_flags_isolated`
   (`test_execution_options.py:785`), which item 9 moves to the public name.
8. `register_numpy` still yields three records in the order `expression`,
   `table_matmul`, `symbolic_matrix`; apart from `stream_lifecycle` on the first
   and third, no record has a key outside the contract table. Replay fidelity:
   replaying those three records onto a fresh `Runtime` through public methods
   yields records equal to the source on every listed key, none carrying
   `stream_lifecycle`, and `modes == ("batch",)` on the replayed `expression@1`
   and `symbolic_matrix@1` capability entries where the source has
   `("batch", "stream")`. The helper's revision stays pinned only by the
   existing `test_capabilities.py:447`; this item does not assert it again.
9. Existing tests move to the public names with assertions unchanged:
   `python/tests/test_runtime.py` (lines 57-206), `test_execution_options.py`
   (629, 717, 728, 800), `test_capabilities.py` (514, 581, 1096), `test_array.py`
   (1288), plus callers `python/calc_flow/array.py` and
   `benchmarks/symbolic_support.py:322`.
10. The native seam is untouched: `test_execution_options.py:104-119` and `:740`
    still reference `_native` `_register_mapping_provider` and pass, and the
    diff contains nothing under `crates/`, `python/calc_flow/_native.pyi`, or
    `python/calc_flow/__init__.py`.

Studio, in `web-ui/backend/tests/test_run_manager.py`:

11. `_restore_registrations` restores a mapping record through
    `register_mapping_provider` on test doubles that define only public names
    (doubles near lines 922, 1052, 1415; direct call at 1571), with the
    arguments asserted today.
12. `RunManager.capabilities()` succeeds against a double that defines only
    `capability_registration_snapshot` (lines 894-898), and
    `test_capability_probe_does_not_block_runs_and_revalidates_revision` passes
    unchanged in behavior.
13. `RunManager.submit` and the tests feeding `_selected_registrations` read
    `registration_snapshot()` (lines 599, 656, 1065, 1080, 1435, 1453, 1542).
14. No file under `web-ui/backend/src/` references the three underscore names.

Docs, for `cf-doc-writer`:

15. `docs/python-api.md`: rename at 743-744. Extend 636-646 and 657-659 to say
    both `register_provider` and `register_mapping_provider` take no lifecycle
    metadata. Reconcile the existing sentence at 642-644 ("A registration
    record carrying forged lifecycle keys is ignored rather than upgraded")
    with the unlisted-keys paragraph, so both describe such keys the same way.
    Add `### Trusted registration records` at the end of
    `## Runtime capabilities` (after line 676) holding the two snapshot
    signatures, the record table with its producer-side lead, the
    compatibility rule, the replay-fidelity and unlisted-keys paragraphs, the
    copy and trust guarantees, the pair bullets, the across-calls paragraph,
    the limit, and the example above. The new section must not state a
    relation between a revision and a record count, a starting revision, a
    prefix relation, or "append-only", and must not call the table exact. The
    existing revision sentences at 668-672 stay as they are.
16. `docs/api-reference.md`: add the three methods to the summary at 247-250
    and state that records are trusted values while `capabilities()` is
    data-only; rename at 258 and 261-262; at 284-289 add that the pair's
    records are the registrations its capabilities describe, that revisions
    are compared for equality only, and the lifecycle limit. Do not add
    `revision == len(records)`.
17. `CHANGELOG.md`: one dated bullet at the top of `## 2026-10` naming the three
    public methods, the removed underscore spellings, and the migration (rename
    the call; arguments unchanged).
18. A repository search for the three underscore names matches only
    `pipeline.py`'s `self._inner._register_mapping_provider`, `_native.pyi`,
    `crates/calc-flow-python/src/config.rs`, the two native tests in item 10,
    the `CHANGELOG.md` migration bullet from item 17, this note, and its
    critique.

Residual risk: the `kafka_decoder` rows are not exercised by any test run.
They come from reading `register_kafka_decoder` (`pipeline.py:500-507`). Item
2's Kafka clause runs only on a wheel built with `connector-kafka`; the
existing test returns early without one (`python/tests/test_runtime.py:44-50`),
the binding's default features exclude it, and no workflow was found that
builds a Kafka-enabled wheel for pytest. The `--features kafka` step at
`.github/workflows/ci-linux.yml:153` is a Rust connector test.

## Findings Outside The Three Capabilities

Found while enumerating the contract; verified on the baseline by calling the
Studio functions directly, not through a full `RunManager.submit`.

- **Studio rejects helper records.** For a session runtime with
  `register_numpy` applied and a project using `numpy:expression@1`,
  `_selected_registrations` keeps `stream_lifecycle` and the worker's
  `_restore_registrations` raises `RunManagerError: worker received an invalid
  registration contract`. `symbolic_matrix@1` fails the same way.
- **Studio assumes every record has `provider`.** A `kafka_decoder` record
  raises `KeyError: 'provider'` in `_selected_registrations` (reached by
  `RunManager.submit`) and in `_capabilities_response` and
  `_worker_registration` (reached by `RunManager.capabilities`). Tested with a
  hand-built decoder-shaped record, because the local wheel has no Kafka
  connector.
- **Studio caches by `(session_id, revision)`.** Given the limit above, a
  `/capabilities` request landing between the helper's last two steps can cache
  a batch-only `symbolic_matrix@1` until the next registration. This follows
  from reading `RunManager.capabilities`; the race was not reproduced.

The first two are not fixed in this work item (decision 2). Both need an
embedder-supplied runtime with a helper or a Kafka decoder applied; Studio's
default runtime is bare (`app.py:143`). Per the critique, the fix belongs at
the parent read sites (`run_manager.py:246-258`, `811-821`), so the worker
preflight (`866-965`) can stay strict.

## Open Questions

None open. Decisions on the three earlier questions:

1. **The underscore spellings are removed with no alias**, including the
   documented `_register_mapping_provider`. `CHANGELOG.md` carries the
   migration line (item 17).
2. **The two Studio consumer gaps are tracked separately.** They must land
   before the repository split and before any additive record change: Studio's
   range `calc-flow-python>=2026.10.12,<2027` admits every later 2026 core
   release, and its tests pin rejection of unexpected keys and unknown kinds
   (`web-ui/backend/tests/test_run_manager.py:1277-1330`). Until then core
   cannot use the freedom the compatibility rule grants.
3. **Revision and the stream proof stay a separate follow-up with no route
   chosen.** The two routes are making the helpers atomic and advancing the
   revision at the proof step. The contract therefore states no relation
   between a revision and a record count, and items 6 and 8 assert none.

Dependency note: tag `calc-flow-python-v2026.10.12` predates this change, so
the public names first ship in the next release. Each CalVer release so far
has raised Studio's `calc-flow-python>=` floor to the release version; the next
bump must do the same, because earlier versions lack the public names.
