from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest


def inputs():
    schema = pa.schema(
        [
            ("account", pa.string()),
            ("stock", pa.string()),
            ("quantity", pa.int64()),
            ("side", pa.string()),
            ("flow_type", pa.string()),
        ]
    )
    batches = [
        [("b", "x", 3, "buy", "trade"), ("a", "x", 2, "sell", "trade")],
        [("a", "x", 4, "buy", "trade"), ("b", "x", 2, "sell", "trade")],
        [("b", "x", 0, "buy", "trade"), ("c", "y", 8, "buy", "quote")],
    ]
    return tuple(
        pa.Table.from_arrays(list(zip(*rows)), schema=schema) for rows in batches
    )


def expected():
    schema = pa.schema(
        [
            ("account", pa.string()),
            ("stock", pa.string()),
            ("position", pa.int64()),
            pa.field("_cf_op", pa.string(), nullable=False),
        ]
    )
    return [
        pa.table(
            [["b", "a"], ["x", "x"], [3, -2], ["upsert", "upsert"]], schema=schema
        ),
        pa.table(
            [["b", "b", "a", "a"], ["x"] * 4, [3, 1, -2, 2], ["retract", "upsert"] * 2],
            schema=schema,
        ),
        pa.Table.from_batches([], schema=schema),
    ]


def test_public_stream_reconstructs_each_prefix_in_group_order():
    from benchmarks.sparse_changelog import observe, validate

    batches = inputs()
    elapsed, actual = asyncio.run(observe(batches))
    assert elapsed > 0
    assert len(actual) == 3
    assert all(a.equals(b, check_metadata=True) for a, b in zip(actual, expected()))
    assert validate(batches, actual) == [2, 4, 0]


@pytest.mark.parametrize(
    "fault", ["order", "old_value", "metadata", "missing", "extra"]
)
def test_oracle_rejects_malformed_changes_even_if_final_values_match(fault):
    from benchmarks.sparse_changelog import validate

    result = expected()
    if fault == "order":
        result[1] = result[1].take([2, 3, 0, 1])
    elif fault == "old_value":
        result[1] = result[1].set_column(
            2, result[1].schema.field(2), pa.array([99, 1, -2, 2])
        )
    elif fault == "metadata":
        result[1] = result[1].replace_schema_metadata({b"unexpected": b"field"})
    elif fault == "missing":
        result.pop()
    else:
        result.append(result[-1])
    with pytest.raises(ValueError):
        validate(inputs(), result)


def test_isolated_worker_retains_per_batch_oracle_evidence(monkeypatch, tmp_path):
    from benchmarks import sparse_changelog

    monkeypatch.setattr(sparse_changelog, "load_fixture", lambda _: inputs())
    active = sparse_changelog.SparseChangelogCase(
        {"fixture": "fixture.arrow"}, tmp_path
    )
    try:
        for sample in (active.sample(), active.sample()):
            assert sample["verified_output_rows"] == [2, 4, 0]
            assert sample["correctness"] == {"passed": True, "rows": 6}
            assert sample["seconds"] > 0
        assert active.finish() == {"state": "completed"}
    finally:
        active.close()
