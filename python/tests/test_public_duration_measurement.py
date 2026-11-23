from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pytest


def test_open_lower_boundary_and_nulls():
    from benchmarks.public_duration_sum import INPUT, expected

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = [0, 0, 29, 30, 60, 90]
    rows = pa.table(
        {
            "event_time": [base + timedelta(seconds=x) for x in ts],
            "sequence": list(range(6)),
            "symbol": ["a", "z", "a", "a", "a", "a"],
            "price": [1.0, None, 3.0, 4.0, None, 2.0],
        },
        schema=INPUT,
    )
    output = expected(rows)
    assert output["sum_30"].to_pylist() == [1.0, None, 4.0, 7.0, None, 2.0]
    assert output["sum_60"].to_pylist() == [1.0, None, 4.0, 8.0, 7.0, 2.0]
    assert output["sum_90"].to_pylist() == [1.0, None, 4.0, 8.0, 8.0, 9.0]


def test_public_declaration_has_three_duration_sums_on_one_stage():
    from benchmarks.public_duration_sum import declaration, plan

    proof = plan(declaration())
    assert proof["partition_by"] == ["symbol"]
    assert [o["frame"]["micros"] for o in proof["outputs"]] == [
        30_000_000,
        60_000_000,
        90_000_000,
    ]


def test_small_public_stream_and_corruption_checks():
    import asyncio

    from benchmarks.public_duration_sum import INPUT, expected, observe, validate

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamps = [0, 0, 29, 30, 60, 90]
    t = pa.table(
        {
            "event_time": [base + timedelta(seconds=x) for x in stamps],
            "sequence": list(range(6)),
            "symbol": ["a", "z", "a", "a", "a", "a"],
            "price": [1.0, None, 3.0, 4.0, None, 2.0],
        },
        schema=INPUT,
    )
    _, out = asyncio.run(observe((t.slice(0, 2), t.slice(2))))
    oracle = expected(t)
    assert validate(out, oracle) == [6, 2, 1, 1]
    rows = pa.concat_tables(out)
    # These faults change a null, an exact old boundary, an entity/order or metadata.
    for idx, col in [
        (0, pa.array([0, 1, 2, 4, 3, 5], type=pa.uint64())),
        (2, pa.array([1.0, 0.0, 4.0, 7.0, None, 2.0])),
        (4, pa.array([1.0, None, 4.0, 8.0, 8.0, 10.0])),
    ]:
        with pytest.raises(ValueError, match="differs"):
            validate([rows.set_column(idx, rows.schema.field(idx), col)], oracle)
    with pytest.raises(ValueError, match="differs"):
        validate([rows.replace_schema_metadata({b"bad": b"schema"})], oracle)


def test_worker_emits_validated_window_count_and_plan(monkeypatch, tmp_path):
    from benchmarks import public_duration_sum as duration
    from scripts.benchmark_suite.worker import finish_case, prepare_case

    source = pa.table(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "sequence": [0],
            "symbol": ["x"],
            "price": [None],
        },
        schema=duration.INPUT,
    )
    monkeypatch.setattr(duration, "load", lambda _path: (source,))
    warmed, worker = prepare_case(
        {"family": "public-duration-sum", "rows": 1, "fixture": "fixture.arrow"},
        tmp_path,
    )
    try:
        for sample in (warmed["warmup"], worker.sample()):
            assert sample["correctness"] == {"passed": True, "rows": 1}
            assert sample["verified_row_count_and_nulls"] == [1, 1, 1, 1]
            assert sample["seconds"] > 0
        assert len(worker.plan["outputs"]) == 3
    finally:
        assert finish_case(worker)["state"] == "completed"
