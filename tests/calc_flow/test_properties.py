from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from calc_flow import (
    Batch,
    BatchMetadata,
    Checkpoint,
    ExpressionOperator,
    Pipeline,
    RunContext,
    StatelessOperator,
)

JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    ),
    max_leaves=15,
)


@st.composite
def arrow_tables(draw: st.DrawFn) -> pa.Table:
    type_name = draw(st.sampled_from(("int64", "float64", "string", "bool")))
    size = draw(st.integers(min_value=0, max_value=20))
    if type_name == "int64":
        values = draw(
            st.lists(
                st.one_of(st.none(), st.integers(-(2**31), 2**31)),
                min_size=size,
                max_size=size,
            )
        )
    elif type_name == "float64":
        values = draw(
            st.lists(
                st.one_of(
                    st.none(),
                    st.floats(allow_nan=False, allow_infinity=False, width=32),
                ),
                min_size=size,
                max_size=size,
            )
        )
    elif type_name == "string":
        values = draw(
            st.lists(
                st.one_of(st.none(), st.text(max_size=20)),
                min_size=size,
                max_size=size,
            )
        )
    else:
        values = draw(
            st.lists(
                st.one_of(st.none(), st.booleans()),
                min_size=size,
                max_size=size,
            )
        )
    return pa.table({"value": pa.array(values, type=pa.type_for_alias(type_name))})


@given(JSON_VALUES)
@settings(max_examples=50)
def test_metadata_preserves_generated_json_values(value: object) -> None:
    metadata = BatchMetadata(cursor=value, attributes={"generated": value})

    document = metadata.to_dict()

    assert document["cursor"] == value
    assert document["attributes"] == {"generated": value}


@given(arrow_tables())
@settings(max_examples=50)
def test_generated_arrow_tables_and_record_batches_normalize_identically(
    table: pa.Table,
) -> None:
    record_batch = pa.RecordBatch.from_arrays(
        [table.column(0).combine_chunks()],
        schema=table.schema,
    )

    table_batch = Batch.table(table)
    record_batch_batch = Batch.table(record_batch)

    assert record_batch_batch.table_payload.equals(table_batch.table_payload)
    assert record_batch_batch.schema == table_batch.schema


@given(st.lists(st.integers(-10, 10), min_size=1, max_size=6))
@settings(max_examples=25, deadline=None)
def test_generated_linear_dags_execute_in_topological_order(
    increments: list[int],
) -> None:
    pipeline = Pipeline("generated-linear-dag")
    input_column = "base"
    expected = 1
    for index, increment in enumerate(increments):
        output_column = f"value_{index}"
        pipeline.then(
            ExpressionOperator(
                f"node_{index}",
                f"{output_column} = {input_column} + ({increment})",
            )
        )
        input_column = output_column
        expected += increment

    result = pipeline.compile().execute({"input": Batch.table(pa.table({"base": [1]}))})

    assert result.output.table_payload[input_column].to_pylist() == [expected]


def identity(inputs: Mapping[str, Batch], _context: RunContext) -> Mapping[str, Batch]:
    return {"output": inputs["input"]}


@given(st.integers(min_value=2, max_value=8))
@settings(max_examples=20)
def test_generated_cycles_are_rejected(node_count: int) -> None:
    pipeline = Pipeline("generated-cycle")
    for index in range(node_count):
        pipeline.add_node(f"node_{index}", StatelessOperator(f"node_{index}", identity))
    for index in range(node_count):
        pipeline.connect(f"node_{index}", f"node_{(index + 1) % node_count}")

    with pytest.raises(ValueError, match="cycle"):
        pipeline.compile()


@given(
    cursor=JSON_VALUES,
    state=st.dictionaries(
        st.from_regex(r"[a-z][a-z0-9_]{0,7}", fullmatch=True),
        st.dictionaries(
            st.from_regex(r"[a-z][a-z0-9_]{0,7}", fullmatch=True),
            JSON_VALUES,
            max_size=4,
        ),
        max_size=4,
    ),
    sequence=st.integers(min_value=0, max_value=1_000_000),
)
@settings(max_examples=50)
def test_generated_checkpoint_state_round_trips(
    cursor: object,
    state: dict[str, dict[str, object]],
    sequence: int,
) -> None:
    checkpoint = Checkpoint(
        pipeline_name="property-checkpoint",
        pipeline_fingerprint="fingerprint",
        source_cursor=cursor,
        sequence=sequence,
        state=state,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    restored = Checkpoint.from_dict(checkpoint.to_dict())

    assert restored == checkpoint
