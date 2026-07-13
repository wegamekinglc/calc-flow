from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from calc_flow import Batch, ExpressionOperator, Pipeline, SqlOperator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "v1"


def write_table(name: str, table: pa.Table) -> str:
    path = OUTPUT / name
    with path.open("wb") as stream, ipc.new_file(stream, table.schema) as writer:
        writer.write_table(table)
    return name


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expression_input = pa.table({"a": [1, 3], "b": [2, 4]})
    expression_output = (
        Pipeline("fixture-expression")
        .then(ExpressionOperator("calculate", "total = a + b"))
        .execute({"input": Batch.table(expression_input)})
        .output.table_payload
    )
    left = pa.table({"id": [1, 2], "amount": [10, 20]})
    right = pa.table({"id": [2, 1], "rate": [3, 4]})
    sql_output = (
        Pipeline("fixture-sql")
        .then(
            SqlOperator(
                "join",
                "SELECT l.id, l.amount * r.rate AS total FROM l JOIN r ON l.id = r.id",
                inputs=("l", "r"),
            )
        )
        .execute({"l": Batch.table(left), "r": Batch.table(right)})
        .output.table_payload
    )
    arrow_files = [
        write_table("expression.arrow", expression_input),
        write_table("expression_expected.arrow", expression_output),
        write_table("sql_left.arrow", left),
        write_table("sql_right.arrow", right),
        write_table("sql_expected.arrow", sql_output),
        write_table("empty.arrow", pa.table({"value": pa.array([], type=pa.int64())})),
    ]
    manifest = {
        "format_version": 1,
        "arrow_files": arrow_files,
        "cases": [
            {
                "name": "expression_assignment",
                "input": "expression.arrow",
                "operation": "total = a + b",
                "expected": "expression_expected.arrow",
                "invariants": ["table_only", "metadata_preserved"],
            },
            {
                "name": "sql_join",
                "input": ["sql_left.arrow", "sql_right.arrow"],
                "operation": "join",
                "expected": "sql_expected.arrow",
                "invariants": ["single_select"],
            },
            {
                "name": "empty_table",
                "input": "empty.arrow",
                "operation": "identity",
                "expected": "empty.arrow",
                "invariants": ["schema_preserved"],
            },
            {
                "name": "metadata_round_trip",
                "input": "expression.arrow",
                "operation": "identity",
                "expected": "expression.arrow",
                "invariants": ["deeply_immutable_json"],
            },
            {
                "name": "state_rollback",
                "input": "expression.arrow",
                "operation": "fail_after_state",
                "expected": {"state": {}},
                "invariants": ["rollback"],
            },
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
