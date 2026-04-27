from __future__ import annotations

import re

_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?<![!<>=])=(?!=)\s*(.+?)\s*$")


def split_assignment(expression: str) -> tuple[str, str] | None:
    match = _ASSIGNMENT_RE.match(expression)
    if match is None:
        return None
    return match.group(1), match.group(2)


def sql_projection(expression: str, table_name: str) -> str:
    assignment = split_assignment(expression)
    if assignment is not None:
        column, value = assignment
        return f"SELECT *, ({value}) AS {column} FROM {table_name}"
    return f"SELECT ({expression}) AS result FROM {table_name}"
