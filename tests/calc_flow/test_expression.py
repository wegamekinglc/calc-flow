from __future__ import annotations

from calc_flow.expression import split_assignment, sql_projection


def test_split_assignment() -> None:
    assert split_assignment("c = a + b") == ("c", "a + b")


def test_split_assignment_ignores_comparison() -> None:
    assert split_assignment("a == b") is None
    assert split_assignment("a != b") is None
    assert split_assignment("a <= b") is None
    assert split_assignment("a >= b") is None


def test_sql_projection_for_assignment() -> None:
    assert sql_projection("c = a + b", "input") == "SELECT *, (a + b) AS c FROM input"


def test_sql_projection_for_expression() -> None:
    assert sql_projection("a + b", "input") == "SELECT (a + b) AS result FROM input"
