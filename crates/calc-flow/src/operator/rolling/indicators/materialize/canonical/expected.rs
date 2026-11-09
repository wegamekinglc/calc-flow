use super::{Check, invalid};
use crate::Result;
use datafusion::arrow::datatypes::{DataType as D, FieldRef, Fields, UnionFields, UnionMode};
use std::{collections::HashMap, sync::Arc};

pub(super) fn validate(actual: &D, expected: &D, node: &str, check: Check<'_>) -> Result<()> {
    check()?;
    if std::ptr::eq(actual, expected) {
        return Ok(());
    }
    match (actual, expected) {
        (D::List(a), D::List(e))
        | (D::LargeList(a), D::LargeList(e))
        | (D::ListView(a), D::ListView(e))
        | (D::LargeListView(a), D::LargeListView(e)) => field(a, e, node, check),
        (D::FixedSizeList(a, n), D::FixedSizeList(e, m)) => fixed(a, e, *n == *m, node, check),
        (D::Struct(a), D::Struct(e)) => fields(a, e, node, check),
        _ => encoded(actual, expected, node, check),
    }
}
fn fixed(
    actual: &FieldRef,
    expected: &FieldRef,
    width_matches: bool,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    same(width_matches, node)?;
    field(actual, expected, node, check)
}
fn encoded(actual: &D, expected: &D, node: &str, check: Check<'_>) -> Result<()> {
    match (actual, expected) {
        (D::Map(a, x), D::Map(e, y)) => fixed(a, e, *x == *y, node, check),
        (D::Dictionary(a, b), D::Dictionary(e, f)) => dictionary((a, b), (e, f), node, check),
        (D::RunEndEncoded(a, b), D::RunEndEncoded(e, f)) => run((a, b), (e, f), node, check),
        (D::Union(a, x), D::Union(e, y)) => union((a, *x), (e, *y), node, check),
        _ => same(actual == expected, node),
    }
}
fn dictionary(actual: (&D, &D), expected: (&D, &D), node: &str, check: Check<'_>) -> Result<()> {
    validate(actual.0, expected.0, node, check)?;
    validate(actual.1, expected.1, node, check)
}
fn run(
    actual: (&FieldRef, &FieldRef),
    expected: (&FieldRef, &FieldRef),
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    field(actual.0, expected.0, node, check)?;
    field(actual.1, expected.1, node, check)
}
fn fields(actual: &Fields, expected: &Fields, node: &str, check: Check<'_>) -> Result<()> {
    same(actual.len() == expected.len(), node)?;
    for (a, e) in actual.iter().zip(expected.iter()) {
        field(a, e, node, check)?;
    }
    Ok(())
}
fn union(
    actual: (&UnionFields, UnionMode),
    expected: (&UnionFields, UnionMode),
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    same(actual.1 == expected.1, node)?;
    same(actual.0.len() == expected.0.len(), node)?;
    for ((a_id, a), (e_id, e)) in actual.0.iter().zip(expected.0.iter()) {
        same(a_id == e_id, node)?;
        field(a, e, node, check)?;
    }
    Ok(())
}
fn field(actual: &FieldRef, expected: &FieldRef, node: &str, check: Check<'_>) -> Result<()> {
    check()?;
    if Arc::ptr_eq(actual, expected) {
        return Ok(());
    }
    same(
        actual.name() == expected.name() && actual.is_nullable() == expected.is_nullable(),
        node,
    )?;
    validate(actual.data_type(), expected.data_type(), node, check)?;
    metadata(actual.metadata(), expected.metadata(), node, check)
}
fn metadata(
    actual: &HashMap<String, String>,
    expected: &HashMap<String, String>,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    same(actual.len() == expected.len(), node)?;
    for (key, value) in actual {
        check()?;
        same(expected.get(key) == Some(value), node)?;
    }
    Ok(())
}
fn same(matches: bool, node: &str) -> Result<()> {
    if matches { Ok(()) } else { Err(invalid(node)) }
}
