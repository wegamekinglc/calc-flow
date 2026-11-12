use super::{
    first_asof, identity_columns, identity_source, join_alias, nested_inputs, program, rejects,
    renamed, select_names, source, spec, verify,
};
use calc_flow::{Result, TableExpr, UdfRegistry};
use std::collections::BTreeMap;

fn appended(input: &TableExpr, from: &str, to: &str) -> Result<TableExpr> {
    input.with_columns(
        ["key", "time", "sequence", "value"]
            .map(|name| {
                input
                    .column(&format!("{from}{name}"))?
                    .alias(&format!("{to}{name}"))
            })
            .into_iter()
            .collect::<Result<_>>()?,
    )
}
fn chain(input: &TableExpr, from: &str, to: &str) -> Result<TableExpr> {
    renamed(&appended(input, from, to)?, to, to)
}

#[tokio::test]
async fn appended_alias_only_selection_executes_both_lifecycles() -> Result<()> {
    let left = chain(&source("a")?, "", "df_")?;
    let joined = left.asof_join(&source("b")?, spec("df_")?)?;
    verify(
        program(BTreeMap::from([("out".into(), joined)]))?,
        super::super::fixture(),
        BTreeMap::from([("out".into(), vec![Some(7)])]),
    )
    .await
}

#[tokio::test]
async fn nested_alias_chain_through_filter_executes_both_lifecycles() -> Result<()> {
    let first = first_asof()?;
    let filtered = filtered_aliases(&first)?;
    let post = chain(&filtered, "stage_", "df_")?;
    let second = join_alias(&post, "c", "df_")?;
    verify(
        program(BTreeMap::from([
            ("first".into(), first),
            ("second".into(), second),
        ]))?,
        nested_inputs(),
        BTreeMap::from([
            ("first".into(), vec![Some(7)]),
            ("second".into(), vec![Some(3)]),
        ]),
    )
    .await
}
fn filtered_aliases(input: &TableExpr) -> Result<TableExpr> {
    let stage = appended(input, "left__", "stage_")?;
    stage.filter(stage.column("stage_key")?.eq(calc_flow::lit("A")?))
}

#[tokio::test]
async fn alias_chain_preserves_complete_composite_sequence() -> Result<()> {
    let left = chain(&identity_source("sequence")?, "", "df_")?;
    let joined = left.asof_join(&source("b")?, composite_spec()?)?;
    verify(
        program(BTreeMap::from([("out".into(), joined)]))?,
        super::super::fixture(),
        BTreeMap::from([("out".into(), vec![Some(7)])]),
    )
    .await
}
fn composite_spec() -> Result<calc_flow::StreamAsofJoinSpec> {
    let base = spec("df_")?;
    calc_flow::StreamAsofJoinSpec::new(
        calc_flow::AsofJoinSide::new(
            vec!["df_key".into()],
            "df_time".into(),
            vec!["df_sequence".into(), "df_value".into()],
            "left".into(),
        )?,
        base.right().clone(),
        std::time::Duration::from_micros(base.tolerance_micros()),
        base.limits(),
    )
}

fn distorted(input: &TableExpr, from: &str, kind: &str) -> Result<TableExpr> {
    let stage = input.with_columns(identity_columns(input, from, kind)?)?;
    select_names(&stage, &["df_key", "df_time", "df_sequence"])
}

#[tokio::test]
async fn alias_chain_cannot_launder_derived_or_incomplete_identity() -> Result<()> {
    for kind in [
        "entity",
        "sequence",
        "cast_key",
        "cast_time",
        "rewrite_sequence",
    ] {
        rejects(distorted(&identity_source(kind)?, "", kind)?, kind).await?;
    }
    let first = first_asof()?;
    rejects(
        distorted(&first, "left__", "rewrite_sequence")?,
        "nested sequence",
    )
    .await
}

#[tokio::test]
async fn ordinary_alias_chain_does_not_gain_proof() -> Result<()> {
    let left = chain(&source("a")?, "", "df_")?;
    let report = program(BTreeMap::from([("out".into(), left)]))?
        .analyze(
            &UdfRegistry::new().snapshot(),
            &calc_flow::AnalysisOptions::default(),
        )
        .await?;
    let node = report
        .nodes
        .iter()
        .find(|n| n.path == "outputs.out")
        .unwrap();
    assert!(node.batch.ordering.temporal_identity.is_none());
    assert!(node.stream.ordering.temporal_identity.is_none());
    Ok(())
}
