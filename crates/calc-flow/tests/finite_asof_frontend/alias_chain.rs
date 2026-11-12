use super::{identity_source, program, renamed, source, spec, verify};
use calc_flow::{CompileMode, Result, TableExpr, UdfRegistry};
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
    let first = source("a")?.asof_join(&source("b")?, super::super::asof_support::spec(2))?;
    let stage = appended(&first, "left__", "stage_")?;
    let filtered = stage.filter(stage.column("stage_key")?.eq(calc_flow::lit("A")?))?;
    let post = chain(&filtered, "stage_", "df_")?;
    let second = post.asof_join(&source("c")?, spec("df_")?)?;
    let mut inputs = super::super::fixture();
    inputs.insert(
        "c".into(),
        super::super::asof_support::batch(&[("A", 8, 3, 3)]),
    );
    verify(
        program(BTreeMap::from([
            ("first".into(), first),
            ("second".into(), second),
        ]))?,
        inputs,
        BTreeMap::from([
            ("first".into(), vec![Some(7)]),
            ("second".into(), vec![Some(3)]),
        ]),
    )
    .await
}

#[tokio::test]
async fn alias_chain_preserves_complete_composite_sequence() -> Result<()> {
    let left = chain(&identity_source("sequence")?, "", "df_")?;
    let base = spec("df_")?;
    let composite = calc_flow::StreamAsofJoinSpec::new(
        calc_flow::AsofJoinSide::new(
            vec!["df_key".into()],
            "df_time".into(),
            vec!["df_sequence".into(), "df_value".into()],
            "left".into(),
        )?,
        base.right().clone(),
        std::time::Duration::from_micros(base.tolerance_micros()),
        base.limits(),
    )?;
    let joined = left.asof_join(&source("b")?, composite)?;
    verify(
        program(BTreeMap::from([("out".into(), joined)]))?,
        super::super::fixture(),
        BTreeMap::from([("out".into(), vec![Some(7)])]),
    )
    .await
}

fn distorted(input: &TableExpr, from: &str, kind: &str) -> Result<TableExpr> {
    let columns = ["key", "time", "sequence"].map(|name| {
        let column = input.column(&format!("{from}{name}"))?;
        let column = match (kind, name) {
            ("cast_key", "key") | ("cast_time", "time") => column.cast(
                super::super::asof_support::schema()
                    .field_with_name(name)
                    .unwrap()
                    .data_type()
                    .clone(),
            )?,
            ("rewrite_sequence", "sequence") => column + calc_flow::lit(1_i64)?,
            _ => column,
        };
        column.alias(&format!("df_{name}"))
    });
    let stage = input.with_columns(columns.into_iter().collect::<Result<_>>()?)?;
    stage.select(
        ["key", "time", "sequence"]
            .map(|name| {
                stage
                    .column(&format!("df_{name}"))?
                    .alias(&format!("df_{name}"))
            })
            .into_iter()
            .collect::<Result<_>>()?,
    )
}
async fn rejects(left: TableExpr) -> Result<()> {
    let joined = left.asof_join(&source("b")?, spec("df_")?)?;
    let p = program(BTreeMap::from([("out".into(), joined)]))?;
    for mode in [CompileMode::Batch, CompileMode::Stream] {
        let error = p
            .for_mode(mode)?
            .collect_schema(&UdfRegistry::new().snapshot())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("ordering_required"),
            "{mode:?}: {error}"
        );
    }
    Ok(())
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
        rejects(distorted(&identity_source(kind)?, "", kind)?).await?;
    }
    let first = source("a")?.asof_join(&source("b")?, super::super::asof_support::spec(2))?;
    rejects(distorted(&first, "left__", "rewrite_sequence")?).await
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
