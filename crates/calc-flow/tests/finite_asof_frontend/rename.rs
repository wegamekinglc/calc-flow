use super::{fixture, source, stream, values};
use calc_flow::{
    AsofJoinSide, CompileMode, CompileOptions, InputOptions, Program, Result, StreamAsofJoinSpec,
    TableExpr, UdfRegistry, table_input,
};
use std::collections::BTreeMap;

#[path = "alias_chain.rs"]
mod chain;

fn renamed(input: &TableExpr, from: &str, to: &str) -> Result<TableExpr> {
    input.select(
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
fn spec(prefix: &str) -> Result<StreamAsofJoinSpec> {
    let original = super::asof_support::spec(2);
    StreamAsofJoinSpec::new(
        AsofJoinSide::new(
            vec![format!("{prefix}key")],
            format!("{prefix}time"),
            vec![format!("{prefix}sequence")],
            "left".into(),
        )?,
        original.right().clone(),
        std::time::Duration::from_micros(original.tolerance_micros()),
        original.limits(),
    )
}
fn program(outputs: BTreeMap<String, TableExpr>) -> Result<Program> {
    Program::new("df_identity_rename", CompileMode::Batch, outputs)
}

#[tokio::test]
async fn source_df_rename_asof_executes_in_both_lifecycles() -> Result<()> {
    let renamed = renamed(&source("a")?, "", "df_")?;
    let joined = renamed.asof_join(&source("b")?, spec("df_")?)?;
    verify(
        program(BTreeMap::from([("out".into(), joined)]))?,
        fixture(),
        BTreeMap::from([("out".into(), vec![Some(7)])]),
    )
    .await
}

#[tokio::test]
async fn asof_df_rename_asof_executes_in_both_lifecycles() -> Result<()> {
    let first = source("a")?.asof_join(&source("b")?, super::asof_support::spec(2))?;
    let post = renamed(&first, "left__", "next_")?;
    let second = post.asof_join(&source("c")?, spec("next_")?)?;
    let mut inputs = fixture();
    inputs.insert("c".into(), super::asof_support::batch(&[("A", 8, 3, 3)]));
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

async fn verify(
    p: Program,
    inputs: BTreeMap<String, calc_flow::Batch>,
    expected: BTreeMap<String, Vec<Option<i64>>>,
) -> Result<()> {
    let udfs = UdfRegistry::new().snapshot();
    let plan = p.compile_batch(&udfs, &CompileOptions::default()).await?;
    let result = plan
        .execute(inputs.clone(), calc_flow::ExecutionOptions::default())
        .await?;
    for (name, expected) in &expected {
        assert_eq!(
            values(std::slice::from_ref(&result.outputs[name])),
            *expected
        );
    }
    let plan = p
        .for_mode(CompileMode::Stream)?
        .compile_stream(
            &udfs,
            &calc_flow::StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    stream::verify_expected(plan, inputs, expected).await
}

#[tokio::test]
async fn df_rename_does_not_prove_incomplete_or_derived_identity() -> Result<()> {
    for missing in [
        "entity",
        "sequence",
        "cast_key",
        "cast_time",
        "rewrite_sequence",
    ] {
        let projected = invalid_projection(missing)?;
        let joined = projected.asof_join(&source("b")?, spec("df_")?)?;
        let p = program(BTreeMap::from([("out".into(), joined)]))?;
        for mode in [CompileMode::Batch, CompileMode::Stream] {
            let error = p
                .for_mode(mode)?
                .collect_schema(&UdfRegistry::new().snapshot())
                .await
                .unwrap_err();
            assert!(
                error.to_string().contains("ordering_required"),
                "{missing} {mode:?}: {error}"
            );
        }
    }
    Ok(())
}

fn identity_source(missing: &str) -> Result<TableExpr> {
    table_input(
        "a",
        super::asof_support::schema(),
        InputOptions {
            entity_by: if missing == "entity" {
                vec!["key".into(), "value".into()]
            } else {
                vec!["key".into()]
            },
            event_time: Some("time".into()),
            sequence_by: if missing == "sequence" {
                vec!["sequence".into(), "value".into()]
            } else {
                vec!["sequence".into()]
            },
        },
    )
}
fn invalid_projection(kind: &str) -> Result<TableExpr> {
    let a = identity_source(kind)?;
    let columns = ["key", "time", "sequence"].map(|name| {
        let column = a.column(name)?;
        let column = match (kind, name) {
            ("cast_key", "key") | ("cast_time", "time") => column.cast(
                super::asof_support::schema()
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
    a.select(columns.into_iter().collect::<Result<_>>()?)
}

#[tokio::test]
async fn ordinary_df_and_sql_renames_do_not_gain_temporal_proof() -> Result<()> {
    let a = source("a")?;
    let outputs = BTreeMap::from([
        ("df".into(), renamed(&a, "", "df_")?),
        (
            "sql".into(),
            a.sql("SELECT key AS df_key, time AS df_time, sequence AS df_sequence FROM input")?,
        ),
    ]);
    let report = program(outputs)?
        .analyze(
            &UdfRegistry::new().snapshot(),
            &calc_flow::AnalysisOptions::default(),
        )
        .await?;
    let outputs = report
        .nodes
        .iter()
        .filter(|n| n.path == "outputs.df" || n.path == "outputs.sql")
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 2);
    for node in outputs {
        assert!(node.batch.ordering.temporal_identity.is_none());
        assert!(node.stream.ordering.temporal_identity.is_none());
    }
    Ok(())
}

#[tokio::test]
async fn additional_aliases_retain_the_original_identity() -> Result<()> {
    let a = source("a")?;
    let aliases =
        ["key", "time", "sequence"].map(|name| a.column(name)?.alias(&format!("aaa_{name}")));
    let appended = a.with_columns(aliases.into_iter().collect::<Result<_>>()?)?;
    let selected = appended.select(
        [
            "aaa_key",
            "aaa_time",
            "aaa_sequence",
            "key",
            "time",
            "sequence",
            "value",
        ]
        .map(|name| appended.column(name)?.alias(name))
        .into_iter()
        .collect::<Result<_>>()?,
    )?;
    let outputs = [("appended", appended), ("selected", selected)]
        .into_iter()
        .map(|(name, input)| {
            Ok((
                name.into(),
                input.asof_join(&source("b")?, super::asof_support::spec(2))?,
            ))
        })
        .collect::<Result<_>>()?;
    verify(
        program(outputs)?,
        fixture(),
        BTreeMap::from([
            ("appended".into(), vec![Some(7)]),
            ("selected".into(), vec![Some(7)]),
        ]),
    )
    .await
}
