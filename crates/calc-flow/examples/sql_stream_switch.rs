//! Python 29: the same SQL graph produces a batch total or cumulative snapshots.
mod support;
use calc_flow::{ExecutionOptions, PipelineBuilder, UdfRegistry};
use std::collections::BTreeMap;
use support::{ExampleResult, ReplaySource, collect, i64_values, ints, sql, stream_plan, table};

fn declaration() -> calc_flow::Result<PipelineBuilder> {
    PipelineBuilder::new("sum-across-modes")?.add_node(
        "sum",
        sql("sum", "SELECT SUM(x) AS total FROM input", &["input"])?,
    )
}

#[tokio::main]
async fn main() -> ExampleResult {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let engine = match args.as_slice() {
        [] => "both",
        [flag, value] if flag == "--engine" => value.as_str(),
        _ => return Err("usage: sql_stream_switch [--engine sql|streaming|both]".into()),
    };
    if !["sql", "streaming", "both"].contains(&engine) {
        return Err("unknown engine".into());
    }
    if engine != "streaming" {
        let result = declaration()?
            .compile_batch(&UdfRegistry::new().snapshot())?
            .execute(
                BTreeMap::from([("input".into(), table(vec![("x", ints(&[1, 2, 3]))]))]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(
            i64_values(&[result.outputs["output"].clone()], "total"),
            [Some(6)]
        );
        println!("sql: [6]");
    }
    if engine != "sql" {
        let source = ReplaySource::batches(vec![
            table(vec![("x", ints(&[1]))]),
            table(vec![("x", ints(&[2, 3]))]),
        ])?;
        let result = collect(
            stream_plan(declaration()?)?,
            BTreeMap::from([("input".into(), source.binding())]),
        )
        .await?;
        assert_eq!(i64_values(&result["output"], "total"), [Some(1), Some(6)]);
        println!("streaming: [1, 6]");
    }
    Ok(())
}
