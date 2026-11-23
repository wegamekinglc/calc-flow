use std::{collections::BTreeMap, env, error::Error, fs, fs::File, path::Path, time::Instant};

use calc_flow::arrow::{
    array::{Float64Array, TimestampMicrosecondArray},
    compute::concat_batches,
    ipc::reader::FileReader,
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, EventTime, ProviderRegistry, SessionOutput, StreamRequirements,
    StreamRuntimeConfig, StreamSession, UdfRegistry, compile_stream_project_graph,
    import_project_json,
};
use serde_json::json;

type Run<T> = Result<T, Box<dyn Error>>;

fn read_table(path: &Path) -> Run<RecordBatch> {
    let reader = FileReader::try_new(File::open(path)?, None)?;
    let schema = reader.schema();
    let records = reader.collect::<Result<Vec<_>, _>>()?;
    Ok(concat_batches(&schema, &records)?)
}

fn verify(result: &SessionOutput, output: &str, expected: &RecordBatch) -> Run<()> {
    assert_eq!(result.iter_batches().count(), 1);
    assert_eq!(expected.num_columns(), 5);
    let batches = result.batches(output);
    let mut offset = 0;
    for batch in batches {
        for record in batch.table_payload()?.batches() {
            let count = record.num_rows();
            let wanted = expected.slice(offset, count);
            assert_eq!(record.schema(), wanted.schema());
            for col in 0..4 {
                assert_eq!(record.column(col).as_ref(), wanted.column(col).as_ref());
            }
            let actual = record
                .column(4)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let reference = wanted
                .column(4)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for (actual, reference) in actual.iter().zip(reference.iter()) {
                match (actual, reference) {
                    (Some(a), Some(b)) => {
                        assert!((a - b).abs() <= 1e-10 + 1e-10 * b.abs(), "{a} vs {b}")
                    }
                    (None, None) => {}
                    other => panic!("different null placement: {other:?}"),
                }
            }
            offset += count;
        }
    }
    assert_eq!(offset, expected.num_rows());
    Ok(())
}

fn main() -> Run<()> {
    let args = env::args_os().nth(1).ok_or("fixture directory required")?;
    let fixture = Path::new(&args);
    let verify_only = env::args_os()
        .nth(2)
        .is_some_and(|arg| arg == "--verify-only");
    let input_columns = read_table(&fixture.join("inputs.arrow"))?;
    let expected = read_table(&fixture.join("expected.arrow"))?;
    let original = read_table(&fixture.join("inputs.arrow"))?;
    let project = import_project_json(&fs::read(fixture.join("project.json"))?)?;
    let compiled = compile_stream_project_graph(
        &project,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &StreamRequirements::default(),
    )?;
    let mut session =
        StreamSession::new(compiled, &StreamRuntimeConfig::default(), BTreeMap::new())?;
    let inputs = session.input_ids();
    let outputs = session.output_ids();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
    let input = inputs[0].to_owned();
    let output = outputs[0].to_owned();
    let times = input_columns
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    let envelope = |start: usize, count: usize| {
        Batch::table(
            vec![input_columns.slice(start, count)],
            BatchMetadata::default(),
        )
    };
    let watermark = |index: usize| Some(EventTime::from_micros(times.value(index)));
    let history = 4096;
    let sample_count = 64;
    assert_eq!(input_columns.num_rows(), history + sample_count);
    let warm = session.push(&input, envelope(0, history)?, watermark(history - 1))?;
    verify(&warm, &output, &expected.slice(0, history))?;
    let mut nanos = Vec::with_capacity(sample_count);
    for row in history..history + sample_count {
        let batch = envelope(row, 1)?;
        let mark = watermark(row);
        let pushed = if verify_only {
            session.push(&input, batch, mark)?
        } else {
            let start = Instant::now();
            let pushed = session.push(&input, batch, mark)?;
            nanos.push(start.elapsed().as_nanos());
            pushed
        };
        verify(&pushed, &output, &expected.slice(row, 1))?;
    }
    assert!(session.finish()?.is_empty());
    assert_eq!(input_columns, original);
    println!(
        "{}",
        json!({"nanoseconds":nanos, "history":history,"appends":sample_count,"all_columns_validated":true,"input_unchanged":true,"finished_empty":true})
    );
    Ok(())
}
