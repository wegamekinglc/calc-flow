use std::{collections::BTreeMap, fs, fs::File, path::Path, time::Instant};

mod support;

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

type Run<T> = support::ExampleResult<T>;
const HISTORY: usize = 4096;
const APPENDS: usize = 64;

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

fn open_session(fixture: &Path) -> Run<StreamSession> {
    let project = import_project_json(&fs::read(fixture.join("project.json"))?)?;
    let compiled = compile_stream_project_graph(
        &project,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &StreamRequirements::default(),
    )?;
    Ok(StreamSession::new(
        compiled,
        &StreamRuntimeConfig::default(),
        BTreeMap::new(),
    )?)
}

fn envelope(table: &RecordBatch, start: usize, count: usize) -> calc_flow::Result<Batch> {
    Batch::table(vec![table.slice(start, count)], BatchMetadata::default())
}

fn watermark(table: &RecordBatch, index: usize) -> Option<EventTime> {
    let times = table
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    Some(EventTime::from_micros(times.value(index)))
}

fn append_samples(
    session: &mut StreamSession,
    input: &str,
    output: &str,
    source: &RecordBatch,
    expected: &RecordBatch,
    verify_only: bool,
) -> Run<Vec<u128>> {
    let mut nanos = Vec::with_capacity(APPENDS);
    for row in HISTORY..HISTORY + APPENDS {
        let batch = envelope(source, row, 1)?;
        let mark = watermark(source, row);
        let pushed = if verify_only {
            session.push(input, batch, mark)?
        } else {
            let start = Instant::now();
            let pushed = session.push(input, batch, mark)?;
            nanos.push(start.elapsed().as_nanos());
            pushed
        };
        verify(&pushed, output, &expected.slice(row, 1))?;
    }
    Ok(nanos)
}

fn run(
    session: &mut StreamSession,
    source: &RecordBatch,
    expected: &RecordBatch,
    verify_only: bool,
) -> Run<Vec<u128>> {
    assert_eq!(source.num_rows(), HISTORY + APPENDS);
    let inputs = session.input_ids();
    let outputs = session.output_ids();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
    let input = inputs[0].to_owned();
    let output = outputs[0].to_owned();
    let warm = session.push(
        &input,
        envelope(source, 0, HISTORY)?,
        watermark(source, HISTORY - 1),
    )?;
    verify(&warm, &output, &expected.slice(0, HISTORY))?;
    let samples = append_samples(session, &input, &output, source, expected, verify_only)?;
    assert!(session.finish()?.is_empty());
    Ok(samples)
}

fn execute(fixture: &Path, verify_only: bool) -> Run<()> {
    let source = read_table(&fixture.join("inputs.arrow"))?;
    let expected = read_table(&fixture.join("expected.arrow"))?;
    let original = read_table(&fixture.join("inputs.arrow"))?;
    let mut session = open_session(fixture)?;
    let nanos = run(&mut session, &source, &expected, verify_only)?;
    assert_eq!(source, original);
    println!(
        "{}",
        json!({"nanoseconds":nanos, "history":HISTORY,"appends":APPENDS,"all_columns_validated":true,"input_unchanged":true,"finished_empty":true})
    );
    Ok(())
}

fn main() -> Run<()> {
    let args = support::arguments()?;
    let directory = args.first().ok_or("fixture directory required")?;
    execute(
        Path::new(directory),
        args.get(1).is_some_and(|arg| arg == "--verify-only"),
    )
}
