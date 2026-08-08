use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, JsonMap,
    LocalStateBackend, OperatorMetadata, StateBackend, StateHandle, StateLineageKey,
    StreamJobContext, StreamOperator, StreamOperatorContext, WindowAggregateOperator, WindowSpec,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const STATE_PIPELINE_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn sha256(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn benchmark_state_handle(
    key: &StateLineageKey,
    operator_id: &str,
    epoch: Epoch,
    segment_id: &str,
    bytes: &[u8],
) -> StateHandle {
    let lineage_hash = sha256(format!(
        "{}\0{}",
        key.pipeline_name(),
        key.pipeline_fingerprint()
    ));
    StateHandle::new(
        operator_id,
        epoch,
        segment_id,
        &format!(
            "committed/{lineage_hash}/{}/{}-{}.arrow",
            sha256(operator_id),
            epoch.as_u64(),
            sha256(segment_id)
        ),
        u64::try_from(bytes.len()).unwrap(),
        &sha256(bytes),
    )
    .unwrap()
}

fn state_backend_io(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let directory = TempDir::new().unwrap();
    let key = StateLineageKey::new("benchmark", STATE_PIPELINE_FINGERPRINT).unwrap();
    let backend = runtime
        .block_on(LocalStateBackend::new(directory.path()))
        .unwrap();
    let lineage = runtime.block_on(backend.open_lineage(&key)).unwrap();
    let incremental_bytes = Arc::new(vec![0x5a; 64 * 1_024]);
    let next_epoch = AtomicU64::new(100);
    c.bench_function("state/incremental_write_64k", |b| {
        b.to_async(&runtime).iter(|| {
            let lineage = lineage.as_ref();
            let epoch = Epoch::new(next_epoch.fetch_add(1, Ordering::Relaxed)).unwrap();
            let segment_id = format!("delta-{:020}", epoch.as_u64());
            let handle =
                benchmark_state_handle(&key, "incremental", epoch, &segment_id, &incremental_bytes);
            let bytes = Arc::clone(&incremental_bytes);
            async move {
                lineage
                    .stage_segment(&handle, bytes.as_slice())
                    .await
                    .unwrap();
                lineage.validate_segment(&handle).await.unwrap();
                lineage.publish_segment(&handle).await.unwrap();
                black_box(handle)
            }
        });
    });

    let restore_bytes = vec![0xa5; 4 * 1_024 * 1_024];
    let restore_handle = benchmark_state_handle(
        &key,
        "restore",
        Epoch::INITIAL,
        "base-restore",
        &restore_bytes,
    );
    runtime.block_on(async {
        lineage
            .stage_segment(&restore_handle, &restore_bytes)
            .await
            .unwrap();
        lineage.validate_segment(&restore_handle).await.unwrap();
        lineage.publish_segment(&restore_handle).await.unwrap();
    });
    c.bench_function("state/full_restore_4m", |b| {
        b.to_async(&runtime)
            .iter(|| async { black_box(lineage.load_segment(&restore_handle).await.unwrap()) });
    });

    let compaction_handles = (0..8)
        .map(|ordinal| {
            let bytes = vec![u8::try_from(ordinal).unwrap(); 64 * 1_024];
            let segment_id = format!("delta-{ordinal:04}");
            let handle = benchmark_state_handle(
                &key,
                "compact",
                Epoch::new(2).unwrap(),
                &segment_id,
                &bytes,
            );
            runtime.block_on(async {
                lineage.stage_segment(&handle, &bytes).await.unwrap();
                lineage.validate_segment(&handle).await.unwrap();
                lineage.publish_segment(&handle).await.unwrap();
            });
            handle
        })
        .collect::<Vec<_>>();
    c.bench_function("state/compaction_read_8x64k", |b| {
        b.to_async(&runtime).iter(|| async {
            let mut restored = Vec::new();
            for handle in &compaction_handles {
                restored.extend(lineage.load_segment(handle).await.unwrap());
            }
            black_box(restored)
        });
    });
}

fn window_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("group", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn window_input(rows: usize) -> Batch {
    let row_count = rows;
    let rows = i64::try_from(row_count).unwrap();
    let record = RecordBatch::try_new(
        window_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![0; row_count])) as ArrayRef,
            Arc::new(Int64Array::from_iter_values(0..rows)),
            Arc::new(Int64Array::from_iter_values(1..=rows)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn window_operator(hopping: bool) -> WindowAggregateOperator {
    let spec = if hopping {
        WindowSpec::hopping(
            "event_time",
            Duration::from_micros(100),
            Duration::from_micros(10),
        )
        .unwrap()
    } else {
        WindowSpec::tumbling("event_time", Duration::from_micros(100)).unwrap()
    }
    .group_by(["group"])
    .unwrap()
    .aggregate(AggregateFunction::Sum, "value", "sum_value")
    .unwrap();
    WindowAggregateOperator::new("window", window_schema(), spec).unwrap()
}

fn window_execution_and_restore(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let job = StreamJobContext::new(
        1,
        STATE_PIPELINE_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let input = window_input(1_024);
    for (name, hopping) in [("tumbling", false), ("hopping", true)] {
        let operator = window_operator(hopping);
        let collector = EdgeCollector::new(operator.output_ports().to_vec());
        let state = tokio::sync::Mutex::new((operator, collector));
        c.bench_function(&format!("window/{name}_1024_rows"), |b| {
            b.to_async(&runtime).iter(|| async {
                let context = StreamOperatorContext::new(&job, "window", None);
                let mut state = state.lock().await;
                let (operator, collector) = &mut *state;
                operator.reset().unwrap();
                operator
                    .process_data("input", input.clone(), &context, collector)
                    .await
                    .unwrap();
                black_box(());
            });
        });
    }

    let mut source = window_operator(false);
    let mut collector = EdgeCollector::new(source.output_ports().to_vec());
    runtime.block_on(async {
        let context = StreamOperatorContext::new(&job, "window", None);
        source
            .process_data("input", input, &context, &mut collector)
            .await
            .unwrap();
    });
    let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();
    let mut target = window_operator(false);
    c.bench_function("state/window_arrow_restore_1024_keys", |b| {
        b.iter(|| {
            target.restore(black_box(&snapshot)).unwrap();
            black_box(());
        });
    });
}

criterion_group!(
    m4_state_window,
    state_backend_io,
    window_execution_and_restore
);
criterion_main!(m4_state_window);
