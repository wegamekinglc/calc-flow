use super::*;
use crate::operator::join::MatchedPair;
use crate::{
    CancellationToken, DataFusionConfig, DataFusionRuntime, EdgeBudget, EventTime, JsonMap,
    StreamJobContext, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Int64Array, ListArray, StringArray},
    buffer::Buffer,
    datatypes::{Field, Int64Type, Schema, SchemaRef},
};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::{AdmittedRow, StoredRow, columnar::RowPayload};

struct Fixture {
    parents: [Arc<RecordBatch>; 2],
    schema: SchemaRef,
    admitted: Vec<AdmittedRow>,
    opposite: Vec<StoredRow>,
    matched: Vec<MatchedPair>,
    charges: Vec<u64>,
}

fn parent(base: i64) -> Arc<RecordBatch> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![base, base + 1, base + 2])),
        Arc::new(StringArray::from_iter_values(
            (0..3).map(|row| format!("padding-{base}-{row}").repeat(256)),
        )),
        Arc::new(Int64Array::from(vec![Some(base), None, Some(base + 2)])),
        Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([
            Some(vec![Some(base), Some(base + 1)]),
            None,
            Some(vec![Some(base + 2)]),
        ])),
    ];
    let fields = arrays
        .iter()
        .enumerate()
        .map(|(index, array)| Field::new(format!("c{index}"), array.data_type().clone(), true))
        .collect::<Vec<_>>();
    Arc::new(RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap())
}

impl Fixture {
    fn new(incoming_left: bool) -> Self {
        let parents = [parent(10), parent(100)];
        let payload = |side: usize, row| RowPayload::Rowed {
            parent: Arc::clone(&parents[side]),
            row,
        };
        let incoming = usize::from(!incoming_left);
        let admitted = [2, 0, 1]
            .into_iter()
            .map(|row| AdmittedRow {
                record: payload(incoming, row),
                event_time: EventTime::from_micros(100),
                row_id: u64::try_from(row).unwrap(),
                retain: true,
            })
            .collect();
        let opposite = [2, 0, 1]
            .into_iter()
            .map(|row| StoredRow {
                record: payload(1 - incoming, row),
                event_time: EventTime::from_micros(100),
                row_id: u64::try_from(row).unwrap(),
                charge: 50_000,
                encoded_key: Arc::new(Vec::new().into()),
            })
            .collect();
        let fields = parents
            .iter()
            .enumerate()
            .flat_map(|(side, record)| {
                record.schema_ref().fields().iter().map(move |field| {
                    field
                        .as_ref()
                        .clone()
                        .with_name(format!("s{side}_{}", field.name()))
                })
            })
            .collect::<Vec<_>>();
        Self {
            parents,
            schema: Arc::new(Schema::new(fields)),
            admitted,
            opposite,
            matched: (0..3)
                .map(|index| MatchedPair {
                    pos: index,
                    opposite_index: index,
                })
                .collect(),
            charges: vec![50_000; 3],
        }
    }

    fn output<'a>(
        &'a self,
        schema: &'a SchemaRef,
        projection: Option<&'a Arc<[usize]>>,
        incoming_left: bool,
    ) -> JoinOutput<'a> {
        JoinOutput {
            schema,
            projection,
            admitted: &self.admitted,
            opposite: &self.opposite,
            matched: &self.matched,
            incoming_is_left: incoming_left,
            operator_id: "projected-join",
            admitted_charges: Some(&self.charges),
        }
    }
}

fn observer(counter: &Arc<AtomicUsize>) -> output_gather::test_support::Hook {
    let counter = Arc::clone(counter);
    Arc::new(move |_, finished| {
        if !finished {
            counter.fetch_add(1, Ordering::SeqCst);
        }
    })
}

fn assert_independent(source: &Fixture, copied: &RecordBatch) {
    let copied = record_buffers(copied);
    for parent in &source.parents {
        for original in record_buffers(parent) {
            for buffer in &copied {
                assert!(!buffers_overlap(&original, buffer));
            }
        }
    }
}

fn record_buffers(record: &RecordBatch) -> Vec<Buffer> {
    record
        .columns()
        .iter()
        .flat_map(|column| data_buffers(&column.to_data()))
        .collect()
}

fn data_buffers(data: &arrow_data::ArrayData) -> Vec<Buffer> {
    data.buffers()
        .iter()
        .cloned()
        .chain(data.nulls().map(|nulls| nulls.buffer().clone()))
        .chain(data.child_data().iter().flat_map(data_buffers))
        .collect()
}

fn buffers_overlap(left: &Buffer, right: &Buffer) -> bool {
    let a = left.as_ptr() as usize;
    let b = right.as_ptr() as usize;
    !left.is_empty() && !right.is_empty() && a < b + right.len() && b < a + left.len()
}

pub(in crate::operator::join) fn serial_columns_and_budgets() {
    retained_state_projects_unused_payloads();
    let counter = Arc::new(AtomicUsize::new(0));
    let _configuration = output_gather::test_support::configure(false, Some(observer(&counter)));
    for incoming_left in [true, false] {
        let fixture = Fixture::new(incoming_left);
        let full = fixture
            .output(&fixture.schema, None, incoming_left)
            .materialize(0..3)
            .unwrap();
        assert_eq!(counter.swap(0, Ordering::SeqCst), 8);
        let columns: Arc<[usize]> = vec![5, 0, 5].into();
        let schema = Arc::new(fixture.schema.project(&columns).unwrap());
        let output = fixture.output(&schema, Some(&columns), incoming_left);
        let actual = output.materialize(0..3).unwrap();
        assert_eq!(actual, full.project(&columns).unwrap());
        assert_eq!(
            counter.swap(0, Ordering::SeqCst),
            3,
            "unselected column kernels ran"
        );
        assert_independent(&fixture, &actual);
        assert_eq!(output.materialize(0..0).unwrap().schema(), schema);
        assert!(output.ranges(EdgeBudget::new(3, 100).unwrap()).is_err());
        counter.store(0, Ordering::SeqCst);
        selected_ranges(&fixture, incoming_left);
        counter.store(0, Ordering::SeqCst);
    }
}

fn retained_state_projects_unused_payloads() {
    let service = crate::runtime::streaming::gather_work::TestService::new(1, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        retained_state_projection().await;
        super::super::state_projection::tests::selected_payload_cases(&service).await;
    });
    drop(executor);
    service.shutdown();
}

async fn retained_state_projection() {
    use crate::{
        JoinStateLimits, JoinTimeBounds, OperatorMetadata, StreamJoinSpec, StreamOperator,
    };
    use datafusion::arrow::{
        array::TimestampMicrosecondArray,
        datatypes::{DataType, TimeUnit},
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Int64, false),
        Field::new("unused", DataType::Utf8, false),
    ]));
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![7])) as ArrayRef,
            Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![42])),
            Arc::new(StringArray::from_iter_values([
                "unused-payload".repeat(1024)
            ])),
        ],
    )
    .unwrap();
    let original = record.clone();
    let input = crate::Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(std::time::Duration::ZERO, std::time::Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
    )
    .unwrap();
    let mut projected = StreamJoinOperator::new(
        "projected-state",
        Arc::clone(&schema),
        Arc::clone(&schema),
        spec.clone(),
    )
    .unwrap();
    projected.set_output_projection(vec![2, 6]).unwrap();
    let mut full =
        StreamJoinOperator::new("projected-state", Arc::clone(&schema), schema, spec).unwrap();
    let job = StreamJobContext::new(
        1,
        "retained-projection",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "projected-state", None);
    let mut collector = crate::EdgeCollector::new(projected.output_ports().to_vec());
    projected
        .process_data("right", input.clone(), &context, &mut collector)
        .await
        .unwrap();
    let mut full_collector = crate::EdgeCollector::new(full.output_ports().to_vec());
    full.process_data("right", input.clone(), &context, &mut full_collector)
        .await
        .unwrap();
    assert_eq!(
        projected.status(),
        full.status(),
        "projection must preserve authentic V1 row/state charges"
    );
    assert_eq!(input.table_payload().unwrap().batches()[0], original);
    let retained = projected
        .state
        .right
        .at(0)
        .record
        .schema_ref()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    let pool = projected
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((projected, full, context));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        retained,
        vec!["key", "time", "value"],
        "unread large payloads must not stay in retained state"
    );
}

fn selected_ranges(fixture: &Fixture, incoming_left: bool) {
    let columns: Arc<[usize]> = vec![0, 4, 6].into();
    let schema = Arc::new(fixture.schema.project(&columns).unwrap());
    let output = fixture.output(&schema, Some(&columns), incoming_left);
    assert_eq!(
        output.ranges(EdgeBudget::new(10, 50).unwrap()).unwrap(),
        vec![0..2, 2..3]
    );
    let columns: Arc<[usize]> = vec![3].into();
    let schema = Arc::new(fixture.schema.project(&columns).unwrap());
    let mut output = fixture.output(&schema, Some(&columns), incoming_left);
    output.admitted_charges = None;
    assert_eq!(
        output.ranges(EdgeBudget::new(3, 1 << 20).unwrap()).unwrap(),
        vec![0..3]
    );
    let actual = output.materialize(0..3).unwrap();
    assert_independent(fixture, &actual);
    let columns: Arc<[usize]> = Vec::new().into();
    let schema = Arc::new(fixture.schema.project(&columns).unwrap());
    let output = fixture.output(&schema, Some(&columns), incoming_left);
    assert_eq!(
        output.ranges(EdgeBudget::new(2, 1).unwrap()).unwrap(),
        vec![0..2, 2..3]
    );
    let actual = output.materialize(0..3).unwrap();
    assert_eq!((actual.num_columns(), actual.num_rows()), (0, 3));
}

pub(in crate::operator::join) fn owned_columns_and_funding() {
    let service = crate::runtime::streaming::gather_work::TestService::new(1, 1).unwrap();
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    executor.block_on(async {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let pool = runtime.incremental_memory_pool();
        let job = StreamJobContext::new(
            1,
            "projected-join",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_gather_owner(service.owner("projected-join".into()));
        let context = StreamOperatorContext::new(&job, "projected-join", None);
        let counter = Arc::new(AtomicUsize::new(0));
        let _configuration = output_gather::test_support::configure(true, Some(observer(&counter)));
        let fixture = Fixture::new(false);
        let columns: Arc<[usize]> = vec![5, 0, 6].into();
        let schema = Arc::new(fixture.schema.project(&columns).unwrap());
        let output = fixture.output(&schema, Some(&columns), false);
        let mut cleanup = None;
        let (actual, funding) =
            output_gather::materialize(&output, 0..3, &runtime, &context, &mut cleanup)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        assert!(pool.reserved() > 0);
        assert!(cleanup.is_none());
        assert_independent(&fixture, &actual);
        let expected = output.materialize(0..3).unwrap();
        assert_eq!(actual, expected);
        drop((actual, funding));
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(executor);
    service.shutdown();
}
