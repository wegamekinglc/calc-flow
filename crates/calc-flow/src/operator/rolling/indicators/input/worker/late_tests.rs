use super::*;
use crate::{
    BatchMetadata, EdgeBudget, EdgeCollector, JsonMap, OperatorMetadata, StreamJobContext,
    StreamOperator,
};
use datafusion::arrow::{array::StringArray, record_batch::RecordBatch};

fn side_case() -> (RollingOperator, Batch) {
    let (prototype, large) = tests::large_string_case();
    let original = &large.table_payload().unwrap().batches()[0];
    let mut columns = original.columns().to_vec();
    columns[4] = Arc::new(StringArray::from(vec!["context"]));
    let record = RecordBatch::try_new(original.schema(), columns).unwrap();
    let mut spec = prototype.spec.clone();
    spec.late_policy = LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    let operator = RollingOperator::new("features", record.schema(), spec).unwrap();
    let input = Batch::table(vec![record.clone(), record], BatchMetadata::default()).unwrap();
    (operator, input)
}

#[tokio::test]
async fn test_late_envelope_scratch_limit_is_preserved_before_any_emission() {
    let (mut operator, input) = side_case();
    let job = StreamJobContext::new(
        71,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(2)))
        .with_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let source = operator.indicator_state_packet().unwrap();
    let error = operator
        .process_data("input", input.clone(), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("late scratch"), "{error}");
    assert!(output.drain("late").is_empty());
    assert!(Arc::ptr_eq(
        &source,
        &operator.indicator_state_packet().unwrap()
    ));
    assert!(!operator.indicator_late_failed);
    let record = input.table_payload().unwrap().batches()[0].clone();
    let single = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    operator
        .process_data("input", single, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("late").len(), 1);
}

#[tokio::test]
async fn test_owned_worker_identity_is_admitted_before_the_first_poll_copies_it() {
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    let (mut operator, input) = side_case();
    operator
        .set_state_budget(StateBudget::new(8, 64 << 10).unwrap())
        .unwrap();
    let job = StreamJobContext::new(
        72,
        "a".repeat(1 << 20),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let record = input.table_payload().unwrap().batches()[0].clone();
    let input = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let mut future = Box::pin(operator.process_data("input", input, &context, &mut output));
    let mut task = Context::from_waker(Waker::noop());
    let mut result = None;
    let measured = allocation_counter::measure(|| {
        result = Some(future.as_mut().poll(&mut task));
    });
    drop(future);
    assert!(
        measured.bytes_max < 64 << 10,
        "unfunded worker identity copy in callback poll: {measured:?}"
    );
    assert!(
        matches!(result, Some(Poll::Ready(Err(CalcFlowError::Operator { node_id, .. }))) if node_id == "features")
    );
}

struct RejectLate;
#[async_trait::async_trait]
impl crate::StreamCollector for RejectLate {
    async fn emit(&mut self, port: &str, _: Batch) -> Result<()> {
        assert_eq!(port, "late");
        Err(operator_error("collector", "late rejected"))
    }
}

#[tokio::test]
async fn test_late_emit_rejection_keeps_the_entire_mixed_candidate_uncommitted() {
    use datafusion::arrow::array::{TimestampMicrosecondArray, UInt64Array};
    let (mut operator, input) = side_case();
    let first = input.table_payload().unwrap().batches()[0].clone();
    let mut columns = first.columns().to_vec();
    columns[0] = Arc::new(TimestampMicrosecondArray::from(vec![3]).with_timezone("UTC"));
    columns[2] = Arc::new(UInt64Array::from(vec![3]));
    let second = RecordBatch::try_new(first.schema(), columns).unwrap();
    let input = Batch::table(vec![first, second], BatchMetadata::default()).unwrap();
    let job = StreamJobContext::new(
        76,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(2)));
    let source = operator.indicator_state_packet().unwrap();
    let error = operator
        .process_data("input", input, &context, &mut RejectLate)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("late rejected"), "{error}");
    assert!(Arc::ptr_eq(
        &source,
        &operator.indicator_state_packet().unwrap()
    ));
    assert!(operator.state.buffer.is_empty());
    assert_eq!(operator.state.metrics.late_rows, 0);
    assert_eq!(operator.state.next_late_output_sequence, 0);
    assert!(operator.indicator_late_failed);
}

#[derive(Clone, Copy, Debug)]
struct DroppedOutput {
    live_payload_bytes: usize,
    admitted_without_output: bool,
}
struct ObservedBytes {
    data: Vec<u8>,
    trace: Arc<std::sync::Mutex<Option<DroppedOutput>>>,
    resources: super::super::super::resources::Resources,
    fixed_bytes: u64,
}
impl AsRef<[u8]> for ObservedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}
impl Drop for ObservedBytes {
    fn drop(&mut self) {
        let budget = StateBudget::new(8, self.fixed_bytes).unwrap();
        let admitted_without_output = self.resources.check(0, budget, "features").is_ok();
        *self.trace.lock().unwrap() = Some(DroppedOutput {
            live_payload_bytes: self.data.capacity(),
            admitted_without_output,
        });
    }
}

fn observed_late_payload(
    operator: &RollingOperator,
    trace: Arc<std::sync::Mutex<Option<DroppedOutput>>>,
    record: &RecordBatch,
    schema: SchemaRef,
) -> (Batch, usize) {
    use datafusion::arrow::{
        array::ListArray,
        buffer::{Buffer, OffsetBuffer, ScalarBuffer},
    };
    type Bytes = <datafusion::object_store::PutPayload as IntoIterator>::Item;
    let bytes = Bytes::from_owner(ObservedBytes {
        data: vec![b'q'; 4096],
        trace,
        resources: operator.indicator_resources.as_ref().unwrap().clone(),
        fixed_bytes: operator.state.charge.bytes,
    });
    let buffer = Buffer::from(bytes);
    let pointer = buffer.as_ptr() as usize;
    let strings = StringArray::new(
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 4096])),
        buffer,
        None,
    );
    let datafusion::arrow::datatypes::DataType::List(field) = schema.field(4).data_type() else {
        panic!("list input");
    };
    let lists = ListArray::new(
        Arc::clone(field),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1])),
        Arc::new(strings),
        None,
    );
    let mut columns = record.columns()[..4].to_vec();
    columns.push(Arc::new(lists));
    let record = RecordBatch::try_new(schema, columns).unwrap();
    (
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
        pointer,
    )
}

fn output_body_pointer(prepared: &Prepared) -> usize {
    use datafusion::arrow::array::ListArray;
    let record = &prepared.late[0].table_payload().unwrap().batches()[0];
    let list = record
        .column(4)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let strings = list
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    strings.value_data().as_ptr() as usize
}

#[tokio::test]
async fn test_real_prepared_drop_keeps_late_output_funded_until_its_last_owner_drops() {
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    let (prototype, ordinary) = side_case();
    let input_record = &ordinary.table_payload().unwrap().batches()[0];
    let mut fields = input_record.schema().fields()[..4].to_vec();
    fields.push(Arc::new(Field::new(
        "context",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )));
    let schema = Arc::new(Schema::new(fields));
    let operator =
        RollingOperator::new("features", Arc::clone(&schema), prototype.spec.clone()).unwrap();
    let trace = Arc::new(std::sync::Mutex::new(None));
    let (input, pointer) =
        observed_late_payload(&operator, Arc::clone(&trace), input_record, schema);
    let job = StreamJobContext::new(
        77,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", Some(EventTime::from_micros(2)));
    let guard = operator
        .native_checkpoint_guard("features", &job)
        .await
        .unwrap();
    let work = operator
        .indicator_input_work(input, context.input_watermark(), &context, guard)
        .unwrap();
    let identity = Arc::clone(&work.node);
    let prepared = operator
        .native_checkpoint_workers()
        .run_rolling_credited(RollingScope::stream(&job), identity, move |check| {
            work.run(check)
        })
        .await
        .unwrap();
    assert_eq!(prepared.late.len(), 1);
    assert_eq!(output_body_pointer(&prepared), pointer);
    assert!(trace.lock().unwrap().is_none());
    drop(prepared);
    let observed = trace.lock().unwrap().unwrap();
    assert_eq!(observed.live_payload_bytes, 4096);
    assert!(
        !observed.admitted_without_output,
        "actual late output was refunded before its owner dropped: {observed:?}"
    );
    operator
        .indicator_resources
        .as_ref()
        .unwrap()
        .check(
            0,
            StateBudget::new(8, operator.state.charge.bytes).unwrap(),
            "features",
        )
        .unwrap();
}
