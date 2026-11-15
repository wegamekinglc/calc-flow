use calc_flow::{
    BatchKind, ExternalOperatorSpec, JsonMap, Port, StreamOperatorFactory, StreamOperatorLifecycle,
    array::NativeArraySequenceFactory,
};
use serde_json::{Value, json};

fn options(kernel: Value) -> JsonMap {
    JsonMap::from([
        ("ir_version".into(), 1.into()),
        ("input_dtype".into(), "float64".into()),
        ("kernel".into(), kernel),
    ])
}

fn ports() -> (Vec<Port>, Vec<Port>) {
    (
        vec![Port::new("input", BatchKind::Array, true, None).unwrap()],
        vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
    )
}

#[test]
fn test_sequence_factory_accepts_closed_kernel_options_and_proves_stateful_stream() {
    let factory = NativeArraySequenceFactory::default();
    for kernel in [
        json!({"function":"rolling_mean","window":u64::MAX,"numerical_profile":"stable_v2"}),
        json!({"function":"ewma","span":1,"min_periods":3}),
    ] {
        let spec =
            ExternalOperatorSpec::new("native", "array_sequence", "1", options(kernel)).unwrap();
        let (inputs, outputs) = ports();
        let operator = factory.create(&spec, inputs, outputs).unwrap();
        assert_eq!(
            operator.lifecycle(),
            StreamOperatorLifecycle::CheckpointedStateful {
                state_version: 1,
                microbatch_invariant: true,
                deterministic: true,
                replay_safe: true
            }
        );
        assert_eq!(operator.input_ports()[0].name(), "input");
        assert_eq!(operator.output_ports()[0].kind(), BatchKind::Array);
    }
}

#[test]
fn test_sequence_factory_rejects_wrong_identity_ports_and_non_kernel_fields() {
    let factory = NativeArraySequenceFactory::default();
    let valid = options(json!({"function":"ewma","span":3}));
    let wrong = ExternalOperatorSpec::new("other", "array_sequence", "1", valid.clone()).unwrap();
    let (inputs, outputs) = ports();
    assert!(factory.create(&wrong, inputs, outputs).is_err());
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", valid.clone()).unwrap();
    for input in [
        Port::new("input", BatchKind::Array, false, None).unwrap(),
        Port::new("weights", BatchKind::Array, true, None).unwrap(),
        Port::new("input", BatchKind::Table, true, None).unwrap(),
    ] {
        assert!(factory.create(&spec, vec![input], ports().1).is_err());
    }
    for bad in [
        json!({"function":"ewma","span":0}),
        json!({"function":"ewma","span":3,"alpha":0.5}),
        json!({"function":"rolling_mean","window":2,"min_periods":3}),
        json!({"function":"rolling_mean","window":2,"numerical_profile":"rebased_v1"}),
    ] {
        let spec =
            ExternalOperatorSpec::new("native", "array_sequence", "1", options(bad)).unwrap();
        let (inputs, outputs) = ports();
        assert!(factory.create(&spec, inputs, outputs).is_err());
    }
    let mut bad = valid;
    bad.insert("input_dtype".into(), "bool".into());
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", bad).unwrap();
    let (inputs, outputs) = ports();
    assert!(factory.create(&spec, inputs, outputs).is_err());
}

#[test]
fn test_sequence_descriptor_declares_fixed_vector_ports_and_strict_stream_options() {
    let descriptor = NativeArraySequenceFactory::capability_descriptor();
    assert_eq!(descriptor["modes"], json!(["stream"]));
    assert_eq!(descriptor["port_binding"], "fixed");
    assert_eq!(descriptor["state_layouts"], json!([1, 2, 3]));
    assert_eq!(descriptor["partition_contract"], "single_ordered_sequence");
    assert_eq!(
        descriptor["input_ports"],
        json!([{"name":"input","kind":"array","required":true}])
    );
    assert!(descriptor["options_schema"].is_null());
    let schema: Value =
        serde_json::from_str(descriptor["options_json_schema"].as_str().unwrap()).unwrap();
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(
        schema["properties"]["input_dtype"]["enum"],
        json!(["float32", "float64"])
    );
    assert_eq!(
        schema["oneOf"][0]["properties"]["kernel"]["oneOf"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

fn job() -> calc_flow::StreamJobContext {
    calc_flow::StreamJobContext::new(
        91,
        "sequence-factory",
        JsonMap::new(),
        None,
        calc_flow::CancellationToken::new(),
    )
}

fn batch(values: Vec<f64>, sequence: u64) -> calc_flow::Batch {
    let len = values.len();
    calc_flow::Batch::external(
        std::sync::Arc::new(
            calc_flow::array::NativeArray::from_f64(values, vec![len], None).unwrap(),
        ),
        calc_flow::BatchMetadata::new("ordered", sequence, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn test_sequence_factory_process_preserves_arrival_order_empty_and_metadata() {
    let spec = ExternalOperatorSpec::new(
        "native",
        "array_sequence",
        "1",
        options(json!({"function":"ewma","span":3})),
    )
    .unwrap();
    let (inputs, outputs) = ports();
    let mut output = calc_flow::EdgeCollector::new(outputs.clone());
    let mut operator = NativeArraySequenceFactory::default()
        .create(&spec, inputs, outputs)
        .unwrap();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    for input in [
        batch(vec![10.0], 99),
        batch(vec![], 1),
        batch(vec![14.0, 18.0], 0),
    ] {
        operator
            .process_data("input", input, &context, &mut output)
            .await
            .unwrap();
    }
    let messages = output.drain("output");
    let actual: Vec<_> = messages
        .iter()
        .map(|message| message.as_data().unwrap())
        .collect();
    assert_eq!(
        actual
            .iter()
            .map(|batch| batch.metadata().sequence())
            .collect::<Vec<_>>(),
        [99, 1, 0]
    );
    let arrays: Vec<_> = actual
        .iter()
        .map(|batch| {
            batch
                .external_payload()
                .unwrap()
                .as_any()
                .downcast_ref::<calc_flow::array::NativeArray>()
                .unwrap()
        })
        .collect();
    match arrays[0].float_view().unwrap().values() {
        calc_flow::array::FloatValues::Float64(values) => assert_eq!(values, &[10.0]),
        _ => panic!("expected float64 output"),
    }
    assert_eq!(arrays[1].shape(), &[0]);
    match arrays[2].float_view().unwrap().values() {
        calc_flow::array::FloatValues::Float64(values) => assert_eq!(values, &[12.0, 15.0]),
        _ => panic!("expected float64 output"),
    }
}

#[tokio::test]
async fn test_sequence_checkpoint_requires_preparation_and_reuses_unchanged_capture() {
    let spec = ExternalOperatorSpec::new(
        "native",
        "array_sequence",
        "1",
        options(json!({"function":"ewma","span":3})),
    )
    .unwrap();
    let (inputs, outputs) = ports();
    let mut operator = NativeArraySequenceFactory::default()
        .create(&spec, inputs, outputs)
        .unwrap();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    assert!(matches!(operator.checkpoint(calc_flow::Epoch::INITIAL),
        Err(calc_flow::CalcFlowError::CheckpointMismatch { message }) if message.contains("prepare_checkpoint_async")));
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let first = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    assert_eq!(first.segments.len(), 1);
    assert_eq!(first.segments["state"].bytes().len(), 92);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let second = operator
        .checkpoint(calc_flow::Epoch::new(2).unwrap())
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &first.segments["state"].bytes_arc(),
        &second.segments["state"].bytes_arc()
    ));
}

fn ewma_operator() -> Box<dyn calc_flow::StreamOperator> {
    let spec = ExternalOperatorSpec::new(
        "native",
        "array_sequence",
        "1",
        options(json!({"function":"ewma","span":3})),
    )
    .unwrap();
    let (inputs, outputs) = ports();
    NativeArraySequenceFactory::default()
        .create(&spec, inputs, outputs)
        .unwrap()
}

async fn output_values(
    operator: &mut dyn calc_flow::StreamOperator,
    values: Vec<f64>,
    context: &calc_flow::StreamOperatorContext<'_>,
) -> Vec<f64> {
    let mut output = calc_flow::EdgeCollector::new(ports().1);
    operator
        .process_data("input", batch(values, 0), context, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    let output = messages[0].as_data().unwrap();
    let array = output
        .external_payload()
        .unwrap()
        .as_any()
        .downcast_ref::<calc_flow::array::NativeArray>()
        .unwrap();
    match array.float_view().unwrap().values() {
        calc_flow::array::FloatValues::Float64(values) => values.to_vec(),
        _ => panic!("expected f64 output"),
    }
}

#[tokio::test]
async fn test_sequence_restore_reuses_complete_seed_and_rejects_corruption_atomically() {
    let mut operator = ewma_operator();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    assert_eq!(
        output_values(operator.as_mut(), vec![10.0], &context).await,
        [10.0]
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [12.0]
    );
    operator.restore_async(&snapshot, &context).await.unwrap();
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [12.0]
    );
    let mut corrupt = snapshot.clone();
    corrupt.segments.clear();
    assert!(matches!(
        operator.restore_async(&corrupt, &context).await,
        Err(calc_flow::CalcFlowError::CheckpointMismatch { .. })
    ));
    let mut corrupt = snapshot.clone();
    let mut bytes = corrupt.segments["state"].bytes().to_vec();
    bytes[8] = 2;
    corrupt
        .segments
        .insert("state".into(), calc_flow::StateSegment::new(bytes));
    assert!(matches!(
        operator.restore(&corrupt),
        Err(calc_flow::CalcFlowError::CheckpointMismatch { .. })
    ));
    assert_eq!(
        output_values(operator.as_mut(), vec![18.0], &context).await,
        [15.0]
    );
}

#[tokio::test]
async fn test_sequence_reset_discards_recurrence_and_preparation() {
    let mut operator = ewma_operator();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    output_values(operator.as_mut(), vec![10.0], &context).await;
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let retained = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    operator.reset().unwrap();
    assert!(operator.checkpoint(calc_flow::Epoch::INITIAL).is_err());
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [14.0]
    );
    assert_eq!(retained.segments["state"].bytes().len(), 92);
}

struct RejectOutput;
#[async_trait::async_trait]
impl calc_flow::StreamCollector for RejectOutput {
    async fn emit(&mut self, _port: &str, _batch: calc_flow::Batch) -> calc_flow::Result<()> {
        Err(calc_flow::CalcFlowError::Compile {
            message: "rejected output".into(),
        })
    }
}
struct AcceptAndCancel(calc_flow::CancellationToken);
#[async_trait::async_trait]
impl calc_flow::StreamCollector for AcceptAndCancel {
    async fn emit(&mut self, _port: &str, _batch: calc_flow::Batch) -> calc_flow::Result<()> {
        self.0.cancel();
        Ok(())
    }
}

#[tokio::test]
async fn test_sequence_transition_commits_only_accepted_output_and_has_no_post_accept_check() {
    let mut operator = ewma_operator();
    let token = calc_flow::CancellationToken::new();
    let first_job =
        calc_flow::StreamJobContext::new(95, "acceptance", JsonMap::new(), None, token.clone());
    let context = calc_flow::StreamOperatorContext::new(&first_job, "sequence", None);
    output_values(operator.as_mut(), vec![10.0], &context).await;
    assert!(
        operator
            .process_data("input", batch(vec![14.0], 0), &context, &mut RejectOutput)
            .await
            .is_err()
    );
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [12.0]
    );
    operator
        .process_data(
            "input",
            batch(vec![18.0], 0),
            &context,
            &mut AcceptAndCancel(token),
        )
        .await
        .unwrap();
    let retry_job = job();
    let context = calc_flow::StreamOperatorContext::new(&retry_job, "sequence", None);
    assert_eq!(
        output_values(operator.as_mut(), vec![19.0], &context).await,
        [17.0]
    );
}

struct HeldOutput(std::sync::Arc<tokio::sync::Notify>);
#[async_trait::async_trait]
impl calc_flow::StreamCollector for HeldOutput {
    async fn emit(&mut self, _port: &str, _batch: calc_flow::Batch) -> calc_flow::Result<()> {
        self.0.notify_one();
        std::future::pending().await
    }
}

#[tokio::test]
async fn test_sequence_transition_lost_output_awaiter_reuses_operator_without_stale_commit() {
    let mut operator = ewma_operator();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    output_values(operator.as_mut(), vec![10.0], &context).await;
    let entered = std::sync::Arc::new(tokio::sync::Notify::new());
    let mut collector = HeldOutput(entered.clone());
    {
        let pending =
            operator.process_data("input", batch(vec![14.0], 0), &context, &mut collector);
        tokio::pin!(pending);
        tokio::select! {
            ()=entered.notified()=>{},
            result=&mut pending=>panic!("output unexpectedly settled: {result:?}"),
        }
    }
    operator.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [12.0]
    );
    operator.reset().unwrap();
    assert_eq!(
        output_values(operator.as_mut(), vec![18.0], &context).await,
        [18.0]
    );
}

#[tokio::test]
async fn test_sequence_transition_output_budget_rejection_does_not_advance_seed() {
    let mut configured = options(json!({"function":"ewma","span":3}));
    configured.insert("execution".into(), json!({"max_elements":1}));
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", configured).unwrap();
    let (inputs, outputs) = ports();
    let mut operator = NativeArraySequenceFactory::default()
        .create(&spec, inputs, outputs)
        .unwrap();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    output_values(operator.as_mut(), vec![10.0], &context).await;
    let result = operator
        .process_data(
            "input",
            batch(vec![14.0, 18.0], 0),
            &context,
            &mut RejectOutput,
        )
        .await;
    assert!(
        matches!(result,Err(calc_flow::CalcFlowError::InvalidArgument{field,..}) if field=="outputs.output.elements")
    );
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [12.0]
    );
}

fn mean_operator(profile: &str) -> Box<dyn calc_flow::StreamOperator> {
    let mut configured =
        options(json!({"function":"rolling_mean","window":3,"numerical_profile":profile}));
    configured.insert("input_dtype".into(), "float32".into());
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", configured).unwrap();
    NativeArraySequenceFactory::default()
        .create(&spec, ports().0, ports().1)
        .unwrap()
}
fn mean_fixture() -> datafusion::arrow::array::ArrayRef {
    use datafusion::arrow::array::Float32Array;
    let mut values = vec![Some(777.0)];
    values.extend((0..130_u16).map(|index| match index % 11 {
        0 => Some(1e20),
        1 => Some(-1e20),
        2 => None,
        3 => Some(f32::NAN),
        4 => Some(f32::INFINITY),
        5 => Some(f32::NEG_INFINITY),
        _ => Some(f32::from(index)),
    }));
    let array: datafusion::arrow::array::ArrayRef = std::sync::Arc::new(Float32Array::from(values));
    array.slice(1, 130)
}
fn array_bits(array: &calc_flow::array::NativeArray) -> Vec<Option<u64>> {
    let view = array.float_view().unwrap();
    let calc_flow::array::FloatValues::Float64(values) = view.values() else {
        panic!("expected f64")
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            view.validity()
                .is_none_or(|mask| mask.is_valid(index))
                .then(|| value.to_bits())
        })
        .collect()
}
async fn mean_chunks(profile: &str, data: &datafusion::arrow::array::ArrayRef) -> Vec<Option<u64>> {
    let mut operator = mean_operator(profile);
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    let mut collector = calc_flow::EdgeCollector::new(ports().1);
    let mut offset = 0;
    for len in [1, 63, 0, 66] {
        let array =
            calc_flow::array::NativeArray::from_arrow(data.slice(offset, len), vec![len]).unwrap();
        let batch = calc_flow::Batch::external(
            std::sync::Arc::new(array),
            calc_flow::BatchMetadata::default(),
        )
        .unwrap();
        operator
            .process_data("input", batch, &context, &mut collector)
            .await
            .unwrap();
        offset += len;
        if offset == 64 {
            operator.prepare_checkpoint_async(&context).await.unwrap();
            let snapshot = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
            operator = mean_operator(profile);
            operator.restore_async(&snapshot, &context).await.unwrap();
        }
    }
    collector
        .drain("output")
        .iter()
        .flat_map(|message| {
            let array = message
                .as_data()
                .unwrap()
                .external_payload()
                .unwrap()
                .as_any()
                .downcast_ref::<calc_flow::array::NativeArray>()
                .unwrap();
            array_bits(array)
        })
        .collect()
}
#[tokio::test]
async fn test_sequence_mean_profiles_chunks_and_restored_global_cadence_match_eager() {
    use calc_flow::array::{ArrayExecutionOptions, NativeArray, RollingMeanOptions, rolling_mean};
    let data = mean_fixture();
    let array = NativeArray::from_arrow(data.clone(), vec![130]).unwrap();
    for profile in [
        calc_flow::RollingNumericalProfile::StableV1,
        calc_flow::RollingNumericalProfile::StableV2Preview,
    ] {
        let options = RollingMeanOptions::new(3)
            .unwrap()
            .with_numerical_profile(profile);
        let expected = rolling_mean(
            array.float_view().unwrap(),
            options,
            &ArrayExecutionOptions::default(),
        )
        .unwrap();
        let name = match profile {
            calc_flow::RollingNumericalProfile::StableV1 => "stable_v1",
            calc_flow::RollingNumericalProfile::StableV2Preview => "stable_v2",
        };
        assert_eq!(mean_chunks(name, &data).await, array_bits(&expected));
    }
}

#[derive(Debug)]
struct Counterfeit(calc_flow::array::NativeArray);
impl calc_flow::ExternalPayload for Counterfeit {
    fn backend(&self) -> &'static str {
        "native"
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn estimated_bytes(&self) -> usize {
        self.0.estimated_bytes()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        &self.0
    }
}
#[tokio::test]
async fn test_sequence_payload_identity_rank_and_declared_dtype_reject_even_empty() {
    use calc_flow::array::NativeArray;
    let mut operator = ewma_operator();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    let counterfeit = calc_flow::Batch::external(
        std::sync::Arc::new(Counterfeit(
            NativeArray::from_f64(vec![], vec![0], None).unwrap(),
        )),
        calc_flow::BatchMetadata::default(),
    )
    .unwrap();
    let result = operator
        .process_data("input", counterfeit, &context, &mut RejectOutput)
        .await;
    assert!(
        matches!(result,Err(calc_flow::CalcFlowError::InvalidArgument{field,..}) if field=="inputs.input.backend")
    );
    for (array, field) in [
        (
            NativeArray::from_f64(vec![], vec![0, 1], None).unwrap(),
            "inputs.input.shape",
        ),
        (
            NativeArray::from_f32(vec![], vec![0], None).unwrap(),
            "inputs.input.dtype",
        ),
    ] {
        let batch = calc_flow::Batch::external(
            std::sync::Arc::new(array),
            calc_flow::BatchMetadata::default(),
        )
        .unwrap();
        assert!(
            matches!(operator.process_data("input",batch,&context,&mut RejectOutput).await,
            Err(calc_flow::CalcFlowError::InvalidArgument{field:actual,..}) if actual==field)
        );
    }
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [14.0]
    );
}

#[tokio::test]
async fn test_sequence_capture_and_restore_share_one_encoded_allocation_credit() {
    let mut configured = options(json!({"function":"ewma","span":3}));
    configured.insert("execution".into(), json!({"max_allocation_bytes":400}));
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", configured).unwrap();
    let mut operator = NativeArraySequenceFactory::default()
        .create(&spec, ports().0, ports().1)
        .unwrap();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "sequence", None);
    output_values(operator.as_mut(), vec![10.0], &context).await;
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    output_values(operator.as_mut(), vec![14.0], &context).await;
    operator.restore_async(&snapshot, &context).await.unwrap();
    let restored = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &snapshot.segments["state"].bytes_arc(),
        &restored.segments["state"].bytes_arc()
    ));
    assert_eq!(
        output_values(operator.as_mut(), vec![14.0], &context).await,
        [12.0]
    );
}

#[tokio::test]
async fn test_sequence_transition_count_overflow_keeps_field_and_names_actual_ingress() {
    let mut operator = ewma_operator();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "prices_ema", None);
    output_values(operator.as_mut(), vec![10.0], &context).await;
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let mut snapshot = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    let mut bytes = snapshot.segments["state"].bytes().to_vec();
    bytes[68..76].copy_from_slice(&u64::MAX.to_le_bytes());
    snapshot
        .segments
        .insert("state".into(), calc_flow::StateSegment::new(bytes));
    operator.restore_async(&snapshot, &context).await.unwrap();
    let result = operator
        .process_data("input", batch(vec![14.0], 0), &context, &mut RejectOutput)
        .await;
    let Err(calc_flow::CalcFlowError::InvalidArgument { field, message }) = result else {
        panic!("expected count overflow")
    };
    assert_eq!(field, "input.rows");
    assert!(message.contains("native:array_sequence@1"));
    assert!(message.contains("prices_ema"));
    assert!(message.contains("ingress input"));
}

#[tokio::test]
async fn test_sequence_restore_error_names_original_codec_field_and_actual_operator() {
    let mut operator = ewma_operator();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "ema_restore_node", None);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let mut snapshot = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    let mut bytes = snapshot.segments["state"].bytes().to_vec();
    bytes[8] = 2;
    snapshot
        .segments
        .insert("state".into(), calc_flow::StateSegment::new(bytes));
    let Err(calc_flow::CalcFlowError::CheckpointMismatch { message }) =
        operator.restore_async(&snapshot, &context).await
    else {
        panic!("expected a checkpoint layout mismatch")
    };
    assert!(message.contains("state.layout"));
    assert!(message.contains("ema_restore_node"));
}
