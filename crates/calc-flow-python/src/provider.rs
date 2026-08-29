use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::json;

use crate::{
    batch::{PyBatch, PythonPayload, rehome_python_payload},
    config::{PythonRoot, json_to_python},
    execution_options::PyProviderContext,
};

#[derive(Clone)]
pub(crate) struct PortContract {
    name: String,
    kind: calc_flow::BatchKind,
}

impl PortContract {
    pub(crate) fn new(name: &str, kind: calc_flow::BatchKind) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }
}

#[derive(Clone)]
enum PythonProviderMode {
    SingleArray,
    Mapping {
        inputs: Vec<PortContract>,
        outputs: Vec<PortContract>,
    },
}

pub(crate) struct PythonOperatorFactory {
    callback: Arc<PythonRoot>,
    provider: String,
    name: String,
    version: String,
    mode: PythonProviderMode,
    accepts_context: bool,
    stream_lifecycle: Option<calc_flow::StreamOperatorLifecycle>,
}

impl PythonOperatorFactory {
    #[cfg(test)]
    pub(crate) fn new(
        callback: Arc<PythonRoot>,
        provider: &str,
        name: &str,
        version: &str,
    ) -> Self {
        Self::new_with_context(callback, provider, name, version, false)
    }

    pub(crate) fn new_with_context(
        callback: Arc<PythonRoot>,
        provider: &str,
        name: &str,
        version: &str,
        accepts_context: bool,
    ) -> Self {
        Self {
            callback,
            provider: provider.into(),
            name: name.into(),
            version: version.into(),
            mode: PythonProviderMode::SingleArray,
            accepts_context,
            stream_lifecycle: None,
        }
    }

    pub(crate) fn new_stateless_stream(
        callback: Arc<PythonRoot>,
        provider: &str,
        name: &str,
        version: &str,
        deterministic: bool,
        replay_safe: bool,
    ) -> Self {
        Self {
            callback,
            provider: provider.into(),
            name: name.into(),
            version: version.into(),
            mode: PythonProviderMode::SingleArray,
            accepts_context: false,
            stream_lifecycle: Some(calc_flow::StreamOperatorLifecycle::Stateless {
                microbatch_invariant: true,
                deterministic,
                replay_safe,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_mapping(
        callback: Arc<PythonRoot>,
        provider: &str,
        name: &str,
        version: &str,
        inputs: Vec<PortContract>,
        outputs: Vec<PortContract>,
    ) -> Self {
        Self::new_mapping_with_context(callback, provider, name, version, inputs, outputs, false)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the factory retains the complete explicit provider registration contract"
    )]
    pub(crate) fn new_mapping_with_context(
        callback: Arc<PythonRoot>,
        provider: &str,
        name: &str,
        version: &str,
        inputs: Vec<PortContract>,
        outputs: Vec<PortContract>,
        accepts_context: bool,
    ) -> Self {
        Self {
            callback,
            provider: provider.into(),
            name: name.into(),
            version: version.into(),
            mode: PythonProviderMode::Mapping { inputs, outputs },
            accepts_context,
            stream_lifecycle: None,
        }
    }
}

impl calc_flow::BatchOperatorFactory for PythonOperatorFactory {
    fn create(
        &self,
        spec: &calc_flow::ExternalOperatorSpec,
        inputs: Vec<calc_flow::Port>,
        outputs: Vec<calc_flow::Port>,
    ) -> calc_flow::Result<Box<dyn calc_flow::BatchOperator>> {
        validate_ports(&self.mode, &inputs, &outputs)?;
        let options_json = encode_provider_options(spec.options())?;
        validate_callback(&self.callback, &options_json).map_err(|message| {
            calc_flow::CalcFlowError::InvalidArgument {
                field: "provider.options".into(),
                message,
            }
        })?;
        Ok(Box::new(PythonOperator {
            callback: Arc::clone(&self.callback),
            provider: self.provider.clone(),
            name: self.name.clone(),
            version: self.version.clone(),
            mode: self.mode.clone(),
            accepts_context: self.accepts_context,
            options: spec.options().clone(),
            options_json,
            inputs,
            outputs,
            stream_lifecycle: self.stream_lifecycle,
        }))
    }
}

impl calc_flow::StreamOperatorFactory for PythonOperatorFactory {
    fn create(
        &self,
        spec: &calc_flow::ExternalOperatorSpec,
        inputs: Vec<calc_flow::Port>,
        outputs: Vec<calc_flow::Port>,
    ) -> calc_flow::Result<Box<dyn calc_flow::StreamOperator>> {
        let lifecycle =
            self.stream_lifecycle
                .ok_or_else(|| calc_flow::CalcFlowError::InvalidArgument {
                    field: "provider.lifecycle".into(),
                    message: "Python provider has no proven stateless stream lifecycle".into(),
                })?;
        validate_ports(&self.mode, &inputs, &outputs)?;
        let options_json = encode_provider_options(spec.options())?;
        validate_stream_callback(&self.callback, &options_json).map_err(|message| {
            calc_flow::CalcFlowError::InvalidArgument {
                field: "provider.options".into(),
                message,
            }
        })?;
        Ok(Box::new(PythonOperator {
            callback: Arc::clone(&self.callback),
            provider: self.provider.clone(),
            name: self.name.clone(),
            version: self.version.clone(),
            mode: self.mode.clone(),
            accepts_context: false,
            options: spec.options().clone(),
            options_json,
            inputs,
            outputs,
            stream_lifecycle: Some(lifecycle),
        }))
    }
}

fn validate_ports(
    mode: &PythonProviderMode,
    inputs: &[calc_flow::Port],
    outputs: &[calc_flow::Port],
) -> calc_flow::Result<()> {
    let error = match mode {
        PythonProviderMode::SingleArray => {
            let expected_inputs = [PortContract::new("input", calc_flow::BatchKind::Array)];
            let expected_outputs = [PortContract::new("output", calc_flow::BatchKind::Array)];
            (!ports_match(inputs, &expected_inputs) || !ports_match(outputs, &expected_outputs))
                .then_some("Python array providers require one required array input named input and one required array output named output")
        }
        PythonProviderMode::Mapping {
            inputs: expected_inputs,
            outputs: expected_outputs,
        } => (!ports_match(inputs, expected_inputs) || !ports_match(outputs, expected_outputs))
            .then_some("Python mapping provider ports do not match the registered contract"),
    };
    error.map_or(Ok(()), |message| {
        Err(calc_flow::CalcFlowError::InvalidArgument {
            field: "provider.ports".into(),
            message: message.into(),
        })
    })
}

fn ports_match(ports: &[calc_flow::Port], expected: &[PortContract]) -> bool {
    ports.len() == expected.len()
        && ports.iter().zip(expected).all(|(port, contract)| {
            port.name() == contract.name
                && port.kind() == contract.kind
                && port.required()
                && port.schema().is_none()
        })
}

fn encode_provider_options(options: &calc_flow::JsonMap) -> calc_flow::Result<String> {
    serde_json::to_string(options).map_err(|source| calc_flow::CalcFlowError::Format {
        message: source.to_string(),
    })
}

fn validate_callback(callback: &PythonRoot, options_json: &str) -> Result<(), String> {
    Python::attach(|py| {
        let callback = callback.object().bind(py);
        if !callback
            .hasattr(pyo3::intern!(py, "validate"))
            .map_err(|error| error.to_string())?
        {
            return Ok(());
        }
        let options = json_to_python(py, options_json).map_err(|error| error.to_string())?;
        callback
            .call_method1(pyo3::intern!(py, "validate"), (options,))
            .map(|_| ())
            .map_err(|error| error.to_string())
    })
}

fn validate_stream_callback(callback: &PythonRoot, options_json: &str) -> Result<(), String> {
    Python::attach(|py| {
        let callback = callback.object().bind(py);
        if !callback
            .hasattr(pyo3::intern!(py, "validate_stream"))
            .map_err(|error| error.to_string())?
        {
            return Err("stream provider callback must define validate_stream(options)".into());
        }
        let options = json_to_python(py, options_json).map_err(|error| error.to_string())?;
        callback
            .call_method1(pyo3::intern!(py, "validate_stream"), (options,))
            .map(|_| ())
            .map_err(|error| error.to_string())
    })
}

struct PythonOperator {
    callback: Arc<PythonRoot>,
    provider: String,
    name: String,
    version: String,
    mode: PythonProviderMode,
    accepts_context: bool,
    options: calc_flow::JsonMap,
    options_json: String,
    inputs: Vec<calc_flow::Port>,
    outputs: Vec<calc_flow::Port>,
    stream_lifecycle: Option<calc_flow::StreamOperatorLifecycle>,
}

impl PythonOperator {
    fn provider_error(&self, message: impl Into<String>) -> calc_flow::CalcFlowError {
        calc_flow::CalcFlowError::ExternalProvider {
            provider: self.provider.clone(),
            name: self.name.clone(),
            version: self.version.clone(),
            message: message.into(),
        }
    }
}

impl calc_flow::OperatorMetadata for PythonOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[calc_flow::Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[calc_flow::Port] {
        &self.outputs
    }

    fn configuration(&self) -> calc_flow::JsonMap {
        BTreeMap::from([
            ("name".into(), json!(self.name)),
            ("options".into(), json!(self.options)),
            ("provider".into(), json!(self.provider)),
            ("version".into(), json!(self.version)),
        ])
    }
}

#[async_trait]
impl calc_flow::StreamOperator for PythonOperator {
    fn lifecycle(&self) -> calc_flow::StreamOperatorLifecycle {
        self.stream_lifecycle.unwrap_or_default()
    }

    async fn process_data(
        &mut self,
        ingress: &str,
        batch: calc_flow::Batch,
        context: &calc_flow::StreamOperatorContext<'_>,
        output: &mut dyn calc_flow::StreamCollector,
    ) -> calc_flow::Result<()> {
        context.check_cancelled()?;
        if ingress != "input" {
            return Err(
                self.provider_error(format!("unexpected stream provider ingress {ingress:?}"))
            );
        }
        let callback = Arc::clone(&self.callback);
        let options_json = self.options_json.clone();
        let result = tokio::task::spawn_blocking(move || {
            Python::attach(|py| call_python_operator(py, &callback, &batch, &options_json, None))
                .map_err(|error| error.to_string())
        })
        .await
        .map_err(|_| self.provider_error("stream callback worker terminated"))?
        .map_err(|message| self.provider_error(message))?;
        context.check_cancelled()?;
        self.outputs[0]
            .validate(&result, "provider.output output")
            .map_err(|error| self.provider_error(error.to_string()))?;
        output.emit("output", result).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: calc_flow::EventTime,
        context: &calc_flow::StreamOperatorContext<'_>,
        _output: &mut dyn calc_flow::StreamCollector,
    ) -> calc_flow::Result<()> {
        context.check_cancelled()
    }

    async fn on_end(
        &mut self,
        context: &calc_flow::StreamOperatorContext<'_>,
        _output: &mut dyn calc_flow::StreamCollector,
    ) -> calc_flow::Result<()> {
        context.check_cancelled()
    }
}

#[async_trait]
impl calc_flow::BatchOperator for PythonOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, calc_flow::Batch>,
        context: &calc_flow::BatchOperatorContext<'_>,
    ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
        if let PythonProviderMode::Mapping {
            inputs: input_contracts,
            outputs: output_contracts,
        } = &self.mode
        {
            return Python::attach(|py| {
                call_python_operator_mapping(
                    py,
                    &self.callback,
                    inputs,
                    input_contracts,
                    output_contracts,
                    &self.outputs,
                    &self.options_json,
                    self.accepts_context.then_some(context.run),
                )
            })
            .map_err(|error| self.provider_error(error.to_string()));
        }
        let input = inputs
            .get("input")
            .ok_or_else(|| self.provider_error("missing required input input"))?;
        let payload = input
            .external_payload()
            .map_err(|error| self.provider_error(error.to_string()))?;
        let _payload = payload
            .as_any()
            .downcast_ref::<PythonPayload>()
            .ok_or_else(|| {
                self.provider_error("input payload was not created by the Python host")
            })?;
        let output = Python::attach(|py| {
            call_python_operator(
                py,
                &self.callback,
                input,
                &self.options_json,
                self.accepts_context.then_some(context.run),
            )
        })
        .map_err(|error| self.provider_error(error.to_string()))?;
        let output_payload = output
            .external_payload()
            .map_err(|error| self.provider_error(error.to_string()))?;
        let _output_payload = output_payload
            .as_any()
            .downcast_ref::<PythonPayload>()
            .ok_or_else(|| {
                self.provider_error("callback output payload was not created by the Python host")
            })?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

fn call_python_operator(
    py: Python<'_>,
    callback: &PythonRoot,
    input: &calc_flow::Batch,
    options_json: &str,
    run: Option<&calc_flow::RunContext>,
) -> PyResult<calc_flow::Batch> {
    let input = Py::new(py, PyBatch::from_inner_python(py, input.clone())?)?;
    let options = json_to_python(py, options_json)?;
    let callback = callback.object().bind(py);
    let output = if let Some(run) = run {
        let context = Py::new(py, PyProviderContext::from_run(run))?;
        callback.call1((input, options, context))?
    } else {
        callback.call1((input, options))?
    };
    let output = output.extract::<PyRef<'_, PyBatch>>()?.python_payload()?;
    rehome_python_payload(py, output)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the callback adapter needs both declared and compiled mapping port contracts"
)]
fn call_python_operator_mapping(
    py: Python<'_>,
    callback: &PythonRoot,
    batches: &BTreeMap<String, calc_flow::Batch>,
    input_contracts: &[PortContract],
    output_contracts: &[PortContract],
    output_ports: &[calc_flow::Port],
    options_json: &str,
    run: Option<&calc_flow::RunContext>,
) -> PyResult<BTreeMap<String, calc_flow::Batch>> {
    let inputs = PyDict::new(py);
    for contract in input_contracts {
        let batch = batches.get(&contract.name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "missing required input {}",
                contract.name
            ))
        })?;
        inputs.set_item(
            &contract.name,
            Py::new(py, PyBatch::from_inner_python(py, batch.clone())?)?,
        )?;
    }
    let options = json_to_python(py, options_json)?;
    let callback = callback.object().bind(py);
    let output = if let Some(run) = run {
        let context = Py::new(py, PyProviderContext::from_run(run))?;
        callback.call1((inputs, options, context))?
    } else {
        callback.call1((inputs, options))?
    };
    let mapping_type = py.import("collections.abc")?.getattr("Mapping")?;
    if !output.is_instance(&mapping_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "callback output must be a mapping",
        ));
    }
    let output = py.get_type::<PyDict>().call1((output,))?;
    let output = output.cast::<PyDict>()?;
    let expected = output_contracts
        .iter()
        .map(|contract| contract.name.as_str())
        .collect::<BTreeSet<_>>();
    let actual = output
        .keys()
        .extract::<Vec<String>>()?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let missing = expected
        .iter()
        .filter(|name| !actual.contains(**name))
        .copied()
        .collect::<Vec<_>>();
    let extra = actual
        .iter()
        .filter(|name| !expected.contains(name.as_str()))
        .map(String::as_str)
        .collect::<Vec<_>>();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "callback outputs must exactly match declared outputs; missing {missing:?}, extra {extra:?}"
        )));
    }
    output_contracts
        .iter()
        .zip(output_ports)
        .map(|(contract, port)| {
            let output = output
                .get_item(&contract.name)
                .map_err(|error| mapping_output_error(&contract.name, &error))?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "output {} is missing",
                        contract.name
                    ))
                })?;
            let batch = output
                .extract::<PyRef<'_, PyBatch>>()
                .map_err(|error| mapping_output_error(&contract.name, &error))?
                .clone_inner()
                .map_err(|error| mapping_output_error(&contract.name, &error))?;
            port.validate(&batch, &format!("provider.output {}", contract.name))
                .map_err(|error| mapping_output_error(&contract.name, &error))?;
            if batch.kind() == calc_flow::BatchKind::Array {
                let payload = batch
                    .external_payload()
                    .map_err(|error| mapping_output_error(&contract.name, &error))?;
                if payload.as_any().downcast_ref::<PythonPayload>().is_none() {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "output {} payload was not created by the Python host",
                        contract.name
                    )));
                }
            }
            let batch = rehome_python_payload(py, batch)
                .map_err(|error| mapping_output_error(&contract.name, &error))?;
            Ok((contract.name.clone(), batch))
        })
        .collect()
}

fn mapping_output_error(name: &str, error: &impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("output {name}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use calc_flow::BatchOperatorFactory as _;
    use pyo3::types::PyDict;

    #[derive(Debug)]
    struct ForeignPayload;

    impl calc_flow::ExternalPayload for ForeignPayload {
        fn backend(&self) -> &'static str {
            "python"
        }

        fn len(&self) -> usize {
            1
        }

        fn estimated_bytes(&self) -> usize {
            8
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn array_port(name: &str) -> calc_flow::Port {
        calc_flow::Port::new(name, calc_flow::BatchKind::Array, true, None).unwrap()
    }

    fn table_port(name: &str) -> calc_flow::Port {
        calc_flow::Port::new(name, calc_flow::BatchKind::Table, true, None).unwrap()
    }

    fn python_array_batch(py: Python<'_>) -> calc_flow::Batch {
        py.get_type::<PyBatch>()
            .call_method1(
                "_from_external",
                (
                    PyDict::new(py).into_any(),
                    "python",
                    1,
                    PyDict::new(py).as_any(),
                ),
            )
            .unwrap()
            .extract::<PyRef<'_, PyBatch>>()
            .unwrap()
            .clone_inner()
            .unwrap()
    }

    #[test]
    fn operator_creation_preencodes_provider_options() {
        let options = BTreeMap::from([("nested".into(), json!({"value": [1, 2, 3]}))]);
        assert_eq!(
            encode_provider_options(&options).unwrap(),
            r#"{"nested":{"value":[1,2,3]}}"#
        );
    }

    #[test]
    fn factory_validates_options_and_exact_array_ports() {
        Python::initialize();
        Python::attach(|py| {
            let callback = py
                .eval(
                    c"type('Callback', (), {'validate': lambda self, options: (_ for _ in ()).throw(ValueError('unsafe'))})()",
                    None,
                    None,
                )
                .unwrap()
                .unbind();
            let root = Arc::new(PythonRoot::new(callback));
            let factory = PythonOperatorFactory::new(root, "python", "expression", "1");
            let spec = calc_flow::ExternalOperatorSpec::new(
                "python",
                "expression",
                "1",
                BTreeMap::from([("expression".into(), json!("x"))]),
            )
            .unwrap();
            let error = match factory.create(
                &spec,
                vec![array_port("input")],
                vec![array_port("output")],
            ) {
                Ok(_) => panic!("validation should reject the options"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("unsafe"));

            let callback = py
                .eval(c"lambda batch, options: batch", None, None)
                .unwrap()
                .unbind();
            let root = Arc::new(PythonRoot::new(callback));
            let factory = PythonOperatorFactory::new(root, "python", "identity", "1");
            let spec =
                calc_flow::ExternalOperatorSpec::new("python", "identity", "1", BTreeMap::new())
                    .unwrap();
            let error = match factory.create(
                &spec,
                vec![array_port("wrong")],
                vec![array_port("output")],
            ) {
                Ok(_) => panic!("invalid ports should be rejected"),
                Err(error) => error,
            };
            assert_eq!(
                error.to_string(),
                "invalid provider.ports: Python array providers require one required array input named input and one required array output named output"
            );
        });
    }

    #[tokio::test]
    async fn operator_rejects_payloads_from_other_hosts() {
        Python::initialize();
        let root = Python::attach(|py| {
            Arc::new(PythonRoot::new(
                py.eval(c"lambda batch, options: batch", None, None)
                    .unwrap()
                    .unbind(),
            ))
        });
        let factory = PythonOperatorFactory::new(root, "python", "identity", "1");
        let spec = calc_flow::ExternalOperatorSpec::new("python", "identity", "1", BTreeMap::new())
            .unwrap();
        let mut operator = factory
            .create(&spec, vec![array_port("input")], vec![array_port("output")])
            .unwrap();
        let input = calc_flow::Batch::external(
            Arc::new(ForeignPayload),
            calc_flow::BatchMetadata::default(),
        )
        .unwrap();
        let cancellation = calc_flow::CancellationToken::new();
        let run = calc_flow::RunContext::new(BTreeMap::new(), None, cancellation).unwrap();
        let context = calc_flow::BatchOperatorContext { run: &run };
        let error = operator
            .process(&BTreeMap::from([("input".into(), input)]), &context)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("not created by the Python host"));
    }

    #[test]
    fn callback_round_trips_python_batches_and_copies_options() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"class Callback:\n    def __call__(self, batch, options):\n        options['changed'] = True\n        return batch\ncallback = Callback()",
                None,
                Some(&locals),
            )
            .unwrap();
            let callback = locals.get_item("callback").unwrap().unwrap().unbind();
            let root = PythonRoot::new(callback);
            let metadata = PyDict::new(py);
            let object = PyDict::new(py).into_any().unbind();
            let batch = py
                .get_type::<PyBatch>()
                .call_method1("_from_external", (object, "python", 1, metadata.as_any()))
                .unwrap()
                .extract::<PyRef<'_, PyBatch>>()
                .unwrap()
                .clone_inner()
                .unwrap();
            let options = BTreeMap::from([("value".into(), json!(1))]);
            let options_json = encode_provider_options(&options).unwrap();

            let output = call_python_operator(py, &root, &batch, &options_json, None).unwrap();

            assert_eq!(output.num_rows(), 1);
            assert_eq!(options, BTreeMap::from([("value".into(), json!(1))]));
        });
    }

    #[tokio::test]
    async fn stateless_stream_factory_calls_once_per_batch_and_replays_immutably() {
        Python::initialize();
        let (root, calls) = Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"class Callback:\n    def __init__(self): self.calls = 0\n    def validate_stream(self, options): pass\n    def __call__(self, batch, options):\n        self.calls += 1\n        return batch\ncallback = Callback()",
                None,
                Some(&locals),
            )
            .unwrap();
            let callback = locals.get_item("callback").unwrap().unwrap();
            (
                Arc::new(PythonRoot::new(callback.clone().unbind())),
                callback.unbind(),
            )
        });
        let factory = PythonOperatorFactory::new_stateless_stream(
            root, "python", "identity", "1", true, true,
        );
        let spec = calc_flow::ExternalOperatorSpec::new("python", "identity", "1", BTreeMap::new())
            .unwrap();
        let mut operator = calc_flow::StreamOperatorFactory::create(
            &factory,
            &spec,
            vec![array_port("input")],
            vec![array_port("output")],
        )
        .unwrap();
        let cancellation = calc_flow::CancellationToken::new();
        let job =
            calc_flow::StreamJobContext::new(1, "fingerprint", BTreeMap::new(), None, cancellation);
        let context = calc_flow::StreamOperatorContext::new(&job, "identity", None);
        let mut output = calc_flow::EdgeCollector::new(vec![array_port("output")]);
        let input = Python::attach(python_array_batch);

        operator
            .process_data("input", input.clone(), &context, &mut output)
            .await
            .unwrap();
        operator
            .process_data("input", input, &context, &mut output)
            .await
            .unwrap();

        let batches = output.drain("output");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].as_data().unwrap().num_rows(), 1);
        assert_eq!(batches[1].as_data().unwrap().num_rows(), 1);
        Python::attach(|py| {
            assert_eq!(
                calls
                    .bind(py)
                    .getattr("calls")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                2
            );
        });
    }

    #[tokio::test]
    async fn stateless_stream_factory_fails_closed_on_ports_errors_and_cancellation() {
        Python::initialize();
        let root = Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"class Callback:\n    def validate_stream(self, options): pass\n    def __call__(self, batch, options): raise ValueError('callback failed')\ncallback = Callback()",
                None,
                Some(&locals),
            )
            .unwrap();
            Arc::new(PythonRoot::new(
                locals.get_item("callback").unwrap().unwrap().unbind(),
            ))
        });
        let factory =
            PythonOperatorFactory::new_stateless_stream(root, "python", "failing", "1", true, true);
        let spec = calc_flow::ExternalOperatorSpec::new("python", "failing", "1", BTreeMap::new())
            .unwrap();
        let wrong_ports = calc_flow::StreamOperatorFactory::create(
            &factory,
            &spec,
            vec![array_port("wrong")],
            vec![array_port("output")],
        )
        .err()
        .unwrap();
        assert!(wrong_ports.to_string().contains("provider.ports"));

        let mut operator = calc_flow::StreamOperatorFactory::create(
            &factory,
            &spec,
            vec![array_port("input")],
            vec![array_port("output")],
        )
        .unwrap();
        let cancellation = calc_flow::CancellationToken::new();
        let job = calc_flow::StreamJobContext::new(
            2,
            "fingerprint",
            BTreeMap::new(),
            None,
            cancellation.clone(),
        );
        let context = calc_flow::StreamOperatorContext::new(&job, "failing", None);
        let mut output = calc_flow::EdgeCollector::new(vec![array_port("output")]);
        let input = Python::attach(python_array_batch);
        let error = operator
            .process_data("input", input.clone(), &context, &mut output)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            calc_flow::CalcFlowError::ExternalProvider { .. }
        ));
        assert!(output.drain("output").is_empty());

        cancellation.cancel();
        let error = operator
            .process_data("input", input, &context, &mut output)
            .await
            .unwrap_err();
        assert!(matches!(error, calc_flow::CalcFlowError::Cancelled { .. }));
    }

    #[tokio::test]
    async fn mapping_operator_round_trips_exact_named_batches() {
        Python::initialize();
        let root = Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"class Callback:\n    def __call__(self, inputs, options):\n        assert sorted(inputs) == ['table', 'weights']\n        assert options == {'columns': ['a']}\n        return {'output': inputs['weights']}\ncallback = Callback()",
                None,
                Some(&locals),
            )
            .unwrap();
            Arc::new(PythonRoot::new(
                locals.get_item("callback").unwrap().unwrap().unbind(),
            ))
        });
        let factory = PythonOperatorFactory::new_mapping(
            root,
            "python",
            "table_matmul",
            "1",
            vec![
                PortContract::new("table", calc_flow::BatchKind::Table),
                PortContract::new("weights", calc_flow::BatchKind::Array),
            ],
            vec![PortContract::new("output", calc_flow::BatchKind::Array)],
        );
        let spec = calc_flow::ExternalOperatorSpec::new(
            "python",
            "table_matmul",
            "1",
            BTreeMap::from([("columns".into(), json!(["a"]))]),
        )
        .unwrap();
        let mut operator = factory
            .create(
                &spec,
                vec![table_port("table"), array_port("weights")],
                vec![array_port("output")],
            )
            .unwrap();

        let table = calc_flow::Batch::table(
            vec![datafusion::arrow::record_batch::RecordBatch::new_empty(
                Arc::new(datafusion::arrow::datatypes::Schema::empty()),
            )],
            calc_flow::BatchMetadata::default(),
        )
        .unwrap();
        let weights = Python::attach(python_array_batch);
        let cancellation = calc_flow::CancellationToken::new();
        let run = calc_flow::RunContext::new(BTreeMap::new(), None, cancellation).unwrap();
        let outputs = operator
            .process(
                &BTreeMap::from([("table".into(), table), ("weights".into(), weights.clone())]),
                &calc_flow::BatchOperatorContext { run: &run },
            )
            .await
            .unwrap();

        assert_eq!(outputs["output"].num_rows(), weights.num_rows());
    }

    #[tokio::test]
    async fn mapping_operator_rejects_extra_outputs_by_name() {
        Python::initialize();
        let (root, input) = Python::attach(|py| {
            let callback = py
                .eval(
                    c"lambda inputs, options: {'output': inputs['input'], 'extra': inputs['input']}",
                    None,
                    None,
                )
                .unwrap()
                .unbind();
            let batch = python_array_batch(py);
            (Arc::new(PythonRoot::new(callback)), batch)
        });
        let factory = PythonOperatorFactory::new_mapping(
            root,
            "python",
            "mapping",
            "1",
            vec![PortContract::new("input", calc_flow::BatchKind::Array)],
            vec![PortContract::new("output", calc_flow::BatchKind::Array)],
        );
        let spec = calc_flow::ExternalOperatorSpec::new("python", "mapping", "1", BTreeMap::new())
            .unwrap();
        let mut operator = factory
            .create(&spec, vec![array_port("input")], vec![array_port("output")])
            .unwrap();
        let cancellation = calc_flow::CancellationToken::new();
        let run = calc_flow::RunContext::new(BTreeMap::new(), None, cancellation).unwrap();

        let error = operator
            .process(
                &BTreeMap::from([("input".into(), input)]),
                &calc_flow::BatchOperatorContext { run: &run },
            )
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("external provider python:mapping@1"));
        assert!(message.contains("extra"));
    }

    #[tokio::test]
    async fn mapping_operator_names_non_batch_output() {
        Python::initialize();
        let (root, input) = Python::attach(|py| {
            let callback = py
                .eval(c"lambda inputs, options: {'output': 1}", None, None)
                .unwrap()
                .unbind();
            (Arc::new(PythonRoot::new(callback)), python_array_batch(py))
        });
        let factory = PythonOperatorFactory::new_mapping(
            root,
            "python",
            "mapping",
            "1",
            vec![PortContract::new("input", calc_flow::BatchKind::Array)],
            vec![PortContract::new("output", calc_flow::BatchKind::Array)],
        );
        let spec = calc_flow::ExternalOperatorSpec::new("python", "mapping", "1", BTreeMap::new())
            .unwrap();
        let mut operator = factory
            .create(&spec, vec![array_port("input")], vec![array_port("output")])
            .unwrap();
        let cancellation = calc_flow::CancellationToken::new();
        let run = calc_flow::RunContext::new(BTreeMap::new(), None, cancellation).unwrap();

        let error = operator
            .process(
                &BTreeMap::from([("input".into(), input)]),
                &calc_flow::BatchOperatorContext { run: &run },
            )
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("external provider python:mapping@1"),
            "{message}"
        );
        assert!(message.contains("output output"), "{message}");
    }

    #[test]
    fn mapping_factory_requires_exact_registered_ports() {
        Python::initialize();
        Python::attach(|py| {
            let root = Arc::new(PythonRoot::new(
                py.eval(c"lambda inputs, options: inputs", None, None)
                    .unwrap()
                    .unbind(),
            ));
            let factory = PythonOperatorFactory::new_mapping(
                root,
                "python",
                "mapping",
                "1",
                vec![
                    PortContract::new("table", calc_flow::BatchKind::Table),
                    PortContract::new("weights", calc_flow::BatchKind::Array),
                ],
                vec![PortContract::new("output", calc_flow::BatchKind::Array)],
            );
            let spec =
                calc_flow::ExternalOperatorSpec::new("python", "mapping", "1", BTreeMap::new())
                    .unwrap();

            let error = match factory.create(
                &spec,
                vec![array_port("weights"), table_port("table")],
                vec![array_port("output")],
            ) {
                Ok(_) => panic!("reordered ports should be rejected"),
                Err(error) => error,
            };

            assert_eq!(
                error.to_string(),
                "invalid provider.ports: Python mapping provider ports do not match the registered contract"
            );
        });
    }

    #[tokio::test]
    async fn mapping_operator_names_foreign_host_output() {
        Python::initialize();
        let (root, input) = Python::attach(|py| {
            let locals = PyDict::new(py);
            let foreign = calc_flow::Batch::external(
                Arc::new(ForeignPayload),
                calc_flow::BatchMetadata::default(),
            )
            .unwrap();
            locals
                .set_item(
                    "foreign",
                    Py::new(py, PyBatch::from_inner(foreign)).unwrap(),
                )
                .unwrap();
            let callback = py
                .eval(
                    c"lambda inputs, options: {'output': foreign}",
                    Some(&locals),
                    None,
                )
                .unwrap()
                .unbind();
            (Arc::new(PythonRoot::new(callback)), python_array_batch(py))
        });
        let factory = PythonOperatorFactory::new_mapping(
            root,
            "python",
            "mapping",
            "1",
            vec![PortContract::new("input", calc_flow::BatchKind::Array)],
            vec![PortContract::new("output", calc_flow::BatchKind::Array)],
        );
        let spec = calc_flow::ExternalOperatorSpec::new("python", "mapping", "1", BTreeMap::new())
            .unwrap();
        let mut operator = factory
            .create(&spec, vec![array_port("input")], vec![array_port("output")])
            .unwrap();
        let cancellation = calc_flow::CancellationToken::new();
        let run = calc_flow::RunContext::new(BTreeMap::new(), None, cancellation).unwrap();

        let error = operator
            .process(
                &BTreeMap::from([("input".into(), input)]),
                &calc_flow::BatchOperatorContext { run: &run },
            )
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("external provider python:mapping@1"),
            "{message}"
        );
        assert!(
            message.contains("output output payload was not created by the Python host"),
            "{message}"
        );
    }

    #[test]
    fn mapping_callback_copies_generic_mapping_outputs() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"callback = lambda inputs, options: __import__('collections').UserDict(output=inputs['input'])",
                None,
                Some(&locals),
            )
            .unwrap();
            let root = PythonRoot::new(locals.get_item("callback").unwrap().unwrap().unbind());
            let outputs = call_python_operator_mapping(
                py,
                &root,
                &BTreeMap::from([("input".into(), python_array_batch(py))]),
                &[PortContract::new("input", calc_flow::BatchKind::Array)],
                &[PortContract::new("output", calc_flow::BatchKind::Array)],
                &[array_port("output")],
                "{}",
                None,
            )
            .unwrap();

            assert_eq!(outputs["output"].num_rows(), 1);
        });
    }

    #[test]
    fn mapping_callback_names_wrong_output_kind() {
        Python::initialize();
        Python::attach(|py| {
            let root = PythonRoot::new(
                py.eval(
                    c"lambda inputs, options: {'output': inputs['table']}",
                    None,
                    None,
                )
                .unwrap()
                .unbind(),
            );
            let table = calc_flow::Batch::table(
                vec![datafusion::arrow::record_batch::RecordBatch::new_empty(
                    Arc::new(datafusion::arrow::datatypes::Schema::empty()),
                )],
                calc_flow::BatchMetadata::default(),
            )
            .unwrap();

            let error = call_python_operator_mapping(
                py,
                &root,
                &BTreeMap::from([("table".into(), table)]),
                &[PortContract::new("table", calc_flow::BatchKind::Table)],
                &[PortContract::new("output", calc_flow::BatchKind::Array)],
                &[array_port("output")],
                "{}",
                None,
            )
            .unwrap_err();

            let message = error.to_string();
            assert!(
                message.contains("provider.output output expects a Array batch, received Table"),
                "{message}"
            );
        });
    }

    #[test]
    fn mapping_callback_round_trips_table_outputs() {
        Python::initialize();
        Python::attach(|py| {
            let root = PythonRoot::new(
                py.eval(
                    c"lambda inputs, options: {'output': inputs['table']}",
                    None,
                    None,
                )
                .unwrap()
                .unbind(),
            );
            let table = calc_flow::Batch::table(
                vec![datafusion::arrow::record_batch::RecordBatch::new_empty(
                    Arc::new(datafusion::arrow::datatypes::Schema::empty()),
                )],
                calc_flow::BatchMetadata::default(),
            )
            .unwrap();

            let outputs = call_python_operator_mapping(
                py,
                &root,
                &BTreeMap::from([("table".into(), table)]),
                &[PortContract::new("table", calc_flow::BatchKind::Table)],
                &[PortContract::new("output", calc_flow::BatchKind::Table)],
                &[table_port("output")],
                "{}",
                None,
            )
            .unwrap();

            assert_eq!(outputs["output"].kind(), calc_flow::BatchKind::Table);
        });
    }
}
