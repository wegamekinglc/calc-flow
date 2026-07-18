use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use pyo3::prelude::*;
use serde_json::json;

use crate::{
    batch::{PyBatch, PythonPayload, rehome_python_payload},
    config::{PythonRoot, json_to_python},
};

pub(crate) struct PythonOperatorFactory {
    callback: Arc<PythonRoot>,
    provider: String,
    name: String,
    version: String,
}

impl PythonOperatorFactory {
    pub(crate) fn new(
        callback: Arc<PythonRoot>,
        provider: &str,
        name: &str,
        version: &str,
    ) -> Self {
        Self {
            callback,
            provider: provider.into(),
            name: name.into(),
            version: version.into(),
        }
    }
}

impl calc_flow::ExternalOperatorFactory for PythonOperatorFactory {
    fn create(
        &self,
        spec: &calc_flow::ExternalOperatorSpec,
        inputs: Vec<calc_flow::Port>,
        outputs: Vec<calc_flow::Port>,
    ) -> calc_flow::Result<Box<dyn calc_flow::Operator>> {
        validate_ports(&inputs, &outputs)?;
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
            options: spec.options().clone(),
            options_json,
            inputs,
            outputs,
        }))
    }
}

fn validate_ports(
    inputs: &[calc_flow::Port],
    outputs: &[calc_flow::Port],
) -> calc_flow::Result<()> {
    let valid = |ports: &[calc_flow::Port], name: &str| {
        ports.len() == 1
            && ports[0].name() == name
            && ports[0].kind() == calc_flow::BatchKind::Array
            && ports[0].required()
            && ports[0].schema().is_none()
    };
    if !valid(inputs, "input") || !valid(outputs, "output") {
        return Err(calc_flow::CalcFlowError::InvalidArgument {
            field: "provider.ports".into(),
            message: "Python array providers require one required array input named input and one required array output named output".into(),
        });
    }
    Ok(())
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

struct PythonOperator {
    callback: Arc<PythonRoot>,
    provider: String,
    name: String,
    version: String,
    options: calc_flow::JsonMap,
    options_json: String,
    inputs: Vec<calc_flow::Port>,
    outputs: Vec<calc_flow::Port>,
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

#[async_trait]
impl calc_flow::Operator for PythonOperator {
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

    async fn process(
        &mut self,
        inputs: &BTreeMap<String, calc_flow::Batch>,
        _context: &calc_flow::OperatorContext<'_>,
    ) -> calc_flow::Result<BTreeMap<String, calc_flow::Batch>> {
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
            call_python_operator(py, &self.callback, input, &self.options_json)
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
) -> PyResult<calc_flow::Batch> {
    let input = Py::new(py, PyBatch::from_inner_python(py, input.clone())?)?;
    let options = json_to_python(py, options_json)?;
    let output = callback.object().bind(py).call1((input, options))?;
    let output = output.extract::<PyRef<'_, PyBatch>>()?.python_payload()?;
    rehome_python_payload(py, output)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use calc_flow::ExternalOperatorFactory as _;
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

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn array_port(name: &str) -> calc_flow::Port {
        calc_flow::Port::new(name, calc_flow::BatchKind::Array, true, None).unwrap()
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
            assert!(error.to_string().contains("Python array providers require"));
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
        let context = calc_flow::OperatorContext { run: &run };
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

            let output = call_python_operator(py, &root, &batch, &options_json).unwrap();

            assert_eq!(output.num_rows(), 1);
            assert_eq!(options, BTreeMap::from([("value".into(), json!(1))]));
        });
    }
}
