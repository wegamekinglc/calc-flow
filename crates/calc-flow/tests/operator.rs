use std::{any::Any, collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken, DataFusionConfig,
    DataFusionRuntime, ExpressionOperator, ExternalOperatorFactory, ExternalOperatorSpec,
    ExternalPayload, JsonMap, Operator, OperatorContext, Port, ProviderRegistry, Result,
    RunContext, SqlOperator, UdfKind, UdfReference,
};
use datafusion::arrow::{
    array::{Array, Int64Array},
    compute::concat_batches,
    datatypes::{DataType, Field},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

#[derive(Debug)]
struct TestArray;

impl ExternalPayload for TestArray {
    fn backend(&self) -> &'static str {
        "test"
    }

    fn len(&self) -> usize {
        1
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn table(columns: Vec<(&'static str, Vec<i64>)>) -> Batch {
    let columns = columns
        .into_iter()
        .map(|(name, values)| (name, Arc::new(Int64Array::from(values)) as Arc<dyn Array>))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_from_iter(columns).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn values(batch: &Batch, name: &str) -> Vec<i64> {
    let table = batch.table_payload().unwrap();
    let concatenated = concat_batches(table.schema(), table.batches()).unwrap();
    concatenated
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .map(Option::unwrap)
        .collect()
}

const INVALID_PORTABLE_IDENTIFIERS: [&str; 7] = [
    "",
    " ",
    "\t",
    "line\nbreak",
    "with/slash",
    "naïve",
    "bad!value",
];

#[test]
fn port_construction_validates_names_and_array_schema_rules() {
    for invalid in ["", " ", "not-valid", "9input"] {
        assert!(matches!(
            Port::new(invalid, BatchKind::Table, true, None),
            Err(CalcFlowError::InvalidArgument { field, .. }) if field == "port.name"
        ));
    }
    assert!(matches!(
        Port::new("input", BatchKind::Array, true, Some(vec![])),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "port.schema"
    ));

    let port = Port::new("input_2", BatchKind::Table, false, None).unwrap();
    assert_eq!(port.name(), "input_2");
    assert_eq!(port.kind(), BatchKind::Table);
    assert!(!port.required());
    assert!(port.schema().is_none());
}

#[test]
fn port_validation_checks_batch_kind_and_exact_arrow_schema() {
    let matching = table(vec![("a", vec![1, 2])]);
    let fields = matching
        .table_payload()
        .unwrap()
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    let port = Port::new("input", BatchKind::Table, true, Some(fields)).unwrap();

    assert!(port.validate(&matching, "node.input").is_ok());
    assert_eq!(
        port.schema().unwrap(),
        matching.table_payload().unwrap().schema()
    );

    let mismatched = Port::new(
        "input",
        BatchKind::Table,
        true,
        Some(vec![Field::new("a", DataType::Utf8, true)]),
    )
    .unwrap();
    assert!(matches!(
        mismatched.validate(&matching, "node.input"),
        Err(CalcFlowError::Compile { message })
            if message.contains("node.input") && message.contains("schema")
    ));

    let array = Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap();
    assert!(matches!(
        port.validate(&array, "node.input"),
        Err(CalcFlowError::Compile { message })
            if message.contains("node.input") && message.contains("Table")
    ));
}

#[test]
fn expression_operator_has_fixed_table_ports_and_one_calculation_mode() {
    let expression = ExpressionOperator::new("calc", "b = a + 1", vec![], None, vec![]).unwrap();
    assert_eq!(expression.name(), "calc");
    assert_eq!(expression.input_ports().len(), 1);
    assert_eq!(expression.input_ports()[0].name(), "input");
    assert_eq!(expression.input_ports()[0].kind(), BatchKind::Table);
    assert!(expression.input_ports()[0].required());
    assert!(expression.input_ports()[0].schema().is_none());
    assert_eq!(expression.output_ports().len(), 1);
    assert_eq!(expression.output_ports()[0].name(), "output");
    assert_eq!(expression.output_ports()[0].kind(), BatchKind::Table);
    assert!(expression.output_ports()[0].required());
    assert!(expression.output_ports()[0].schema().is_none());

    assert!(ExpressionOperator::new("projection", "", vec!["a".into()], None, vec![]).is_ok());
    for invalid in [
        ExpressionOperator::new("bad", "", vec![], None, vec![]),
        ExpressionOperator::new("bad", "a + 1", vec!["a".into()], None, vec![]),
    ] {
        assert!(matches!(
            invalid,
            Err(CalcFlowError::InvalidArgument { field, .. })
                if field == "operator.calculation"
        ));
    }
    assert!(matches!(
        ExpressionOperator::new("", "a + 1", vec![], None, vec![]),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "operator.name"
    ));
}

#[test]
fn sql_operator_aliases_are_unique_required_table_ports() {
    let operator = SqlOperator::new(
        "join",
        "SELECT * FROM left JOIN right USING(id)",
        vec!["left".into(), "right".into()],
        vec![],
    )
    .unwrap();
    assert_eq!(operator.name(), "join");
    assert_eq!(
        operator
            .input_ports()
            .iter()
            .map(Port::name)
            .collect::<Vec<_>>(),
        ["left", "right"]
    );
    assert!(
        operator
            .input_ports()
            .iter()
            .all(|port| port.kind() == BatchKind::Table && port.required())
    );
    assert_eq!(operator.output_ports().len(), 1);
    assert_eq!(operator.output_ports()[0].name(), "output");

    for aliases in [vec![], vec!["left".into(), "left".into()]] {
        assert!(matches!(
            SqlOperator::new("bad", "SELECT 1", aliases, vec![]),
            Err(CalcFlowError::InvalidArgument { field, .. })
                if field == "operator.inputs"
        ));
    }
    assert!(matches!(
        SqlOperator::new("bad", "SELECT 1", vec!["not-valid".into()], vec![]),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "port.name"
    ));
}

#[test]
fn built_in_with_ports_rejects_optional_inputs() {
    let expression = ExpressionOperator::new("calc", "a + 1", vec![], None, vec![]).unwrap();
    assert!(matches!(
        expression.with_ports(
            Port::new("input", BatchKind::Table, false, None).unwrap(),
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        ),
        Err(CalcFlowError::InvalidArgument { field, .. })
            if field == "operator.input_ports"
    ));

    let sql = SqlOperator::new("sql", "SELECT * FROM input", vec!["input".into()], vec![]).unwrap();
    assert!(matches!(
        sql.with_ports(
            vec![Port::new("input", BatchKind::Table, false, None).unwrap()],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        ),
        Err(CalcFlowError::InvalidArgument { field, .. })
            if field == "operator.input_ports"
    ));
}

#[test]
fn operators_reject_invalid_queries_during_construction() {
    for query in [
        "DELETE FROM input",
        "CREATE TABLE bad AS SELECT 1",
        "SELECT 1; SELECT 2",
        "SELECT FROM",
    ] {
        assert!(matches!(
            SqlOperator::new("sql", query, vec!["input".into()], vec![]),
            Err(CalcFlowError::InvalidArgument { field, .. }) if field == "query"
        ));
    }

    for operator in [
        ExpressionOperator::new("calc", "result = (", vec![], None, vec![]),
        ExpressionOperator::new("calc", "", vec!["(".into()], None, vec![]),
        ExpressionOperator::new("calc", "", vec!["value".into()], Some("(".into()), vec![]),
    ] {
        let diagnostic = format!("{operator:?}");
        assert!(
            matches!(
                operator,
                Err(CalcFlowError::InvalidArgument { field, .. }) if field == "query"
            ),
            "{diagnostic}"
        );
    }
}

#[test]
fn sql_operator_configuration_preserves_the_validated_source_query() {
    let query = "  SELECT * FROM input;  ";
    let operator = SqlOperator::new("sql", query, vec!["input".into()], vec![]).unwrap();

    assert_eq!(
        operator.configuration()["query"],
        Value::String(query.into())
    );
}

#[tokio::test]
async fn expression_operator_processes_assignment_with_real_datafusion() {
    let input = table(vec![("a", vec![1, 2])]);
    let original_batches = input.table_payload().unwrap().batches().to_vec();
    let inputs = BTreeMap::from([("input".into(), input)]);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new())
        .unwrap()
        .for_node("calculate")
        .unwrap();
    let context = OperatorContext {
        run: &run,
        datafusion: &runtime,
    };
    let mut operator =
        ExpressionOperator::new("calc", "total = a + 1", vec![], None, vec![]).unwrap();

    let outputs = operator.process(&inputs, &context).await.unwrap();

    assert_eq!(
        outputs.keys().map(String::as_str).collect::<Vec<_>>(),
        ["output"]
    );
    assert_eq!(values(&outputs["output"], "total"), [2, 3]);
    assert_eq!(
        inputs["input"].table_payload().unwrap().batches(),
        original_batches
    );
    assert!(
        inputs["input"]
            .table_payload()
            .unwrap()
            .schema()
            .field_with_name("total")
            .is_err()
    );
    assert_eq!(runtime.metrics()[0].node_id.as_deref(), Some("calculate"));
}

#[tokio::test]
async fn expression_operator_processes_projection_and_filter() {
    let inputs = BTreeMap::from([(
        "input".into(),
        table(vec![("a", vec![1, 2]), ("b", vec![10, 20])]),
    )]);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let context = OperatorContext {
        run: &run,
        datafusion: &runtime,
    };
    let mut operator = ExpressionOperator::new(
        "project",
        "",
        vec!["a".into(), "b * 2 AS doubled".into()],
        Some("a >= 2".into()),
        vec![],
    )
    .unwrap();

    let output = operator.process(&inputs, &context).await.unwrap();

    assert_eq!(values(&output["output"], "a"), [2]);
    assert_eq!(values(&output["output"], "doubled"), [40]);
}

#[tokio::test]
async fn sql_operator_processes_join_with_real_datafusion() {
    let inputs = BTreeMap::from([
        (
            "left_table".into(),
            table(vec![("id", vec![1, 2]), ("value", vec![10, 20])]),
        ),
        (
            "right_table".into(),
            table(vec![("id", vec![1, 2]), ("value", vec![5, 7])]),
        ),
    ]);
    let input_keys = inputs.keys().cloned().collect::<Vec<_>>();
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new())
        .unwrap()
        .for_node("join")
        .unwrap();
    let context = OperatorContext {
        run: &run,
        datafusion: &runtime,
    };
    let mut operator = SqlOperator::new(
        "join",
        "SELECT l.id, l.value + r.value AS total FROM left_table l \
         JOIN right_table r ON l.id = r.id ORDER BY l.id",
        vec!["left_table".into(), "right_table".into()],
        vec![],
    )
    .unwrap();

    let output = operator.process(&inputs, &context).await.unwrap();

    assert_eq!(values(&output["output"], "id"), [1, 2]);
    assert_eq!(values(&output["output"], "total"), [15, 27]);
    assert_eq!(inputs.keys().cloned().collect::<Vec<_>>(), input_keys);
    assert_eq!(runtime.metrics()[0].node_id.as_deref(), Some("join"));
}

#[test]
fn built_in_configuration_and_udf_references_are_data_only() {
    let reference = UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let operator = ExpressionOperator::new(
        "calc",
        "result = score(a)",
        vec![],
        Some("a > 0".into()),
        vec![reference.clone()],
    )
    .unwrap();

    assert_eq!(reference.provider(), "rust");
    assert_eq!(reference.name(), "score");
    assert_eq!(reference.version(), "1");
    assert_eq!(reference.kind(), UdfKind::DataFusionScalar);
    assert_eq!(operator.udf_references(), vec![reference]);
    assert_eq!(
        serde_json::to_value(operator.configuration()).unwrap(),
        json!({
            "expression": "result = score(a)",
            "filter_expression": "a > 0",
            "select": [],
            "udfs": [{
                "kind": "data_fusion_scalar",
                "name": "score",
                "provider": "rust",
                "version": "1"
            }]
        })
    );
}

#[test]
fn stateless_operator_lifecycle_is_object_safe_and_rejects_state() {
    let mut operator: Box<dyn Operator> =
        Box::new(ExpressionOperator::new("calc", "a + 1", vec![], None, vec![]).unwrap());

    assert_eq!(operator.snapshot().unwrap(), Value::Null);
    assert!(matches!(
        operator.restore(&json!({"state": 1})),
        Err(CalcFlowError::Format { message }) if message.contains("stateless")
    ));
    operator.reset().unwrap();
    assert_eq!(operator.snapshot().unwrap(), Value::Null);
}

#[derive(Debug)]
struct PassthroughFactory {
    marker: &'static str,
}

impl ExternalOperatorFactory for PassthroughFactory {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn Operator>> {
        Ok(Box::new(PassthroughOperator {
            name: spec.name().into(),
            marker: self.marker,
            inputs,
            outputs,
        }))
    }
}

#[derive(Debug)]
struct PassthroughOperator {
    name: String,
    marker: &'static str,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

#[async_trait]
impl Operator for PassthroughOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::from([("marker".into(), json!(self.marker))])
    }

    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &OperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let output =
            inputs
                .get("input")
                .cloned()
                .ok_or_else(|| CalcFlowError::InvalidArgument {
                    field: "inputs".into(),
                    message: "missing input".into(),
                })?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

#[test]
fn external_operator_spec_is_strict_data_only_configuration() {
    let spec = ExternalOperatorSpec::new(
        "numpy",
        "expression",
        "1",
        BTreeMap::from([("expression".into(), json!("x + 1"))]),
    )
    .unwrap();
    let value = serde_json::to_value(&spec).unwrap();
    assert_eq!(
        value,
        json!({
            "provider": "numpy",
            "name": "expression",
            "version": "1",
            "options": {"expression": "x + 1"}
        })
    );
    let restored: ExternalOperatorSpec = serde_json::from_value(value).unwrap();
    assert_eq!(restored.provider(), "numpy");
    assert_eq!(restored.name(), "expression");
    assert_eq!(restored.version(), "1");
    assert_eq!(restored.options()["expression"], "x + 1");
    assert!(
        serde_json::from_value::<ExternalOperatorSpec>(json!({
            "provider": "python",
            "name": "callable",
            "version": "1",
            "options": {},
            "callable": "os.system"
        }))
        .is_err()
    );
}

#[test]
fn external_operator_spec_deserialization_rejects_invalid_identity_components() {
    for (field, index) in [("provider", 0), ("name", 1), ("version", 2)] {
        for invalid in INVALID_PORTABLE_IDENTIFIERS {
            let mut identity = ["python", "expression", "1"];
            identity[index] = invalid;
            assert!(matches!(
                ExternalOperatorSpec::new(identity[0], identity[1], identity[2], BTreeMap::new()),
                Err(CalcFlowError::InvalidArgument { field: actual, .. }) if actual == field
            ));

            let mut value = json!({
                "provider": "python",
                "name": "expression",
                "version": "1",
                "options": {}
            });
            value[field] = json!(invalid);

            assert!(
                serde_json::from_value::<ExternalOperatorSpec>(value).is_err(),
                "accepted invalid {field} {invalid:?}"
            );
        }
    }
}

#[test]
fn external_operator_spec_and_registry_round_trip_portable_identities() {
    let value = json!({
        "provider": "python-3",
        "name": "_array.expression",
        "version": "1.2_rc-1",
        "options": {"expression": "x + 1"}
    });
    let spec: ExternalOperatorSpec = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(spec.provider(), "python-3");
    assert_eq!(spec.name(), "_array.expression");
    assert_eq!(spec.version(), "1.2_rc-1");
    assert_eq!(spec.options()["expression"], "x + 1");
    assert_eq!(serde_json::to_value(&spec).unwrap(), value);

    let registry = ProviderRegistry::default();
    let factory: Arc<dyn ExternalOperatorFactory> =
        Arc::new(PassthroughFactory { marker: "portable" });
    registry
        .register(
            "python-3",
            "_array.expression",
            "1.2_rc-1",
            Arc::clone(&factory),
        )
        .unwrap();
    let resolved = registry
        .resolve("python-3", "_array.expression", "1.2_rc-1")
        .unwrap();
    assert!(Arc::ptr_eq(&resolved, &factory));
}

#[test]
fn provider_registry_rejects_invalid_registration_without_replacement() {
    let registry = ProviderRegistry::default();
    let original: Arc<dyn ExternalOperatorFactory> =
        Arc::new(PassthroughFactory { marker: "original" });
    registry
        .register("python", "passthrough", "1", Arc::clone(&original))
        .unwrap();

    for (field, index) in [("provider", 0), ("name", 1), ("version", 2)] {
        for invalid in INVALID_PORTABLE_IDENTIFIERS {
            let mut identity = ["python", "passthrough", "1"];
            identity[index] = invalid;
            let rejected: Arc<dyn ExternalOperatorFactory> =
                Arc::new(PassthroughFactory { marker: "rejected" });

            assert!(matches!(
                registry.register(identity[0], identity[1], identity[2], rejected),
                Err(CalcFlowError::InvalidArgument { field: actual, .. }) if actual == field
            ));
            let resolved = registry.resolve("python", "passthrough", "1").unwrap();
            assert!(Arc::ptr_eq(&resolved, &original));
        }
    }
}

#[test]
fn provider_registry_validates_resolution_before_lookup() {
    let registry = ProviderRegistry::default();

    for (field, index) in [("provider", 0), ("name", 1), ("version", 2)] {
        for invalid in INVALID_PORTABLE_IDENTIFIERS {
            let mut identity = ["python", "passthrough", "1"];
            identity[index] = invalid;

            assert!(matches!(
                registry.resolve(identity[0], identity[1], identity[2]),
                Err(CalcFlowError::InvalidArgument { field: actual, .. }) if actual == field
            ));
        }
    }

    assert!(matches!(
        registry.resolve("python", "unavailable", "1"),
        Err(CalcFlowError::Compile { message })
            if message.contains("python:unavailable@1")
    ));
}

#[tokio::test]
async fn provider_registry_resolves_factory_without_replacing_duplicates() {
    let registry = ProviderRegistry::default();
    let first: Arc<dyn ExternalOperatorFactory> = Arc::new(PassthroughFactory { marker: "first" });
    let duplicate: Arc<dyn ExternalOperatorFactory> = Arc::new(PassthroughFactory {
        marker: "duplicate",
    });
    registry
        .register("python", "passthrough", "1", Arc::clone(&first))
        .unwrap();

    assert!(matches!(
        registry.register("python", "passthrough", "1", duplicate),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "provider"
    ));
    let resolved = registry.resolve("python", "passthrough", "1").unwrap();
    assert!(Arc::ptr_eq(&resolved, &first));
    assert!(matches!(
        registry.resolve("numpy", "expression", "1"),
        Err(CalcFlowError::Compile { message })
            if message.contains("numpy:expression@1")
    ));

    let spec = ExternalOperatorSpec::new("python", "passthrough", "1", BTreeMap::new()).unwrap();
    let ports = || vec![Port::new("input", BatchKind::Table, true, None).unwrap()];
    let mut operator = resolved
        .create(
            &spec,
            ports(),
            vec![Port::new("output", BatchKind::Table, true, None).unwrap()],
        )
        .unwrap();
    assert_eq!(operator.configuration()["marker"], "first");

    let input = table(vec![("a", vec![1, 2])]);
    let inputs = BTreeMap::from([("input".into(), input)]);
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
    let context = OperatorContext {
        run: &run,
        datafusion: &runtime,
    };
    let outputs = operator.process(&inputs, &context).await.unwrap();
    assert_eq!(values(&outputs["output"], "a"), [1, 2]);
}

#[test]
fn provider_registry_is_shareable_across_threads() {
    let registry = Arc::new(ProviderRegistry::default());
    let factory: Arc<dyn ExternalOperatorFactory> =
        Arc::new(PassthroughFactory { marker: "shared" });
    registry
        .register("python", "passthrough", "1", Arc::clone(&factory))
        .unwrap();

    let mut threads = Vec::new();
    for _ in 0..4 {
        let registry = Arc::clone(&registry);
        let expected = Arc::clone(&factory);
        threads.push(std::thread::spawn(move || {
            let resolved = registry.resolve("python", "passthrough", "1").unwrap();
            assert!(Arc::ptr_eq(&resolved, &expected));
        }));
    }
    for thread in threads {
        thread.join().unwrap();
    }
}
