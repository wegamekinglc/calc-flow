mod support;

use std::collections::{BTreeMap, HashMap};

use calc_flow::{
    BatchMetadata, BatchOperator, CalcFlowError, Checkpoint, DataFusionConfig, DataSourceSpec,
    Edge, NodeSpec, OperatorSpec, PROJECT_FORMAT_VERSION, PipelineBuilder, PipelineSpec,
    PortEndpoint, ProjectSpec, RunOptions, UdfRegistry, canonical_json, export_project_json,
    export_project_yaml, import_project_json, import_project_yaml,
};
use chrono::{TimeZone, Utc};
use proptest::{
    collection::vec,
    prelude::*,
    test_runner::{Config as ProptestConfig, RngAlgorithm, RngSeed},
};
use serde_json::{Map, Value};

const PROPERTY_CASES: u32 = 48;
const PROPERTY_SEED: u64 = 0xCA1C_F10A_0000_0014;

fn property_config() -> ProptestConfig {
    ProptestConfig {
        cases: PROPERTY_CASES,
        failure_persistence: None,
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(PROPERTY_SEED),
        ..ProptestConfig::default()
    }
}

fn json_value() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i32>().prop_map(|value| Value::Number(value.into())),
        "[a-zA-Z0-9 _.-]{0,16}".prop_map(Value::String),
    ];
    leaf.prop_recursive(4, 32, 4, |inner| {
        prop_oneof![
            vec(inner.clone(), 0..4).prop_map(Value::Array),
            vec(("[a-z]{1,8}", inner), 0..4)
                .prop_map(|entries| { Value::Object(entries.into_iter().collect::<Map<_, _>>()) }),
        ]
    })
    .boxed()
}

fn json_map() -> BoxedStrategy<BTreeMap<String, Value>> {
    vec(("[a-z]{1,8}", json_value()), 0..5)
        .prop_map(|entries| entries.into_iter().collect())
        .boxed()
}

fn endpoint(node_id: &str, port: &str) -> PortEndpoint {
    PortEndpoint::new(node_id, port).unwrap()
}

fn edge(source: &str, target: &str) -> Edge {
    Edge::new(endpoint(source, "output"), endpoint(target, "input"))
}

fn graph_builder(name: &str, nodes: usize) -> PipelineBuilder {
    (0..nodes).fold(PipelineBuilder::new(name).unwrap(), |builder, index| {
        let node = format!("node_{index}");
        builder
            .add_node(
                &node,
                Box::new(support::TestOperator::transform(
                    &node,
                    support::Action::Pass,
                    std::sync::Arc::default(),
                )) as Box<dyn BatchOperator>,
            )
            .unwrap()
    })
}

fn project(id: String, name: String, description: String, data: Value) -> ProjectSpec {
    ProjectSpec {
        format_version: PROJECT_FORMAT_VERSION,
        id,
        name,
        description,
        pipeline: PipelineSpec {
            name: "property-pipeline".into(),
            nodes: vec![NodeSpec {
                id: "calculate".into(),
                operator: OperatorSpec::Expression {
                    expression: "result = value + 1".into(),
                    select: Vec::new(),
                    filter: None,
                    udfs: Vec::new(),
                },
                input_ports: Vec::new(),
                output_ports: Vec::new(),
                position: None,
            }],
            edges: Vec::new(),
            datafusion: DataFusionConfig::default(),
        },
        data_sources: vec![DataSourceSpec {
            id: "source".into(),
            input: "input".into(),
            format: "inline_json".into(),
            data,
        }],
        run_options: RunOptions::default(),
    }
}

proptest! {
    #![proptest_config(property_config())]

    #[test]
    fn canonical_json_round_trips_recursively_sorted_values(value in json_value()) {
        let document = canonical_json(&value).unwrap();
        let decoded: Value = serde_json::from_str(&document).unwrap();

        prop_assert_eq!(&decoded, &value);
        prop_assert_eq!(canonical_json(&decoded).unwrap(), document);
    }

    #[test]
    fn project_json_and_yaml_round_trip_exactly(
        id in "[a-z][a-z0-9_-]{0,11}",
        name in "[a-zA-Z0-9 _.-]{1,16}",
        description in "[a-zA-Z0-9 _.-]{0,24}",
        data in json_value(),
    ) {
        let project = project(id, name, description, data);
        let json = export_project_json(&project).unwrap();
        let yaml = export_project_yaml(&project).unwrap();

        prop_assert!(json.ends_with('\n'));
        prop_assert!(yaml.ends_with('\n'));
        prop_assert_eq!(import_project_json(json.as_bytes()).unwrap(), project.clone());
        prop_assert_eq!(import_project_yaml(yaml.as_bytes()).unwrap(), project.clone());
        prop_assert_eq!(export_project_json(&project).unwrap(), json);
        prop_assert_eq!(export_project_yaml(&project).unwrap(), yaml);
    }

    #[test]
    fn generated_acyclic_graphs_compile(
        node_count in 1_usize..9,
        connect_forward in vec(any::<bool>(), 8),
    ) {
        let mut builder = graph_builder("generated-dag", node_count);
        let mut expected_edges = Vec::new();
        for (index, should_connect) in connect_forward
            .iter()
            .enumerate()
            .take(node_count.saturating_sub(1))
        {
            if *should_connect {
                let source = format!("node_{index}");
                let target = format!("node_{}", index + 1);
                builder = builder.connect(edge(&source, &target)).unwrap();
                expected_edges.push((source, target));
            }
        }
        let plan = builder.compile_batch(&UdfRegistry::new().snapshot()).unwrap();
        let order = plan.topological_order();
        let positions: HashMap<_, _> = order
            .iter()
            .enumerate()
            .map(|(index, node)| (*node, index))
            .collect();

        prop_assert_eq!(order.len(), node_count);
        for (source, target) in expected_edges {
            prop_assert!(positions[source.as_str()] < positions[target.as_str()]);
        }
    }

    #[test]
    fn generated_cycles_are_rejected(node_count in 2_usize..9) {
        let mut builder = graph_builder("generated-cycle", node_count);
        for index in 0..node_count - 1 {
            builder = builder
                .connect(edge(&format!("node_{index}"), &format!("node_{}", index + 1)))
                .unwrap();
        }
        builder = builder
            .connect(edge(&format!("node_{}", node_count - 1), "node_0"))
            .unwrap();

        let error = match builder.compile_batch(&UdfRegistry::new().snapshot()) {
            Ok(_) => return Err(TestCaseError::fail("generated cycle compiled")),
            Err(error) => error,
        };
        let message = error.to_string();
        let is_compile_error = matches!(error, CalcFlowError::Compile { .. });
        prop_assert!(is_compile_error);
        prop_assert!(message.contains("cycle"));
    }

    #[test]
    fn batch_metadata_owns_generated_json_values(
        source in "[a-z]{0,12}",
        sequence in any::<u64>(),
        attributes in json_map(),
    ) {
        let mut caller_attributes = attributes.clone();
        let metadata = BatchMetadata::new(&source, sequence, attributes.clone()).unwrap();
        caller_attributes.insert("caller_mutation".into(), Value::Bool(true));
        let wire = serde_json::to_vec(&metadata).unwrap();
        let decoded: BatchMetadata = serde_json::from_slice(&wire).unwrap();

        prop_assert_eq!(metadata.source(), source);
        prop_assert_eq!(metadata.sequence(), sequence);
        prop_assert_eq!(metadata.attributes(), &attributes);
        prop_assert_eq!(&decoded, &metadata);
        prop_assert_ne!(metadata.attributes(), &caller_attributes);
    }

    #[test]
    fn checkpoints_round_trip_generated_json_state_and_enforce_invariants(
        pipeline in "[a-z][a-z0-9_-]{0,11}",
        fingerprint in "[a-f0-9]{1,24}",
        has_cursor in any::<bool>(),
        cursor_value in json_value(),
        sequence in any::<u64>(),
        state in json_map(),
    ) {
        let cursor = has_cursor.then_some(cursor_value);
        let checkpoint = Checkpoint::new(
            &pipeline,
            &fingerprint,
            cursor,
            sequence,
            state.clone(),
            Utc.with_ymd_and_hms(2026, 7, 14, 0, 0, 0).unwrap(),
        )
        .unwrap();
        let wire = serde_json::to_vec(&checkpoint).unwrap();
        let decoded: Checkpoint = serde_json::from_slice(&wire).unwrap();

        prop_assert_eq!(&decoded, &checkpoint);
        prop_assert_eq!(checkpoint.pipeline_name, pipeline);
        prop_assert_eq!(checkpoint.pipeline_fingerprint, fingerprint);
        prop_assert_eq!(checkpoint.sequence, sequence);
        prop_assert_eq!(checkpoint.state, state);
        prop_assert!(Checkpoint::new(
            "",
            "fingerprint",
            None,
            sequence,
            BTreeMap::new(),
            Utc.with_ymd_and_hms(2026, 7, 14, 0, 0, 0).unwrap(),
        ).is_err());
        prop_assert!(Checkpoint::new(
            "pipeline",
            "fingerprint",
            None,
            sequence,
            BTreeMap::from([(String::new(), Value::Null)]),
            Utc.with_ymd_and_hms(2026, 7, 14, 0, 0, 0).unwrap(),
        ).is_err());
    }
}
