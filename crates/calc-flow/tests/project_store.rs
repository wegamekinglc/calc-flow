use std::{collections::BTreeMap, path::Path, sync::Arc};

use calc_flow::{
    CalcFlowError, DataFusionConfig, FileProjectStore, MAX_JSON_DEPTH, NodeSpec, OperatorSpec,
    PipelineSpec, ProjectSpec, ProjectStore, RunOptions, export_project_json, export_project_yaml,
    import_project_json, import_project_json_with_limit, import_project_yaml,
    import_project_yaml_with_limit,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn project(id: &str, name: &str) -> ProjectSpec {
    ProjectSpec {
        format_version: 2,
        id: id.into(),
        name: name.into(),
        description: "project".into(),
        pipeline: PipelineSpec {
            name: "pipeline".into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            datafusion: DataFusionConfig::default(),
        },
        data_sources: Vec::new(),
        run_options: RunOptions::default(),
    }
}

fn hashed_path(directory: &Path, key: &str) -> std::path::PathBuf {
    directory.join(format!(
        "{}.json",
        hex::encode(Sha256::digest(key.as_bytes()))
    ))
}

fn nested_arrays(depth: usize) -> Value {
    (0..depth).fold(Value::Null, |value, _| Value::Array(vec![value]))
}

#[test]
fn project_json_and_yaml_round_trip_through_strict_project_spec() {
    let value = project("demo", "Demo");
    let json = export_project_json(&value).unwrap();
    let yaml = export_project_yaml(&value).unwrap();

    assert_eq!(import_project_json(json.as_bytes()).unwrap(), value);
    assert_eq!(import_project_yaml(yaml.as_bytes()).unwrap(), value);
    assert!(json.ends_with('\n'));
    assert!(yaml.ends_with('\n'));
}

#[test]
fn canonical_project_json_is_recursive_pretty_sorted_and_exact() {
    let mut value = project("demo", "Demo");
    value.data_sources.push(calc_flow::DataSourceSpec {
        id: "source".into(),
        input: "input".into(),
        format: "inline_json".into(),
        data: json!({"z": {"b": 2, "a": 1}, "a": [{"d": 4, "c": 3}]}),
    });
    let document = export_project_json(&value).unwrap();
    let parsed: Value = serde_json::from_str(&document).unwrap();
    let expected = format!("{}\n", serde_json::to_string_pretty(&parsed).unwrap());
    assert_eq!(document, expected);
    assert!(document.find("\"a\"").unwrap() < document.find("\"z\"").unwrap());
    assert!(document.find("\"c\"").unwrap() < document.find("\"d\"").unwrap());
}

#[test]
fn project_import_caps_bytes_before_parsing_with_inclusive_boundary() {
    let document = export_project_json(&project("demo", "Demo")).unwrap();
    assert_eq!(
        import_project_json_with_limit(document.as_bytes(), document.len())
            .unwrap()
            .id,
        "demo"
    );
    assert!(matches!(
        import_project_json_with_limit(document.as_bytes(), document.len() - 1),
        Err(CalcFlowError::Format { .. })
    ));
    assert_eq!(
        import_project_yaml_with_limit(
            export_project_yaml(&project("yaml", "Yaml"))
                .unwrap()
                .as_bytes(),
            calc_flow::MAX_PROJECT_DOCUMENT_BYTES
        )
        .unwrap()
        .id,
        "yaml"
    );
}

#[test]
fn yaml_import_rejects_v1_unknown_alias_tag_include_bomb_and_multiple_documents() {
    let valid = export_project_yaml(&project("demo", "Demo")).unwrap();
    let v1 = valid.replacen("format_version: 2", "format_version: 1", 1);
    assert!(matches!(
        import_project_yaml(v1.as_bytes()),
        Err(CalcFlowError::UnsupportedVersion {
            expected: 2,
            found: 1
        })
    ));
    let cases = [
        format!("{valid}unknown: true\n"),
        valid.replacen("name: Demo", "name: &shared Demo\ndescription: *shared", 1),
        valid.replacen("id: demo", "id: !evil demo", 1),
        valid.replacen("id: demo", "id: !include outside.yaml", 1),
        format!("{valid}---\n{valid}"),
        format!(
            "bomb: &a [{}]\n{valid}",
            (0..128).map(|_| "*a").collect::<Vec<_>>().join(",")
        ),
    ];
    for document in cases {
        assert!(
            matches!(
                import_project_yaml(document.as_bytes()),
                Err(CalcFlowError::Format { .. })
            ),
            "unexpectedly accepted:\n{document}"
        );
    }

    let deeply_nested = format!("{}deep: {}0{}\n", valid, "[".repeat(40), "]".repeat(40));
    assert!(matches!(
        import_project_yaml(deeply_nested.as_bytes()),
        Err(CalcFlowError::Format { .. })
    ));
}

#[test]
fn project_json_import_rejects_duplicate_keys_at_every_extensible_depth() {
    let documents = [
        r#"{"format_version":2,"id":"demo","id":"other","name":"Demo","pipeline":{"name":"pipeline","nodes":[]}}"#,
        r#"{"format_version":2,"id":"demo","name":"Demo","pipeline":{"name":"pipeline","nodes":[{"id":"node","operator":{"kind":"external","provider":"python","name":"custom","version":"1","options":{"nested":{"value":1,"value":2}}}}]}}"#,
        r#"{"format_version":2,"id":"demo","name":"Demo","pipeline":{"name":"pipeline","nodes":[]},"data_sources":[{"id":"source","input":"input","format":"inline_json","data":{"nested":{"value":1,"value":2}}}]}"#,
    ];
    for document in documents {
        assert!(
            matches!(
                import_project_json(document.as_bytes()),
                Err(CalcFlowError::Format { .. })
            ),
            "duplicate key was accepted in {document}"
        );
    }
}

#[test]
fn project_json_import_enforces_the_full_document_depth_limit() {
    let mut boundary = serde_json::to_value(project("demo", "Demo")).unwrap();
    boundary["data_sources"] = json!([{
        "id": "source",
        "input": "input",
        "format": "inline_json",
        "data": nested_arrays(MAX_JSON_DEPTH - 3),
    }]);
    assert!(import_project_json(&serde_json::to_vec(&boundary).unwrap()).is_ok());

    boundary["data_sources"][0]["data"] = nested_arrays(MAX_JSON_DEPTH - 2);
    assert!(matches!(
        import_project_json(&serde_json::to_vec(&boundary).unwrap()),
        Err(CalcFlowError::Format { .. })
    ));
}

#[test]
fn yaml_import_rejects_duplicate_keys_and_merge_keys() {
    let duplicate = b"format_version: 2\nid: demo\nid: other\nname: Demo\npipeline:\n  name: pipeline\n  nodes: []\n";
    let nested_duplicate = b"format_version: 2\nid: demo\nname: Demo\npipeline:\n  name: pipeline\n  name: other\n  nodes: []\n";
    let merge = b"format_version: 2\nid: demo\nname: Demo\n<<: {description: merged}\npipeline:\n  name: pipeline\n  nodes: []\n";
    for document in [
        duplicate.as_slice(),
        nested_duplicate.as_slice(),
        merge.as_slice(),
    ] {
        assert!(matches!(
            import_project_yaml(document),
            Err(CalcFlowError::Format { .. })
        ));
    }
}

#[test]
fn project_export_preflights_external_json_values_and_wire_document_depth() {
    let mut boundary = project("demo", "Demo");
    boundary.data_sources.push(calc_flow::DataSourceSpec {
        id: "source".into(),
        input: "input".into(),
        format: "inline_json".into(),
        data: nested_arrays(MAX_JSON_DEPTH - 3),
    });
    assert!(export_project_json(&boundary).is_ok());
    boundary.data_sources[0].data = nested_arrays(MAX_JSON_DEPTH - 2);
    assert!(matches!(
        export_project_json(&boundary),
        Err(CalcFlowError::Format { .. })
    ));

    let mut external = project("external", "External");
    external.pipeline.nodes.push(NodeSpec {
        id: "node".into(),
        operator: OperatorSpec::External {
            provider: "python".into(),
            name: "custom".into(),
            version: "1".into(),
            options: BTreeMap::from([("deep".into(), nested_arrays(MAX_JSON_DEPTH + 1))]),
        },
        input_ports: Vec::new(),
        output_ports: Vec::new(),
        position: None,
    });
    assert!(matches!(
        export_project_json(&external),
        Err(CalcFlowError::Format { .. })
    ));
}

#[tokio::test]
async fn project_store_crud_conflicts_missing_and_sorted_listing() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileProjectStore::new(directory.path()).await.unwrap();
    store.create(&project("zulu", "Zulu")).await.unwrap();
    store.create(&project("alpha", "Alpha")).await.unwrap();
    assert!(matches!(
        store.create(&project("alpha", "Again")).await,
        Err(CalcFlowError::Conflict { resource, key }) if resource == "project" && key == "alpha"
    ));
    assert_eq!(
        store
            .list()
            .await
            .unwrap()
            .into_iter()
            .map(|item| item.id)
            .collect::<Vec<_>>(),
        ["alpha", "zulu"]
    );
    assert_eq!(store.get("alpha").await.unwrap().name, "Alpha");

    store.put(&project("alpha", "Updated")).await.unwrap();
    assert_eq!(store.get("alpha").await.unwrap().name, "Updated");
    store.delete("alpha").await.unwrap();
    assert!(matches!(
        store.get("alpha").await,
        Err(CalcFlowError::NotFound { resource, key }) if resource == "project" && key == "alpha"
    ));
    assert!(matches!(
        store.delete("alpha").await,
        Err(CalcFlowError::NotFound { .. })
    ));
}

#[tokio::test]
async fn project_store_hashes_hostile_windows_style_ids_and_leaks_no_temps() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileProjectStore::new(directory.path()).await.unwrap();
    let id = "../../C:\\CON\\project";
    store.create(&project(id, "Hostile")).await.unwrap();

    let entries = std::fs::read_dir(directory.path())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path(), hashed_path(directory.path(), id));
    assert_eq!(store.get(id).await.unwrap().id, id);
}

#[tokio::test]
async fn project_store_rejects_corrupt_non_object_v1_unknown_oversize_and_key_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileProjectStore::new(directory.path()).await.unwrap();
    let path = hashed_path(directory.path(), "demo");
    for invalid in [b"{".as_slice(), b"[]", b"null"] {
        std::fs::write(&path, invalid).unwrap();
        assert!(matches!(
            store.get("demo").await,
            Err(CalcFlowError::Format { .. })
        ));
    }

    let mut raw = serde_json::to_value(project("demo", "Demo")).unwrap();
    raw["format_version"] = json!(1);
    std::fs::write(&path, serde_json::to_vec(&raw).unwrap()).unwrap();
    assert!(matches!(
        store.get("demo").await,
        Err(CalcFlowError::UnsupportedVersion {
            expected: 2,
            found: 1
        })
    ));
    raw["format_version"] = json!(2);
    raw["unknown"] = json!(true);
    std::fs::write(&path, serde_json::to_vec(&raw).unwrap()).unwrap();
    assert!(matches!(
        store.get("demo").await,
        Err(CalcFlowError::Format { .. })
    ));
    std::fs::write(&path, vec![b' '; calc_flow::MAX_PROJECT_DOCUMENT_BYTES + 1]).unwrap();
    assert!(matches!(
        store.get("demo").await,
        Err(CalcFlowError::Format { .. })
    ));

    std::fs::write(
        &path,
        export_project_json(&project("other", "Other")).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        store.get("demo").await,
        Err(CalcFlowError::Format { .. })
    ));
    assert!(matches!(
        store.list().await,
        Err(CalcFlowError::Format { .. })
    ));
}

#[tokio::test]
async fn project_list_reads_only_canonical_json_names_and_surfaces_corruption() {
    let directory = tempfile::tempdir().unwrap();
    let store = FileProjectStore::new(directory.path()).await.unwrap();
    store.create(&project("demo", "Demo")).await.unwrap();
    std::fs::write(directory.path().join("ignored.json"), "not-json").unwrap();
    std::fs::write(directory.path().join(".write.tmp"), "not-json").unwrap();
    assert_eq!(store.list().await.unwrap().len(), 1);

    std::fs::write(hashed_path(directory.path(), "bad"), "{").unwrap();
    assert!(matches!(
        store.list().await,
        Err(CalcFlowError::Format { .. })
    ));
}

#[cfg(unix)]
#[tokio::test]
async fn project_store_rejects_outside_symlink_and_put_replaces_entry() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        outside.path(),
        export_project_json(&project("demo", "Outside")).unwrap(),
    )
    .unwrap();
    let path = hashed_path(directory.path(), "demo");
    symlink(outside.path(), &path).unwrap();
    let store = FileProjectStore::new(directory.path()).await.unwrap();

    assert!(matches!(
        store.get("demo").await,
        Err(CalcFlowError::Format { .. })
    ));
    assert!(matches!(
        store.list().await,
        Err(CalcFlowError::Format { .. })
    ));
    store.put(&project("demo", "Inside")).await.unwrap();
    assert!(
        !std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(store.get("demo").await.unwrap().name, "Inside");
}

#[tokio::test]
async fn concurrent_project_create_is_race_safe_and_puts_remain_parseable() {
    let directory = tempfile::tempdir().unwrap();
    let store = Arc::new(FileProjectStore::new(directory.path()).await.unwrap());
    let creates = (0..24).map(|index| {
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            store
                .create(&project("same", &format!("Name {index}")))
                .await
        })
    });
    let results = futures::future::join_all(creates).await;
    assert_eq!(
        results
            .into_iter()
            .filter(|result| result.as_ref().unwrap().is_ok())
            .count(),
        1
    );

    let puts = (0..24).map(|index| {
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            store
                .put(&project("same", &format!("Put {index}")))
                .await
                .unwrap();
        })
    });
    futures::future::join_all(puts)
        .await
        .into_iter()
        .for_each(|result| result.unwrap());
    assert!(store.get("same").await.unwrap().name.starts_with("Put "));
    assert!(
        serde_json::from_slice::<Value>(
            &std::fs::read(hashed_path(directory.path(), "same")).unwrap()
        )
        .is_ok()
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}
