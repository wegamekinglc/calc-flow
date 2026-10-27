//! Python 14: JSON/YAML/store round trips and execution of the loaded project.
mod support;
use calc_flow::{
    ExecutionOptions, FileProjectStore, ProjectSpec, ProjectStore, ProviderRegistry, UdfRegistry,
    compile_project, export_project_json, export_project_yaml, import_project_json,
    import_project_yaml,
};
use serde_json::json;
use std::collections::BTreeMap;
use support::{ExampleResult, i64_values, ints, table};

#[tokio::main]
async fn main() -> ExampleResult {
    let project: ProjectSpec = serde_json::from_value(json!({
        "format_version":3,"id":"saved-totals","name":"saved-totals","runtime":{"mode":"batch","options":{}},
        "data_sources":[{"id":"numbers","input":"input","format":"inline_json","data":[{"a":1,"b":2},{"a":3,"b":4}]}],
        "graph":{"name":"saved-totals","nodes":[{"id":"calculate","operator":{"kind":"expression","expression":"total = a + b"}}]}
    }))?;
    let canonical = export_project_json(&project)?;
    let from_json = import_project_json(canonical.as_bytes())?;
    let from_yaml = import_project_yaml(export_project_yaml(&project)?.as_bytes())?;
    assert_eq!(export_project_json(&from_json)?, canonical);
    assert_eq!(export_project_json(&from_yaml)?, canonical);
    let root = tempfile::tempdir()?;
    let store = FileProjectStore::new(root.path()).await?;
    store.create(&from_yaml).await?;
    let loaded = store.get(&project.id).await?;
    assert_eq!(export_project_json(&loaded)?, canonical);
    assert_eq!(
        store
            .list()
            .await?
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>(),
        ["saved-totals"]
    );
    let plan = compile_project(
        &loaded,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
    )?;
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                table(vec![("a", ints(&[1, 3])), ("b", ints(&[2, 4]))]),
            )]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(
        i64_values(&[result.outputs["output"].clone()], "total"),
        [Some(3), Some(7)]
    );
    assert_eq!(export_project_json(&project)?, canonical);
    println!("JSON, YAML, and file-store round trips agree");
    support::print(&[result.outputs["output"].clone()])
}
