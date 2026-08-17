fn main() {
    let schema = calc_flow::project_v3_json_schema().expect("generates");
    let pretty = serde_json::to_string_pretty(&schema).expect("pretty");
    let target = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../schemas/project-v3.schema.json"
    );
    std::fs::write(target, pretty).expect("writes");
    println!("schema written to {target}");
}
