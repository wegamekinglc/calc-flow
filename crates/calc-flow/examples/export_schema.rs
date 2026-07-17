use calc_flow::project_json_schema;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(&project_json_schema()?)?);
    Ok(())
}
