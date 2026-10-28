//! Python 15: CSV, JSON Lines, and Parquet to exactly-once Parquet output.
mod support;
use calc_flow::DeliveryGuarantee;
use support::ExampleResult;
#[tokio::main]
async fn main() -> ExampleResult {
    let root = tempfile::tempdir()?;
    let registry = support::registry()?;
    for format in ["csv", "json", "parquet"] {
        let directory = root.path().join(format);
        support::write_orders(&directory, format, false).await?;
        support::execute(
            &directory,
            support::file_source(&directory, format),
            support::file_sink(&directory, "exactly_once"),
            &registry,
            false,
            DeliveryGuarantee::ExactlyOnce,
        )
        .await?;
        support::check_totals(&directory).await?;
        println!("{format} -> Parquet (exactly_once)");
    }
    Ok(())
}
