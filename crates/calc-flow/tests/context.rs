use std::collections::BTreeMap;

use calc_flow::{CalcFlowError, CancellationToken, RunContext, canonical_json};
use serde_json::json;

#[test]
fn canonical_json_sorts_mapping_keys() {
    assert_eq!(
        canonical_json(&json!({"z": 1, "a": 2})).unwrap(),
        "{\"a\":2,\"z\":1}"
    );
}

#[tokio::test]
async fn node_context_shares_cancellation() {
    let token = CancellationToken::new();
    let context = RunContext::new(BTreeMap::new(), None, token.clone()).unwrap();
    let node = context.for_node("calculate").unwrap();
    token.cancel();
    assert!(matches!(
        node.check_cancelled(),
        Err(CalcFlowError::Cancelled { .. })
    ));
}
