use std::collections::BTreeMap;

use calc_flow::{CalcFlowError, CancellationToken, MAX_JSON_DEPTH, RunContext, canonical_json};
use serde_json::{Value, json};

fn nested_arrays(depth: usize) -> Value {
    (0..depth).fold(Value::Null, |value, _| Value::Array(vec![value]))
}

#[test]
fn canonical_json_sorts_mapping_keys() {
    assert_eq!(
        canonical_json(&json!({"z": 1, "a": 2})).unwrap(),
        "{\"a\":2,\"z\":1}"
    );
}

#[test]
fn canonical_json_enforces_an_inclusive_iterative_depth_limit() {
    assert!(canonical_json(&nested_arrays(MAX_JSON_DEPTH)).is_ok());
    assert!(matches!(
        canonical_json(&nested_arrays(MAX_JSON_DEPTH + 1)),
        Err(CalcFlowError::Format { .. })
    ));
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
