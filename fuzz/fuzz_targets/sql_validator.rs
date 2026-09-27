#![no_main]

use calc_flow::SqlOperator;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 16 * 1024 {
        return;
    }
    if let Ok(query) = std::str::from_utf8(data) {
        let _ = SqlOperator::new("fuzz_sql", query, vec!["input".into()], Vec::new());
    }
});
