#![no_main]

use calc_flow::import_project_json;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    let _ = import_project_json(data);
});
