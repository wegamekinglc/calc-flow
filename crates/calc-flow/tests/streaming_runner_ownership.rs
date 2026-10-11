use std::path::Path;

#[test]
fn one_shot_runner_reuse_ui_is_a_move_error() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = manifest_dir.join("tests/ui/one_shot_runner_reuse.rs");
    let output_dir = manifest_dir.join("../../target/ui-tests");
    std::fs::create_dir_all(&output_dir).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = std::process::Command::new(rustc)
        .arg("--edition=2024")
        .arg("--crate-name=one_shot_runner_reuse")
        .arg(&fixture)
        .arg("--out-dir")
        .arg(output_dir)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "fixture unexpectedly compiled");
    assert!(stderr.contains("error[E0382]"), "{stderr}");
}
