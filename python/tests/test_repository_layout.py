from pathlib import Path


def test_v2_repository_has_no_python_core_implementation() -> None:
    assert not Path("src/calc_flow").exists()
    assert not Path("tests/calc_flow").exists()
    assert Path("python/calc_flow").is_dir()
    assert Path("crates/calc-flow/src/lib.rs").is_file()
