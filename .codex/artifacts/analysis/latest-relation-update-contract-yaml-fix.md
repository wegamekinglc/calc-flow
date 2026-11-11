# DAL-513 YAML 导入诊断修正（Finding 1）

仅修正 PR #595 主审唯一 Finding 1。修正源码提交 `d14285a3d78c71f42b78ee1b56c8426b6743ce77`；定向验证提交 `51d0488c83e10e837b22194c642726d3b94cb90f`；基线 `729d7302a4684d9626c8d55cdf8f4acb4baed0a9`。原主审报告 `.codex/artifacts/reviews/latest-relation-update-contract.md` 原样保留并纳入提交，SHA256 `d5716544214fce7e54643d94d054601371e820b1a235f5ef59df94fcd323e4f9`。冻结 API note、前序 analysis 和合同均未修改。

## 行为与直接影响

`project_store.rs` 的 YAML 类型解析失败后，用相同预算解析原始值，检查版本、禁止的 tag/alias，再复用 JSON 的 `validate_update_semantics_value`。未知字符串、布尔、整数、null、数组、对象均返回 `ProjectValidation`，公开 Python 为带 `issues` 的 `ConfigError`；两格式的 path/code/message 相同，path=`graph.update_semantics`，code=`invalid_literal`。

抽取错误分支仅为保持函数复杂度门槛，原 Join 原始诊断保持。回归同时覆盖显式 latest/legacy 往返、legacy 默认字段省略，以及错误更新值与超限、深度、tag/alias、多文档的组合拒绝。原 store 模块的版本、重复键、merge key、include、alias bomb、大小/深度等拒绝继续通过。YAML 解析预算和 strict options 未修改。

改动：`crates/calc-flow/src/project_store.rs`、`crates/calc-flow/tests/project_store.rs`、`crates/calc-flow/tests/stream_join_validation.rs`、`python/tests/test_store.py`。未更改绑定、schema、构建或示例；原独立 core/consumer 和完整传播/恢复证据按前序 analysis 复用。

## RED / GREEN

RED 在基线产品代码加新回归后执行；Rust 1 个预期失败，Python 6 个预期失败，均准确复现 YAML 通用错误而缺少稳定诊断。RED Python 明确复用旧 native；没有把它当作修正验证。

GREEN：Rust store/import **18 通过**，Join 导入直接回归 **1 通过**；本次源码重建的 native 上公开 Python store 模块 **15 通过**。JSON/YAML 六类非法值的实际完整诊断收录于 JSON 的 `public_python.diagnostics`，两格式逐项相等。公开 probe 的解释器、包和扩展路径也收录其中。

初次匹配 native 验证为 13 通过、2 个新增往返断言失败：测试混用了 compact `canonical_json` 与 pretty `export_project_json`。`51d0488c83e10e837b22194c642726d3b94cb90f` 仅将两边统一为同一导出接口的完整字节比较，随后 15 通过；未减弱断言，未修改产品代码。初次结果及日志哈希也保留在 JSON。native 构建提交与验证提交的 core/binding/Python 三个产品源码树及全部打包源文件均逐项相等，验证复用有据。

以下命令均从指定 checkout 根目录执行，日志哈希在同名 JSON；Rust GREEN 的产品和测试字节与修正提交相同。

```bash
CARGO_TARGET_DIR=target/latest/cargo CARGO_BUILD_JOBS=4 cargo test -p calc-flow --test project_store update_semantics_import_diagnostics_match_json_and_yaml --release --config 'profile.release.package.calc-flow.opt-level=0' --config 'profile.release.package.calc-flow.codegen-units=256'
```

1 expected failure: YAML Format instead of ProjectValidation

```bash
PYTHONPATH=target/latest/final-python /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python -m pytest python/tests/test_store.py -q -k import_diagnostics_agree
```

6 expected failures: YAML ConfigError lacks issues

```bash
CARGO_TARGET_DIR=target/latest/cargo CARGO_BUILD_JOBS=4 cargo test -p calc-flow --test project_store --release --config 'profile.release.package.calc-flow.opt-level=0' --config 'profile.release.package.calc-flow.codegen-units=256'
```

18 passed

```bash
CARGO_TARGET_DIR=target/latest/cargo CARGO_BUILD_JOBS=4 cargo test -p calc-flow --test stream_join_validation yaml_join_import_preserves_raw_validation_diagnostics --release --config 'profile.release.package.calc-flow.opt-level=0' --config 'profile.release.package.calc-flow.codegen-units=256'
```

1 passed; 16 filtered

```bash
CARGO_TARGET_DIR=target/latest/cargo CARGO_BUILD_JOBS=4 cargo clippy -p calc-flow --lib --test project_store --test stream_join_validation --release --config 'profile.release.package.calc-flow.opt-level=0' --config 'profile.release.package.calc-flow.codegen-units=256' -- -D warnings
```

passed

native 构建及 Python 定向验证使用以下实际命令，完整 argv、环境、退出状态及日志哈希见同名 JSON：

```bash
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python target/latest/yaml-fix/verify.py
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python target/latest/yaml-fix/verify_followup.py
```

首个脚本执行 Maturin build、解包到全新目录、比对全部 Python/stub 源文件并运行初次 pytest。后续脚本先核对产品源码树及所有打包文件相等，再运行 `pytest python/tests/test_store.py -q` 和公开 API probe；只重跑本次受影响模块，没有重建内容一致的 native。

## 匹配 native 与静态检查

- Python：`/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python`，`3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]`。
- 构建源码：`d14285a3d78c71f42b78ee1b56c8426b6743ce77`；core tree `f5142c563e5ecc6750181ef8ccd4660c5dbd3d12`；binding tree `c2115655d8b102dcf2dbe9f585a203d6e1abce31`；Python tree `59fe42a9a77a19e788917e332701de9f34440909`。
- wheel：`target/latest/yaml-fix/wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl`；SHA256 `688977a6a2f986a3c9eccb53e7ad38fc9869593cef962e17d8a4156453f5c05a`。
- package：`target/latest/yaml-fix/python/calc_flow`；native SHA256 `c9648f102357c3e55d7871e57e031667f91f71890a84ff037afb869fba11b021`；全部 **54** 个打包 Python/stub 源文件与当前 checkout 逐字节相等。
- 受影响 core/store/Join clippy warnings denied、Ruff check/format、cargo fmt、schema 无漂移、diff whitespace 均通过。Lizard 1.26：改动入口 CCN 8，新增错误分支 helper CCN 7，所有新增回归函数不超过 2；无 suppression。

## 剩余事项

本次未重跑前序完整验证、coverage、外部服务和跨平台矩阵；既有有效结果见前序同名 analysis。当前修正尚需 reviewer 仅复核 Finding 1 和直接影响，以及当前 PR head 的 required checks。推送后的一次 CI 快照及原持久唤醒状态在交接评论中提供；未将 pending CI 宣称为通过。issue 保持 `in_progress`，未合并、扩派或启动 DAL-514。
