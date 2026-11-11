# PR #591 独立聚焦主审

评审时间：2026-11-11。Verdict：**Request Changes**。

PR：https://github.com/wegamekinglc/calc-flow/pull/591
分支：`feature/dal-512-relational-frontends` → `main`。
实际 reviewed HEAD：`e12fc8ba6f8d45c83cd462628fde46481fc56ced`。
基线：`775ea9738265597de492b848f8e0763962c7041b`。
复用指定 checkout；开始时干净，未另建分支/worktree，未修改产品、提交 GitHub review、扩派或合并。

## Findings

以下均为本片公开合同或实际回归，须在合并前处理；没有将一般风格建议列为代码 blocker。

1. **[P2] Rust typed 输入丢失 metadata 后拒绝调用方的原始 Batch。**
   `crates/calc-flow/src/program/lower.rs:504`，`native_port`，特别是第 510 行。lowerer 将用户的精确 Arrow schema 先转换为只含 name/type/nullable 的 PortSpec，再重建 Port，丢失 schema 和 field metadata。编译成功，但计划入口的 Port.validate 比较完整 Arrow schema，因此用同一份 schema 构造的 RecordBatch 也被拒绝。最小公开调用是 `table_input("events", schema.clone(), ...)` → 单输出 Program → compile_batch → execute 原 Batch。无 metadata 对照成功；仅加 schema metadata 或 field metadata 均报 `graph input "events" (...) schema mismatch`。这使常见带来源/单位/pandas metadata 的 Arrow 输入无法使用新 Rust 接口。应在内部端口保留必要的精确 schema，或在自有边界实现一致的 metadata normalization，且不修改调用方对象；补 schema/field metadata 的真实执行回归。项目序列化格式不需要因此扩展。

2. **[P2] Python 新 schema API 无法返回已知空 schema。**
   `python/calc_flow/symbolic/program.py:649`，`Program.collect_schema`（TableExpr 委托同一路径）。`cf.table_input("events", schema=pa.schema([])).collect_schema()` 抛 `ConfigError: invalid operator.calculation: provide exactly one expression or non-empty select list`；batch/stream Program 均如此。已知空 schema 被无条件交给 executable document lowering，生成空 select，尚未到 schema planner 就失败。该输入声明合法，现有文档也承诺零列表输入；Rust 本片已专门测试已知空 schema 不等于 unresolved。应使这个纯 schema 查询返回 `pa.schema([])`，并保留未知 schema 的拒绝。对照 `source.select(n=cf.lit(1)).collect_schema()` 可以返回非空 schema，因此问题是已知空输出路径，而非运行环境缺依赖。

3. **[P2] Python schema planning 的不支持类型错误泄露生成 ID，未落到逻辑输出。**
   `python/calc_flow/symbolic/program.py:649`，`Program.collect_schema`。输入为 int64，输出为 `source.sql("SELECT CAST(x AS DECIMAL(20,2)) AS n FROM input")` 时，新接口抛裸 ValueError：`inputs.cf_sql_input_2c5e5dadccee944d44eee8b4.schema[0].data_type: unsupported_type: unknown portable Arrow type 'decimal128(20, 2)'`。将逻辑输出名从 result 改成 a.b，错误仍没有逻辑输出位置。拒绝 Decimal 本身正确；问题是冻结 note 要求 planning 错误保留逻辑 role/path 和 code，不能只暴露 cf_sql 生成地址。应在新公开 schema 边界保留/转换为现有 CompileError 及正确 escaped output/schema 路径，补不支持输出类型的定位测试；无需新增 Decimal 支持。

4. **[P2] 新公开导出未同步既有 frozen export 合同测试。**
   `python/calc_flow/symbolic/__init__.py:62`；对应未改的 `python/tests/test_symbolic_expr.py:80`。新增 CompileMode 是冻结 note 的正确要求，但既有 `test_public_surface_is_the_frozen_export_list` 仍要求旧列表。匹配本 HEAD 的 wheel 上独立执行该测试得到 **1 failed**，首个差异为 CompileMode / AnalysisOptions。151 项定向检查没有包含这个直接相关模块。应将既有精确导出期望同步为已批准的新公共合同并运行该测试，不删除或放宽断言。

## Open Questions

无需要用户决策的阻塞问题。以上可由实现者在同一 PR 局部修正后复核影响。

同 mode 的 Python `for_mode("batch")` 在同一 Runtime 后续 compile_batch 时仍可命中既有缓存并得到同一 batch plan；这是已有显式 batch cache 行为，不能混同为跨 mode 的 plan/job 共享。跨 mode 的 key、stream 每次新 owning plan，以及同 batch plan 的 execute 状态隔离证据有效。关系指南“compiled plans ... never shared by the wrappers”的绝对措辞宜收窄到跨生命周期/stream owner；此处作为非阻塞文档精度提醒，不要求改动既有 batch cache 合同。

## Tests

### 本轮实际执行

- `git status --short`、`git branch --show-current`、`git rev-parse HEAD`；GitHub PR metadata 与 `gh pr diff 591 --repo wegamekinglc/calc-flow`。本地 HEAD 与 PR head 相同，目标 main，标题/body 满足格式且只关闭 DAL-512。
- 读取 DAL-512、父 DAL-498 及相关评论线程；核对冻结 API note、分析 md/json、仓库 AGENTS 和 cf-reviewer 合同。身份模板的 style-review/component-boundaries reference 在此仓库不存在；无 dal-cpp/dal-public 组件，未套用其约束。
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=target/relational/wheel-python JAX_PLATFORMS=cpu python -m pytest python/tests/test_symbolic_expr.py::test_public_surface_is_the_frozen_export_list -q -p no:cacheprovider`：**1 failed**，见 Finding 4。
- 使用同一匹配 wheel 的 `python - <<'PY'` 小脚本检查已知空 schema（batch/stream）、空 schema 上 literal 投影、不支持 SQL 输出类型及两种逻辑输出名；结果见 Findings 2–3。没有读取数据或启动 job。
- 使用已有最终 core rlib 编译并运行一个 stdin Rust consumer，对无 metadata/schema metadata/field metadata 三种输入做相同公开执行。编译命令及完整用例在后文；对照成功，两种 metadata 用例失败，见 Finding 1。未重建 core/native。
- Python 校验脚本逐一验证 evidence JSON 中 14 个 check 的日志 SHA256：均找到相符本地日志；核对三棵产品源码 Git tree、wheel/native 哈希、全部 55 个 tracked Python package 文件逐字节一致（含 py.typed）、三个 JUnit 结果和冻结 note 哈希。consumer Cargo.lock 全量解析，263 个 package 中仅 consumer 自身不在根 lock，无 PyO3/Python 依赖。
- `git diff --check 775ea9738265597de492b848f8e0763962c7041b HEAD` 通过；`git diff 775ea9738265597de492b848f8e0763962c7041b HEAD --exit-code -- schemas/project-v3.schema.json .codex/agents .claude/agents Cargo.toml Cargo.lock pyproject.toml` 无差异。

### 复用的有效验证

产品提交 `e5ef248b9853e095be651b5bf06c2286dbadbf86` 与 reviewed HEAD 的三棵源码 tree 完全一致：

- core：`75d312dd6ebc376a424bdecc45dee726d4fa1065`
- PyO3：`0598f7aaefb029408ada38d02dd6d51bf6c78289`
- Python：`1bcebca776b58b2f70bf285addb477510e9dfc35`

wheel SHA256：`c3f6dc948dadc5380147a96bd33b16ef89faeb4f00be565e7501dfe31f8c2fe8`。
native SHA256：`8faf47d39df917ead9aa837b57890fef89597894323eaee673cc111b9721a543`。
冻结 note SHA256：`f3185b9217233a10f2fc584ecceb383cafbef8345bc92a3a8ea3e979b7084e71`。

复用分析证据所列实际命令和日志，未因角色切换重复执行：Rust 18 项 typed frontend/Program/validation、9 项未受后续修正影响的低层 history 测试、独立公开 consumer、standalone default core build、1 项 rustdoc、core/PyO3 Clippy、格式/complexity/docs 检查、两语言原位置 example 32；匹配 wheel 的 135 + 11 + 5 = 151 项 Python 检查。JUnit 均为零 failure/error/skip。既有有效检查仍然有效，但不覆盖本轮发现的缺口。

独立 consumer 的四种图、共享命名输出、batch/owned stream、NULL/空行/重复行、data-only export/reload 执行均有实际结果和独立期望。两语言证据保留同 batch plan 1/5、同 stream job 1/6、新 job 5、旧 SUM 链 1/7、exact fingerprint、strict gate 在 Source 前及 async capture/cancellation/owned cleanup。未发现新的共享运行状态或取消清理回归。未本地运行全仓、服务矩阵、覆盖率、跨平台或性能套件。

### Rust metadata 最小复现命令

以下命令在指定 checkout 根目录执行，复用其现有匹配产物；输出二进制仅为本地验证产物。

```bash
rustc --edition=2024 --crate-name relational_review_metadata - \
  -L dependency=target/relational/cargo/release/deps \
  --extern calc_flow=target/relational/cargo/release/deps/libcalc_flow-8195b3c90974754d.rlib \
  --extern tokio=target/relational/cargo/release/deps/libtokio-b8407d61ce751429.rlib \
  -o target/relational/review-metadata <<'RS'
use std::{collections::{BTreeMap, HashMap}, sync::Arc};
use calc_flow::{*, arrow::{
    array::Int64Array, datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
}};
fn main() -> Result<()> {
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        for (label, schema) in [
            ("plain", Schema::new(vec![Field::new("x", DataType::Int64, true)])),
            ("schema metadata", Schema::new_with_metadata(
                vec![Field::new("x", DataType::Int64, true)],
                HashMap::from([("origin".into(), "consumer".into())]),
            )),
            ("field metadata", Schema::new(vec![
                Field::new("x", DataType::Int64, true).with_metadata(
                    HashMap::from([("unit".into(), "USD".into())]),
                ),
            ])),
        ] {
            let schema = Arc::new(schema);
            let source = table_input("events", schema.clone(), InputOptions::default())?;
            let program = Program::new("metadata", CompileMode::Batch,
                BTreeMap::from([("out".into(), source)]))?;
            let plan = program.compile_batch(&UdfRegistry::new().snapshot(),
                &CompileOptions::default()).await?;
            let record = RecordBatch::try_new(schema.clone(), vec![
                Arc::new(Int64Array::from(vec![Some(1)])),
            ]).unwrap();
            let batch = Batch::table(vec![record], BatchMetadata::default())?;
            match plan.execute(BTreeMap::from([("events".into(), batch)]),
                ExecutionOptions::default()).await {
                Ok(_) => println!("{label}: execute OK"),
                Err(e) => println!("{label}: execute ERROR {e}"),
            }
        }
        Ok(())
    })
}
RS
target/relational/review-metadata
```

## CI gate（与代码 verdict 分开）

复用触发评论的初始快照；结束前仅追加一次非阻塞快照，时间约 06:55 UTC，HEAD 未变化：**13 SUCCESS、7 IN_PROGRESS、1 ACTION_REQUIRED**，无 COMPLETED/FAILURE、CANCELLED 或 SKIPPED。required checks 尚未全部完成，不能称 merge-ready。未 watch、轮询或等待 CI。

ACTION_REQUIRED 为 [Codacy Static Code Analysis](https://app.codacy.com/gh/wegamekinglc/calc-flow/pull-requests/591)，check run **130370405171**，标题为 **28 new issues (0 max.) of at least minor severity.**。进一步读取该 check output 和 annotations，属于诊断同一快照而非轮询。协调器应交实现者处理该门禁，不能把本地 complexity 脚本通过当作 Codacy 已通过。具体线索：

- 25 条 complexity 告警；例如 `dataframe.rs:492` dtype 为 21（limit 8）、`dataframe.rs:163` table_input 为 14、`program/lower.rs:158` add_error 为 13、`symbolic/program.py:45` _engine_selection 为 9，另含测试/示例。
- 3 条 Python 示例 assert 告警：`examples/32_sql_expression_pipeline.py:44`、46、56。
- 获取命令：`gh api repos/wegamekinglc/calc-flow/commits/e12fc8ba6f8d45c83cd462628fde46481fc56ced/check-runs` 中选该 run；`gh api repos/wegamekinglc/calc-flow/check-runs/130370405171/annotations`。

这些外部门禁告警与上面独立复现的四项代码/合同问题分别报告；没有将其逐条转成新的设计 blocker。

## Summary

公开 Rust 模块/re-export、typed DAG、命名绑定进入 runtime 的位置、物理 binding 发现接口、两语言原位置示例及文档总体覆盖 S0b；共享权威执行内核，未改 project schema、历史 fixtures、依赖或门禁。兼容参数矩阵、跨 mode plan 身份及已有状态/所有权证据可复用。

本次四项可复现缺口尚未修复，代码结论为 **Request Changes**；Codacy ACTION_REQUIRED 与尚在运行的 required CI 另阻止合并。没有修改产品或合并。DAL-512 应保持 **in_progress**，局部修正后复核受影响路径即可，无需默认全量、本地性能或另派 doc-writer。

