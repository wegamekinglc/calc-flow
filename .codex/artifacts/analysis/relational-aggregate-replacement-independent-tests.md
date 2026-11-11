# DAL-514 independent replacement tests

独立结论：指定的五个现有替换测试通过；新增四个预算、恢复及取消边界测试首次执行全部通过。未发现本片 operator 原子性产品 blocker，未修产品。仅提交必要测试与本报告/JSON，交协调器接 reviewer；DAL-514 保持 `in_progress`，唯一 PR 为 [#597](https://github.com/wegamekinglc/calc-flow/pull/597)。

## Source and native provenance

进入共享 checkout 时工作树干净，分支 `feature/dal-514-aggregate-replacement`，HEAD `e9ca4ac754d26b4584e6f11a5640e0c1f0107139`。`ddd8bd872ad4d1524cf5f97e8e6112197d71584d..HEAD` 仅两个前序证据文件变化。已读 API note、前序 MD/JSON、cf-tester、AGENTS 与相邻源码/测试。

实核四个 component tree 与前序记录相同：core `a1249e6cb31facf704ff5dbe698cbc48749b76bf`，binding `fd8dd7cc31ccf1d867c5df0f99cc87c5bca5ec9c`，Python `522c347be1ce7bff94ffb0c4ea1fd1ac6486c263`，snapshot policy `8b2ca0e0185e8a521366501d93ab96b150230f51`。原 snapshot policy 与其 12 项测试的源码提交 `456a64c033826a419fb64493598123fa7cf04321` 逐字节一致。

原 native 构建源码为 `ddd8bd87`。实核 wheel SHA256 `f930a713bff3ca6b3f0947c3a0f96fb2084147cf4c30e61e2573a5e9e06d882b`，extension SHA256 `f36ea47ddeba95d918274493acda92945120bed86449a94a71cc59f3277cacaa`，55 个打包 Python/stub 文件及全部 21 个前序日志摘要。原 1154 文件 source manifest 在补测试后仅本测试文件不同。导入探针显式使用 `PYTHONPATH=target/aggregate/final-python JAX_PLATFORMS=cpu`，CPython 3.13.9 / PyArrow 24.0.0；未采用环境默认 calc_flow 包。实际 interpreter、wheel、extension 路径和完整哈希保存在 JSON。

**编译期间出现并行产品改动：** `crates/calc-flow/src/dataframe/aggregate.rs`、`crates/calc-flow/src/query_analysis/sql/replacement.rs`、`python/calc_flow/symbolic/aggregate.py` 被其他写入者修改。本轮保留并排除这些改动，JSON 记录观察到的 SHA256。所测 `operator/sql.rs`、`operator/{sql,snapshot_sql}/` 的生产实现、`state_memory_pool.rs`、`pipeline/stream.rs` 仍与原 manifest 相同。新测试 binary 和日志哈希已记录；编译期间发生并行修改，因此不声称该 binary 等于最后的完整 dirty tree。原 wheel/119 项 Python 结果仍只属于原源码，不能扩展为这些并行改动的验证。其负责者/协调器需接续最新全树/native 来源核对。

后续还观察到并行修改 `crates/calc-flow/examples/32_sql_expression_pipeline.rs`、`crates/calc-flow/tests/latest_relation.rs`、`crates/calc-flow/tests/relational-consumer/src/main.rs`；同样保留、排除，不能把原相应示例/集成/consumer 结果当作这些新版本的验证。

## Running existing tests

```sh
target/aggregate/cargo/release/deps/calc_flow-70207ccc03de92aa operator::snapshot_sql::replacement_tests:: --nocapture
```

结果 **5 passed, 0 failed**。独立重跑包含：候选 emit 拒绝、行预算失败、预先取消、查询失败和初始化区分、checkpoint/restore 准备失败、layout6 恢复后下一更新、未知/旧布局拒绝、字节 ledger 与完整输出 edge 限制。仅运行该入口，未重跑整套。

前序 **119 Python、24 Rust 集成、12 snapshot 单测、独立 core build/外部 consumer、示例及 lint** 按各自源码 SHA 复用；21 项日志哈希核对通过。原记录的 core/consumer 源码较早，后续有诊断解释变化；不把旧结果重标为本轮新执行或完整最终树证据。JSON 保留全部原命令、SHA、结果与日志摘要。

## Authoring focused tests

全部位于 `crates/calc-flow/src/operator/snapshot_sql/replacement_tests.rs`，使用局部 owner、collector 和 cancellation token，不共享 singleton/store。独立预期来自 API note 的 C13/C14/A06/O01 与小表手算，没有让另一执行前端充当唯一 oracle。

- `test_replacement_legacy_input_budget_limits_each_candidate`：普通 SqlOperator 转发可选旧 `sql_state_budget`。R1=[1] 后，[2,3] 在 rows=1 或 bytes=15 被拒绝；无 emit，checkpoint signature 不变。rows=2/bytes=16 精确允许 [2,3]→5，再允许 [4,5]→9，证明限制每次 latest 输入而非历史累计；错误含 alias/预算字段和 required/limit。
- `test_replacement_restore_honors_tighter_legacy_input_budget`：保存 [2,3] 的 layout6，目标已有 [9]。收紧 rows 或 bytes 后 restore 拒绝，保留目标 checkpoint；精确预算恢复成功且无重发。原 owner 与恢复 owner 的下一事件各含两个 RecordBatch [4]、[5]，均恰好一次输出9。
- `test_replacement_cancelled_pending_emit_refunds_candidate_and_output`：在 Notify 确定的 pending emit 点，显式核对旧关系/capture 加候选 backing、输出 backing 和 8192 metadata 的共同计费。取消并 drop process future 后精确退款到旧 charge、旧 checkpoint 不变、无输出；同 owner 换有效 context 可重试 [2,3]→5。
- `test_replacement_successful_emit_commits_when_collector_cancels`：collector 接收一次 sum=5 后取消并 yield，再返回成功。process 仍成功提交两行 R2；恢复该 checkpoint 后，原 owner 与恢复 owner 下一次 [4] 都只输出4。覆盖成功 emit 后不再插入失败取消检查的合同。

```sh
CARGO_TARGET_DIR=target/aggregate/cargo CARGO_BUILD_JOBS=4 cargo test --locked -p calc-flow --lib --release --config profile.release.package.calc-flow.opt-level=0 --config profile.release.package.calc-flow.codegen-units=256 operator::snapshot_sql::replacement_tests::test_replacement_ -- --nocapture
rustfmt --edition 2024 --check crates/calc-flow/src/operator/snapshot_sql/replacement_tests.rs
git diff --check -- crates/calc-flow/src/operator/snapshot_sql/replacement_tests.rs
```

结果 **4 passed, 0 failed**，编译 4m28s，测试 0.01s；最终格式/空白检查均退出0。没有新增公共接口或行为修复，因此没有伪造实现前 RED。

## Repair and handoff limits

未发生产品失败或修复；初次 rustfmt check 仅发现一条断言的折行，最终检查通过。没有改前序测试/证据、建环境/PR、扩派、性能测试或全仓覆盖率扫描。

新增用例是确定的 operator 局部边界证据。pending emit 测试明确由测试 drop future，不证明 managed runner 自动取消收敛、任意并发调度或外部 Sink 副作用回滚。公共图准入/启动前 preflight 与端到端行为复用前序证据。

CI 沿交接为“未确认”，保留既有 `checks_finished` 条件唤醒，本轮未在线查询或轮询 CI。main 前序观察为 `e7e771ac0efd0b44af936353a130f316c46a535b`，此前 merge preview 无冲突但未运行 merged tree；此限制保留。全量、coverage 和跨平台结果仍待 CI，独立主审仍待协调器接续。

本仓库无 DAL 专用 `.codex/references/{write-tests,unit-test-style,component-boundaries,run-tests}.md`；适用规则采用实际 AGENTS、cf-tester 和 `.claude/rules/code-style.md`。本轮按明确授权复用原专用 checkout/分支，未另建 worktree。
