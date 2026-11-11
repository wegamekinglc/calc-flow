# DAL-514 implementation and verification

验证源码：`ddd8bd872ad4d1524cf5f97e8e6112197d71584d`；基线：`c4f1749c0595ed3df0fe7eaf4c6331c6093dcfab`。分支 `feature/dal-514-aggregate-replacement`。
本报告的后续提交只添加证据；产品源码以 JSON 的 component trees 和 native 来源核对。

## Behavior and changed files

Rust `dataframe/aggregate.rs` 与 Python `symbolic/aggregate.py` / `expr.py` 提供命名 SUM/COUNT/MIN/MAX/AVG、count_rows 和分组。两端均生成安全 quoted SQL，由同一个 native planner 推导 schema、同一 SqlOperator 执行。新 row lineage 不继承时间排序。

`query_analysis` 使用严格 AST/类型证明，analysis revision=3；低层 compiler、typed Program 和 project reload 使用同一准入。只有 complete_snapshot → 获准 aggregate 在 latest 下选择替换分支。row-local SQL 不增加预算要求；原 append 默认与身份保持。

`operator/sql`、内部 relation/capture engine 与 `pipeline/stream` 接入显式 `sql_replacement_state_budget`、候选/输出/活 capture 共同计费、emit 后提交和新 state_version=3 / layout=6 / accounting=1。旧普通 SQL layouts 3/4/5 和 snapshot layout1 保留。Python runtime、PyO3 parser、stub 和 Source 前 preflight 同步。

`docs/relational-frontends.md`、`docs/streaming-guide.md`、两端 example32 及外部 Rust consumer 已更新。完整逐文件列表在 JSON；前序 artifacts 保留。

## Acceptance mapping

| Mapping         | Observable evidence / entry                                                                                                                                                                                     |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A04 / C02 / O01 | Python mixed-chain 三种组合、共享输出；Rust latest_relation chain；latest 1/6、legacy 1/7，夹 projection/filter；SQL→DF aggregate→SQL 不 collect 中间表。                                                                           |
| A05 / C04       | Python 小表/类型/NULL group 与 SQL 对照；独立 Rust consumer 检查全部 reducer、COUNT schema、空/全 NULL/重复与 NULL group。均有手算 oracle。                                                                                                |
| A06 / L01–L03   | 同一暖 batch plan 连续执行 1/5，完整 H=6；同 stream job producer 1/6；fresh job 和无 Data 不继承状态。                                                                                                                               |
| C12             | replacement_commit_restore_and_failed_candidates 的多 RecordBatch 单次输出；grouped_empty 的清空与 EOF 不重发；初始化 false 与 true/0 行实际恢复区分。                                                                                     |
| C13 budget      | byte_ledger 的 input backing+4096 metadata、old+candidate+prepared output inclusive limit 与 limit−1；capture 存活仍收费、失败归还；grouped_empty 的 edge bytes exact/不足一字节；相邻 snapshot slice/metadata/capture tests 复用相同源码并运行。 |
| C13 atomicity   | commit_restore 的 emit 拒绝/行预算/取消；initialization_and_query_failure 的首次失败、执行失败、合法重试及 checkpoint/restore preparation failure。签名仍为旧 checkpoint，pool charge 不泄漏。                                                      |
| C14 / A06       | layout6 checkpoint→restore→下一 replacement 与未恢复相同 alias 历史相同；legacy↔replacement 实际 restore 拒绝；未知 kind/layout/accounting/initialized、恢复预算不足拒绝且旧状态不变。                                                              |
| A09             | 十类非白名单查询保持准确拒绝且合法 finite SQL 仍编译；缺预算/compact 冲突在 iterator/bindings/job 前拒绝；pair parser、暖缓存、project reload、低层 compiler 受同一 gate 约束。                                                                              |
| A08 / identity  | 外部 Cargo consumer、Python 根包导出、annotations/stub、project roundtrip、docs/examples；legacy 声明/native/canonical-byte goldens 通过；默认 None 不增 runtime hash 字段。                                                           |

## RED and GREEN

RED 使用 git 导出的精确 c4f1749c Python 源码；native 仅供 import，不执行缺失 API。
`PYTHONPATH=target/aggregate/baseline python -m pytest python/tests/test_aggregate_replacement.py -k dataframe_aggregate_nulls -q`
得到预期的 3 个 `AttributeError: TableExpr has no attribute aggregate`。

| Check                               | Result                                                                       |
|-------------------------------------|------------------------------------------------------------------------------|
| RED exact baseline                  | Expected failure: 3 AttributeError, aggregate absent before native execution |
| Rust replacement/adjacent snapshots | 12 passed, including 5 new replacement tests                                 |
| Rust frontend/update/history        | 24 passed (11 + 4 + 9)                                                       |
| Library/binding lint                | passed                                                                       |
| Affected target lint                | passed                                                                       |
| core-standalone                     | exit=0                                                                       |
| rust-example                        | exit=0                                                                       |
| consumer                            | exit=0                                                                       |
| rustdoc                             | exit=0                                                                       |
| clippy-final                        | exit=0                                                                       |
| rust-example-final                  | exit=0                                                                       |
| native-final                        | exit=0                                                                       |
| python-complete                     | exit=0                                                                       |
| python-example-final                | exit=0                                                                       |
| ruff-final                          | exit=0                                                                       |
| ruff-format-final                   | exit=0                                                                       |
| rust-format-final                   | exit=0                                                                       |
| consumer-format-final               | exit=0                                                                       |
| complexity-final                    | exit=0                                                                       |
| schema-no-drift                     | exit=0                                                                       |
| whitespace-final                    | exit=0                                                                       |

所有命令、源码 SHA、输出摘要及 log SHA256 在 JSON。Rust 构建复用 Cargo cache，但编译路径和 wheel 源码属于本 checkout。成本控制 flags 为 `CARGO_BUILD_JOBS=4`、`--release --config profile.release.package.calc-flow.opt-level=0 --config profile.release.package.calc-flow.codegen-units=256`；绑定同样使用 package opt-level=0/codegen-units=256，不作性能声明。

## Native provenance

- Interpreter: `/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python`；Python `3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]`；PyArrow `24.0.0`。
- Wheel: `target/aggregate/final-wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl`，SHA256 `f930a713bff3ca6b3f0947c3a0f96fb2084147cf4c30e61e2573a5e9e06d882b`。
- Extension SHA256: `f36ea47ddeba95d918274493acda92945120bed86449a94a71cc59f3277cacaa`。
- 已逐字节核对本 checkout 打包的 `55` 个 Python/stub 文件；新 wheel 是本源码构建，未以其他 checkout 的已安装扩展冒充。

## Independent tester handoff

优先入口为 `crates/calc-flow/src/operator/snapshot_sql/replacement_tests.rs` 中的五个 `replacement_*` 测试。
可用最终构建的 lib test binary 只选择 `operator::snapshot_sql::replacement_tests::`；不必重跑全套。
预算测试以 accounting-v1 backing helper 加确定性 metadata 预付量作 ledger，显式验 inclusive 与不足一单位，不用只有巨大通过值的探针。
补充验证可集中在非终态 managed replay、并发取消/emit 交界、optional legacy input budget 与恢复预算收紧，不需要新增性能阶段。

## Design decisions and remaining limits

没有改变冻结公共合同。内部以 crate-private policy 复用现有 relation/capture owner，避免复制预算和 prepare/install 生命周期；普通 SQL 仍由 SqlOperator 持有并使用独立 policy/configuration/state version，未绕过 public snapshot_sql 的输出限制。
当前无独立持久 proof cache；latest 必须在 preparation cache 前重新证明并匹配 revision3，旧报告不能作许可。

全量、coverage、服务容器和跨平台由 CI 承担；Python 3.9 只有语法证据。未宣称 CI 已绿或可合并。
外部 sink 副作用不属于本次 operator 候选事务。未扩一般 delta/retraction、增量多输入 JOIN 或 batch 中途恢复。
独立 tester → reviewer 仍待协调器接续；本轮没有合并、扩派或启动后续任务，DAL-514 保持 in_progress。

最终 fetch 观察 main 已到 `e7e771ac0efd0b44af936353a130f316c46a535b`。只读 `git merge-tree --write-tree --messages HEAD origin/main` 退出0，预览 tree=`42c44fe60dc482442797df3c6befddc78bf32788`，无冲突；未实际合并或对该 merged tree 运行验证，集成结果留给 CI。

Python 验证显式使用 `PYTHONPATH=target/aggregate/final-python JAX_PLATFORMS=cpu` 和上列 interpreter；不得直接使用环境默认的 calc_flow import。独立 core/consumer 命令移除 `PYO3_PYTHON`。完整环境键在 JSON 的 verification_environment。merge preview 当时的 head 为 `a45c6d60da9ff876ec2719bc991de79216dccd28`；后续仅证据文本改变。
