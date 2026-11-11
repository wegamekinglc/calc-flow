# DAL-514 implementation and verification

最新验证源码：`1d3147fc5375b83964158833e0129129c8337c89`；原始基线：`c4f1749c0595ed3df0fe7eaf4c6331c6093dcfab`。分支 `feature/dal-514-aggregate-replacement`。
以下历史验证保留各自 SHA；最新主审 F1/F2 修复、component trees、匹配 native 和最小补验见末节及 JSON 的 `review_followup`，不把旧结果重标为新源码通过。

## Behavior and changed files

Rust `dataframe/aggregate.rs` 与 Python `symbolic/aggregate.py` / `expr.py` 提供命名 SUM/COUNT/MIN/MAX/AVG、count_rows 和分组。两端均生成安全 quoted SQL，由同一个 native planner 推导 schema、同一 SqlOperator 执行。新 row lineage 不继承时间排序。

`query_analysis` 使用严格 AST/类型证明，analysis revision=3；低层 compiler、typed Program 和 project reload 使用同一准入。只有 complete_snapshot → 获准 aggregate 在 latest 下选择替换分支。row-local SQL 不增加预算要求；原 append 默认与身份保持。

`operator/sql`、内部 relation/capture engine 与 `pipeline/stream` 接入显式 `sql_replacement_state_budget`、候选/输出/活 capture 共同计费、emit 后提交和新 state_version=3 / layout=6 / accounting=1。旧普通 SQL layouts 3/4/5 和 snapshot layout1 保留。Python runtime、PyO3 parser、stub 和 Source 前 preflight 同步。

`docs/relational-frontends.md`、`docs/streaming-guide.md`、两端 example32 及外部 Rust consumer 已更新。完整逐文件列表在 JSON；前序 artifacts 保留。

## Acceptance mapping

| Mapping           | Observable evidence / entry                                                                                                                                                                                       |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A04 / C02 / O01   | Python mixed-chain 三种组合、共享输出；Rust latest_relation chain；latest 1/6、legacy 1/7，夹 projection/filter；SQL→DF aggregate→SQL 不 collect 中间表。                                                                             |
| A05 / C04         | Python 小表/类型/NULL group 与 SQL 对照；独立 Rust consumer 检查全部 reducer、COUNT schema、空/全 NULL/重复与 NULL group。均有手算 oracle。                                                                                                  |
| A06 / L01–L03     | 同一暖 batch plan 连续执行 1/5，完整 H=6；同 stream job producer 1/6；fresh job 和无 Data 不继承状态。                                                                                                                                 |
| C12               | replacement_commit_restore_and_failed_candidates 的多 RecordBatch 单次输出；grouped_empty 的清空与 EOF 不重发；初始化 false 与 true/0 行实际恢复区分。                                                                                       |
| C13 budget        | byte_ledger 的 input backing+4096 metadata、old+candidate+prepared output inclusive limit 与 limit−1；capture 存活仍收费、失败归还；grouped_empty 的 edge bytes exact/不足一字节；相邻 snapshot slice/metadata/capture tests 复用相同源码并运行。   |
| C13 atomicity     | commit_restore 的 emit 拒绝/行预算/取消；initialization_and_query_failure 的首次失败、执行失败、合法重试及 checkpoint/restore preparation failure。签名仍为旧 checkpoint，pool charge 不泄漏。                                                        |
| C14 / A06         | layout6 checkpoint→restore→下一 replacement 与未恢复相同 alias 历史相同；legacy↔replacement 实际 restore 拒绝；未知 kind/layout/accounting/initialized、恢复预算不足拒绝且旧状态不变。                                                                |
| A09               | 十类非白名单查询保持准确拒绝且合法 finite SQL 仍编译；缺预算/compact 冲突在 iterator/bindings/job 前拒绝；pair parser、暖缓存、project reload、低层 compiler 受同一 gate 约束。                                                                                |
| A08 / identity    | 外部 Cargo consumer、Python 根包导出、annotations/stub、project roundtrip、docs/examples；legacy 声明/native/canonical-byte goldens 通过；默认 None 不增 runtime hash 字段。                                                             |

## Original RED and GREEN (per-command SHA in JSON)

RED 使用 git 导出的精确 c4f1749c Python 源码；native 仅供 import，不执行缺失 API。
`PYTHONPATH=target/aggregate/baseline python -m pytest python/tests/test_aggregate_replacement.py -k dataframe_aggregate_nulls -q`
得到预期的 3 个 `AttributeError: TableExpr has no attribute aggregate`。

| Check                                 | Result                                                                         |
|---------------------------------------|--------------------------------------------------------------------------------|
| RED exact baseline                    | Expected failure: 3 AttributeError, aggregate absent before native execution   |
| Rust replacement/adjacent snapshots   | 12 passed, including 5 new replacement tests                                   |
| Rust frontend/update/history          | 24 passed (11 + 4 + 9)                                                         |
| Library/binding lint                  | passed                                                                         |
| Affected target lint                  | passed                                                                         |
| core-standalone                       | exit=0                                                                         |
| rust-example                          | exit=0                                                                         |
| consumer                              | exit=0                                                                         |
| rustdoc                               | exit=0                                                                         |
| clippy-final                          | exit=0                                                                         |
| rust-example-final                    | exit=0                                                                         |
| native-final                          | exit=0                                                                         |
| python-complete                       | exit=0                                                                         |
| python-example-final                  | exit=0                                                                         |
| ruff-final                            | exit=0                                                                         |
| ruff-format-final                     | exit=0                                                                         |
| rust-format-final                     | exit=0                                                                         |
| consumer-format-final                 | exit=0                                                                         |
| complexity-final                      | exit=0                                                                         |
| schema-no-drift                       | exit=0                                                                         |
| whitespace-final                      | exit=0                                                                         |

所有命令、源码 SHA、输出摘要及 log SHA256 在 JSON。Rust 构建复用 Cargo cache，但编译路径和 wheel 源码属于本 checkout。成本控制 flags 为 `CARGO_BUILD_JOBS=4`、`--release --config profile.release.package.calc-flow.opt-level=0 --config profile.release.package.calc-flow.codegen-units=256`；绑定同样使用 package opt-level=0/codegen-units=256，不作性能声明。

## Original native provenance (ddd8bd87)

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
原始交接时独立 tester → reviewer 尚待接续；本轮没有合并、扩派或启动后续任务，DAL-514 保持 in_progress。

前次交接的最终 fetch 观察 main 已到 `e7e771ac0efd0b44af936353a130f316c46a535b`。只读 `git merge-tree --write-tree --messages HEAD origin/main` 退出0，预览 tree=`42c44fe60dc482442797df3c6befddc78bf32788`，无冲突；未实际合并或对该 merged tree 运行验证，集成结果留给 CI。

原始 Python 验证显式使用 `PYTHONPATH=target/aggregate/final-python JAX_PLATFORMS=cpu` 和上列 interpreter；不得直接使用环境默认的 calc_flow import。独立 core/consumer 命令移除 `PYO3_PYTHON`。原环境键保存在 JSON 的 prior_validation.verification_environment。merge preview 当时的 head 为 `a45c6d60da9ff876ec2719bc991de79216dccd28`；当次交接后续提交仅证据文本改变，本轮产品修复另见下节。


## Codacy follow-up — 217040d9

本轮修复源码 `217040d9b91635a410c64b416da4046dbac550bd` 保留 tester `016e96fa` 及独立测试 MD/JSON。旧 e9ca4ac7 的 Codacy check 130556786333 报 12 条新增问题：11 处 CCN 超过8，1条 Bandit B608。按函数职责拆分 aggregate 构造/校验、AST 准入/拒绝解释和 example/consumer/测试 helpers；值、错误优先级和全部现有断言保持。SQL 构造使用固定子句、统一双引号转义与内建 reducer 白名单，新增恶意引号/SQL片段的 schema 名称测试。未改阈值、排除项、CI/覆盖率设置或冻结公共合同。

**RED → GREEN：** Lizard 1.26.0 精确复现远端11处计数，修改/新增函数最大 CCN=8；Bandit 1.9.4 B608 在旧源码1条、当前0条，nosec=0。初次新增用例误走项目执行，触发既有 portable 字段校验；最终用例覆盖公共 collect_schema，精确核对 group/result 名称，未扩执行字段域。

本轮新执行（完整命令、环境、日志摘要、SHA 在 JSON 的 phase=codacy_followup）：

| Check                                                                               | Result                                                                    |
|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Independent Rust core build (PYO3_PYTHON removed)                                   | passed                                                                    |
| latest_relation + query_history_portability                                         | 20 passed (11+9)                                                          |
| Rust example32 + external Cargo consumer (PYO3_PYTHON removed)                      | passed; retained all manual reducer/schema/NULL/group/latest/warm oracles |
| New matching native wheel + packaged Python source equality                         | passed; 55 Python/stub files                                              |
| Focused Python aggregate/update/history/snapshot + export/capability tests          | 120 passed                                                                |
| Python example32                                                                    | passed                                                                    |
| Affected Clippy/Ruff/format, complexity ratchet + Lizard, Bandit, schema/whitespace | passed                                                                    |

新 wheel `target/aggregate/ci-wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl`，SHA256 `33ba324b056c3c300ec54d620d0b4ad16b0a28c3449318db985f1b55ac757f50`；extension SHA256 `dde797aca5962739faf5c136bbd6383492a733f43e429f439a886a0e302e61e8`。`PYTHONPATH=target/aggregate/ci-python JAX_PLATFORMS=cpu`，interpreter 与原记录相同（CPython3.13.9/PyArrow24）；所有55个 Python/stub 与本 checkout 逐字节一致，source manifest SHA256 `314b1eae2a3e50e3cd2d0ae829c1c8ccfd8967e691028fcbeb7674ef42740626`。核心源码 tree `124068c90da0662733f998d6ec34751835be6388`，binding tree `fd8dd7cc31ccf1d867c5df0f99cc87c5bca5ec9c`，Python tree `168d5c6cbde1764f42ade4acb8beab4a2f95d2ca`，snapshot subtree（含 tester 新测试）`48e35d5cab7b5213ab93c841f9e8f80d94c5f00e`。构建 flags 仍仅控制本地构建成本，无性能声明。

**复用边界：** 原119项 Python、旧示例/consumer/集成结果继续只归属其原 SHA，未重标；新 frontend/proof/example/consumer 均已重建验证。operator 的执行/预算/commit/restore 生产文件与 e9ca4ac7 逐字节一致；原12项 snapshot 及 tester 独立5项原用例、4项新用例按其报告范围复用，不重跑未受影响原子性。新增测试入口和日志详见 `relational-aggregate-replacement-independent-tests.{md,json}`；本报告保留原五个失败注入/恢复入口。tester 的并行来源限制已由本轮 commit、manifest、matching wheel 及定向构建收尾，未将其旧 binary 声称为新全树 binary。

额外核对 manifest 中全部 1154 个 Rust/Python/stub 源文件与 `217040d9` 的 Git blob 一致；新 wheel 的 55 个 Python/stub 文件与同一 manifest 一致。

本轮只做必要定向验证；full/coverage/服务容器/跨平台仍由 CI 承担，Python3.9仍只有语法证据。旧 clean merge-tree preview 只属于原 head，未验证 merged tree。推送后仅取一次非阻塞 CI 快照，pending 交持久条件唤醒；远端新 head 通过与否以最终 issue 评论为准。DAL-514 保持 in_progress，交协调器接独立 reviewer，不合并或扩派。

## Review blocker follow-up — 1d3147fc

针对主审 F1/F2，复用本 runtime 已有专用 checkout，确认远端仍为 `12ca086f0de24211f76f5168e5d7767e92fafc43` 后提交产品修复 `1d3147fc5375b83964158833e0129129c8337c89`。保留 tester `016e96fa` 和全部后续修复；仅修改以下四个产品/回归/指南文件，另更新本分析 MD/JSON。用户已明确要求复用可用 checkout，没有另建环境、扩派或合并。

- `crates/calc-flow/src/query_analysis/sql/replacement.rs`：先沿现有单 alias/限定符检查证明绑定，再通过同一 schema 字段比较投影列与分组列；复用既有未加引号小写归一化和引号精确匹配。合法限定名、反向限定、大小写等价及表别名不再被原始 AST 相等比较误拒；没有扩大类型、函数、关系或预算白名单，也没有改变恢复身份。
- `crates/calc-flow/src/program/admission_tests.rs`：新增一个含七种合法 SQL 写法的公共 stream 编译回归。
- `python/tests/test_aggregate_replacement.py`：新增七种合法写法在 latest/legacy 两种选择下的手算执行对照，以及四种未分组、错误限定符、引号大小写错误的拒绝对照。局部观察 helper 按实际命名输出初始化，原 mixed-chain 断言保持。
- `docs/relational-frontends.md`：同步旧范围、SUM 链和 batch 段落；有显式预算的支持域 latest 为 `1,6`，legacy 为 `1,7`，域外仍在 Source 前拒绝，batch 每次独立且不要求 replacement 状态预算。

**RED：** 在 `12ca086f` 的生产源码上仅加入新 Rust 回归，运行：

```sh
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/pr-repair-20261112/cargo-debug CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib program::admission_tests::replacement_group_keys_resolve_equivalent_bound_columns -- --exact
```

正常同形对照通过，随后 `SELECT input.g, SUM(input.x) AS total FROM input GROUP BY g` 以 `update_interpretation_mismatch` 失败；退出101，0 passed / 1 failed。不是构建中止或测试错误。原生产文件及新测试 overlay 哈希保存在 JSON。

**GREEN：** 同一 CARGO 环境运行 `cargo test --locked -p calc-flow --lib program::admission_tests::`，六项既有测试加新增回归 **7 passed / 0 failed**。首轮构建完成后只改最小字段身份比较，再运行受影响入口，没有运行全量测试。

新 native 使用 CPython3.13.9 / PyArrow24.0.0，显式 interpreter `/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python`，从本 checkout 执行 `python -m maturin build --locked --profile dev --interpreter <同一 interpreter> --out target/review-fixes/wheels`。`PYO3_PYTHON` 为同一 interpreter，`CARGO_TARGET_DIR` 复用上列依赖缓存，`CARGO_BUILD_JOBS=2`。core/binding 编译路径均为本 checkout；没有使用别处的现成 native。此 dev 构建只用于行为验证，不作性能或发布验证声明。

从新 wheel 解包至 `target/review-fixes/site`，明确验证实际导入路径与 extension 哈希，再执行：

```sh
PYTHONPATH=target/review-fixes/site JAX_PLATFORMS=cpu /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python -m pytest python/tests/test_aggregate_replacement.py python/tests/test_sql_checkpoint_layout_inventory.py -q -k "mixed_chain or budget_preflight or replacement_analysis_edges or native_budget_pairs or grouped_replacement or sql_layout_inventory"
```

结果 **29 passed / 18 deselected**（6.61s）：原 mixed-chain 6项、预算 preflight 2项、alias metadata 1项、paired budget/project reload 1项、新合法等价写法 14项、新非法写法4项、真实 layouts3/4/5/6 inventory 1项。新分组链逐事件手算 latest `1,6` / legacy `1,7`；有限 `g=[a,a], x=[1,2]` 得到 `g=[a], total=[3]`。

定向 core Clippy `cargo clippy --locked -p calc-flow --lib -- -D warnings` 退出0；两份 Rust 文件 rustfmt、受影响 Python Ruff lint/format、complexity ratchet、spec references、schema 无漂移及 whitespace 检查通过。完整命令、日志哈希和结果在 JSON 的 `phase=review_followup`。

来源：wheel SHA256 `257f9008de1c8551bf49edc2807643440915efaa5376a20bf9b3bc429b221945`；extension SHA256 `9343f4a367e95056f1e8188b9aad6e4447f6584dedf9d75c2806111bcecbccdb`。55个打包 Python/stub 与源码逐字节一致；构建前后2115个 Rust/Python/stub/Cargo/toolchain 清单文件稳定，全部与 `1d3147fc` Git blob 匹配，manifest SHA256 `e1195409f2970014db15a4ba6821677c004c068f8008051027f6cc1d42de40f8`。

component trees：core `3b4ff71c8031e3078dc1f83cdb5ab3fa85b50365`，binding `105c8ddb89f68cdf6a839e5a78903ddf0311e83a`，Python `83d14621b5c32dff61b3d713ebe5d65ad9311c40`；snapshot policy 仍为 `48e35d5cab7b5213ab93c841f9e8f80d94c5f00e`。

**复用边界：** 原119/120 Python、20 Rust 集成、core/consumer/示例及先前 Arc/storage 结果保留原始来源，不扩为本轮完整源码验证。未变 snapshot 子树的原12项与独立 tester 9项原子性/恢复证据继续按主审确认的范围复用，没有重跑。此前 exit −15 的准入构建和旧 native 不计作通过；本轮新7项 Rust和匹配 native 的29项 Python 补齐主审列出的最小缺口。无公共设计偏离。

全量、combined coverage、服务及跨平台仍交 CI；Python3.9 仍不具有本轮运行证据。历史 merge preview 不作为本次候选集成验证。推送后仅取一次非阻塞快照，沿既有 checks_finished 条件接续；最终 head/检查事实见交付评论。DAL-514 保持 in_progress，交协调器安排 F1/F2 与直接影响的独立复核。
