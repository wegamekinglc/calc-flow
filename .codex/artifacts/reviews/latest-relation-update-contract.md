# DAL-513 / PR #595 独立主审

## Findings

1. **[P2] YAML 导入遗漏冻结的更新语义配置诊断。**

   位置：`crates/calc-flow/src/project_store.rs:273`，`validate_update_semantics_value`；遗漏入口为同文件 `import_project_yaml_with_limit:344`，尤其 `:356` 的原始值错误投影分支。

   本片只在 JSON 导入调用新增验证器。将一个正常导出的 YAML 项目的 `graph.update_semantics` 从 `latest_relation` 改成 `future` 后，公开 `calc_flow.store.import_project_yaml` 返回通用 `ConfigError`，内容为 serde 的 `unknown variant`，且没有 `issues`。同一项目经 JSON 导入则返回 `issues=({'path': 'graph.update_semantics', 'code': 'invalid_literal', ...},)`。Rust 对应分别是 `CalcFlowError::Format` 和 `ProjectValidation`。

   冻结 API note 的 Stable preflight codes 表（`:173`）给该配置规定了 `graph.update_semantics / invalid_literal`，没有按持久化格式拆分合同。YAML 是既有公开项目入口；选择该格式后，调用者无法继续使用约定的字段路径和稳定代码定位同一种配置错误。这是可复现的公共诊断验收缺口，非法值的安全拒绝仍有效。

   建议在 YAML 的有预算原始值解析成功后复用 `validate_update_semantics_value`，保留现有版本、tag、alias、深度和大小限制；补 JSON/YAML 对未知字符串及非字符串值的同义诊断断言。当前 `crates/calc-flow/tests/latest_relation.rs:510` 的回归只覆盖 JSON，已有 project-store 测试没有捕获此缺口。复核仅需该导入边界及其直接回归。

   本次匹配 wheel 的实际观察：

   ```text
   JSON: ConfigError; issues[0].path = graph.update_semantics
                      issues[0].code = invalid_literal
   YAML: ConfigError; unknown variant `future`; issues absent
   ```

   可通过公开 Python API 独立复现：

   ```python
   import json
   import calc_flow as cf
   import pyarrow as pa
   from calc_flow.store import export_project_yaml, import_project_json, import_project_yaml

   source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
   program = cf.Program("yaml-literal", mode="batch", outputs={"result": source})
   document = program.with_update_semantics("latest_relation").to_project()
   valid_yaml = export_project_yaml(document)
   import_project_yaml(valid_yaml)  # 正常往返成功
   bad_yaml = valid_yaml.replace("update_semantics: latest_relation", "update_semantics: future")
   bad_json = document.model_dump()
   bad_json["graph"]["update_semantics"] = "future"
   for load, value in [(import_project_json, json.dumps(bad_json)), (import_project_yaml, bad_yaml)]:
       try:
           load(value)
       except cf.ConfigError as error:
           print(str(error), getattr(error, "issues", None))
   ```

未发现其他需要阻塞本片的正确性、兼容性或必要证据问题。

## Open Questions

无须追加公共合同决策。普通 latest 替换聚合与 DataFrame aggregate 继续属于 DAL-514；本片的拒绝边界符合冻结范围。

## Tests

### 审查对象与证据来源

- PR：<https://github.com/wegamekinglc/calc-flow/pull/595>，作者 Cheng Li；`feature/dal-513-latest-relation` → `main`，40 个改动文件。GitHub head、本地 HEAD 和指定提交均为 `729d7302a4684d9626c8d55cdf8f4acb4baed0a9`。PR body 有 Summary、Test plan、Closes DAL-513；GitHub 无既有 review，元数据报告可合并。
- 三点 diff 的共同祖先为 `f524493cb80546f35b65db0ecb9d2c86f1a2ea2e`；读取时 GitHub/base 与本地 `origin/main` 均为 `eb29d655b1c7040b647138488249a8055e660a8c`。本次审查以 PR head 的本片改动为对象。
- 已读取同名冻结 API note、analysis Markdown/JSON，并核对父任务的生命周期、四单元和分片讨论，以及仓库 AGENTS、`cf-reviewer.toml`。该仓库没有 `.codex/references/style-review.md` 或 `component-boundaries.md`；实际是 Rust/PyO3/Python 边界，未出现 C++/dal-public 依赖或示例搬迁。
- 实证复用经过内容核对：head 的 core 源码树 `e0f12de0e7f05f81439db2f1ced3a241ee7f651e`、Python 树 `59fe42a9a77a19e788917e332701de9f34440909` 与 native build `7ef99c99e3931379ec45f43515a8717fe84dfc88` 一致；binding 的生产代码前缀逐字节一致，后续改动仅为 `cfg(test)` 的 revision 断言。54 个打包 Python 源文件哈希、wheel 和扩展哈希均与 evidence JSON 一致。
- validation SHA `8f76053f5d517a33c95449834646c8de7141b56b` 到本 head 仅新增两个 analysis 文件。检查日志哈希及 Python JUnit 计数吻合；clippy 的成功日志对应 `target/latest/clippy-libs-green.log`。

### 直接复用的本地结果

| 验证                          | 来源及结果                                                                 |
| ----------------------------- | -------------------------------------------------------------------------- |
| Rust 受影响测试                | 初始 SHA `f16f7314`：65 通过；后续 serde 修正后：25 通过，含 latest 与 store。两轮存在重叠，不相加为独立用例。 |
| snapshot SQL 兼容性            | 16 通过，含 whole-event 预算、候选失败/取消、初始化、空替换及真实恢复。          |
| 匹配最终 wheel 的 Python       | 六个受影响模块 164 通过，0 failure/error/skip；导出清单另 1 通过。             |
| PyO3 / 文档                   | 错误映射 5 通过；新增 selector doctest 2 通过；rustdoc warnings denied 通过。  |
| 独立 Rust 构建及公开 consumer | core 独立构建通过；consumer 覆盖 DF、SQL、mixed DAG、schema、stream、导出重载及完整 struct literal 迁移。 |
| 静态与生成物                  | 复用受影响 clippy、格式、schema 生成及复杂度等成功结果；本次 diff whitespace 检查通过。 |

这些是按记录源码内容复用的结果，本轮没有重新运行上述产品测试或完整矩阵。

### 本次定向补查

仅对两个具体缺口运行现有环境中的 Python 探针：

```text
PYTHONPATH=target/latest/final-python JAX_PLATFORMS=cpu \
  /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python \
  target/latest/reviewer-probes.py
```

- `SUM → DISTINCT → project`：终端支持事实仍为 unknown，编译按 `update_semantics_unproved` 拒绝，证明下游投影没有洗掉该 unknown。
- 有效 latest YAML 往返成功；同一未知值分别经公开 JSON/YAML 入口观察，复现 Finding 1。
- 探针脚本 SHA256：`faf1a59175a90f4f97ab5b828755bd3e155e328e3999a5e4d570305ce9bdad9e`；输出 SHA256：`f34dd3030b4dca98e6ae3156dd432fd85732f8b157d99a56aa60e76ced4aeae6`。上述复现代码和观察已包含在本附件，运行临时文件留在忽略的 target 目录。

## Summary

强制准入的主体实现与冻结合同一致：Rust typed lowering 在 `program/lower.rs:375` 按本 mode gate，低层 compiler 在 `pipeline/compile.rs:77` 再检查，project builder 在 `config.rs:1311` 保留选择；Python `symbolic/query_analysis.py:556` 的 gate 先于公开编译缓存及 collection 包装，注册锁覆盖分析和准备。普通及 strict 的拒绝、共享消费边、稳定逻辑路径和 stale v1 拒绝均有匹配测试证据。

逐边规则将快照经已证明的投影、过滤和 row-local SQL/CTE 保留为完整替换；普通累计消费者明确 mismatch，未知消费者继续拒绝。`latest_relation.rs:330`、`:417` 及 Python 新测试覆盖真实命名空事件、多 RecordBatch 单事件和无事件；既有 snapshot SQL 的预算和原子性边界保持。

legacy 默认省略字段、声明及 native 指纹的编码路径保持；Python 旧基线 golden 值与当前结果相同。`latest_relation.rs:366` 实际保存并恢复双向错配，均在 Source open/read 计数为 0 时拒绝，相同选择的终端恢复成功。原 SQL state/layout 未变。公开 selector、根包/模块导出、stub、schema、文档及 Rust 完整 struct literal 的迁移均已覆盖。

剩余风险：Finding 1 待修；完整 workspace/Python 回归、coverage、服务容器与跨平台结果仍由 CI 承担，Python 3.9 只有语法证据。本轮按要求复用 **10:42 UTC 的 5 通过、16 运行、0 失败** 快照，未查询新检查结果、等待或轮询；该快照不能证明当前 required checks 已绿。已有持久唤醒和协调器接续 CI、修正及合并。

未改产品代码、提交 GitHub review、改 PR、扩派或合并；交付本报告，DAL-513 保持 `in_progress`。

## Verdict

**Request Changes** — 修复 Finding 1 并补该导入边界的定向回归后复核；本次没有其他主审 blocker。
