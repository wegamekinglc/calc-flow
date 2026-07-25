# DAL-5 公共接口与覆盖面审计

## 范围与基线

- 任务窗口：2026-07-18 15:00 至 2026-07-25 15:00（Asia/Shanghai）。
- 审计基线：本地最新可用 `origin/main`，提交
  `f71e49d7ccba117ade55270bd9df7499c04769cb`（2026-07-25 04:00:52 +08:00）。
- 当前工作分支 `agent/cf-orchestrator/fae83363` 与该 `origin/main` 提交完全一致；开始时
  代码工作树干净，因此没有切换分支、覆盖未提交内容或获取远端更新。
- 本文只做证据整理和机会识别，不对最终三个新功能排序或定案。

## 当前公共功能版图

### Rust crate

`crates/calc-flow/src/lib.rs:18-50` 将 v2 引擎的主要能力统一重导出：

- 不可变 `Batch`、表/外部载荷和元数据；
- 严格项目模型、JSON Schema、校验和编译；
- `PipelineBuilder`、表达式/SQL/外部算子、端口和边；
- DataFusion 配置、执行计划、节点计时、查询计划指标；
- UDF/外部 provider 注册；
- source、sink、micro-batch/streaming runner 和 checkpoint；
- 项目/checkpoint 文件存储及 JSON/YAML 导入导出；
- 分类错误、取消、截止时间和运行上下文。

本周最重要的公共边界变化来自 lazy DataFusion 工作：

- 新增并导出 `OperatorDefinition`，将内建 DataFusion 算子与外部 `Operator` 显式分类
  （`crates/calc-flow/src/operator.rs:438-473`）。
- `PipelineBuilder::add_node` 接受 `Into<OperatorDefinition>`，仍保留常见 boxed 调用形式
  （`crates/calc-flow/src/pipeline.rs:1286-1314`）。
- `ExecutionPlan::datafusion_config()` 现在返回 `Option<DataFusionConfig>`，
  `requires_datafusion()` 明确外部-only 计划不拥有表引擎资源
  （`crates/calc-flow/src/pipeline.rs:225-258`）。
- `OperatorContext` 只暴露 engine-neutral `RunContext`
  （`crates/calc-flow/src/operator.rs:110-130`）。

### Python binding

`python/calc_flow/__init__.py:5-52` 的顶层入口覆盖 Batch、函数式 builder、Runtime、
执行计划/结果、文件存储、两类 runner、异常层级和 NumPy/JAX 注册。完整项目能力仍可通过
`ProjectDocument` 加 `Runtime.compile_project()` 使用；便捷 `PipelineBuilder` 提供
`expression`、`sql`、`external`、`connect` 和 `compile`
（`python/calc_flow/pipeline.py:237-398`）。

PyO3 边界已经支持：

- Python provider 和向量化标量 UDF 注册；
- GC-safe callback ownership；
- 同步/异步执行及状态 snapshot/restore/reset；
- Python provider 注册快照随 Studio spawned worker 序列化；
- table 与 external array 结果投影。

本周 `main` 没有修改顶层 Python 适配器或 `_native.pyi`；Python 侧近期改动只优化了
provider options 的预编码和 engine-neutral context，不改变已检查的 Python 签名。

### Studio REST / OpenAPI / Web UI

已签入的 OpenAPI 含 11 个 `/api/v2` path、16 个 GET/POST/PUT/DELETE 操作：

- catalog、项目 schema；
- 项目 CRUD、JSON/YAML 导入导出和校验；
- checkpoint 检查/删除；
- 有资源上限的 preview run、查询、SSE 事件和取消。

React Studio 已能编辑表达式/SQL 图、SQL aliases、端口 Arrow schema、多数据源、项目文件、
checkpoint 信息、结果行、节点计时、DataFusion logical/physical plan，并能比较 benchmark
报告。窗口内主要新增多数据源编辑、graph-safe SQL alias 操作和可调整面板；这些都是
前端覆盖增强，`web-ui/openapi.json` 与项目 schema 在窗口内未变。

## 已确认的覆盖缺口或不一致

以下结论是当前代码事实，不是功能提案：

1. **catalog 的实际契约比文档和 UI 名称窄。**
   `Runtime.catalog()` 只序列化已注册的 scalar UDF
   （`crates/calc-flow-python/src/config.rs:280-307`）；provider 注册、内建算子、Arrow
   类型和限制都不在返回值中。FastAPI 直接以 `list[dict[str, object]]` 返回该值
   （`web-ui/backend/src/calc_flow_studio/app.py:259-261`），OpenAPI 因而只生成
   `array<object>`。但 `docs/api-reference.md:141` 描述 `/catalog` 为“Operators,
   UDFs, Arrow types, limits”。

2. **三类关键 REST 响应没有单一生成类型。**
   `CatalogResponse`、`ValidationReport`、`RunResultPreview` 由前端手写
   （`web-ui/src/types.ts:23-85`）；后端 catalog/validate 使用宽泛 dict，
   `RunResponse.result` 也是 `dict[str, JSONValue] | None`
   （`web-ui/backend/src/calc_flow_studio/models.py:91-110`）。结果面板必须把生成的
   `RunResponse.result` 再断言成手写类型
   （`web-ui/src/components/ResultsPanel.tsx:24`）。字段漂移不会由
   `openapi-typescript` 自动暴露。

3. **Python 执行没有投影 Rust 的 `ExecutionOptions`。**
   Rust 执行公开 `settings`、`deadline` 和 `CancellationToken`
   （`crates/calc-flow/src/pipeline.rs:23-28,288-307`），但 PyO3 的同步和异步
   `execute` 都固定使用 `ExecutionOptions::default()`
   （`crates/calc-flow-python/src/pipeline.rs:144-170`）。Python task cancellation
   已有支持，但用户不能传递 run settings 或显式 deadline。

4. **Python 便捷 builder 只覆盖项目模型的一部分。**
   Rust builder 可设置 DataFusion config，项目 schema 还包含 description、
   run options、显式 ports/schema 和数据源；Python 便捷 builder 不提供这些 modifier。
   这不阻断功能，因为 `ProjectDocument`/原始 project JSON 是完整逃生口，但会让常见
   高级配置从 fluent API 跳回文档字典。

5. **Studio schema 能表示 external/array，默认 UI 和 preview 不能完成该路径。**
   `OperatorSpec` 含 external variant，Python Runtime 也能注册 provider；但是浏览器只
   能新建 expression/sql 节点（`web-ui/src/App.tsx:89,112-144,806-810`），导入的
   external 节点只显示 provider 摘要、不可编辑 options
   （`web-ui/src/components/NodeInspector.tsx:171-177`），preview 输入又明确拒绝
   非 table graph input
   （`web-ui/backend/src/calc_flow_studio/run_manager.py:322-329`）。

6. **run manager 有历史记录，但 REST/UI 无法列出。**
   `RunManager` 默认保留最多 100 个本地 run，并会裁剪终态记录
   （`web-ui/backend/src/calc_flow_studio/run_manager.py:628-646,942-960`），REST
   只有按 ID 查询，没有 run collection/list route。页面刷新后客户端没有可发现方式
   恢复尚在 manager 中的 run 或比较同项目的最近运行。

7. **Studio 只观察 runner checkpoint，不运行 runner 生命周期。**
   Rust/Python 已有 micro-batch 和 streaming runner，Studio 仅提供 stateless preview
   加 checkpoint inspect/reset；没有 source open/next、stream step 或 runner status
   REST 契约。这是产品覆盖边界，不是现有实现错误。

## 候选机会池（未排序、未选择最终三个）

### A. 生成式、类型化的 Studio 能力与结果契约

候选方向：

- 为 validation、catalog/capabilities、table/array output、node timing 和
  DataFusion metric 建立 Pydantic response models；
- 让 `RunResponse.result` 使用判别联合，而不是宽泛 JSON；
- 由 OpenAPI 生成现有手写 TypeScript 类型，删除强制断言；
- 对 catalog 的兼容方案二选一：保持 `/catalog` 为 UDF-only 并修正文档，或新增
  `/api/v2/capabilities` 返回内建算子、已注册 UDF/provider、Arrow 类型和资源限制。

价值：降低 Rust → PyO3 → FastAPI → TypeScript 漂移风险，并为后续 external/array
Studio 支持提供可信的能力发现。直接把现有 `/catalog` 从数组改成对象会破坏 REST 和
Python 客户端，应避免。

### B. Python 执行上下文对齐

候选方向：

- 给 `ExecutionPlan.execute[_async]` 增加 keyword-only、数据化的 options，或新增小型
  `ExecutionOptions` Python value object；
- 首批只投影 strict-JSON `settings` 与 timezone-aware `deadline`；继续把 asyncio
  cancellation 映射到 native cancellation；
- 同步更新 `_native.pyi`、纯 Python wrapper、示例和取消/超时测试。

价值：让 Python 自定义 provider 能使用 Rust 已有的 `RunContext.settings/deadline`，
消除跨语言功能缺口。不要把 Studio 的 CPU/memory/output preview limits 混入引擎
`ExecutionOptions`，两者职责不同。

### C. 注册驱动的 external/array Studio 工作流

候选方向：

- 先基于类型化 capabilities 暴露数据化 provider metadata（绝不暴露 callback、源码或
  import path）；
- 只允许从已注册 provider 新建 external 节点，编辑其 schema/options；
- 扩展 preview 的 array input 解码，同时复用现有 spawned-worker、大小/时间/内存限制
  和 array result 序列化；
- 默认 CLI 未注册 provider 时隐藏该功能，而不是生成无法编译的图。

价值：打通核心/Python 已实现但 Studio 未覆盖的 NumPy/JAX 与可信 provider 能力。
风险集中在 callback 序列化、array 大小核算和受信任注册边界。

### D. 可发现的本地 run 历史

候选方向：

- 在现有内存 `max_history` 上增加分页 `GET /api/v2/runs`，支持 `project_id`、状态和
  cursor 过滤；
- UI 恢复当前项目最近 run，比较节点耗时和 DataFusion plans；
- 首版明确“随服务重启消失”，不要把内存列表误称为持久化审计日志。

价值：复用已经维护的状态，API/实现增量相对小，并显著改善刷新恢复和性能回归观察。

### E. Python fluent builder 的增量补齐

候选方向：

- 优先增加与 Rust 命名对齐的 `with_datafusion_config` 和显式 port/schema modifier；
- project description、run options 和 data sources 仅在有清晰常用流程时加入，避免把
  builder 变成完整 schema 的第二套表示；
- 保持 builder 返回新值，不修改 caller-owned mapping。

价值：减少高级用户在 fluent builder 与原始项目字典之间切换。应先用使用场景验证优先级，
因为完整 `ProjectDocument` 已能表达全部配置。

### F. Studio runner 生命周期

候选方向：为 micro-batch/streaming 增加独立的 runner resource、start/step/status/stop
和 checkpoint 语义，而不是扩充 stateless preview run。

价值高但范围最大；需要 source/sink 信任模型、背压、断线恢复、进程生命周期和至少一次
交付语义的独立规格。它不适合作为“顺手扩展 preview endpoint”的小改动。

## 兼容性与设计护栏

- Rust/Python/Studio 均为 `2.0.0`；Rust/Python 公共签名优先做 additive、keyword-only
  扩展。`ExpressionOperator`/`SqlOperator` 与外部 `Operator` 的新分类应继续保持。
- `/api/v2` 字段重命名或类型收窄会破坏生成客户端；优先新增 route/optional field，
  任何替换必须有版本迁移。
- `ProjectSpec` 使用 `deny_unknown_fields`，项目格式不是天然前向兼容；新增项目字段即使
  有默认值，也会被旧 reader 拒绝。真正的格式扩展需要明确版本策略。
- 项目和 catalog 必须保持纯数据；callback、源码、import path 只存在受信任的进程内
  注册边界。
- Studio 仍是 loopback-only、无认证的本地工具；不要把本地 API 提案误包装成公共托管
  服务。
- external-only 计划不得因能力发现、校验或 UI 预览而隐式初始化 DataFusion。
- REST/Python/Rust 共同变化必须同步 schema、`web-ui/openapi.json`、
  `web-ui/src/api/schema.d.ts`、`_native.pyi`、示例和受影响测试。
- 依赖边界固定为 Rust 1.88、DataFusion 54、Python 3.13+、PyArrow 24.x；设计需保留
  当前 ABI/Arrow C Data Interface 边界。

## 近期窗口证据

`origin/main` 的 first-parent 记录在窗口内包含 PR #14 至 #26：

- #14：lazy DataFusion / external-only engine isolation，并引入上述 Rust 公共分类；
- #15-16：Windows Studio launchers 与 release 环境修复；
- #17：Studio 多数据源、SQL alias 和面板交互；
- #18：全规模 benchmark CI；
- #19、#25：agent-team 基础设施；
- #20-24、#26：文档、示例和 Markdown 一致性。

从 `5be0e5a6`（窗口开始前最后一条 `main` 提交）到 `f71e49d7`：

- Rust core/Python provider 和 React UI 有变更；
- `schemas/project-v2.schema.json`、`web-ui/openapi.json`、
  `python/calc_flow/_native.pyi`、`python/calc_flow/__init__.py` 均未变。

因此本周公共契约的实质变化集中在 Rust engine boundary；Studio 进展主要补齐既有 v2
REST/项目契约的 UI 覆盖。

## 验证记录与限制

- `multica issue get`：确认 DAL-5 描述和固定窗口；metadata `{}`、recent comments `[]`。
- `git status --short --branch`、`git rev-parse`、`git for-each-ref`：确认开始时代码树干净，
  当前 HEAD 与 `origin/main` 都是 `f71e49d7`。
- `git log --first-parent --since ... --until ... origin/main`：确认窗口内 PR #14-#26。
- `git diff --quiet 5be0e5a6..origin/main -- <contract>`：项目 schema、OpenAPI 和 Python
  adapter/stub 均未改变。
- `jq` 检查 OpenAPI：版本 `2.0.0`，所有 11 个 path 均以 `/api/v2/` 开头；
  `/catalog` 的 200 schema 为 `array<object>`。
- `python scripts/test_release_config.py`：11 tests，全部通过。
- `cargo metadata --no-deps --format-version 1`：`calc-flow` 与
  `calc-flow-python` 均为 `2.0.0`、`rust_version=1.88.0`。
- 没有运行完整编译/测试矩阵：本任务是只读接口审计，结论依赖当前 `main` 源码、已签入
  schema/OpenAPI、静态测试和 git 历史，而不是新实现验证。
- 未查询 GitHub 实时 open/closed PR 状态；“近期窗口证据”仅陈述本地
  `origin/main` first-parent 可证明的合并记录。
- 本次没有修改源代码、生成文件、分支、提交或 PR；唯一有意写入是本审计文档。
  共享工作树中的 `.codex/artifacts/analysis/` 是并发任务产生，本文未触碰。
