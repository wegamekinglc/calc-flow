# DAL-5：每周能力进展批评与新功能优先级

## 结论

两份上游材料在固定窗口、代码基线、合并集合和主要公共接口缺口上相互一致，可以作为
本周决策输入。报告窗口为 2026-07-18 15:00 至 2026-07-25 15:00
（Asia/Shanghai，即 2026-07-18T07:00:00Z 至
2026-07-25T07:00:00Z），审计修订为
`f71e49d7ccba117ade55270bd9df7499c04769cb`。本地 `HEAD` 与
`origin/main` 都指向该提交。

本周交付结论必须维持一条硬边界：PR #14–#26 已合并到 `main`，PR #27 仍然开放，
不能列为已交付能力。#27 当前 12 个检查中 11 个通过、`Rust core parity` 失败；
失败发生在 `Run Rust tests`，后续 Rust coverage 与 rustdoc 步骤被跳过。PR 分支的
Rust job 只安装 PyArrow，而改动新增了由 rust-numpy 支撑的 NumPy owned-array Rust
测试，因此“测试环境缺少 NumPy”这一诊断有代码、工作流和失败步骤三方支持。

本批评选择且只选择三个按序提案：

1. 类型化的 Studio capabilities、validation 与 run-result 契约
   （立即可靠性/契约工作）；
2. Python `ExecutionOptions` 对齐
   （立即可靠性/公共 API 契约工作）；
3. 跨 Rust、Python 与 Studio 的逐批次 `head` operator
   （用户可见功能）。

在开始任何新功能实现之前，应先恢复 #27 的绿色门禁。该修复是发布前置条件，不计入
上述三个新功能提案。

## 证据复核

### 固定窗口与仓库历史

| 检查项 | 复核结果 | 评价 |
| --- | --- | --- |
| 时区换算 | `2026-07-18 15:00 +08:00` 至 `2026-07-25 15:00 +08:00` 等于 `07:00Z` 至 `07:00Z` | 正确 |
| 审计修订 | `HEAD == origin/main == f71e49d7`；提交时间为 2026-07-25 04:00:52 +08:00 | 正确，早于窗口终点 |
| 窗口前基线 | `git rev-list -1 --before=...` 返回 `5be0e5a6` | 正确 |
| 可达提交 | 窗口内 `origin/main` 可达提交数为 104 | 与进展报告一致 |
| first-parent | 13 个 merge commit，覆盖 PR #14–#26 | 与进展报告一致；实际合并次序中 #22 早于 #21，不影响集合 |
| 净变更 | 110 files changed，19,871 insertions，598 deletions | 与进展报告一致 |
| `main` CI | run `30122407953` 在 `f71e49d7` 上完成且成功 | 支持“该审计修订门禁绿色” |

窗口边界应在后续周报中明确写成闭区间还是半开区间。当前没有提交恰好落在两个端点，
所以该歧义不改变本期结果。

### GitHub PR 与 CI

- GitHub PR 元数据确认 #14–#26 均为 `merged=true`，merge SHA 与本地
  first-parent 历史一致。
- #27 为 `state=open`、`merged=false`、`draft=false`，head 为
  `412798264506e4e5a585462014aaea2d2d7fe389`，包含 12 个提交。
- `gh pr checks 27` 返回 12 行：11 个 `pass`、1 个 `fail`。唯一失败为
  `Rust core parity`，run `30146006588`、job `89647987554`。
- job API 显示 format 与 clippy 成功，`Run Rust tests` 失败；coverage 工具安装、
  coverage 门禁、coverage 上传和 rustdoc 全部跳过。
- #27 版本的 CI workflow 在 Rust job 中执行
  `python -m pip install "pyarrow==24.0.0"`，没有安装 NumPy；PR diff 同时新增
  rust-numpy 依赖及 `owned_numpy_adoption_preserves_identity...` Rust 测试。

GitHub check API 没有返回 commit status context，且本次 `gh run view --log` 未返回可引用
的原始 stderr 行。因此“缺少 NumPy”的精确异常文本来自上游进展报告；本次独立复核确认
了失败步骤、依赖差异和触发该依赖的新测试，但没有重新获得原始日志文本。该限制不影响
“#27 当前不能合并或视为已交付”的结论。

## 对上游分析的批评

### 证据质量

上游进展报告的优点是把 `main` 已交付能力、开放 PR 和尚未实现的规格明确分开，并用
源码、git 历史、GitHub PR/CI 与成功的 `main` run 交叉验证。API 审计进一步定位了
Rust、PyO3/Python、REST/OpenAPI 和 TypeScript 之间的具体漂移点；两份材料对
`f71e49d7` 的事实描述没有实质冲突。

以下措辞需要收紧：

- “14 个活跃 PR”应定义为“在窗口内合并，或在窗口内创建/更新且窗口结束时仍开放”。
  #14 的创建时间早于窗口，但合并时间在窗口内；若把“活跃”理解为“窗口内创建”，计数
  会产生歧义。
- “当前已实现能力”准确含义是“审计修订的 `main` 源码已实现且该修订 CI 成功”，不是
  “已发布到包仓库”或“所有已安装客户端已获得”。材料没有检查 crate、wheel 或 Studio
  发布渠道的最新公开版本。
- 19,871 行新增不能直接解释为同等规模的用户功能进展。窗口包含 agent-team 迁移、
  设计记录和大范围文档工作，代码量只能描述变更规模，不能充当价值或速度指标。
- benchmark 四档运行目前仍是信息性 artifact；在同机可比 `main` 样本达到既定数量前，
  不应把它们表述为性能回归门禁。
- 源码行号在未来提交上会漂移。两份材料已经给出审计 SHA，后续引用时应继续将行号与
  `f71e49d7` 绑定，而不是把它们当作浮动 `main` 的永久位置。

### 兼容性与可行性

1. `/api/v2/catalog` 的现有数组响应已经是公共行为。把它直接改成能力对象会破坏 REST
   客户端；新增 `/api/v2/capabilities` 才是可接受的 additive 路径。
2. `RunResponse.result`、validation 与 catalog 使用宽泛 JSON/手写 TypeScript 类型，
   会把跨层漂移推迟到运行时。该风险在扩展 array/external Studio 流程前应先处理。
3. Python 执行对齐可以复用 Rust 已有 `ExecutionOptions`，不需要修改项目格式或
   DataFusion 计划结构，技术范围相对可控。必须继续使用 additive、keyword-only 入口。
4. `head` 规格声称变更“additive”，只对旧项目由新 reader 加载成立。新增
   `kind: "head"` 后，旧 v2 reader 会拒绝未知 operator variant；这不是双向兼容。
   实现前必须明确：v2 是否允许 capability-dependent 的新增 operator kind，以及旧
   runtime 的确定性失败是否为正式兼容策略。
5. `head` 规格中“R <= N 时 emitted unchanged”不能暗示 Python/Rust 对象身份不变；
   DataFusion 路径可能产生新的 Arrow batch。验收条件应约束行内容、顺序、schema 与
   metadata，而不是对象 identity。
6. `head` 的 DataFusion-vs-direct-Arrow 路径与 Studio 范围仍是阻断设计问题。为了
   保持“table operator 由 DataFusion 执行”的现有边界，本批评推荐使用 DataFusion
   `LIMIT`，并将 `requires_datafusion()` 明确为 `true`；为了兑现用户可见功能，Studio
   palette 与 inspector 也应在本提案范围内。

## 阻断项与非阻断限制

### 阻断交付或实施的事项

- **#27 门禁失败。** 在 Rust job 显式安装与 all-features Rust 测试匹配的 NumPy
  依赖并重新跑绿之前，#27 不可合并、不可列为已交付；coverage 与 rustdoc 也必须实际
  执行而不是保持 skipped。
- **`head` 引擎语义尚未定案。** 实现前需批准 DataFusion `LIMIT` 路径，或正式修改
  “table-only 使用 DataFusion”的架构约束；不能由实现者临时选择。
- **v2 operator 扩展策略尚未明文化。** 需明确旧 v2 runtime 对 `kind: "head"` 的
  unsupported-capability 错误是预期行为，并记录最低 runtime/capability 要求；否则
  “format_version 不变”与“旧 reader 可用”的含义不清。

没有阻断证据缺口妨碍完成本周能力盘点或给出以下排序；上述是发布/实施门禁，而不是要求
重做本期报告。

### 非阻断证据限制

- 未取得用户采用率、支持工单、运行频率或失败率，因此排序依据是技术风险、公共契约
  缺口、依赖关系和可交付范围，不代表经用户研究验证的需求强度。
- 未审阅 PR discussion、review thread 或外部发布数据；PR 描述不是独立验证。
- #27 的原始 NumPy ImportError 文本未在本次复核中重新获得，但依赖与失败路径已
  交叉支持诊断。
- GitHub 状态是 2026-07-25 本次审计时的快照；开放 PR 后续变化不得反写为窗口结束时
  已交付。
- 性能基线样本不足，`head` 的性能验收应使用专门、可重复的微基准并报告区间，不能用
  单次“无可测回归”声明代替。

## 按优先级的新功能提案

### 1. 类型化的 Studio capabilities、validation 与 run-result 契约

**类别：** 立即可靠性/契约工作。

**用户价值：** Studio 能准确发现当前进程真正支持的 operator、UDF、external provider、
Arrow 类型和 preview 限制；前端在编译期发现响应漂移，而不是在结果面板中依赖
`as unknown as` 后于运行时失败。该基础同时降低后续 external/array Studio 功能生成
不可运行项目的概率。

**提议范围：**

- 保持现有 `/api/v2/catalog` 的 UDF-only 数组形状不变，并修正文档对它的过宽描述。
- 新增 additive 的 `/api/v2/capabilities`，返回版本化、纯数据、确定排序的
  `CapabilitiesResponse`。
- 用 Pydantic 模型表达 capabilities、`ValidationReport` 和 run result；run result
  使用带 `kind` 判别字段的 table/array 联合，而不是 `dict[str, JSONValue]`。
- 从 OpenAPI 生成对应 TypeScript 类型，删除 `ValidationReport`、
  `RunResultPreview` 的重复手写形状和结果面板的强制断言。
- provider 能力只暴露已注册的 provider/name/version、端口种类、数据化 options
  schema 等白名单元数据；不暴露 callback、源码、import path 或 secrets。

**受影响表面：** PyO3/Python runtime 的数据化能力快照、Studio backend models/routes、
`web-ui/openapi.json`、生成的 TypeScript schema、API client、结果/校验 UI、API 文档
与相应测试。项目 `format_version` 与 checkpoint 不在范围内。

**验收条件：**

1. 现有 `/api/v2/catalog` 的响应兼容测试保持通过；新 route 具有具体 OpenAPI schema，
   不再是 `array<object>` 或无约束 object。
2. capabilities 对无注册、仅 UDF、仅 external provider 和混合注册 runtime 返回
   准确、稳定排序的纯数据结果。
3. `ValidationReport` 和 `RunResponse.result` 全部由 OpenAPI 生成 TypeScript 类型；
   `web-ui/src/types.ts` 不再重复声明这些形状，结果面板不再使用双重类型断言。
4. table 与 array 结果分别通过判别联合序列化、反序列化和前端渲染测试；未知 kind
   确定性失败。
5. capability JSON 中没有 callable、源码、import path 或注册时 secrets；快照可由
   spawned worker 安全重建。
6. external-only runtime 的 capabilities 查询、验证和结果建模不创建 DataFusion
   session，现有 lazy-DataFusion 回归测试保持通过。
7. OpenAPI、生成 TypeScript、backend、frontend 和文档同步门禁全部绿色。

**依赖：** 需要 `cf-api-designer` 先固定 response model、capability version 与
provider metadata 白名单；不依赖 #27 合并，但后续 array Studio 工作依赖本提案。

**风险与缓解：**

- 动态注册会让不同进程的能力不同；响应必须标识 runtime/session 范围，不能宣传为全局
  安装清单。
- 直接扩充 `/catalog` 风险最高，因此明确禁止改变其顶层形状。
- provider metadata 可能泄露受信任边界；采用显式白名单和序列化测试。
- 把所有结果一次建模会扩大改动面；按 capabilities/validation/result 三个独立模型
  实现，但在同一 OpenAPI 一致性门禁下交付。

**排序理由：** 这是 Studio 后续 external/array 编辑、运行历史和 runner UI 的共同
契约基础，也是当前已有的具体跨层漂移风险；先做可避免后续功能继续复制手写类型。

**证据追踪：** API 审计缺口 1、2、5；兼容性护栏中的 `/api/v2` additive 原则、纯数据
注册边界与 external-only DataFusion 隔离。

### 2. Python `ExecutionOptions` 对齐

**类别：** 立即可靠性/公共 API 契约工作。

**用户价值：** Python 用户可以给自定义 provider 传递每次运行的严格 JSON settings
和绝对 deadline，获得与 Rust `RunContext` 相同的可配置和超时语义；同步和异步执行不再
被固定在 `ExecutionOptions::default()`。

**提议范围：**

- 新增不可变、数据化的 Python `ExecutionOptions` value object，首版只包含
  `settings` 与 `deadline`。
- 给 `ExecutionPlan.execute` 和 `execute_async` 增加 keyword-only
  `options: ExecutionOptions | None = None`；省略时完全保持当前默认行为。
- `settings` 只接受严格 JSON-compatible mapping，并在进入 native 边界时防御性复制；
  `deadline` 只接受 timezone-aware `datetime`，规范化到 UTC。
- 保留当前 asyncio task cancellation 到 native cancellation 的映射；不向 Python 暴露
  Rust `CancellationToken`，也不把 Studio CPU/memory/output preview limits 混入该类。

**受影响表面：** PyO3 execution binding、纯 Python wrapper/top-level exports、
`_native.pyi`、Python API 文档、examples 与 Python/Rust binding 测试。项目 schema、
Studio REST 和 Rust `ExecutionOptions` 结构不需要改变。

**验收条件：**

1. 不传 options 的同步/异步执行与当前结果、错误和取消行为完全一致。
2. 注册的测试 provider 在同步和异步路径中收到相同的 settings 与 UTC deadline；
   provider 不能修改 caller-owned mapping。
3. 非 JSON 值、非 mapping settings、naive datetime 和无效 deadline 产生字段明确的
   Python 异常；已经过期的 deadline 在 operator 执行前失败。
4. deadline 到达时执行按现有 rollback 语义退出；有状态 operator 的 snapshot 恢复测试
   通过。
5. asyncio cancellation 继续优先映射到 native cancellation，且不会因 options 新增
   重复 token 或后台 task 泄漏。
6. `ExecutionOptions` 在 `__init__.py`、wrapper 签名和 `_native.pyi` 中一致，文档明确
   engine deadline 与 Studio preview limits 的职责边界。
7. Rust/PyO3、Python pytest、ruff、stub/文档同步门禁绿色。

**依赖：** 需要 `cf-api-designer` 固定 value object 的构造、deadline 精度/时区转换和
错误映射；底层 Rust 能力已存在，不依赖 #27 或提案 1。

**风险与缓解：**

- Python wall-clock datetime 与 Rust deadline 表示转换可能产生精度或时钟语义差异；
  明确使用绝对 UTC deadline，并覆盖已过期/将来/边界测试。
- settings 可能含共享可变对象；执行前严格 JSON 编码或深复制，禁止 callback。
- 为未来字段过度设计会扩大 API；首版只投影 Rust 已消费的 settings/deadline。

**排序理由：** 缺口已由源码直接证明，底层能力现成、改动独立、兼容路径清晰，能以较小
范围消除真实的 Rust/Python 功能差异；排在提案 1 之后是因为它不解决当前 Studio 的
跨层类型漂移。

**证据追踪：** API 审计缺口 3；`crates/calc-flow-python/src/pipeline.rs` 的同步与异步
路径均使用 `ExecutionOptions::default()`，而 Rust `ExecutionOptions` 已包含 settings、
deadline 与 cancellation。

### 3. 跨 Rust、Python 与 Studio 的逐批次 `head` operator

**类别：** 用户可见功能。

**用户价值：** 用户可以显式限制每个 table batch 的前 N 行，用于预览、调试和下游成本
控制，不再手写 `SELECT * FROM input LIMIT N`；图、项目文件和 Studio inspector 都能
直接显示这一行数边界。

**提议范围：**

- 按 `.codex/artifacts/specs/head-operator.md` 实现 stateless、per-batch 的
  `head(n)`，只接受 table input，N 为正整数。
- 采用 DataFusion `LIMIT` 执行路径，保持现有 table operator 架构；head-only plan 的
  `requires_datafusion()` 为 `true`。
- 新增 `kind: "head"` 项目 variant、Rust builder/operator、Python functional builder，
  以及 Studio palette node 与 N inspector。
- 明确不是跨 batch 的 global head，不改变 runner、source、checkpoint 或 array/external
  语义。

**受影响表面：** Rust operator/config/pipeline 与测试、项目 JSON Schema、
Python binding/wrapper/stub、Studio backend validation/OpenAPI、React palette/inspector、
文档、examples 与 targeted benchmark。checkpoint 格式不变。

**验收条件：**

1. 对 R=0、R<N、R=N、R>N 的每个 batch，输出为前 `min(N,R)` 行，顺序、值、Arrow
   schema 与 batch metadata 保持一致；不要求输出对象 identity 与输入相同。
2. Rust `n=0` 以及 Python 的零、负数、bool 和非整数输入在构造期以字段为 `n` 的明确
   invalid-argument 错误失败。
3. array-to-head wiring 在编译期以 port kind mismatch 失败；head-only plan 明确需要
   DataFusion，且能报告现有 DataFusion/node timing/row-count 指标。
4. JSON/YAML 项目 round trip 保留 `kind: "head"` 与 N；未知字段失败。旧 runtime
   对 head 项目返回 documented unsupported-operator/capability 错误，不静默降级。
5. micro-batch 与 streaming 按每个输入 batch 独立截断；sink 失败后的 at-least-once
   redelivery 产生相同输出，snapshot 为 null、非 null restore 失败。
6. Python builder 保持 functional/immutable；Studio 可新建、编辑、保存、导入、预览
   head node，N 的客户端与服务端校验一致。
7. 现有项目、checkpoint 和不含 head 的 Python/Rust 调用全部回归通过；schema、
   OpenAPI、TypeScript 与 `_native.pyi` 同步。
8. 增加专门的 head overhead 微基准并报告样本、区间和基线；现有四档 benchmark 继续
   运行，但在样本门槛满足前不声称其构成统计性能门禁。

**依赖：** 实现前由 `cf-api-designer` 固定 v2 capability 扩展策略和 public shape，
由 `cf-critic`/架构维护者批准 DataFusion 路径；Studio 能力发现最好依赖提案 1，但核心、
Python 和固定 palette 实现不必等待 #27。

**风险与缓解：**

- 为简单 slice 启动 DataFusion 有固定开销；用 targeted benchmark 量化，不能以破坏
  既有 table-engine 边界的临时优化规避。
- 新 operator kind 对旧 v2 reader 不是前向兼容；通过 capabilities、最低 runtime
  要求和确定性 unsupported 错误显式管理。
- per-batch head 容易被误解为全局 LIMIT；API、node label、文档和 runner 测试都使用
  “per batch”措辞。
- Studio 若只支持 schema 而不提供 palette，会让“所有公共表面”目标落空；本提案明确
  纳入 palette/inspector。

**排序理由：** 规格已存在、行为边界小且用户可见，但仍有 DataFusion 路径、v2
capability 兼容和 Studio 范围三个设计决定；因此排在两个可独立交付的契约修复之后。

**证据追踪：** 进展报告缺口 2；`.codex/artifacts/specs/head-operator.md` 的 goals、
FR1–FR9、acceptance criteria 与两个 open questions；API 审计的严格项目兼容护栏。

## 取舍说明

Studio external/array 工作流具有较高用户价值，但它同时依赖一个绿色、已合并的 #27 和
提案 1 的可信 capabilities/typed result 契约；本期不将它列入前三，避免在两个未稳定
基础上扩大 callback 序列化、array 大小核算和信任边界。Studio runner 生命周期需要独立
的 source/sink、背压、恢复和进程生命周期规格，范围明显大于一周提案。内存 run 历史
列表与 Python fluent builder 补齐都可继续保留在机会池，但当前风险降低或用户价值证据
弱于以上三项。

## 实际检查与写入边界

- 读取并逐节对照：
  - `.codex/artifacts/analysis/dal-5-weekly-capability-progress-2026-07-25.md`
  - `.codex/artifacts/api-notes/dal-5-public-surface-audit.md`
  - `.codex/artifacts/specs/head-operator.md`
- 运行只读 git 检查：`rev-parse`、`show`、窗口内 `log`/`rev-list`、
  first-parent log、`diff --shortstat`、`diff --name-only` 与 `status --short`。
- 运行只读 GitHub 检查：PR #14–#27 metadata、`gh pr checks 27`、
  Actions run/job metadata、#27 workflow 与 PR diff 的依赖交叉检查，以及
  `main` CI run `30122407953`。
- 用 `rg` 对 Runtime catalog、REST/Pydantic/TypeScript 结果类型、Python
  `ExecutionOptions::default()`、runner single-input 与 head 规格引用做静态追踪。
- 未运行 build、test 或 benchmark；本任务不修改实现，现有 CI 证据用于确认审计修订与
  PR 状态，不用于宣称新功能通过。
- 未修改源码、测试、生成文件、分支、提交、PR 或远端状态。唯一有意写入是本批评文件；
  上游 analysis/api-note 是共享工作树中已存在的并发产物，本任务未改动。
