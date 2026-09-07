# DAL-189 文档与 Agent 一致性审计（2026-09-07）

本次从远端最新 `main` 提交
`5c1cac1255e826ea4655244963c891681b14b97c` 建立独立 worktree，
工作分支为 `docs/DAL-189-audit-20260907`。开始时工作区干净，
`git fetch origin main` 后的 `origin/main` 与
`git ls-remote origin refs/heads/main` 一致。
首次审计提交为 `c0e3cc77ca8ac2aa05f8fe056d7b78d5e99f928a`。
本轮按 `dal-189-independent-review-20260907.md` 的 Request Changes
修复 B1/B2/B3，继续使用该独立分支；fetch 与 ls-remote 确认 main 基线未变。
修订提交和推送结果在 issue 交接中记录；未创建 PR、未合并，交回协调器
安排 `cf-reviewer` 定向复审。本报告将首次审计证据与本轮补充检查分开列出。

## 枚举范围和审计边界

基线共有 **114 个 Markdown 文件、43,693 行**。枚举使用
`git ls-files -z`，按大小写无关的 `.md` / `.markdown` 后缀筛选，
包含被忽略规则覆盖但仍受版本控制的文件，未只扫描常见文档目录。

| 范围          | 数量 |
|---------------|------|
| 根目录        | 4    |
| `docs/`       | 23   |
| `.claude/`    | 14   |
| `.codex/`     | 59   |
| `.agents/`    | 1    |
| `benchmarks/` | 6    |
| `crates/`     | 2    |
| `design/`     | 2    |
| `examples/`   | 1    |
| `web-ui/`     | 2    |

下文列出全部路径和基线摘要；完整 SHA-256、外链响应、字段同步结果及
检查统计见 [审计证据](dal-189-docs-audit-20260907-evidence.json)。
本报告是新增加的第 115 个 Markdown 文件，不计入基线 114 个。

全部文件接受链接、锚点、空白及表格结构扫描。当前用户文档 30 份、
维护/agent 指南 16 份按当前源码、配置与生成契约核对。其余 67 份为
工程或历史记录，另有一份 CHANGELOG；其日期、旧签名、未完成计划和
当时的测量结论不作为当前 API 承诺，也不据此改写历史。
两个保留的 Claude 工程记录仍与 canonical 副本逐字节一致。
基线历史记录中 16 个文件的 62 个表格存在管道对齐差异，作为历史样式
遗留记录保留；当前 benchmark README 的一处表格已随修订对齐。

这是全仓静态文档审计，不构成全量运行时回归、发布安装或性能结果证明。
没有改写示例程序、产品源代码、测试或生成契约。

## 发现、依据与修订

| 发现                                                                                    | 当前依据                                                                                                           | 修订                                                                                      |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| API reference 仍写 capability schema 2，与 Python API 页冲突                            | `python/calc_flow/capabilities.py`、Studio `models.py`、`web-ui/openapi.json`、`web-ui/src/api/decoders.ts` 均为 3 | 更新 schema 值，补齐 connector 与 `state_layouts` 说明                                    |
| Runner 签名未反映连接器项目的默认绑定                                                   | `python/calc_flow/runtime.py` 的 `StreamingRunner.__init__` 区分项目计划与图计划                                   | 列出实际参数名和默认值，说明项目计划拒绝外部绑定/config 覆盖                              |
| Studio 后端 README 指向未发布的 PyPI 包，并误写核心依赖名                               | `web-ui/backend/pyproject.toml`、`.github/workflows/release.yml` 只发布 `calc-flow-python`                         | 改为源码 wheel 安装入口及已有的 `--no-sync` 启动命令                                      |
| Studio README 的构建命令没有安装 wheel，启动还会重新同步环境                            | `web-ui/package.json`、托管启动器和 getting-started 安装流程                                                       | 区分构建与安装，复用已准备环境，明确两个终端的根目录                                      |
| AGENTS/CLAUDE 连续执行命令时，backend 后的 `cd web-ui` 路径错误                         | 仓库目录、两个 pyproject、frontend package.json                                                                    | 用 subshell 限定 Studio 工作目录；两份命令块保持一致                                      |
| Implementer 列出的 `checkpoint` / `io` 核心模块已不存在                                 | `crates/calc-flow/src/lib.rs` 模块列表                                                                             | 同步当前 `continuous/`、`state/`、`connector/`、`static_input`、`time/`                   |
| 三个 agent 仍把直接 llvm-cov 命令当作完整覆盖率入口                                     | `scripts/run_rust_coverage.py`、Linux CI、AGENTS 要求四类服务和合并 profile                                        | 指向受维护的 coverage harness 及其服务环境前提                                            |
| Reviewer 暗示 unsafe 可用注释豁免，且所有旧检查点都可恢复                               | Cargo 的 `unsafe_code = "forbid"`；manifest 版本校验及 state backend 的 v2 拒绝用例                                | 保持禁止 unsafe；按受支持版本、lineage、fingerprint 和布局核验恢复                        |
| 性能 agent 仍称 CI 仅提供信息、等待 20 个样本才有门禁                                   | benchmark suite catalog、report、statistics、aggregate 及工作流                                                    | 说明当前 engine/warm 两轮 +5% 置信下界门禁；把独立 contract-v2 最小值诊断限定在其自身范围 |
| benchmark README 下半部分仍写 PR lifecycle 用 overhead、Rust 只跑两个目标且没有其他门禁 | `scripts/benchmark_suite/catalog.py`、`legacy.py`、`rust.py` 和工作流                                              | 对齐 standard lifecycle、动态 Rust 目标清单及各自门禁范围                                 |
| Doc writer 表格仍将已归档发布页写作当前文件名                                           | canonical 已有的固定提交文档归档链接                                                                               | 同步表格及 Multica 中落后的归档段落                                                       |
| Claude 样式指南有 `Result<T}` 拼写错误且遗漏文件末尾规则                                | AGENTS 与 `.agents/skills/code-style/SKILL.md`                                                                     | 修复类型拼写并补齐无尾随空白、末尾换行要求                                                |

其余当前表面按以下源码组交叉核对：Rust crate exports 和 state/continuous
入口；Python `__init__.py`、`_native.pyi`、pipeline/runtime/store/capabilities；
project-v3 schema；Studio OpenAPI 的 15 个路径、19 个操作；connector
Cargo feature 和当前 MySQL 注册；symbolic rolling 布局常量、lowering
声明与 Python capability 页已写明的 catalog 限制；14 个编号 Python
示例及 4 个 Rust 用户示例清单；发布工作流及 benchmark catalog。
workspace、Python、Studio、frontend 和 OpenAPI 版本均为 `4.0.0`。
未发现需要本角色修改示例程序的阻断项。

### CHANGELOG 判定

本次样式、路径和维护说明修订不单独增加 changelog。
审计发现 **2026-09-03 的公共 capability schema 3 变更漏记**，因此补一条
该日期的基本契约变更记录。依据是当前 main 的两个祖先提交
`e019cf5685b34067180fde2afbfa43e3ed69806a`、
`58a4c13032327b4b24b494e5ba00da09952b4a53`，以及
`.codex/artifacts/api-notes/symbolic-computation-engine.md` 的既有修订。
没有为本次文档同步虚构新的 API 发布。

## Agent 映射及同步

每个名称映射到 `.codex/agents/<名称>.toml`、
`.claude/agents/<名称>.md` 和下表唯一 Multica ID；该表记录首次审计的文本写入。
字段语义为 canonical `description` → Multica `description`，
canonical `developer_instructions` → Multica `instructions`。
Claude 保留 frontmatter、工具名称、worktree 工具及样式指南入口等客户端差异。

| 名称            | Multica ID                           | 首次审计文本结果                           |
|-----------------|--------------------------------------|--------------------------------------------|
| cf-orchestrator | a6c22a43-3dc1-4338-8581-3e1317dd9519 | 摘要一致；未写入                           |
| cf-implementer  | 227e58dc-a05a-48b6-bd20-94af140da251 | instructions：模块路径                     |
| cf-tester       | a553f49e-ae3c-401e-88d4-919d9e078703 | instructions：覆盖率入口与门槛             |
| cf-api-designer | ea5b9d56-eb67-4d40-ba30-d07054d02f77 | 摘要一致；未写入                           |
| cf-critic       | d29d22b4-ec2a-4a2c-ab7b-861f7f4d0805 | 摘要一致；未写入                           |
| cf-reviewer     | a274f526-62bc-481f-93ef-67b4a63d9f3a | instructions：unsafe 与检查点契约          |
| cf-doc-writer   | 1b71d01f-a1e8-4e73-97f0-3a2576a7378c | instructions：归档位置及表格               |
| cf-simplifier   | 113b511d-172f-4673-becd-cc0db10b70b8 | 摘要一致；未写入                           |
| cf-performancer | b37225a5-8640-4da1-9500-89adfab9c66b | instructions：benchmark 范围与当前 CI 门禁 |
| cf-spec-writer  | 62f89f6b-8158-4c5d-ac1d-8ed6d092473f | 摘要一致；未写入                           |

10 个 canonical/Claude/Multica 目录摘要完全一致，无需 description 写入。
首次审计的 17 项仓库运行说明修订已从 canonical 同步到 Claude。5 个 Multica agent
仅发送 `--instructions`；每次写入前重新读取记录，确认名称、ID、原文本
未被并发修改，再读回验证预期文本。除 `instructions` 和服务端
`updated_at` 外，所有可读取字段逐值相同，目录摘要也未改变。
未发送模型、thinking level、service tier、runtime、并发、技能、
权限、环境或 MCP 配置的任何更新参数；未读取明文环境秘密，也未改动 squad。

### 独立评审修复与差异分类

同步验收采用**canonical 语义一致、授权适配逐项列明**，不要求运行说明
字节相同。评审已确认 114 项基线清单、行数/哈希和现有事实修订；本轮
没有重做全仓审计，也没有将现行紧凑协议删去或重插默认本地全矩阵。

| 阻断                     | canonical 修复及下游证据                                                                                                                                                                                                                                                                  | 状态               |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| B1 本地验证范围冲突      | `AGENTS.md` Verification、canonical code-style、implementer Phase 3/4/5、tester Phase 4/4B、reviewer Step 4、simplifier Apply Mode 统一最小定向检查；spec-writer 验收模板也明确 CI 职责。完整命令、Rust 90%/Studio 85% 门槛保留；Claude、Multica 对应段落同步                             | 已修订，待定向复审 |
| B2 协调工具自相矛盾      | orchestrator 的 Tool Restrictions、Analyze、Plan、Delegate、Do/Do NOT、Remember 同时收口；只允许授权上下文/issue/check 元数据读取、派发和交接，唯一文件例外为平台要求的临时评论正文。仍禁止实现、测试、专业 artifact 审查、git、任意仓库编辑、PR/merge；明确 Multica 不得后台退出等待回调 | 已修订，待定向复审 |
| B3 checkpoint 场景误路由 | performancer Phase 1 将 batch fan-out 留给 `benchmarks/test_runtime.py`，stream/checkpoint/recovery 指向 `benchmarks/test_symbolic_baseline.py::test_stream_window_checkpoint_and_recovery` 和 unified `lifecycle`，必要时补相关 Rust target；保留专用证据合同                            | 已修订，待定向复审 |

B1 的三类本地全量例外为：明确用户请求、复现/诊断 CI 失败、无 CI 覆盖且
明确高风险；需先写原因与限定范围。提交/推送或评审交接最多一次非阻塞
CI 快照；pending 可交接，不能当作 green 或 merge-ready。required
test/coverage/cross-platform/performance failure 仍阻止合并，最终 reviewer
不可跳过。AGENTS 完整命令块与 CLAUDE 副本保持不变且相等。

B2 的共享流程与适配规则写入 `.codex/agents/README.md`，再同步 Claude
README；清晰任务默认 implementer → reviewer → 必要时 doc-writer，
spec/API/critic/tester 按实际需要介入。协调工具可用不构成额外专业权限。

B3 依据 `scripts/benchmark_suite/legacy.py` 的 lifecycle pytest 节点、
catalog 和 `scripts/verify_stream_lifecycle_evidence.py`：standard scale，
至少 20 轮与诊断样本，阶段耗时/分位数、checkpoint bytes、RSS、恢复正确性
及 provenance。该证据合同不能替换成 standalone minima 判定。本轮未测性能。

| Agent           | 保留差异与本轮处理                                                                                  |
|-----------------|-----------------------------------------------------------------------------------------------------|
| cf-orchestrator | 修复 B2；保留紧凑协议和既有 CI 责任附录，平台持久 issue 派发与生命周期为授权适配                    |
| cf-implementer  | B1 定向 RED/GREEN、refactor 和交接已进入 canonical；保留紧凑协议与角色补充                          |
| cf-tester       | B1 最小验证及完整 Playwright 的 CI/例外职责已对齐；补回测试失败诊断和生产 bug 交 implementer 的边界 |
| cf-reviewer     | B1 最终质量门保留、避免重复检查；pending/失败分类与不合并边界一致                                   |
| cf-simplifier   | B1 coherent batch 与最小检查已对齐；仅授权 apply、按需非阻断的补充保留                              |
| cf-spec-writer  | 验收模板明确 local/CI 分工；轻量主稿、最多一次阻断校正作为有依据补充保留                            |
| cf-api-designer | 无需写入；公共契约变化时介入和一次校正的流程补充保留                                                |
| cf-critic       | 无需写入；Blocking 限错误实现/兼容性/验收失败，其他发现为 caveat 的补充保留                         |
| cf-doc-writer   | 无需写入；常规文档一次收口、真正新概念才需 critique 的补充保留                                      |
| cf-performancer | 修复 B3；保留按需性能派发补充，不豁免已失败 required gate                                           |

本轮 7 个 Multica agent 仅更新 `instructions`，其余 3 个只读核对。
10 个运行说明去掉已列明的紧凑协议/角色补充后，与 canonical 正文在首尾
空白归一化后相等；所有既有补充逐字节保留。10 个 description 均一致。
Claude 保留工具名、EnterWorktree、样式入口、CLAUDE 指引及换行/表格格式
差异；frontmatter 未改。其他非文本配置没有写入，读回可见字段无变化。
证据 JSON 的 `agent-sync` 为本轮逐项读回、哈希与保护字段比较；
`initial-agent-sync` 保留首次修补证据。哈希不相等本身不是阻断。

## 首次审计实际检查及结果

以下为首次提交已经完成、独立评审已核对或沿用的证据，本轮不重复执行。

- `git fetch origin main`、`git ls-remote origin refs/heads/main`、
  `git rev-parse origin/main`：基线相同；从该提交建立独立分支。
- 全仓 Markdown 解析：使用 `MarkdownIt("commonmark").enable("table")`，
  遍历内联/引用链接及图片，按目标文件解析 heading slug 和重复标题编号。
  基线 552 个链接中，499 个本地目标/锚点全部通过；末轮检查结果见证据。
  示例代码块中的占位路径、loopback 服务和历史源码路径不当作在线资源运行。
- 41 个唯一 HTTP(S) Markdown 目标逐一 GET：40 个成功；Coveralls 项目页
  返回 403，徽章可读取。三个 GitHub 评论锚点用
  `gh api repos/wegamekinglc/calc-flow/issues/comments/<id>`
  核对 `html_url`，Cargo 的 `artifact-messages` 锚点也已确认。
- `python3 -m unittest scripts.test_codex_agents`：修订前、后各运行一次，
  均为 10 项通过；含 TOML、角色清单、技能声明、表格和两个保留 artifact
  的逐字节镜像约束。没有编写或更改测试。
- `python3 scripts/run_examples.py --help`：参数入口可运行。
  14 个编号 Python 示例经 `ast.parse` 语法检查，4 个 Rust 用户示例路径存在；
  API reference 的现有 Python 示例块与基线逐字节相同。
  没有执行需要 native 构建的示例，也没有把语法检查称为运行成功。
- 对受影响 Markdown 的 28 段可直接解析 shell 命令执行 `bash -n`；
  11 段含显式占位参数的模板单列。AGENTS 与 CLAUDE 首个命令块完全一致；
  用工作目录断言替代 uv/npm 执行验证 Studio subshell，未触发安装或构建。
- 静态比对 capability 常量、backend model、OpenAPI、frontend decoder；
  提取 pipeline/runtime/store/native stub 的签名核对文档默认参数；
  核对 package 版本、benchmark catalog/判定公式和 coverage harness 的前提。
- 变更文件表格对齐、无尾随空白、末尾换行；`git diff --check` 通过。
  `schemas/project-v3.schema.json`、`web-ui/openapi.json`、
  `web-ui/src/api/schema.d.ts`、产品源码和示例未改动。
- 未运行全仓测试、构建、发布安装、soak 或性能测量。广泛回归交 CI。
  当前 Linux/Windows 工作流只对 PR 或 main/master push 触发；独立分支
  push 的一次非阻塞状态快照在 issue 交接中报告，不等待或轮询。
  后续 PR 含 agent TOML，按现有分类器会进入非 docs-only 路径。

## 本轮补充检查与剩余限制

- 读取独立评审附件；核对原分支 c0e3cc7 与远端 main、分支 ref，基线不变。
- `python3 -m unittest scripts.test_codex_agents`：仅运行一次，10 项通过。
  不新增/修改测试，不执行产品构建、全量测试或性能测量。
- 仅检查本轮修改文档和 agent 正文的结构、链接/锚点、空白和 diff；
  具体计数与路径在 `final-static-checks`。报告表格按 Unicode 显示宽度对齐；
  首次表格检查仅按字符位置检查，不能用来证明中文显示宽度正确。
- 同步复核 10 个映射/摘要、7 个写入后的 instructions 和全部既有附录；
  比较 canonical 非运行说明字段、Claude frontmatter、Multica 可见保护字段。
  源码、示例、测试、生成契约和非文本 agent 配置未改动。
- 本轮新的 CI 状态仅在推送后取一次非阻塞快照并在 issue 交接报告。
  广泛回归与完整门禁由 CI 负责，无 CI 最终通过证据。

三项阻断均已按评审方向修订并同步，是否通过由 cf-reviewer 定向复审判定。
PR 仍待独立评审通过后再创建；保留工作分支，不合并。
**Coveralls 项目页继续记为未验证（首次 GET 403）**；本轮未重复访问，
徽章可读不证明页面可达或 coverage 通过。没有新增产品行为/API，
本轮不增 CHANGELOG 条目；此前补记的 schema 3 条目保留。

## 基线 Markdown 全量清单

摘要为原始文件字节的 SHA-256 前 12 位；完整值在证据文件中。

| 路径                                                                                | 分类            | 基线摘要       |
|-------------------------------------------------------------------------------------|-----------------|----------------|
| `.agents/skills/code-style/SKILL.md`                                                | 维护/agent 指南 | `eafe9d2475b9` |
| `.claude/agents/README.md`                                                          | 维护/agent 指南 | `6d84af54e343` |
| `.claude/agents/cf-api-designer.md`                                                 | 维护/agent 指南 | `ab288cb49ee4` |
| `.claude/agents/cf-critic.md`                                                       | 维护/agent 指南 | `ff02a04e9c72` |
| `.claude/agents/cf-doc-writer.md`                                                   | 维护/agent 指南 | `f621f284b381` |
| `.claude/agents/cf-implementer.md`                                                  | 维护/agent 指南 | `92e346939a9f` |
| `.claude/agents/cf-orchestrator.md`                                                 | 维护/agent 指南 | `984ada1396b9` |
| `.claude/agents/cf-performancer.md`                                                 | 维护/agent 指南 | `3f069831af6c` |
| `.claude/agents/cf-reviewer.md`                                                     | 维护/agent 指南 | `52d5dc5ceb7a` |
| `.claude/agents/cf-simplifier.md`                                                   | 维护/agent 指南 | `75ee039b48ef` |
| `.claude/agents/cf-spec-writer.md`                                                  | 维护/agent 指南 | `10244bb314df` |
| `.claude/agents/cf-tester.md`                                                       | 维护/agent 指南 | `93b9c22ff9dd` |
| `.claude/api-notes/docs-examples.md`                                                | 工程/历史记录   | `9dffa84fa1e2` |
| `.claude/rules/code-style.md`                                                       | 维护/agent 指南 | `cb248a66377b` |
| `.claude/specs/head-operator.md`                                                    | 工程/历史记录   | `4385cdd2eb98` |
| `.codex/agents/README.md`                                                           | 维护/agent 指南 | `e0b93eb8530a` |
| `.codex/artifacts/analysis/a6-public-continuous-runtime-implementation-evidence.md` | 工程/历史记录   | `af6191cd8279` |
| `.codex/artifacts/analysis/dal-5-5-2-acceptance-audit.md`                           | 工程/历史记录   | `ec9be052a4a9` |
| `.codex/artifacts/analysis/dal-5-weekly-capability-progress-2026-07-25.md`          | 工程/历史记录   | `52fbe05aafaa` |
| `.codex/artifacts/analysis/m3-delta-implementation-evidence.md`                     | 工程/历史记录   | `23e49162a8a0` |
| `.codex/artifacts/analysis/m4-state-window-implementation-evidence.md`              | 工程/历史记录   | `9a4ce845a3c4` |
| `.codex/artifacts/analysis/sce-11-static-stream-inputs-review-record.md`            | 工程/历史记录   | `69eb915d389f` |
| `.codex/artifacts/api-notes/a6-public-continuous-runtime.md`                        | 工程/历史记录   | `faae44001e1d` |
| `.codex/artifacts/api-notes/continuous-streaming-m2-completion.md`                  | 工程/历史记录   | `088033964aa8` |
| `.codex/artifacts/api-notes/continuous-streaming-runtime.md`                        | 工程/历史记录   | `e493456c8570` |
| `.codex/artifacts/api-notes/dal-45-provider-option-error-paths-contract.md`         | 工程/历史记录   | `696db3daaf43` |
| `.codex/artifacts/api-notes/dal-5-5-2-main-reconciliation.md`                       | 工程/历史记录   | `e7181fc5ca51` |
| `.codex/artifacts/api-notes/dal-5-5-2.md`                                           | 工程/历史记录   | `a9477bd20d59` |
| `.codex/artifacts/api-notes/dal-5-public-surface-audit.md`                          | 工程/历史记录   | `859ea4d05690` |
| `.codex/artifacts/api-notes/dal-5-studio-capabilities-contract.md`                  | 工程/历史记录   | `2537ca69a006` |
| `.codex/artifacts/api-notes/docs-examples.md`                                       | 工程/历史记录   | `9dffa84fa1e2` |
| `.codex/artifacts/api-notes/m3-delta-api-note.md`                                   | 工程/历史记录   | `f8f3f8730a23` |
| `.codex/artifacts/api-notes/m4-state-window.md`                                     | 工程/历史记录   | `f3bde952eedd` |
| `.codex/artifacts/api-notes/m5-epoch-checkpoint.md`                                 | 工程/历史记录   | `98d1534ecae7` |
| `.codex/artifacts/api-notes/m6-connectors-project-v3.md`                            | 工程/历史记录   | `3a5e427a84ae` |
| `.codex/artifacts/api-notes/sce-11-static-stream-inputs-dictionary-addendum.md`     | 工程/历史记录   | `656c5ac65d7b` |
| `.codex/artifacts/api-notes/sce-11-static-stream-inputs.md`                         | 工程/历史记录   | `da9a9ec1687f` |
| `.codex/artifacts/api-notes/symbolic-computation-engine.md`                         | 工程/历史记录   | `0f5d0e95d335` |
| `.codex/artifacts/api-notes/symbolic-exponential-indicators.md`                     | 工程/历史记录   | `6604c3f4f580` |
| `.codex/artifacts/api-notes/symbolic-relational-dag.md`                             | 工程/历史记录   | `e42f4bc17f93` |
| `.codex/artifacts/api-notes/symbolic-stream-joins.md`                               | 工程/历史记录   | `25e7d048e599` |
| `.codex/artifacts/api/sce-13-static-array-snapshot.md`                              | 工程/历史记录   | `37c7f4ec10df` |
| `.codex/artifacts/critiques/a6-public-continuous-runtime.md`                        | 工程/历史记录   | `a075cdb83222` |
| `.codex/artifacts/critiques/continuous-streaming-m2-completion.md`                  | 工程/历史记录   | `5ef2c9d0c208` |
| `.codex/artifacts/critiques/continuous-streaming-runtime.md`                        | 工程/历史记录   | `c0ea09e54a9b` |
| `.codex/artifacts/critiques/dal-45-provider-option-error-paths-critique.md`         | 工程/历史记录   | `1e1389dc9265` |
| `.codex/artifacts/critiques/dal-47-execution-options-stub-defaults.md`              | 工程/历史记录   | `7050b99163e6` |
| `.codex/artifacts/critiques/dal-5-5-2.md`                                           | 工程/历史记录   | `f64e1e7a0405` |
| `.codex/artifacts/critiques/dal-5-studio-capabilities-contract-critique.md`         | 工程/历史记录   | `a6395ccad02e` |
| `.codex/artifacts/critiques/dal-5-weekly-capability-critique-2026-07-25.md`         | 工程/历史记录   | `ee204ab9f6a8` |
| `.codex/artifacts/critiques/github-36-claude-artifact-paths.md`                     | 工程/历史记录   | `16c079037ecd` |
| `.codex/artifacts/critiques/m3-delta-critique.md`                                   | 工程/历史记录   | `afa42d11b6e3` |
| `.codex/artifacts/critiques/m4-state-window.md`                                     | 工程/历史记录   | `40b122051ffb` |
| `.codex/artifacts/critiques/m5-epoch-checkpoint.md`                                 | 工程/历史记录   | `bc1a458c169e` |
| `.codex/artifacts/critiques/m6-connectors-project-v3.md`                            | 工程/历史记录   | `2f85dc0b49f5` |
| `.codex/artifacts/critiques/sce-07-rolling-inf-mean.md`                             | 工程/历史记录   | `aab34c2cc0a5` |
| `.codex/artifacts/critiques/sce-13-static-array-snapshot.md`                        | 工程/历史记录   | `32219206092d` |
| `.codex/artifacts/critiques/symbolic-computation-engine.md`                         | 工程/历史记录   | `b4a7d307b31e` |
| `.codex/artifacts/specs/a6-public-continuous-runtime.md`                            | 工程/历史记录   | `43a8465e5d51` |
| `.codex/artifacts/specs/continuous-streaming-m2-completion.md`                      | 工程/历史记录   | `917a29248dd6` |
| `.codex/artifacts/specs/continuous-streaming-runtime.md`                            | 工程/历史记录   | `a635847d480d` |
| `.codex/artifacts/specs/dal-45-provider-option-error-paths.md`                      | 工程/历史记录   | `48dfa854380d` |
| `.codex/artifacts/specs/dal-47-execution-options-stub-defaults.md`                  | 工程/历史记录   | `d7f2d8c2d699` |
| `.codex/artifacts/specs/dal-5-5-2.md`                                               | 工程/历史记录   | `e21da0b766bb` |
| `.codex/artifacts/specs/github-36-claude-artifact-paths.md`                         | 工程/历史记录   | `cc48aa29119a` |
| `.codex/artifacts/specs/head-operator.md`                                           | 工程/历史记录   | `4385cdd2eb98` |
| `.codex/artifacts/specs/m3-delta-spec.md`                                           | 工程/历史记录   | `ab7ff946d674` |
| `.codex/artifacts/specs/m4-state-window.md`                                         | 工程/历史记录   | `a417f98917c7` |
| `.codex/artifacts/specs/m5-epoch-checkpoint.md`                                     | 工程/历史记录   | `87496fec7e57` |
| `.codex/artifacts/specs/m6-connectors-project-v3.md`                                | 工程/历史记录   | `af8749e686f3` |
| `.codex/artifacts/specs/symbolic-computation-contract.md`                           | 工程/历史记录   | `e62d128b7658` |
| `.codex/artifacts/specs/symbolic-exponential-indicators.md`                         | 工程/历史记录   | `fe7bce2bebe6` |
| `.codex/artifacts/specs/symbolic-relational-dag.md`                                 | 工程/历史记录   | `4c475d6db5ef` |
| `.codex/artifacts/specs/symbolic-stream-joins.md`                                   | 工程/历史记录   | `86065565280d` |
| `AGENTS.md`                                                                         | 维护/agent 指南 | `e06629dff091` |
| `CHANGELOG.md`                                                                      | 变更历史        | `760162a013d8` |
| `CLAUDE.md`                                                                         | 维护/agent 指南 | `6350962f5ffe` |
| `README.md`                                                                         | 当前用户文档    | `1111c75cd7c2` |
| `benchmarks/README.md`                                                              | 当前用户文档    | `651a389286b1` |
| `benchmarks/symbolic/BASELINE.md`                                                   | 工程/历史记录   | `7a3806ab09b9` |
| `benchmarks/symbolic/SCE05.md`                                                      | 工程/历史记录   | `5aa2879e9be9` |
| `benchmarks/symbolic/SCE08.md`                                                      | 工程/历史记录   | `cf350bedaaf5` |
| `benchmarks/symbolic/SCE14.md`                                                      | 工程/历史记录   | `80576f3b9c5c` |
| `benchmarks/symbolic/SCE16.md`                                                      | 工程/历史记录   | `9ea696d610d7` |
| `crates/calc-flow-connectors/README.md`                                             | 当前用户文档    | `2a0851c061a6` |
| `crates/calc-flow/examples/README.md`                                               | 当前用户文档    | `9a8eed114662` |
| `design/tech_spec.md`                                                               | 工程/历史记录   | `ce05943a80c0` |
| `design/v0.2-refactor-plan.md`                                                      | 工程/历史记录   | `b22b5b5fcabf` |
| `docs/README.md`                                                                    | 当前用户文档    | `1caad359900d` |
| `docs/api-reference.md`                                                             | 当前用户文档    | `1aeb267ce344` |
| `docs/array-guide.md`                                                               | 当前用户文档    | `7134421be416` |
| `docs/batch-guide.md`                                                               | 当前用户文档    | `2ba53c411623` |
| `docs/benchmark-suite.md`                                                           | 当前用户文档    | `c5ec11ef7fba` |
| `docs/connectors.md`                                                                | 当前用户文档    | `1b12176f660f` |
| `docs/design.md`                                                                    | 当前用户文档    | `f3dbe669d017` |
| `docs/examples.md`                                                                  | 当前用户文档    | `08e59d30fc64` |
| `docs/getting-started.md`                                                           | 当前用户文档    | `f70b149a03ef` |
| `docs/introduction.md`                                                              | 当前用户文档    | `d1da4ce77ecd` |
| `docs/projects-guide.md`                                                            | 当前用户文档    | `6f5b76a5ce52` |
| `docs/python-api.md`                                                                | 当前用户文档    | `b3511f87b540` |
| `docs/python-release.md`                                                            | 当前用户文档    | `29fb29b1ee56` |
| `docs/runtime-envelope.md`                                                          | 当前用户文档    | `ceac40699f36` |
| `docs/rust-api.md`                                                                  | 当前用户文档    | `e2747ddd5e7f` |
| `docs/sql-datafusion-performance.md`                                                | 当前用户文档    | `ebe882ca2d91` |
| `docs/streaming-guide.md`                                                           | 当前用户文档    | `2282245243db` |
| `docs/studio-guide.md`                                                              | 当前用户文档    | `92e89c936d58` |
| `docs/symbolic-api.md`                                                              | 当前用户文档    | `ce542fc2e112` |
| `docs/symbolic-design.md`                                                           | 当前用户文档    | `eb37534dbfbd` |
| `docs/symbolic-workflows.md`                                                        | 当前用户文档    | `f13e94a2770e` |
| `docs/verification.md`                                                              | 当前用户文档    | `654a419c9d9a` |
| `docs/warm-stream-performance.md`                                                   | 当前用户文档    | `0adad07352cd` |
| `examples/README.md`                                                                | 当前用户文档    | `ed9187c9b8d0` |
| `web-ui/README.md`                                                                  | 当前用户文档    | `ccda0368186d` |
| `web-ui/backend/README.md`                                                          | 当前用户文档    | `0c3adf6f94fa` |
