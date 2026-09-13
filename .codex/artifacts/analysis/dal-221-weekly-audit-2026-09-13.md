# DAL-221 全仓 Markdown 与 agent 文本审计（2026-09-13）

文档审计、版本合同测试与 CLAUDE 触发条件已在提交 `f23791e` 获得
独立 Approve；该结论不覆盖后续 lint 和文档修订。现有 PR 为 #274，分支
`docs/DAL-221-weekly-audit-2026-09-13`，目标 main。本轮从测试专员提交
`ca36523c97734b3882644a52e6476e9e22fc431b` 继续，收口 Copilot 两条文档
意见并更新证据，交回协调器安排最终 head 的统一复核及授权合并。
agent 文本同步集合仍为空；本轮只改文档和证据，不修改测试、运行时、
canonical agent 定义或 CI/Codacy 配置。新提交与快照随 issue 交接提供。

## 基线和范围

首次审计通过 `multica repo checkout` 获取仓库，再执行 `git fetch origin main`。
`git rev-parse origin/main` 与 `git ls-remote origin refs/heads/main` 均为
`f9b608559714cde474f883344835427778f584e3`；从该提交建立本轮新分支和
`.worktrees/` 隔离工作树。原 checkout 干净，没有覆盖既有工作。
已从这个基线读取 `.codex/agents/README.md`，没有沿用上周的审计结论。

使用 `rg --files --hidden` 发现文件，以 `git ls-files -z` 的完整受控清单
为准，按大小写无关的 `.md` / `.markdown` 后缀枚举。基线共
**133 个 Markdown 文件、51,870 行**，包括受忽略规则影响但已入库的文件。

| 范围            | 数量  |
|---------------|-----|
| 根目录           | 4   |
| `docs/`       | 33  |
| `.codex/`     | 67  |
| `.claude/`    | 14  |
| `.agents/`    | 1   |
| `benchmarks/` | 6   |
| `crates/`     | 2   |
| `design/`     | 2   |
| `examples/`   | 1   |
| `web-ui/`     | 2   |
| `tests/`      | 1   |

全部文件接受 Markdown 链接、标题锚点、表格、尾随空白、末尾换行扫描。
其中 54 份现行用户/维护指南接受示例语法、公开导入名称、具体文件路径及
跨文件合同检查；另 79 份为工程记录、历史测量、兼容性 fixture 说明及
CHANGELOG。后者的日期、旧 API、设计选项和测量值保留其记录语境，
不作为当前 API 承诺，也没有改写冻结发布记录。

完整逐文件路径、行数、SHA-256、逐链接结果、源文件证据和 agent 字段结果
见 [机器可读证据](dal-221-weekly-audit-2026-09-13-evidence.json)。
本报告是新增的第 134 个 Markdown 文件，不计入基线 133 个。

### 已审主线增量基线（b3cc07c8）

本轮先 fetch 并以 `git ls-remote` 确认：远端审查分支为
`4d948d26921d4d3fd9e9604008efa53a93142e0b`，最新 main 为
`7b5c3e912f3df11bccbafcddff546dc278abee4e`（DAL-211 / #273）。
工作树干净，远端分支没有他人新增提交；已将本任务的唯一文档提交无冲突
rebase 到该 main，保留 SQL 修正和无归因 trailer 的提交正文。
相对原基线 `f9b608559714cde474f883344835427778f584e3`，主线增量为
24 个源码、测试或生成契约文件，没有 Markdown 或 agent 定义变更。

以 `git ls-tree -r --name-only` 重新枚举最新 main、`git ls-files -z` 枚举
当前分支全部受控 Markdown，并重新计算逐文件行数和 SHA-256。最新 main
仍为 133 份、51,870 行，各文件字节与首次审计基线相同；分支含报告共
134 份，最终逐文件清单、统计及新源文件哈希见 JSON。清单更新不代表
重复执行整套语义、外链或示例审计。

### 已审 PR 与 Codacy 收口（f23791e）

独立评审评论 `01a0995f-ba46-766c-9fd9-20710797de6f` 于
2026-09-13 06:06:22 UTC 对 `b3cc07c8b4cc206127cbd792f775a48361a50c7d`
给出“Approve，阻断项清零”。该轮 fetch 后 main 仍为 `7b5c3e9`，PR head
已前进到 `5a1c8a3`；工作树仅快进接收该测试提交，没有改写他人提交。

测试专员在诊断评论 `01a0996d-c215-79f5-9fae-691b554a6460` 确认：
Linux package job 停在脚本 unittest，版本合同测试的五个子测试仍要求
v4 文案；`uv build`、artifact inspectors 和隔离 wheel smoke 未执行。
这是本 PR 的文档更新触发的测试维护问题，不能称为 wheel 编译/安装缺陷。
`5a1c8a3` 仅修改 `scripts/test_release_config.py` 的
`ReleaseConfigTests.test_normative_docs_use_final_package_and_project_versions`，
从 Cargo、Python core 和 Studio manifests 读取源码版本，保留版本一致、
project-v3、安装入口和过期声明约束，并区分源码构建与已发布包。
整个 PR 因此包含 18 个文档/证据文件和 1 个测试文件，不再是纯文档 diff。

复用测试专员交接评论 `01a09972-fb95-7a01-a265-70093da2d37f` 的实际结果：
`python3.13 -m unittest scripts.test_release_config.ReleaseConfigTests.test_normative_docs_use_final_package_and_project_versions`
运行 1 项测试、0.003 秒、OK；
`UV_CACHE_DIR=target/uv-cache uv tool run --from ruff==0.16.0 ruff format --check scripts/test_release_config.py`
报告 1 file already formatted；`git diff --check` 通过，三者均退出 0。
这些结果仅证明该定向测试和文件格式，未验证 wheel 构建或安装。本轮不重跑。

Codacy 的新增问题指向 `CLAUDE.md` 的 `Specialist agents` 段落，要求将
“documentation when needed” 改为具体触发条件。本轮明确：行为、公共
Rust/Python API、Studio REST/OpenAPI 合同、命令或其他用户可见能力变化
后进入文档收口；纯测试或保留这些表面的重构可跳过文档收口，但独立评审
仍然适用。纯文档仍经文档专员和 reviewer。该修订不改变 canonical 定义，
也不放宽 Codacy 或 CI 门禁；是否清除远端检查失败由新 head 的检查确定。

## 发现与修订

| 发现                                                  | 当前依据                                                                    | 修订                                                      |
|-----------------------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------|
| 多份指南仍指向源码版本 4.0.0                                   | workspace、Python、Studio、frontend、OpenAPI 均为 5.0.0                       | 更新版本引用；一般介绍去掉不必要的版本标签                                   |
| 源码安装检查与发布安装共用固定版本断言                                 | 当前源码 5.0.0；本次 PyPI 元数据返回 4.0.0                                          | 明确源码构建的预期版本与包管理器选择的发布版本；API 页声明描述当前 checkout            |
| Rust 参考页以旧 registry 安装命令开头                          | 当前参考面是仓库源码；发布工作流仅 dry-run crate                                         | 删除旧 `cargo add` 行，明确仓库根目录及按需验证入口；未假定 v5 crate 已发布       |
| Studio 依赖主版本与 release tag 说明落后                      | backend pyproject 为 `>=5.0.0,<6`；release workflow 使用 `v5.*`             | 同步 release guide、AGENTS 与 Studio README                 |
| late-output 文档仍称执行禁用                                | `LatePolicySpec`、图路由、operator task、snapshot restore、事务 Parquet recovery | 对齐 Rust/API/架构/文件 connector 指南中的执行、控制、恢复及限制             |
| Rust 页仍将 SQL rewrite 仅描述为 AVG                       | `datafusion_rolling.rs` 和 `sql_aggregate.rs` 支持兼容 COUNT/AVG             | 同步 SQL 路径与 Native 数值路径的区别，并链接现行详细边界                     |
| Rust 双生命周期列表漏列 cross-section                        | crate exports 与 `CrossSectionOperator` 两个 trait 实现                      | 补齐类型                                                    |
| CLAUDE 架构漏列 connector crate，且仍写固定全流程                | AGENTS、crate manifests、canonical team README                            | 补齐依赖关系；按 canonical 的条件路由同步流程                            |
| 根 README 与 CLAUDE 表格将 benchmarks 一概称为 informational | benchmark suite catalog 与 Linux CI                                      | 区分 unified regression gates 和 informational comparisons |

现行合同核验还覆盖 Python `compute`/collection/stream 的签名、root exports
与 `_native.pyi`；`Program.to_project`、store 与 runner 的逻辑/物理绑定边界；
project/manifest 均为 3；ASOF 类型、schema 与状态表达；Studio 15 个路径的
19 个操作；七类 connector feature 与示例入口；AGENTS/CLAUDE 命令镜像。
Python capability catalog 的 rolling layouts 仍为 1/2，而 native writer 为 3；
现有 Python 文档已经明确这一限制，本轮保留。

最新主线已支持 native rolling/cross-section 的 `SideOutput` 执行和恢复。
`late` 及其受支持的单输入 expression/SQL 后继不传 watermark/idle，但
参与 FIFO barrier/end 和与正常输出相同的检查点 epoch。严格的
`late_output` version-1 元数据记录独立 `next_sequence`；即使没有迟到行
也写入，恢复先校验再安装状态。error/drop 不携带该对象；批模式仍拒绝
side output，Python symbolic lowerer 仍只接受 error/drop。

事务 Parquet sink 恢复校验 epoch/output 身份、manifest 和完整文件清单，
可从已准备的 staging 完成提交。正常与诊断输出的 delivery 分别证明，
不能把双输出或共同 checkpoint 解释为普通 sink 的全局 exactly-once。
现行指南已去掉“执行禁用”结论；CHANGELOG 中前一条 staged 合同记录保留
为历史，并由新的启用/恢复条目承接。历史 spec/API note 中的旧阶段约束
继续按首次审计已认可的工程记录边界保留，不改写为现行使用承诺。
文档专员没有设计新示例、编辑示例程序、产品源码、测试或生成契约；
测试文件的后续变化由 `5a1c8a3` 及下述 `ca36523` 单独记录。

### CHANGELOG 判定

文档修订本身不单独记入 CHANGELOG。首次补记 **2026-09-13** 的基本变更：
已合入的公共 `LatePolicySpec::SideOutput`、project schema/端口合同、v5 包版本
及内部 envelope transaction，并明确执行仍禁用。依据为首次审计基线中的
`f8ca77eaf241523c67ebd9156866b5bafeed07f6` 和 `f9b60855`；两项实现均漏记。
没有把 5.0.0 源码版本写成已完成的公开发布或 late-output 恢复验收。
SQL 措辞与提交元数据修正不构成基本能力变更。主线增量收口为 `7b5c3e9` 已合入的
执行、检查点元数据和恢复能力新增一条同日记录；项目/manifest 格式仍为
3，rolling writer layout 仍为 3、cross-section state layout 仍为 1。
版本合同测试/lint 修复、CLAUDE 触发条件与本轮审计/SQL 澄清均不构成基本能力变化，
不新增 CHANGELOG 条目。

### 已审的 P2 与提交归因修正（b3cc07c8）

`docs/rust-api.md` 原先把物理改写失败时的整查询回退保证泛化到了所有查询。
`CalcFlowQueryPlanner::create_physical_plan` 实际先进行逻辑资格检查；
进入物理改写后，只有 `rewritten_windows != candidate_windows` 且
`contains_count` 为真，才清零改写计数并返回 `original`。AVG-only 路径
返回 `transformed.data`，可以保留成功的局部物理改写。Rust 参考页已明确
这一区别，保留原有 SQL 数值语义和详细边界链接，没有修改或新增示例。

旧提交的 `Co-authored-by` 来自工作区 `prepare-commit-msg` 自动归因钩子，
不符合 AGENTS 的默认不添加工具归因要求。前轮已仅以单次 Git 调用覆盖
钩子路径并移除 trailer；本轮 rebase/amend 保留作者身份，并将父提交
更新为已确认的最新 main，提交继续不含归因 trailer。没有
编辑钩子、持久化 Git 设置、runtime 或 agent 配置。新提交和最终提交消息
核验结果随 issue 交接评论提供，避免将自身提交哈希写入提交内形成自引用。

## Agent 逐字段映射与同步方案

每个名称的 canonical 文件为 `.codex/agents/<名称>.toml`，Claude 镜像为
`.claude/agents/<名称>.md`。对应 Multica ID 如下；十个 ID 均经
`multica agent list --output json` 发现、逐个 `multica agent get <id> --output json`
读取并校验名称，没有猜测 ID。

| 名称              | Multica ID                           | description 差异 | instructions 差异/动作                    |
|-----------------|--------------------------------------|----------------|---------------------------------------|
| cf-orchestrator | a6c22a43-3dc1-4338-8581-3e1317dd9519 | 无              | canonical 正文一致；保留紧凑协议及 CI 责任附录；不写入    |
| cf-implementer  | 227e58dc-a05a-48b6-bd20-94af140da251 | 无              | canonical 正文一致；保留紧凑协议及实现角色补充；不写入      |
| cf-tester       | a553f49e-ae3c-401e-88d4-919d9e078703 | 无              | canonical 正文一致；保留紧凑协议及最小验证补充；不写入      |
| cf-api-designer | ea5b9d56-eb67-4d40-ba30-d07054d02f77 | 无              | canonical 正文一致；保留紧凑协议及合同设计补充；不写入      |
| cf-critic       | d29d22b4-ec2a-4a2c-ab7b-861f7f4d0805 | 无              | canonical 正文一致；保留紧凑协议及阻断门槛补充；不写入      |
| cf-reviewer     | a274f526-62bc-481f-93ef-67b4a63d9f3a | 无              | canonical 正文一致；保留紧凑协议及质量门补充；不写入       |
| cf-doc-writer   | 1b71d01f-a1e8-4e73-97f0-3a2576a7378c | 无              | canonical 正文一致；保留紧凑协议及文档收口补充；不写入      |
| cf-simplifier   | 113b511d-172f-4673-becd-cc0db10b70b8 | 无              | canonical 正文一致；保留紧凑协议及按需 apply 补充；不写入 |
| cf-performancer | b37225a5-8640-4da1-9500-89adfab9c66b | 无              | canonical 正文一致；保留紧凑协议及专项性能补充；不写入      |
| cf-spec-writer  | 62f89f6b-8158-4c5d-ac1d-8ed6d092473f | 无              | canonical 正文一致；保留紧凑协议及轻量 spec 补充；不写入  |

字段语义：TOML `description` 对应目录摘要 `description`，
`developer_instructions` 对应运行说明 `instructions`。十个 description
与 Claude frontmatter、Multica 完全一致；Multica instructions 在保留既有
中文 overlay、仅归一化正文首尾空白后，十个均与 canonical 正文相等。
证据逐项记录当前/拟同步字段 SHA-256、canonical 正文 SHA-256、overlay
SHA-256；每项拟同步值都等于当前值，写入集合为空。

Claude 十个角色的语义与 canonical 一致。保留的差异包括 `Agent`/`SendMessage`
等工具名、`EnterWorktree`、`.claude/rules/code-style.md` 入口、AGENTS 权威下
的 CLAUDE 指引，以及换行/表格排版。没有将这些适配误判为待覆盖的配置。
两个保留的 `.claude/api-notes/docs-examples.md`、`.claude/specs/head-operator.md`
与对应 canonical artifact 仍逐字节相等。团队 README 的条件路由也已核对；
本次只修复根 `CLAUDE.md` 中重复说明的偏差，不需要修改角色定义文件。

模型、thinking level、service tier、runtime/config、并发、技能、权限、环境、
MCP、名称、ID 及 squad 均没有写入。没有读取明文环境秘密。
后续若独立评审要求新的 canonical 文本修订，应先重新读取对应 agent，
复核映射和并发修改，再由协调器安排仅文本字段同步；本阶段不执行同步。

## 首次审计已获认可的检查（本轮复用）

- `git fetch origin main`、`git rev-parse origin/main`、
  `git ls-remote origin refs/heads/main`：三者确认同一基线。
- `python3 target/doc-audit/scan.py --baseline`：133 文件、51,870 行；
  888 处 Markdown 链接中，825 处本地文件/锚点通过，63 处为外部链接。
  解析使用 MarkdownIt CommonMark + table，覆盖 inline/reference/image 链接、
  heading slug、重复标题及显式 HTML 锚点。末轮包含本报告的结果见证据。
- `python3 target/doc-audit/external.py`：51 个唯一 HTTP(S) 目标，50 个成功，
  Coveralls 项目页 403；其 badge 正常。三处 GitHub comment 锚点通过
  `gh api repos/wegamekinglc/calc-flow/issues/comments/<id>` 验证 `html_url`；
  Cargo `artifact-messages` 锚点存在。没有把 fenced code、示例服务地址或
  loopback 占位 URL 当作应启动的服务。
- PyPI 元数据 GET 返回 4.0.0；crates.io API 返回 403，未确认 registry 的
  crate 版本。用户文档不宣称源码 v5 已公开发布，也不新增依赖该假设的命令。
- `python3 target/doc-audit/facts.py`：54 份现行 Markdown 中的 35 个 Python
  代码块语法通过，公开导入和 `cf` root 名称存在；runner 的 30 个 Python
  程序语法通过，4 个 Rust 用户示例路径存在。没有执行这些程序。
- API reference 的 HTTP 表格与 OpenAPI 19 个操作集合一致；AGENTS/CLAUDE
  首个完整命令块逐字节一致；7 个 Bash 命令块经 `bash -n` 解析通过。
- generated project schema 的六项 late-policy 字段核验通过：error/envelope、
  drop/1、side_output/1/1 有效；缺少 scope 或 side_output 非 1 版本无效。
  这是 schema 静态检查，不是 native 执行或恢复验收。
- `python3 scripts/verify_python_release.py --version-only`：输出 `5.0.0`。
  `python3 scripts/run_examples.py --help`：入口正常，包含 surface/services 参数。
- `facts.unchanged_example_blocks` 仅比较 Python/Rust/JSON/YAML 应用示例，
  这些代码块与基线相同，不代表所有 fenced block 均未变。安装/构建命令的
  唯一变化是 `docs/rust-api.md` 的 `Build and document` Bash 块删除
  `cargo add calc-flow@4.0.0`；另有 CLAUDE 的 `text` 架构图补入 connector。
  未重写应用示例，也未验证被删除的 registry 安装命令。
- `git diff --check` 通过；生成 schema、OpenAPI、TypeScript contract 无 diff。
  变更文件表格已对齐，无尾随空白，均有末尾换行。基线其余 18 份历史工程/
  测量记录的 73 个表格存在排版遗留，已逐项记录，不为本轮重排历史。

检查辅助脚本保存在本轮隔离树的 `target/doc-audit/`，随 issue 的证据附件
提供以便复核，不作为产品代码入库。本次没有运行 build、产品 test suite、
全仓回归、覆盖率或 benchmark。全量回归和常规性能门禁属于 CI。
推送后只取一次非阻塞状态快照，结果写在交接评论；不等待或轮询。

## 已审主线增量检查（b3cc07c8，本轮复用）

- fetch/ls-remote 确认 main 和原审查 head；读取最新 `.codex/agents/README.md`。
  rebase 无冲突，比较确认本任务相对新 main 仍只有文档和证据改动。
- 阅读主线增量涉及的配置/图路由、operator callback、控制转发、snapshot
  恢复和 file sink recovery；对照已有控制流、空/混合/全迟到 epoch、终态
  恢复、损坏 snapshot 的测试断言作为静态依据，没有执行这些测试。
- `CalcFlowQueryPlanner::create_physical_plan`、SQL aggregate 及详细指南
  与原基线字节相同，SQL 修正段落与 `4d948d26` 一致，复用前轮核验。
- `python3 target/doc-audit/incremental.py` 重枚举全仓 Markdown 清单并更新
  全部行数、哈希及范围统计；仅对本轮 6 份变动 Markdown 核验格式及
  102 处本地出站链接/锚点、38 处入站链接；2 处外部链接保留原核验结果。
  其余链接/格式结果依据文件字节未变且目标未删除复用；外链结果保留原
  查询时点。当前文档中的已有示例和命令代码块没有变化，不重复执行。
- schema/OpenAPI 的主线增量仅改变 side-output 的 description，去除
  description 后结构与旧基线相同，复用六项字段约束和 19 个 REST 操作
  的已认可检查。原 42 份源证据更新变化项哈希，并补入增量事实依据。
  canonical/Claude agent 文件没有变化，逐字段同步方案沿用，不读取或写入
  Multica agent；没有新待同步字段。
- JSON 可解析，清单和源证据哈希与实际字节一致；`git diff --check` 通过。
  SQL 修正及无归因 trailer 保留；新提交及附件哈希随 issue 交接提供。
- 未重试两个 403，未重复整套审计，未运行产品测试、构建或性能门禁。
  推送后最多一次非阻塞 CI 快照，结果随交接提供，不等待或轮询。

## 已审 Codacy 文案与证据检查（f23791e，本轮复用）

该提交汇总为 **134 份 Markdown、52,279 行**，837 处本地链接、63 处外链，
51 份源文件证据。本轮重新核验 19 处本地出站链接/锚点，入站引用为 0。

- 仅核验 CLAUDE 与本报告的链接/锚点、入站引用、表格、尾随空白和末尾
  换行；更新 JSON 内这两份 Markdown 的行数、SHA-256 和链接位置，再从
  既有清单汇总全仓统计。其余 132 份 Markdown 没有变化，复用已审证据，
  不重复全仓审计。CLAUDE 的命令和示例代码块未变。
- 将已继承测试文件的当前哈希纳入证据；原 50 项源码哈希及 agent 字段
  结果保持不变。本轮 JSON 可解析、相关哈希与实际字节一致，diff 格式
  及提交 trailer 经核验；所有本地验证限于本次文档变化。
- 更新现有 PR #274 的描述，补记版本合同测试修复、定向验证和评审边界。
  推送后最多获取一次非阻塞 CI 快照，不等待、轮询或重跑 CI。

## 本轮 Copilot 意见与 lint 证据收口

本轮 fetch/ls-remote 确认 main 仍为 `7b5c3e912f3df11bccbafcddff546dc278abee4e`，
PR head 为 `ca36523c97734b3882644a52e6476e9e22fc431b`，工作树干净后仅快进接收。
Copilot review `5189811036` 针对 `b3cc07c8` 的两条 suppressed 文档意见均已处理：

- 代码块表述：上述应用示例比较本身有效，但不能推广到 Bash 和 `text`。
  对照本 PR 受改 Markdown 的实际 fenced-block diff，列明 Rust 安装行删除
  和 CLAUDE 架构图两项例外；“所有代码块不变”及“所有 fenced block 只有
  一个例外”均不适用。JSON 增补语言范围和例外的前后内容哈希。
- AVG-only 边界：意见有效。`CalcFlowQueryPlanner::create_physical_plan`
  在任意逻辑 fallback 或没有 candidate 时直接返回 DataFusion plan；
  只有通过逻辑资格后才进入 `transform_up`。逻辑层拒绝的 filter、distinct、
  cast 等不能保留局部改写。逻辑合格后物理 shape 未覆盖全部 candidate 时，
  COUNT 路径返回 `original` 并清零，AVG-only 才可保留 `transformed.data`。
  Rust 参考页已补足该前提，与详细边界一致；源码和数值语义未改。

测试专员在评论 `01a09998-9d58-7f00-b146-dc8a31b35484` 确认，后续 CI 的
Ruff E501 是 `5a1c8a3` 新增断言字符串长 89 列、超过 88 列的测试 lint 问题；
job 停在 `ruff check .`，后续测试未运行。`ca36523` 仅将字符串拆为两个相邻
字面量，1 个文件、2 行新增/1 行删除；不是生产缺陷，也没有放宽断言或门禁。
复用该提交实际结果，以下命令均退出 0：

- `UV_CACHE_DIR=target/uv-cache uv tool run --from ruff==0.16.0 ruff check scripts/test_release_config.py`：All checks passed!
- `UV_CACHE_DIR=target/uv-cache uv tool run --from ruff==0.16.0 ruff format --check scripts/test_release_config.py`：1 file already formatted。
- `python3.13 -m unittest scripts.test_release_config.ReleaseConfigTests.test_normative_docs_use_final_package_and_project_versions`：1 项、0.002 秒、OK。
- `git diff --check`：通过；测试专员另以 `ast.dump(ast.parse(...))` 比较修订前后
  整个测试文件，AST 完全相同。以上检查本轮不重跑；本轮只改文档叙述与证据，
  被测版本/安装文本、示例和测试逻辑保持原值。

本轮仅核验 Rust 参考页和本报告的链接/锚点、入站引用、表格、空白及末尾换行，
更新这两份 Markdown 和继承的测试文件 SHA-256，核对 JSON 与实际字节一致。
其余 132 份 Markdown、原 50 项源证据、agent 字段和有效静态检查按未变范围复用。
最新清单汇总为 **134 份 Markdown、52,323 行**，837 处本地链接、63 处外链、
51 份源文件证据；聚合既有清单，不重复全仓审计。提交/diff/trailer 及推送后的
唯一非阻塞 CI 快照随交接提供；不等待、轮询或触发重跑。

## 评审与剩余事项

独立评审评论 `01a09984-ea96-7057-9269-5fd754cad6bf` 已对 `f23791e55abe407b59e2de39b1c53f46f823d810`
给出 Approve、评审阻断项 0；该结论不覆盖后续 `ca36523` 和本轮文档修订。
定向 unittest 与 lint 通过不等于 wheel 构建、安装或 CI 全部通过。
Coveralls 和 crate registry 的 403 是外部
核验限制，保留原始状态，未声明通过。例子语法检查不替代运行结果，静态
审计不证明产品行为、覆盖率、发布安装或性能门禁通过。

用户评论 `01a09992-05dd-7604-b353-f38208c0e758` 已授权修复全部问题并合并，
覆盖先前禁止合并的交付限制。交回协调器安排 `cf-reviewer` 复核最终 head、
关闭已修复的版本测试讨论，并在最终评审与门禁通过后安排合并。本阶段没有
创建 PR、合并或触发其他成员，issue 保持 `in_progress`。
