# DAL-5：Calc Flow 每周功能进展与能力盘点

## 范围与基线

- 固定窗口：2026-07-18 15:00 至 2026-07-25 15:00（Asia/Shanghai，即
  2026-07-18T07:00:00Z 至 2026-07-25T07:00:00Z）。
- 检查对象：`origin/main` / `f71e49d7ccba117ade55270bd9df7499c04769cb`
  （2026-07-25 04:00:52 +08:00，合并 PR #26）。任务分支
  `agent/cf-orchestrator/fae83363` 与该提交一致。
- 窗口起点之前的基线提交：
  `5be0e5a6b84a01162d180ce737a6868de1113890`
  （2026-07-18 11:14:23 +08:00）。
- 本窗口 `main` 有 104 个可达提交、13 个 first-parent 变更（对应 PR
  #14–#26 的合并），净变更为 110 个文件、19,871 行新增、598 行删除。
- GitHub 全状态快照显示窗口内共有 14 个活跃 PR：#14–#26 已合并，#27
  仍开放；没有窗口内关闭但未合并的 PR。

## 当前已实现能力（`main`）

| 核心模块 | 已实现能力 | 主要证据 |
| --- | --- | --- |
| 数据与批次 | 统一的不可变 `Batch` 封装 Arrow 表和外部数组载荷，包含来源、序列和严格 JSON 元数据；表批次校验 schema 一致性，外部载荷保留后端身份。 | `crates/calc-flow/src/batch.rs` |
| 图编译与执行 | 类型化端口、DAG 边和拓扑编译；内置表达式、DataFusion SQL、多输入 SQL，以及显式注册的外部 operator；执行返回命名输出、节点行数/耗时、运行身份和 DataFusion 指标，并在失败/取消时回滚有状态 operator。 | `crates/calc-flow/src/operator.rs`, `crates/calc-flow/src/pipeline.rs` |
| DataFusion、UDF 与外部 provider | 表与混合图按运行创建 DataFusion session；外部数组专用图不保存/创建 DataFusion 资源。支持版本化 UDF 引用、编译时选择、名称冲突检查，以及 Rust/Python 外部 provider 注册。 | `crates/calc-flow/src/datafusion.rs`, `crates/calc-flow/src/udf.rs`, `crates/calc-flow-python/src/provider.rs` |
| 状态运行与恢复 | 拉取式 `MicroBatchRunner`、推送式 `StreamingRunner`、按输出路由的 sink、独占 plan lease、至少一次投递；sink 成功后才提交 checkpoint，失败会恢复 plan 和持久状态。 | `crates/calc-flow/src/runtime/`, `crates/calc-flow/src/checkpoint.rs`, `crates/calc-flow/src/io.rs` |
| 项目契约与存储 | 严格的 `format_version: 2` JSON/YAML 项目文档、生成式 JSON Schema、图/端口/UDF/provider 校验、规范化导入导出；内置原子文件项目库和 checkpoint 库，包含大小、深度和路径边界。 | `crates/calc-flow/src/config.rs`, `crates/calc-flow/src/project_store.rs`, `schemas/project-v2.schema.json` |
| Python 包 | PyO3 绑定复用 Rust 引擎；Python 层提供函数式 builder、同步/异步执行、PyArrow 批次、可信 Python scalar UDF、NumPy/JAX 数组 provider，以及 runner/store 适配器。 | `crates/calc-flow-python/src/`, `python/calc_flow/`, `python/tests/` |
| Studio 后端 | 本地 FastAPI `/api/v2` 提供 catalog/schema、项目 CRUD、严格导入导出、校验、checkpoint 查看/清除、预览提交/查询/SSE/取消；预览在有输入、行数、时间、内存和输出边界的 spawned worker 中执行。 | `web-ui/backend/src/calc_flow_studio/app.py`, `web-ui/backend/src/calc_flow_studio/run_manager.py` |
| Studio 前端 | React Flow 图编辑器可创建表达式/SQL 节点、连接端口、编辑 schema 和 SQL aliases；支持多数据源编辑/文件加载、项目保存/导入/导出、checkpoint 控制、预览结果/指标、benchmark 报告比较，以及持久化的可调整三栏布局。 | `web-ui/src/App.tsx`, `web-ui/src/components/` |
| 发布与质量 | CI 覆盖 Python、Rust（含 90% 行覆盖门槛）、Studio 后端/前端/E2E、wheel 隔离 smoke、OpenAPI 同步、依赖审计，以及四档 benchmark smoke；最新 `main` CI run `30122407953` 成功。 | `.github/workflows/ci.yml`, `benchmarks/README.md` |

## 窗口内进展

| PR / 状态 | 合并或 head 提交 | 本周进展 |
| --- | --- | --- |
| #14 `MERGED` | merge `beea14f`; 功能提交 `9af2ec5`, `15a30d3` | 外部数组专用 plan 延迟/绕过 DataFusion session；混合图仍保留单一 DataFusion runtime；补充 UDF namespace 排队校验和回归测试。 |
| #15 `MERGED` | merge `2aa51d8`; `93214b3` | 增加原生 Windows PowerShell Studio start/stop 包装器和稳定进程身份测试。 |
| #16 `MERGED` | merge `5b3606f`; `f99360d` | 修复 release 安装下 Studio worker 环境被启动器覆盖的问题，并补充安装文档/配置测试。 |
| #17 `MERGED` | merge `af4a33c`; `0784997`–`6200c4f` | Studio 增加多数据源编辑和保存后预览、图安全 SQL alias 增删改、可持久化/可复位的面板尺寸，以及 UI/E2E 覆盖。 |
| #18 `MERGED` | merge `8b2c1b1`; `5d9fa77` | CI benchmark 从单一 smoke 扩展到 `overhead/small/standard/nightly` 四档并分别保存结果。 |
| #19 `MERGED` | merge `590c5a5` | 增加从规格到评审/文档的 Calc Flow 专员团队配置；同时形成尚未实现的 head operator 规格。 |
| #20 `MERGED` | merge `df38ce5` | 文档统一到 Rust-native v2 架构并新增文档索引/架构图。 |
| #21 `MERGED` | merge `8cacc60` | 删除不存在的 Python binding crate 测试路径说明。 |
| #22 `MERGED` | merge `abe9a4a` | 修正文档链接大小写，使 release-config 校验通过。 |
| #23 `MERGED` | merge `2c1bc51` | 在主文档树中交叉链接 Rust/Python 示例。 |
| #24 `MERGED` | merge `0e64ca7`; `a82b1a1` | 对齐 Rust/Python canonical examples，并加入可执行 Rust SQL join 示例。 |
| #25 `MERGED` | merge `6f05b71`; `49abff6`, `e68ea4c`, `2eed250` | 专员团队迁移为 Codex-native 配置；Rust coverage 前后回收磁盘并使迁移校验可移植。 |
| #26 `MERGED` | merge `f71e49d`; `a12bd43`, `d23bc4a` | 全仓 Markdown 完整性/一致性复核并修正 Rust v2 迁移 PR 历史。 |
| #27 `OPEN` | head `4127982`（12 commits） | 正在增加 Arrow table × NumPy/JAX matrix、mapped provider、多输出/所有权边界和示例；尚未进入 `main`。当前 11 个检查通过，`Rust core parity` 失败：新 NumPy ownership Rust 测试所需的 `numpy` 未安装（binding crate 58 passed / 1 failed），后续 coverage/docs 步骤因此跳过。 |

## 值得关注的缺口与风险

1. **表与数组混合计算尚未交付。** `main` 的外部数组执行隔离已完成，但
   table × NumPy/JAX matrix 仍在 PR #27，且有阻断 CI。近期优先级应是修正
   Rust job 的测试依赖/隔离方式并重新建立绿色门禁，而不是把开放 PR 当作已发布能力。
2. **head operator 只有规格。** `.codex/artifacts/specs/head-operator.md` 已存在，
   但 `crates/`、`python/` 和 Studio 中没有 `HeadOperator` 或 `kind: head`
   实现。它是明确、边界较小的下一项引擎功能候选。
3. **runner 只接受一个外部输入。** `ExecutionPlan::single_external_input` 明确要求
   exactly one external input；SQL/直接 plan 执行支持多输入，Studio 也已能编辑多数据源，
   但 micro-batch/streaming runner 还不能原生协调多源游标、水位或恢复。若目标包含持续多源
   处理，需要先定义同步与 checkpoint 语义。
4. **Studio 编辑面小于引擎能力面。** 核心契约支持 `external` operator，当前 toolbox
   只创建 expression/SQL 节点。可考虑由 runtime catalog 驱动外部 operator 的可视化创建与
   options 编辑，避免项目只能通过导入/代码方式使用 provider。
5. **部署与持久化仍是本地形态。** Studio 明确为无认证、仅 loopback、single-user；
   内置项目/checkpoint 实现是文件系统。若要团队协作或远程运行，应作为独立产品方向处理，
   不能直接放宽监听地址。
6. **兼容性与性能门禁仍有限。** v2 不加载 v1 项目或 checkpoint，只保留 fixture
   对照；benchmark CI 虽已覆盖四档规模，但文档明确说明在获得至少 20 个同机可比的
   `main` 样本前仅发布信息性 artifact，不会按性能变化阻断合并。

## 证据与来源限制

- 源码/本地历史命令：
  - `multica repo checkout git@github.com:wegamekinglc/calc-flow.git --ref main`
  - `git status --short --branch`
  - `git rev-parse HEAD`、`git rev-parse origin/main`
  - `git log origin/main --since='2026-07-18T15:00:00+08:00' --until='2026-07-25T15:00:00+08:00'`
  - `git log origin/main --first-parent ...`
  - `git diff --stat 5be0e5a..origin/main`、`git diff --name-status 5be0e5a..origin/main`
  - `find`/`rg`/`sed` 检查 `crates/`、`python/`、`web-ui/`、`schemas/`、测试、
    README、CI workflow 和 benchmark 文档。
- GitHub 命令：
  - `gh pr list --repo wegamekinglc/calc-flow --state all --limit 100 --json ...`
  - `gh pr view 14` 至 `gh pr view 27`（状态、时间、commits、files）
  - `gh pr checks 27`
  - `gh run view 30146006588 --job 89647987554` 及 job log
  - `gh run list --branch main --workflow CI --limit 8`
- GitHub 数据可用，因此不是 local-git-only 分析；但没有审阅 issue/PR discussion、
  review thread 或外部发布指标。开放 PR 的描述仅按提交、改动文件和 CI 证据归纳。
- 为保护工作区，本次未切换分支、未修改源码、未运行会生成 `target/`、`.venv/` 或
  `node_modules/` 的本地 build/test。能力正确性采用源码与已提交测试交叉检查，并引用
  最新 `main` CI 成功结果。唯一有意写入的是本分析文件。
