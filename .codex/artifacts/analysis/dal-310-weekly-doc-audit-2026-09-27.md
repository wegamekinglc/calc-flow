# DAL-310 文档一致性审计（2026-09-27）

## 基线与枚举

本次从远端最新 `main` 的 `ed169219182b1d08b8187e70e0e64f92fb41753e` 建立独立工作树和 `feature/2026-09-27-doc-audit-docs` 分支。`git fetch origin main` 后，`git rev-parse origin/main` 与 `git ls-remote origin refs/heads/main` 一致。原托管 checkout 干净，未覆盖既有改动。

用 `rg --files --hidden --no-ignore -g '*.md' -g '!.git/**'` 枚举基线全部 135 个 Markdown 文件；本证据文件加入后为 136 个。最终按首级目录统计为：根目录 4、`docs/` 33、`.codex/` 69、`.claude/` 14、`.agents/` 1、`benchmarks/` 6、`crates/` 2、`design/` 2、`examples/` 1、`scripts/` 1、`tests/` 1、`web-ui/` 2。枚举命令和基线提交可复核完整清单；扫描覆盖隐藏目录、根目录以及其他路径。

## 事实核验与修订

以 `crates/calc-flow/src/lib.rs` 的导出、`python/calc_flow/__init__.py` 和 `pyproject.toml`、`web-ui/backend/pyproject.toml`、`web-ui/backend/src/calc_flow_studio/app.py`、`web-ui/openapi.json`、`web-ui/package.json`、`.github/workflows/ci-linux.yml` 和仓库脚本为依据，复核当前 API、版本、命令、REST 路由、安全要求和 CI 超时。核对 `AGENTS.md`、`docs/introduction.md`、`docs/README.md`、`README.md`、相关用户指南及同名 agent 定义；`.codex/artifacts/`、`design/` 的旧路径和旧版本描述按历史决策记录处理，不改写为当前规范。

- `CLAUDE.md` 把核心 Python 最低版本写成 3.13，并把 Linux 预编译和执行都写成 30 分钟；已对齐包要求的 CPython 3.9+、Studio 的 Python 3.13+，以及 CI 的 45/30 分钟。
- `web-ui/backend/README.md` 称核心依赖为 “v5”；已按 Studio 的 `pyproject.toml` 写明 `calc-flow-python>=2026.9.25,<2027`。
- `docs/api-reference.md` 遗漏实际存在的 `GET /api/v3/session` 和写请求 token 要求；已补齐路由、`X-Calc-Flow-Session`、本地 Host/Origin 限制及该路由不进入 OpenAPI 的事实，并删除规范页中的旧版本迁移叙述。
- `.claude/agents/README.md` 的固定全阶段流程与 reviewer 职责描述落后于 `.codex/agents/README.md`；已从 canonical 语义同步为按需路由和独立评审。

本次只修正文档，无新引擎能力、公共 API 变更或格式变化；`CHANGELOG.md` 不增加条目。没有改动 Rust、Python、TypeScript 或生成契约。

## Agent 对应和同步

十个 `.codex/agents/cf-*.toml` 与十个 `.claude/agents/cf-*.md` 的名称和 `description` 完全对应；Claude 正文与 canonical 的差异为客户端工具名、工作树入口、规则文件名或换行适配。下表逐项给出文件、名称和 Multica 记录 ID。全部 Multica `description` 已与 canonical 一致。`current` 表示 canonical 正文已在当前 `instructions` 中，`synced instructions` 表示本次把过期的 canonical 语句更新到运行时说明，同时保留中文紧凑交付覆盖层。

| Agent             | Codex file                                  | Claude file                                | Multica ID                             | State               |
|-------------------|---------------------------------------------|--------------------------------------------|----------------------------------------|---------------------|
| `cf-orchestrator` | `.codex/agents/cf-orchestrator.toml`        | `.claude/agents/cf-orchestrator.md`        | `a6c22a43-3dc1-4338-8581-3e1317dd9519` | synced instructions |
| `cf-implementer`  | `.codex/agents/cf-implementer.toml`         | `.claude/agents/cf-implementer.md`         | `227e58dc-a05a-48b6-bd20-94af140da251` | synced instructions |
| `cf-tester`       | `.codex/agents/cf-tester.toml`              | `.claude/agents/cf-tester.md`              | `a553f49e-ae3c-401e-88d4-919d9e078703` | current             |
| `cf-api-designer` | `.codex/agents/cf-api-designer.toml`        | `.claude/agents/cf-api-designer.md`        | `ea5b9d56-eb67-4d40-ba30-d07054d02f77` | current             |
| `cf-critic`       | `.codex/agents/cf-critic.toml`              | `.claude/agents/cf-critic.md`              | `d29d22b4-ec2a-4a2c-ab7b-861f7f4d0805` | current             |
| `cf-reviewer`     | `.codex/agents/cf-reviewer.toml`            | `.claude/agents/cf-reviewer.md`            | `a274f526-62bc-481f-93ef-67b4a63d9f3a` | synced instructions |
| `cf-doc-writer`   | `.codex/agents/cf-doc-writer.toml`          | `.claude/agents/cf-doc-writer.md`          | `1b71d01f-a1e8-4e73-97f0-3a2576a7378c` | current             |
| `cf-simplifier`   | `.codex/agents/cf-simplifier.toml`          | `.claude/agents/cf-simplifier.md`          | `113b511d-172f-4673-becd-cc0db10b70b8` | current             |
| `cf-performancer` | `.codex/agents/cf-performancer.toml`        | `.claude/agents/cf-performancer.md`        | `b37225a5-8640-4da1-9500-89adfab9c66b` | synced instructions |
| `cf-spec-writer`  | `.codex/agents/cf-spec-writer.toml`         | `.claude/agents/cf-spec-writer.md`         | `62f89f6b-8158-4c5d-ac1d-8ed6d092473f` | current             |

四次 Multica 更新都先以 `multica agent get <id> --output json` 读取原记录，只传 `--instructions`；回读证实除 `instructions` 与自动 `updated_at` 外无字段变化。未修改 agent 名称、ID、模型、运行时、技能、权限、并发或 squad。

## 检查结果与边界

- Markdown 解析器逐文件检查基线 135 个文件：905 个本地链接、其中 353 个锚点，均解析成功；71 个外链引用合并为 57 个唯一 URL。HTTP HEAD 检查 56 个返回 200；Coveralls 页面返回 403，无法据此判定链接失效。
- 行内代码路径筛出 301 个不同的仓库根相对候选；75 次不存在的引用位于历史 spec/analysis/design、未来示例占位或旧路径说明，现行用户文档没有此类候选。历史记录保留原貌。
- 40 个现行根目录、`docs/`、Studio、examples 与 benchmarks Markdown 文件中的 99 个 Bash 代码块经 `bash -n` 检查，98 个通过；`docs/python-release.md` 的 `<version>` 是发布者须替换的占位符，原样执行会触发 shell 重定向语法错误。
- `scripts/run_rust_tests.py`、`scripts/run_rust_coverage.py`、`scripts/run_examples.py`、Studio 启停脚本、schema、OpenAPI 与生成类型路径存在；`web-ui/package.json` 包含文档引用的 npm 命令。此环境有 `uv`、`cargo`、`npm`、`python`、`bash`、`curl`，没有 `pwsh`，故未在本机执行 Windows 命令。
- 仅进行文档相关结构、链接、锚点、配置和 diff 检查；未运行构建或测试套件。独立 cf-reviewer 评审和随后 PR 尚待协调器安排，本分支不自行合并。
