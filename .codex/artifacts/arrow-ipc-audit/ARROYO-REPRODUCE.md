# Arroyo 邻近补测的复现与离线审计

先读 [预记](ARROYO-EXTENSION-PLAN.md) 和 [结果/边界](ARROYO-RESULTS.md)。
所有下面路径从 calc-flow repo root 解释。正式数据/失败预检分开，禁止混合旧样本。
原 `reproduce.sh` 会跑原 CF 内部矩阵；本次接续**没有运行它**。

## 先离线验证（不需要引擎或数据库）

需要 Python 3.13、numpy 2.5.4、pyarrow 24.0.0。可使用已有 venv 或 `uv run --no-project
--python 3.13 --with numpy==2.5.4 --with pyarrow==24.0.0 python` 前缀。

```bash
sha256sum --check .codex/artifacts/arrow-ipc-audit/data/arroyo-extension/SHA256SUMS
mkdir -p target/arroyo-adjacent-offline
tar xzf .codex/artifacts/arrow-ipc-audit/data/arroyo-extension/DAL-417-arroyo-adjacent-raw.tar.gz \
  -C target/arroyo-adjacent-offline
target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/audit_adjacent.py \
  target/arroyo-adjacent-offline/raw-extension/results
target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/summarize_adjacent.py \
  target/arroyo-adjacent-offline/raw-extension/results
```

期望 `audit=passed,samples=20,pairs=10,output_rows=327680`。脚本核对源/输出 hashes、
所有列和值/原序、batch 边界、native seal、计时边界、源仍开放、资源、exit 和五对 AB/BA。
`pairs.json`、`phase-summary.json` 从原始时钟重算。归档包含采样时精确 harness 副本，
identity 的旧文件 hashes 不因新增离线摘要脚本而变化。

窗口单次实际证据在 `window-probe-v2/{input.arrow,cf,arroyo,alignment.json}`；
失败窗口声明及 CF 初次缺 Cursor 的日志保留在各自 probe/preflight 目录，不算正式样本。
新离线验证还用一个值被破坏、但 hashes 同步更新的副本验证 oracle 必须拒绝，见验证记录。

## 本机实际构建配置

复用 `target/arroyo-arrow-source` 的 v0.15.0/4ef15f6 缓存。最终完整补丁：

```bash
git -C target/arroyo-arrow-source apply --unidiff-zero \
  ../../.codex/artifacts/arrow-ipc-audit/arroyo-ipc-observed.patch
```

仅对 pristine 固定 ref 应用一次；如果工作树已有旧 IPC 补丁，应先在另一个隔离 upstream
checkout 还原到同 tag，再应用完整 observed patch，不能叠加新旧补丁。
本机未重置 cache checkout，实际是在旧适配上增量修改后生成完整 diff。

真实 CLI 的 build.rs 要求 SQL codegen 数据库和真实 webui/dist。我们使用独立 PG 18
cluster（initdb `-U dal417 -A trust`，仅 loopback，port 47836），新建 arroyo_build，按 V 数字
顺序执行 `crates/arroyo-api/migrations/V*.sql` 共 25 条；**不要指向已有产品数据库**。
实际 `postgres.json`、setup/server/stop 日志在包内。环境变量为：

```bash
CARGO_BUILD_JOBS=8 CFLAGS=-std=gnu17 \
  CMAKE="$PWD/.codex/artifacts/arrow-ipc-audit/cmake-compat.sh" \
  DATABASE_URL=postgresql://dal417@127.0.0.1:47836/arroyo_build \
  cargo build --release --locked --manifest-path target/arroyo-arrow-source/Cargo.toml \
    -p arroyo --bin arroyo
```

实际不是无限制运行：`incremental_build.py target/arrow-ipc-extension/build` 为每次命令
加剩余总预算的 timeout，并保存起止/退出/log。输入 `build/budget.json` 是有状态累计账本，
`budget_seconds=1800,spent_seconds=0,attempts=[]` 仅可在一个**新实验**开始前初始化；
不得删除已有失败记录/复位预算。helper 读取 `build/postgres.json` 的 `database_url`。
需要 frontend 时使用同 helper 加 `--frontend`，它调用 `frontend_build.sh`，官方锁文件
pnpm 9.7.1 frozen install + build，输出与命令同样计入该累计预算。
本次六次命令精确见归档 JSON，不另跑冷构建验证。

构建完 stop 私有 PG 并确认退出，然后才测量；数据库准备墙钟不冒充编译。
缺 PG18/pnpm/cache 或新环境冷构建不能在预算内完成时应保留错误/时长，不能换 fake UI、
改算法或降优化级别填排名。构建和测量预算各自独立。

## 本机实际采集命令

复用 `target/arroyo-attribution/venv` 和固定 CF wheel/native（hash 在结果报告和 identity）。
两个引擎不与编译竞争。先单次预检与 window probe，最后才正式五对：

```bash
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/collect_adjacent.py \
  --binary target/arroyo-arrow-source/target/release/arroyo \
  --output target/arrow-ipc-extension/preflight-v2 --preflight --budget 150 \
  > target/arrow-ipc-extension/preflight-v2.log 2>&1
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/probe_window_alignment.py \
  --binary target/arroyo-arrow-source/target/release/arroyo \
  --output target/arrow-ipc-extension/window-probe-v2 \
  > target/arrow-ipc-extension/window-probe-v2.log 2>&1
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/collect_adjacent.py \
  --binary target/arroyo-arrow-source/target/release/arroyo \
  --output target/arrow-ipc-extension/results --budget 500 \
  > target/arrow-ipc-extension/acquisition.log 2>&1
target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/audit_adjacent.py target/arrow-ipc-extension/results
target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/summarize_adjacent.py target/arrow-ipc-extension/results
bash .codex/artifacts/arrow-ipc-audit/extract_arroyo_plans.sh
```

上述目录必须尚不存在。共享 `measurement-budget.json` 在采样前创建为
`{"limit_s":600,"prior_phase_reserve_s":120,"spent_s":120,"entries":[]}`，每次自动累计，
本机实际账本保留所有失败/预检/正式观测而非这个空初始化示例。
window probe 与 collector 均需要 `multica daemon status`，仅用来保护工作区 daemon，
不读业务数据；进程以精确 child PID 管理。native CLI 显式 stop=128 按上游合同接受。
plans 脚本只读提取当前实际 job 的 protobuf，需 upstream api.proto 和 protoc；原始包已含
这些提取结果，不要求离线审计者安装 protoc/SQLite job database。

新环境先按旧 README 固定 CF 源码/依赖并单独构建 wheel，不能假设新 binary bytes 与本机相同。
extension 的 `fixtures.NATIVE` 是严格防混用 seal；如果新构建不匹配，必须在新实验目录内
明确将该常量改为新 native SHA，并重新预记/采样、保存精确脚本和 identity，不能改本次原始
归档/旧结果或绕过校验。这个新数据集用于当地环境复现趋势，不冒充本机相同 fingerprint。

## 文件与解释

- `cf_batch_worker.py`：真实 CF native Batch 源/汇，输出绑定在 core 后。
- `extension_transport.py`：受控进程、IPC 消费、EOF gate、RSS/退出记录；所有表数据 Arrow。
- `arroyo-ipc-observed.patch`：真实 single_file 接入与分层时钟，不是产品/public connector 改动。
- `collect_adjacent.py`：固定两场景五对，拒绝产品基线变化、剩余预算不足、错值/错序。
- `probe_window_alignment.py`：单语义探针，trace/EOF/值对齐，不产生排名。
- `audit_adjacent.py`：采集前已冻结的离线正确性/时钟/配对审计。
- `summarize_adjacent.py`：采集后新增的纯离线统计，不参与 acquisition timers。
- `raw-extension/build/`：六次完整日志及私有 DB 已清理证据；不包含数据库目录/依赖缓存。

本次没有全套性能、nightly、全量 pytest/Cargo 回归或产品优化。使用已有 CI 门禁；
本地做定向脚本静态/复杂度、补丁可回退校验、IPC 全量/反例和归档离线复核。
