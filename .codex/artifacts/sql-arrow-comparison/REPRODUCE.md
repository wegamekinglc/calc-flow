# SQL / Arroyo 复现与离线复核

固定范围、计时与样本数见 `PLAN.md`，结论及所有边界见 `RESULTS.md`。
这是 #511 的新增范围，不运行旧 `reproduce.sh` 或 streaming/window collectors。

## 本次实际命令

repo root，沿用已有 venv/wheel 和真实 Arroyo release binary，未构建：

```bash
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/sql-arrow-comparison/collect_sql.py \
  --binary target/arroyo-arrow-source/target/release/arroyo \
  --output target/sql-arrow-comparison/preflight --preflight \
  > target/sql-arrow-preflight.log 2>&1
target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/sql-arrow-comparison/audit_sql.py \
  target/sql-arrow-comparison/preflight
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/sql-arrow-comparison/collect_sql.py \
  --binary target/arroyo-arrow-source/target/release/arroyo \
  --output target/sql-arrow-comparison/results \
  > target/sql-arrow-comparison/acquisition.log 2>&1
target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/sql-arrow-comparison/audit_sql.py target/sql-arrow-comparison/results
```

六次预检及 30 次正式分别保存，采集脚本拒绝已存在的 output，预算账本位于其父目录，
自动累计，不得删除/复位已有账本。启动前预留 60s，单 invocation 在 45s 发出超时、由
既有 owned process 逻辑收集至多约 9s 的显式清理。预算 600s 包括新观测预检/失败/正式，
不包括读源码/撰写报告；实际 80.284s，无失败，无额外采样，无新产品构建。

要求 Linux/WSL、CPU 0–7、Multica CLI（只查询 daemon PID 防误停）、固定 Python 3.13.9、
PyArrow 24.0.0、NumPy 2.5.4、psutil 7.2.2，以及 #511 的 sealed binary/wheel。
逐个 worker 核对 native seal，采集器核对实际 Arroyo binary seal 和产品基线；不接受错误
安装/别的构建冒充本次结果。在新环境重建得到新 seal 时必须新实验、新 provenance，
明确适配 constants 后重新预记，不能回写本批旧 seal/数据。构建复现细节沿用
`../arrow-ipc-audit/ARROYO-REPRODUCE.md`，不在本轮预算中偷跑编译。

## 离线复核，无引擎/数据库需求

保持 PR 里的 `sql-arrow-comparison` 与兄弟 `arrow-ipc-audit` 源文件，使用包含上述 Python
数据依赖的环境。共享模块只导入旧 fixture/transport/identity，不执行 native 引擎。

```bash
sha256sum --check .codex/artifacts/sql-arrow-comparison/data/SHA256SUMS
mkdir -p target/sql-arrow-offline
tar xzf .codex/artifacts/sql-arrow-comparison/data/DAL-417-sql-arrow-raw.tar.gz \
  -C target/sql-arrow-offline
target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/sql-arrow-comparison/audit_sql.py \
  target/sql-arrow-offline/raw-sql/results
```

期望 `audit=passed,preflight=False,samples=30,output_rows=491520`。
summary/paired 从原始 span 重算；包括所有列/值/原序、相同输入 bytes/batch、hashes、
正常 API 子阶段计数/包含关系、完成消费时钟、exit/affinity/RSS 及固定 ABC/CBA inventory。
预算、原始每次 IPC、六次预检、正式脚本副本及共享脚本 hashes 随包保留。预检另有其实际
四份采集脚本快照，audit_sql 是预检之后、正式之前冻结的第五份脚本，不把预检纳入正式比值。

`common.py` 复用旧证据 helper；`sql_worker.py` 只运行两个真实 CF SQL 路径；
`sql_transport.py` 将 CF 改为和现有 Arroyo 一样的输入 ready / 输出 file IPC 消费观察，
`collect_sql.py` 固定邻近观测及累计预算；`audit_sql.py` 是纯离线核对与统计。
包装/IPC 已分列；嵌套正常成本仍在 API 总时钟内，绝不从总计扣去再称公开 API 的速度。
