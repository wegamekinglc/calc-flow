# 构建与重现

从含本目录的 checkout 根目录运行。依赖 Python 3.13、uv、Git、仓库固定 Rust
toolchain 和 x86_64 Linux 的 CPU 0–7。固定依赖见 `requirements.txt`。
产品基线为 `d9e2759fb31bfd0c915d31f0cc534665ece72ee9`，候选产品提交为
`9db6d10b29bc6349b529d2807cfd23ba4dd6aace`；后续证据提交不改产品代码。
原 wheel 构建分别为 908.746 秒、572.901 秒，不计入采集预算。

在两个独立环境安装匹配源码的 release wheel；构建/安装记录、实际 native/wheel 哈希
和依赖版本见数据包的 `build-*.json`、`seals.json`、`dependencies-*.json`。
本次为减少路径导致的缓存失效，在同一个专用构建 worktree 先构建干净基线，再复制
已核对哈希的候选 Rust 文件构建候选。主任务 checkout 和 DAL-475 工作区未混用。

新机器可在独立 worktree 构建同一版本：

```bash
git worktree add --detach target/sql-prepare-reuse/source-base d9e2759fb31bfd0c915d31f0cc534665ece72ee9
git worktree add --detach target/sql-prepare-reuse/source-head 9db6d10b29bc6349b529d2807cfd23ba4dd6aace
uv venv --python 3.13 target/sql-prepare-reuse/env-base
uv venv --python 3.13 target/sql-prepare-reuse/env-head
uv pip install --python target/sql-prepare-reuse/env-base/bin/python -r .codex/artifacts/sql-prepare-reuse/requirements.txt
uv pip install --python target/sql-prepare-reuse/env-head/bin/python -r .codex/artifacts/sql-prepare-reuse/requirements.txt
```

分别在对应 source worktree 执行，解释器与输出目录替换为上一步各 side 的绝对路径：

```bash
CARGO_BUILD_JOBS=4 maturin build --release --locked --interpreter <python> --out <wheels>
uv pip install --python <python> --no-deps <wheel>
```

新的二进制哈希通常与原实验不同，应生成自己的 `seals.json`、身份记录及预算账本。
`seals.json` 的 `base`/`head` 各含 `python` 绝对路径、`native_sha256`、`wheel_sha256`
和 `source_ref`；`budget.json` 初值为 `{"limit_s":600,"spent_s":0,"stages":[]}`。
环境准备、诊断、预检、失败、正式采集和清理耗时都纳入该账本，构建时间单列。
先确认无 cargo/rustc/maturin 构建运行；原实验的无构建观测记录保存在原始包中。

实际预检和正式命令如下。两个场景、每场景两轮十对 AB/BA，共 80 次正式观测，
每个新 worker 在 ready-input gate 前暖身一次；冷准备也有独立时钟及指标。

```bash
taskset -c 0-1 target/sql-prepare-reuse/env-base/bin/python .codex/artifacts/sql-prepare-reuse/collect.py --seals target/sql-prepare-reuse/seals.json --budget target/sql-prepare-reuse/budget.json --output target/sql-prepare-reuse/preflight --preflight
taskset -c 0-1 target/sql-prepare-reuse/env-base/bin/python .codex/artifacts/sql-prepare-reuse/collect.py --seals target/sql-prepare-reuse/seals.json --budget target/sql-prepare-reuse/budget.json --output target/sql-prepare-reuse/results
target/sql-prepare-reuse/env-base/bin/python .codex/artifacts/sql-prepare-reuse/summarize.py target/sql-prepare-reuse/results/samples.json > target/sql-prepare-reuse/summary.json
target/sql-prepare-reuse/env-base/bin/python .codex/artifacts/sql-prepare-reuse/audit.py --directory target/sql-prepare-reuse/results --seals target/sql-prepare-reuse/seals.json --summary target/sql-prepare-reuse/summary.json
```

诊断对同样两个 workload 分别调用公开同步和异步接口，保存每次 DataFusion 指标：

```bash
taskset -c 4-7 <python> .codex/artifacts/sql-prepare-reuse/diagnose.py --native-sha <native-sha256> --output <diagnostic.json>
```

诊断的 API 时钟包含逐次指标 getter，不用作正式速度排名或独占 GIL/线程投递计时。
正式 worker 复用已审 `sql_worker.api_calls`，只在输出 IPC 后提取最后一次详细指标。
API、输入绑定、native execute、结果包装及完整消费时钟保持原边界；末次查询指标
单独标记，不能乘以批次数当作整段计时。
