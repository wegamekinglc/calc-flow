# DAL-417 Arrow/IPC 重测与计时审计

[中文报告](REPORT.md) 给出本轮结果、Arroyo 实际缺口和调整后的优化优先级；
[benchmark 逐组审计](BENCHMARK-AUDIT.md) 追踪维护入口到 CI 汇总的完整边界。
[预记约定](PLAN.md) 与[计时修正记录](MEASUREMENT-CORRECTION.md) 同时保留。
本轮只新增研究/测量产物；#494/#510、维护中的 benchmark 和门禁未修改。

四个场景、每 arm 五次，最终 40 次、496,640 条输出逐值校验。
原始数据包保留所有 Arrow IPC 输入/输出、单次时钟/命令/RSS、graph/native seal，
并保留排除的首轮全部 40 次及其精确脚本，不能将两轮合并统计。
数据通路为 Arrow/Batch；JSON 文件只保存配置、图和统计元数据。

## 离线复核

从 repo root 使用 Python 3.13、numpy 2.5.4、pyarrow 24.0.0 运行；不需要 native：

```bash
sha256sum --check .codex/artifacts/arrow-ipc-audit/data/SHA256SUMS
mkdir -p target/arrow-ipc-evidence
tar xzf .codex/artifacts/arrow-ipc-audit/data/DAL-417-arrow-raw.tar.gz \
  -C target/arrow-ipc-evidence
uv run --no-project --python 3.13 --with numpy==2.5.4 --with pyarrow==24.0.0 \
  python .codex/artifacts/arrow-ipc-audit/audit_results.py \
  target/arrow-ipc-evidence/raw-data/results
uv run --no-project --python 3.13 --with numpy==2.5.4 --with pyarrow==24.0.0 \
  python .codex/artifacts/arrow-ipc-audit/verify_columns.py \
  target/arrow-ipc-evidence/raw-data/results
```

成功输出 `audit=passed, samples=40, output_rows=496640`；summary/paired 也会离线重算。
核对 IPC 输入分批、完整输出数学值、安全整型规范化、校验和、sample inventory、worker
seal、时钟顺序和资源。审计不要求 native 已安装；读取当前脚本哈希以确保没有采样后改写。
补充 verifier 核对全部列名和所有 window_end−window_start=4 秒，覆盖原生窗口额外列。
首轮的历史审计可使用包内 `first-pass/harness/audit_results.py` 及 `first-pass` 路径；
其 audit passed 只证明原始记录自洽，`exclusion.json` 仍禁止将它纳入最终统计。

## 重新测量

仅需要新证据时运行；不为交接重复采样。要求 x86_64 Linux/WSL、CPU 0–7、uv、
Python 3.13、仓库 Rust toolchain。先从 PR checkout 复制本目录到固定基线 worktree：

```bash
git worktree add --detach target/arrow-ipc-baseline \
  b5ae227e48ac9bb8a81583a23485e2c6064cee52
cp -R .codex/artifacts/arrow-ipc-audit \
  target/arrow-ipc-baseline/.codex/artifacts/
cd target/arrow-ipc-baseline
bash .codex/artifacts/arrow-ipc-audit/reproduce.sh
```

脚本先拒绝错误 HEAD 和 tracked staged/unstaged diff，使用新目录。
默认 locked release 构建另计；若已有本轮兼容 wheel，可设置 `DAL417_WHEEL`，脚本会
检查固定 wheel SHA。新建构建的 native hash 会作为明确参数传给每个 worker，并记录到
新的 identity；不能冒用本次 native seal 或将不同主机的新数据混入此报告。

本次实际正式命令：

```bash
taskset -c 0-1 target/arroyo-attribution/venv/bin/python \
  .codex/artifacts/arrow-ipc-audit/collect.py \
  --output target/arrow-ipc-audit/results --pairs 5 --budget 500
```

## Arroyo 证据重现

`probe_arroyo.py` 对固定 release binary 尝试 arrow/arrow_ipc/ipc 三种格式，SQL
声明、退出码和错误日志均保留。`arroyo-ipc-adapter.patch` 是固定 v0.15.0 源码上的
最小 single_file source/sink 基准适配；真正调用 SourceCollector/真实图，未替换计算算法。
只对 `.arrow` 路径生效；SQL 的 format=json 仅为声明占位，适配分支不会调用 JSON codec。
该补丁通过 cargo check，release 构建未在本轮预算内完成，未获得运行时/窗口语义证据。

```bash
git clone --depth 1 --branch v0.15.0 https://github.com/ArroyoSystems/arroyo.git \
  target/arroyo-ipc-reproduction
test "$(git -C target/arroyo-ipc-reproduction rev-parse HEAD)" = \
  4ef15f6336a596651f10cf27c1ee29c29bf81577
git -C target/arroyo-ipc-reproduction apply --unidiff-zero \
  ../../.codex/artifacts/arrow-ipc-audit/arroyo-ipc-adapter.patch
CARGO_BUILD_JOBS=4 CFLAGS=-std=gnu17 \
  CMAKE="$PWD/.codex/artifacts/arrow-ipc-audit/cmake-compat.sh" \
  cargo check --locked --manifest-path target/arroyo-ipc-reproduction/Cargo.toml \
  -p arroyo-connectors
```

上面相对补丁路径由 git 在目标仓库内解释。`cmake-compat.sh` 只给旧 librdkafka 的 CMake
配置补 policy minimum；不修改依赖源码/lock。实际受限 release 构建命令和 exit=124 在
原始包 `arroyo/adapter-release.json`，没有从未完成构建给出速度结论。
