# PR #482 修复后的性能复核

**总体结论：不确定。** 五项固定配置均未达到“连续两轮配对中位数置信区间下界超过 +5%”的回退条件；其中四项为 `inconclusive`，Interval 单 CPU 为 `no-confirmed-regression`。这份证据没有确认回退，也没有证明全部路径无回退、等价或获得加速。保留既定样本，不追加测量追求确定结论。

比较基线为实际 release 构建来源 `61aab0e571800df8e8a18ad2244993d0929f5247`，候选为 `25e8b4c33114f1f85a8473136bf3c1f7991873e5`。候选包含 main 集成及 PR 的辅助函数重构、测试修复；结果描述这两个版本的整体差异，不能把每个波动单独归因于复杂度重构。历史运行的时间值没有混入本次比较。

以下为每版本 20 个正式样本的 P50；每项有两轮，每轮十个相邻 A/B 配对。单位为毫秒，正差值表示候选更慢。

| 配置              | 每侧行数  | 基线 P50 ms | 候选 P50 ms | P50 差值 | 维护规则判定                          |
|-------------------|-----------|-------------|-------------|----------|---------------------------------------|
| ASOF / 1 CPU      | 1,000,000 | 94.539      | 95.605      | +1.13%   | 不确定（inconclusive）                |
| ASOF / 32 CPU     | 1,000,000 | 63.670      | 64.713      | +1.64%   | 不确定（inconclusive）                |
| Interval / 1 CPU  | 262,144   | 218.873     | 216.178     | -1.23%   | 未确认回退（no-confirmed-regression） |
| Interval / 8 CPU  | 262,144   | 158.159     | 152.362     | -3.67%   | 不确定（inconclusive）                |
| Interval / 32 CPU | 262,144   | 156.708     | 159.142     | +1.55%   | 不确定（inconclusive）                |

描述性 P95 使用预先声明的 nearest-rank：20 个样本排序后取第 19 个。区间为每轮十个“候选/基线 − 1”配对变化的中位数区间，第 2、9 个秩统计量，实际覆盖率 97.8515625%；它不是汇总 P50 比值的置信区间。该区间依赖配对样本独立且变化分布稳定的假设；交替执行本身不能证明这些假设。原始 maintained statistics 中的线性插值 `head_p95` 保留，但未替换此处的 nearest-rank P95。

| 配置              | 基线 P95 ms | 候选 P95 ms | 第一轮配对区间     | 第二轮配对区间    |
|-------------------|-------------|-------------|--------------------|-------------------|
| ASOF / 1 CPU      | 99.414      | 105.996     | [-5.68%, +13.80%]  | [-2.62%, +11.36%] |
| ASOF / 32 CPU     | 70.737      | 73.160      | [-12.02%, +15.18%] | [-3.15%, +23.42%] |
| Interval / 1 CPU  | 231.105     | 224.169     | [-3.08%, +0.11%]   | [-5.70%, +1.01%]  |
| Interval / 8 CPU  | 163.299     | 168.003     | [-6.99%, +11.01%]  | [-9.18%, +4.28%]  |
| Interval / 32 CPU | 196.355     | 204.279     | [-3.11%, +8.48%]   | [-1.04%, +11.15%] |

原样使用维护中的 `scripts.benchmark_suite.report.comparison` 判定：两轮下界均超过 +5% 才确认回退；任一轮上界超过 +5% 时保留 `inconclusive`；两轮上界均低于 −5% 才确认改善；其余为 `no-confirmed-regression`。本次没有确认改善。没有以单次结果、最小值或另加的等价条件替换该规则。

跨 CPU 的 P50 比值仅作描述；不同 CPU 配置没有相邻配对，不能由下表推断统计显著性。

| CPU 配置变化        | 基线 P50 加速比 | 候选 P50 加速比 |
|---------------------|-----------------|-----------------|
| ASOF 1 → 32 CPU     | 1.485×          | 1.477×          |
| Interval 1 → 8 CPU  | 1.384×          | 1.419×          |
| Interval 1 → 32 CPU | 1.397×          | 1.358×          |
| Interval 8 → 32 CPU | 1.009×          | 0.957×          |

候选 ASOF 的 1→32 CPU 比值为 1.477×，Interval 的 1→8 CPU 比值为 1.419×、1→32 CPU 为 1.358×。候选 Interval 32 CPU 的 P50 比 8 CPU 高约 4.45%；本次没有阶段诊断，不能据此确定原因。

**测量边界与环境。** 复用维护中的 EngineCase：ASOF 每侧 1,000,000 行、Interval 每侧 262,144 行，64 个实体，每侧先合并为一个 RecordBatch；保留现有 1 GiB 状态预算。ASOF 使用原 bounded-feed adapter，Interval 使用 exact-cursor adapter。ready-source 提交至预期输出送达并完成 Arrow concat 为计时区间；构建 fixture、预热、完整 oracle、启动、watermark、EOF 和清理在计时区间外、总预算内。构建时间单独报告。没有诊断 feature 构建或额外 profile 采集。

机器为 WSL 环境，32 个可用逻辑 CPU。1 CPU 固定为 `[0]`，8 CPU 为 `[0,2,4,6,8,10,12,14]`，32 CPU 为 `[0..31]`。系统报告的 16 个 core-ID 分组不代表已确认的宿主机物理核/SMT 拓扑。主代理在可见宿主 `/proc` 的环境启动，已结束拥有的构建/测试进程；控制器采集前及每对样本前后未发现其识别范围内的竞争编译/CPU 工作负载。宿主机整体安静程度和电源模式未知，不把宽区间解释成确定回退。

Linux load average（1/5/15 分钟）采集前为 `[0.28076171875, 0.3212890625, 0.72607421875]`，采集后为 `[0.71533203125, 0.48876953125, 0.73681640625]`。每项两版本、两轮的 machine/dependency/workload 三类 SHA-256 指纹逐项相同；不同 CPU affinity 的 machine 指纹不同，未用于版本回退分类之外的显著性比较。

**预算与一次技术恢复。** 原总预算固定为 600 秒，含 30 秒进程清理预留。首次尝试错误地从 relocated manifest 所在目录推导包路径；该目录只存证明文件，worker 的严格 native 来源检查在 hello 前拒绝了错误导入，尚未执行预热或正式样本。原报告、stdout/stderr、控制器、计划和 manifest 的精确字节保存在 `failed-import-attempt/`，没有覆盖或丢弃。

窄修复从 manifest 的 `native` 路径推导真实 sealed site，保持原 wheel/native 字节及 worker 严格 SHA 检查。恢复由主代理明确授权并经独立审查，五个配置、200 个正式样本、20 个预热和所有计时/oracle 代码不变。新增路径/预算测试先 RED 后 GREEN，连同原来源测试共 13 项通过；没有第二次自动恢复授权。

首次失败耗时 **2.265888925 秒**，恢复可用预算为 **597.734111075 秒**（工作 567.734111075 秒、清理 30 秒）。成功采集耗时 **120.411500199 秒**，两次合计 **122.677389124 秒**。基线复用原 release；候选 release 构建耗时 **654.316 秒**，单独计入构建成本。

**来源封存与离线核验。** 两臂为 `default` feature、release profile，Cargo.lock、Rust/Python 工具链和 RUSTFLAGS 一致。基线 manifest 的原 source 路径已由后续开发使用，因此新 manifest 仅把 source 指向同一 SHA/tree 的干净 detached checkout，并记录原始 manifest 的 SHA；原构建、wheel/native 和原 manifest 全部保持原样。候选原生测量绑定的是上述 `25e8b4c3`，后续文档或 CI 修复提交不能被当作已重新测量的 native。

- 基线 source：`61aab0e571800df8e8a18ad2244993d0929f5247`；tree：`15932415e462de46e0612ec26b724abf78ee82ae`。
- 基线 native SHA-256：`1c1ad9934bdbb229976ff4d0d283e751d5ce65dd80fdf2e6437014a54dfe2fca`。
- 基线 wheel SHA-256：`6dfca099e9fd0176baed387dd5c10b9e55eca7afa329ebefed61abede362cf3a`；manifest SHA-256：`5c1cd81f61fdf577fb8988871ac4937ca9089b0b872d55d7b56a74f195655449`。
- 候选 source：`25e8b4c33114f1f85a8473136bf3c1f7991873e5`；tree：`39920b4d44ba5b2bcf6111320c0df0187eb17b27`。
- 候选 native SHA-256：`bfab76292304d732306e3adb6ac0ffca6ddfd7d851a1c238e178be212a29010e`。
- 候选 wheel SHA-256：`19c7cd5829013f16482e4dddbff4c35e3d90ff287e3adf93cff2933896c901a8`；manifest SHA-256：`75af93b58c8dc7c66d0da7acbabe6c86579ecf95a42cc539d402daf532527640`。

离线审计逐项验证 source SHA/tree/cleanliness、真实 native/wheel 字节、原 baseline relocation 证明、全部 frozen harness 哈希和每个 worker 实际导入证明；核对两轮交替 AB/BA 且第二轮反向、20 个 worker 各 13 条原始响应和各 11 个独立状态目录、200 正式样本及 20 预热的完整 oracle 和完成证明。ASOF 每个输入 identity 与值精确匹配；Interval 每次 2,881,664 行的完整 pair multiset、值和 schema 通过，状态/匹配预算失败为零，watermark/eviction/late/null 证据满足固定契约。逐样本统计重算与维护函数完全一致，独立秩统计复算两轮区间也一致。离线审计没有导入 calc_flow native、构建或采集。

- 原始结果：reports/measurement.json（归档成员 `reports/measurement.json`），SHA-256 `13b1e2afddc6fe1019edc0bd4fff183d3f1ba3357d71d8f26280079700c2d74e`。
- 冻结计划：plan.json（归档成员 `plan.json`），SHA-256 `3c0e89048f27dfc866094ba27d16f136081a768805494a734078684bf3273950`。
- 控制器：measure.py（归档成员 `measure.py`），SHA-256 `4476b61628a7eae91017d82d10367cf9a0bd5b4ab7671aa6d1c738282cbc83af`。
- 正式审计：formal-offline-audit.json（归档成员 `formal-offline-audit.json`），SHA-256 `12e2515a6b8e801d87ae2f14eb224deb14e067767fdf2f55fa8e66a11fcebf0d`。
- 恢复证明：import-recovery.json（归档成员 `import-recovery.json`）、恢复离线审计（归档成员 `import-recovery-offline-audit.json`）、原失败清单（归档成员 `failed-import-attempt/inventory.json`）。

**覆盖范围与后续建议。** 本次五项配置覆盖 ASOF segment/prefix/shared-output 与 Interval retained admission/key dictionary/morsel helper 的已有热路径；固定大小直接复用此前测量，未扩展规模或组合。

- 优先补充 ASOF copied row-gather 的明确端到端案例到维护中的 join/ASOF acceptance harness，采用强制复制且 oracle 可独立核验的固定 payload；当前 all-shared fixture 不测这条路径。
- rolling entity-parallel helper 本次未测；需要性能证据时优先复用维护的 rolling kernel/engine 案例，以刚好触发实体并行的最小工作负载另行限定预算。
- 小批次、倾斜/宽表/嵌套 payload、长期状态及 checkpoint/recovery 本次未测。既有 correctness 证据不替代这些性能结论；checkpoint/lifecycle 测量应保留专门的生命周期证据契约。

上述归档成员均可从同目录的压缩包中获取。紧凑归档为 [pr482-remediation-evidence.tar.gz](pr482-remediation-evidence.tar.gz)，其 [pr482-remediation-evidence.tar.gz.sha256](pr482-remediation-evidence.tar.gz.sha256)、[pr482-remediation-bundle-index.json](pr482-remediation-bundle-index.json) 与 [pr482-remediation-bundle-verification.json](pr482-remediation-bundle-verification.json) 随附。归档保留本次 raw JSON、逐 worker stdout/stderr、封存 plan/controller/maintained harness、build manifests、审计、原失败及恢复、复杂度前后快照和主代理的测试/构建记录。大型 wheel/native 与状态 payload 不打包，保留其哈希和目录库存；原始 JSON 中的绝对路径保持原字节，归档清单提供可移植成员映射。

修复范围记录（归档成员 `remediation-scope.json`） 中 25 项报告过的复杂度问题已由主代理记录为本地修复，完整 CI 在此归档冻结时仍为 pending。本性能报告不把本地检查当作远程 Codacy/全 CI 已通过，不授予合并结论。
