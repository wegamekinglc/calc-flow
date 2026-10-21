# DAL-417 测量前约定

本文件保留原始测量阶段的范围与契约：只生成调研、脚本和数据，不修改产品，
当时不提交、推送或合并。随后追加的 DAL-417 PR 交付授权不改变下述测量契约；
仓库中的阅读与重现入口见 [README.md](README.md)。

固定 calc-flow `dab98a559314bd46d866ed7a6489abcdd3810afe`，Arroyo
`v0.15.0 / 4ef15f6336a596651f10cf27c1ee29c29bf81577`。
前者从 clean tracked source 以默认 file feature 构建 release wheel，后者用官方
Linux x86_64 release 二进制；保留 lock、构建日志、wheel/native/binary SHA-256。

比较问题是 live WebSocket NDJSON → 集成计算 → stdout JSON 可见的应用成本。
calc-flow 使用 Python websockets + PyArrow JSON → 现有 native stream → Python JSON
适配；Arroyo 使用现有 native WebSocket/JSON 和 stdout connector。
它们的适配层不同，不据此判定纯 runtime/算子性能。

同机顺序运行，各子进程 CPU affinity 4-7，Arroyo parallelism=1；两边最多使用相同
4 个逻辑 CPU，内存使用相同主机可用资源，不设单独硬上限。生产者/观察者在另一进程。
预定 32,768 条有效输入，16 key，512 行一个 NDJSON frame，固定整数值及事件时间。
追加 512 条被过滤的 progress marker；保持连接开放直至全部输出校验及 100ms 重复
观察结束。每条输出按独立 Python 整数 oracle 精确比较，包括窗口键、sum 和 count。

4 个 case 为投影/过滤及四秒 tumbling SUM/COUNT，各有 burst 和每 frame 5ms 的 paced
输入。窗口内跨两个输入 frame，每 frame 事件时间推进两秒，最后 progress marker
关闭末窗口。窗口进度来自 timestamp - 1us；calc-flow 显式发源 watermark，Arroyo
按其现有生成器处理。没有 EOF 促成结果。

每 case 固定 5 组相邻交替 AB/BA（共 40 次全新进程运行）；先用 1,024/2,048 行做
定向正确性 preflight，旧一秒窗口 preflight 数据仅诊断。构建时的 preflight 不用于
性能结论。正式数据只在构建完成后获取；全部 acquisition 预算 600 秒，不包括构建
和人工编写脚本/源码调研。未完成时保留错误和样本，不补跑至有利结果。

计时从首 frame 的 send 调用前到最后正确输出行被父进程收到。包含 producer pacing、
loopback、JSON 解码编码、Arrow 转换、通道、watermark、计算和 pipe/观察成本；不含
输入 fixture 生成、进程启动/规划、完整 oracle 比较、关闭/清理、检查点。启动到源连接
另记诊断值，不作完整就绪或生命周期 gate。Arroyo checkpoint interval=1h，calc-flow
checkpointing=False；两者耐久能力不等价，SIGINT 后的 Arroyo final checkpoint 不计时。
不声称恢复/检查点性能已测；若测 lifecycle，必须另按仓库 20 轮契约获取完整证据。

吞吐分母为完整有效输出可见时间、分子有效输入行数；包含 rate limit，不称最大持续
吞吐。每行延迟用对应 frame 的 send 起点；窗口用最后贡献 frame 起点，另记最早可
关闭窗口的 progress frame 起点到可见输出的延迟。RSS 每 10ms 采样整个子进程树，
包含解释器/控制面，可能漏掉更短峰值，不能当状态内存或长期资源曲线。

数据是自定义跨引擎 informational contract，不冒充仓库同依赖、同 workload fingerprint
的 revision regression gate，也不把全套、minima 或厂商数字用作流式实测。
使用各轮中位数、min/max、延迟分位数及成对结果描述重复范围，不作 +5% 回归分类。
