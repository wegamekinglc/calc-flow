# PR #482 payload 游标修复

Windows Rust CI 在 `c0cd5875` 的 `test_output_payload_access_keeps_order_without_reading_metadata_or_the_unused_side` 失败。输出 gather 通过 `AdmittedCursor::row` 获取 payload，默认实现转而调用完整 `row()`，违反了只读取 payload 的约定。本地复现得到同一调用栈和 panic。

修复源码为 `90c06e09412ef225dc5941e6b18bf1ab1e668409`。新增 `cursor_payload` 的通用实现直接委托 `payload`；`AdmittedBatch` 的实现与元数据游标复用原有段定位算法，随后只解析 payload。输出 gather 使用该游标，固定宽度、变长和嵌套分块预检的三个读取点也改用 payload。原有排序、预算、缓存、分配与 await 边界保持不变。保留状态仍通过元数据游标读取所需行信息；字典准备所需的坐标和保留标记访问接口保持不变。

验证始终绑定各自源码，不合并为最终版本的唯一测试总数：

| Verification          | Source                         | Result                  |
|-----------------------|--------------------------------|-------------------------|
| Original gather RED   | Compatible pre-fix core binary | 1 failed, exit 101      |
| New preflight RED     | c0cd5875 plus test-only patch  | 1 failed, exit 101      |
| Focused Rust GREEN    | 90c06e09                       | 18 passed, 0 failed     |
| Scoped core Clippy    | 90c06e09                       | Passed with -D warnings |
| Format / whitespace   | 90c06e09                       | Passed                  |
| Complexity comparison | Five changed Rust files        | No new/increased >8 CC  |

18 项定向测试覆盖原 CI 失败、新增的双输入方向固定/变长/嵌套/重复/零列投影及范围预算、稀疏/投影/复制偏移布局的游标一致性、原有分配和缓存计数、固定宽度的常量访问次数、稀疏状态下的工作量，以及选中切片行的段查找上限。两个 RED 均在生产代码修改前执行；原失败测试的断言保持原样。GREEN 编译约 130 秒、运行 0.78 秒；Clippy 约 138 秒。没有运行本地全仓库或跨平台矩阵，最终平台结果由 GitHub CI 验证。

性能保护采用直接受影响路径的工作量测试和源码对比：原段定位循环及向后查找逻辑原样提取，gather 不增加段查找、遍历或分配，并移除输出路径不需要的元数据读取。既有测试仍满足至多 12 次 admission segment lookups，以及分块预检分配和缓存计数约束。此处没有采集新的 wall-clock 数据，也不宣称获得额外加速或证明时间等价。

[此前的性能复核](pr482-remediation.md) 仍只对应 `61aab0e5` → `25e8b4c3`：四项结论不确定，Interval 单 CPU 未确认回退。这些时间不能视为 `90c06e09` 的重新测量。

[证据包](pr482-payload-cursor-evidence.tar.gz) 包含原 Windows CI 日志、本地两次 RED 和 GREEN、Clippy、原测试补丁、源码/文件哈希、复杂度比较和精确命令/feature/环境记录。包内 `manifest.json` 列出每个成员的 SHA-256；`verification.json` 保留命令和源码边界。证据包在推送前冻结，最终 CI 状态与合并结果见 [PR #482](https://github.com/wegamekinglc/calc-flow/pull/482)。

[独立审查](../reviews/pr482-payload-cursor.md) 评估源码及定向验证。合并以最新提交的必需 CI、覆盖率和 Codacy 全部通过为条件。

归档 SHA-256：`be713e4028995b66b9145ee1862dada09ee16d2da7a1c0167782226ffc38c94d`。
