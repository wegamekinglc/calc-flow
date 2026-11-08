# Native array runtime 设计审查记录

本记录汇总 A3b 的设计结论；实现源码审查和 CI 是独立合并条件。

A3b bounded IR、共享预算和原生 batch factory 设计通过。factory 为其所有
operator 共享两个私有 worker slots。取消及 waiter 丢失时，独立 supervisor
持有 blocking worker，permit 和输入输出所有者保留到实际结束；长循环及
包含 null 的历史扫描必须观察取消。默认 Tokio Runtime Drop 负责排空 blocking
pool；调用者显式选择 shutdown_timeout/background 时，不保证返回前 worker
结束，但必须已请求合作取消且 closure 持有资源。不得为此引入额外 cleanup
线程或通用 runtime 框架。

capability schema 4 明确加入 options 定义端口及不可变 JSON schema 文本。
`native_provider` 保存真实注册身份，不借用 Python callback 或 stateless
stream 证明；描述查询不构成注册。失败注册不推进记录、revision 或缓存。

## 已解析字面量修订

专项审查通过 [A3 字面量边界修订](../api-notes/native-array-runtime.md#a3-已解析数值字面量边界)。
这是对原始 token 保证的明确更正：当前 JsonMap 和锁定的 serde_json 解析器
已丢失部分整数拼写，无法在保留完整有限 f64 范围的同时恢复它。必须公开
保留整数表示与已舍入 f64 的非单调原始拼写边界，并补充真实解析到执行 bits
的回归。不得声称 JSON Schema 单独表达该表示相关规则。

本修订不需更改数值算法、启用全局任意精度解析或增加依赖；Python 声明前端
仍在编码前限制 Python 整数。保留原生数组数据、f32 有限转换及 pow 界限。
