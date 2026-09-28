# Runtime Simulator Correctness Review — Phase 2

审查对象：`src/runtime/simulator.py`（未重构，审查 + 最小语义修正）。
验证方法：`tests/test_{control,retry}_semantics.py`（14 用例）+
`scripts/fuzz_taskir.py`（10000 随机程序，IV1–IV6 不变量，0 失败）。

## Question 1: memo 是否等价于 SSA value cache？

**基本等价，但 rollback 使它变成"带版本的存储"。**

- 键是 SSA 值名（`%id`），pull-based 求值 = 惰性求值：每个值至多计算一次、
  定义点唯一——这正是 SSA value cache 的语义。
- 例外：VERIFY 驱动的 rollback 会清除并**重定义**同名值（同一 assignment
  的重执行）。严格说这破坏了 "single static assignment" 的运行时字面义。
  更准确的架构类比是**乱序处理器的 speculative execution + squash/replay**：
  值带版本，trace 保留全部历史（`superseded` 标记），对外可见的最终态
  仍然是确定性的。
- 推论（v0.2 设计输入）：真实 runtime 应做 checkpoint/版本化存储，
  rollback = 恢复检查点 + 重放，而不是暴力清缓存。模拟器的清缓存是实现
  该语义的最简方式。

## Question 2: `_invalidate_dependents()` 覆盖是否完整？

覆盖边集 = **inputs + after + guard.cond**（与 validator 的 `_dep_edges`
一致，retry 边按设计排除——它是时间回边）。

**"SELECT dependency" 无遗漏**：SELECT 没有独立的边种类，它的分支输入就是
普通 data inputs，天然在覆盖内。fuzzer IV1（顺序不变量，SELECT 非选中分支
豁免）10000 例通过证实了这一点。

但审计发现两处**曾缺失**的语义（已修，附回归测试）：

1. **非 output 可达的被清除节点会留下 stale 状态**。回滚原本只重求值
   output 需求链；不在链上的被清除依赖者旧事件残留、memo 为空——内存态
   与 trace 都不一致。修复：回滚后**按程序序重求值全部被清除节点**，
   并给清除前的历史事件打 `superseded` 标记（trace 留史，语义以终态为准）。
2. **多轮回回滚下，重执行时被 guard 跳过的节点**同样产生 superseded 历史，
   不变量检查只看未 superseded 事件（fuzzer IV1 相应更新）。

## Question 3: critical path 是否应包含 retry 边？

**不应包含。理由：**

- critical path 的用途是"无限并行下的 wall-clock 下界"，是 **DAG 调度性质**。
  retry 边是运行期时间回边（attempt 维度），把它计入会使"下界"依赖执行
  历史（跑了才知道重试几次），不再是可静态分析的调度界。
- 重试成本已经体现在 sequential 总延迟与 trace 里；报告同时给出两者，
  读者可以区分"结构下界"与"含重试的实际耗时"。
- 若 v0.2 scheduler 需要**期望延迟**（expected latency with retry
  probability），那是一个独立的概率分析 pass（依赖 profiling 得到各 skill
  的失败率），不属于 critical path 定义。已在 scheduler 占位接口中记为
  规划项。

## 本轮发现并修复的语义缺陷（附回归测试）

| # | 缺陷 | 触发场景 | 修复 |
|---|---|---|---|
| 1 | **谓词传播缺失**：guard 的 cond 值或 SELECT 的 cond 输入本身被 skip 时，`bool(<skipped>)` 默认为 True，节点被错误执行 | guard(%c)，%c 被上游 guard 跳过 | cond 为 skipped ⇒ 本节点 skip（`predication propagation`），见 test_control_semantics Case D |
| 2 | **内联 verify 探测造成假环失败**：`%A→%B→%C(VERIFY)`、retry 挂 `%A` 时，`%A` 内联探测 `%C` 会 demand `%C→%B→%A`，而 `%B` 还在求值栈上 → 误报 runtime cycle | reviewer Case B（中间隔节点的回滚链） | rollback 重构为 run() 级外层 checkpoint/restore 阶段：先完整求值，再对 verify=False 且有预算的节点失效+重放 |
| 3 | **回滚后非 output 可达依赖者 stale**（见 Question 2） | 多轮 rollback + 跳过节点 | 程序序全量重求值 + superseded 标记 |

## 已知限制（记录，不修）

- **递归深度**：pull-based 求值是递归的，程序深度上限 ~900 节点；v0.2 改
  迭代 worklist。
- **SELECT 双分支都求值**（eager）：非选中分支经 guard 跳过无成本，但选中
  分支的延迟照收；真实 scheduler 可推测执行或惰性化。
- **peak memory = 最大单节点足迹**，无驻留重叠建模（scheduler 工作）。
- **回滚全量重算被清除节点**：保守；更聪明的 runtime 只重算 live 值。
- **mock executor 不会自发失败**（by design：失败只来自 fail_plan 注入），
  确保模拟器结果是确定性的、可断言的。

## 结论

guard + VERIFY + SELECT + retry 的组合在上述修复后，对 v0.1 范围
（无循环、谓词分支、VERIFY 回滚）语义自洽：10000 fuzz + 14 语义用例 +
52 全量测试零失败。"作为 AI execution compiler IR 的基础"目前最大的
表达力缺口是 **effect ordering**（见 validator-audit.md M1）与**循环**
（v0.2 region/back-edge）。
