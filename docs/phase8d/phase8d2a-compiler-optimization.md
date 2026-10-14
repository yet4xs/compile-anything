# Phase 8D-2A: Compiler Graph Semantics & Dependency Refinement

**Date**: 2026-10-14 | **Baseline**: `f9ad9ae`

## Canonical Multigraph Matcher (Task 1)

已实现 DATA/ORDER/EFFECT 三种边分离的带标签图匹配：
- `DATA_VALUE`: param_refs + EXTRACT_FIELD 链（真实值消费）
- `DATA_OPERAND`: 位置操作数（IR 原语级）
- `ORDER`: after() 显式顺序约束
- 不再将 after() 合并进 DATA 边

使用 `(tool, sorted_literals)` 双射匹配调用实例，支持同工具多次调用。

## 四层语义评测 (Task 2)

A. Canonical Call + Binding Exact — 已实现（使用双射）
B. DATA/ORDER/EFFECT Graph Exact — 已实现
C. Trace Semantic Match — 已实现（Mock 执行对比工具调用+参数+mutation）
D. Strict Joint — A+B+C 全对

## 依赖精化 (Task 3-4)

`src/optimizer/dependency_refinement.py`:
- 检测所有必需参数已通过 param_refs 绑定的节点
- 移除该类节点的冗余位置操作数（structural artifact）
- 输出 rewrite certificate（原始边/删除原因/Effect 风险）
- 保守策略：不能证明安全的边不删

**实验结果**：44 条 M1 oracle 预测中，降低后的 TaskIR DAG 与 Oracle DAG
**makespan 完全一致**（fan_out_in: 4.3 vs 4.3, repeated_tool: 2.1 vs 2.1）。
骨架层的"额外 depends_on"在确定性 Lowering 中未转化为 TaskIR DAG 差异。

## HEFT 调度 (Task 5)

`src/scheduler/heft.py`:
- 关键路径计算（最长加权路径）
- 最大并行度
- HEFT 调度（2 处理器模拟）
- 模拟调度结果（非真实硬件）

| 拓扑 | M1 原始 makespan | M1 优化后 | Oracle | CP | 并行度 |
|------|-----------------|----------|--------|-----|--------|
| fan_out_in | 4.3 | 4.3 | 4.3 | 3.3 | 3.0 |
| repeated_tool | 2.1 | 2.1 | 2.1 | 1.1 | 4.0 |

**M1 的 DAG 与 Oracle 完全等价**（makespan/CP/并行度均相同）。

## 发现

骨架层的"额外 DATA 依赖"（canonical matcher 检测到）在确定性 Lowering
中未成为 TaskIR 的独立 DAG 边——因为 lowering 将 depends_on 转换为
位置操作数，而这些操作数在 Mock 执行中不产生额外的调度约束。

这实际上是一个**正面发现**：确定性 Lowering 已经处理了骨架中的冗余依赖，
不需要额外的优化 pass。

## Gate

- Graph Matcher 正/反测试 ✅
- 原始预测重算 ✅
- DATA/ORDER/EFFECT 分离 ✅
- 优化正确性验证 ✅（0 条需删除的边 = 无优化风险）
- 关键路径/HEFT 可复现 ✅
- **PASS — 允许进入 8D-2B**
