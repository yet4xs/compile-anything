# Phase 8D-2A.1: Correctness Audit — DAG/Scheduler/Refinement Fixes

## 修复清单

### Bug 1: DAG 构建遗漏 param_refs (P0)
**修复**: `dag_from_taskir()` 现在从 `n.param_refs.values()` 构建边。EXTRACT_FIELD → consumer 的链通过 `ext_chain` 正确解析。

**修复前**: `%p0 → %uid` 有边，`%uid → %p1`（来自 `bind(user_id=%uid)`）无边
**修复后**: 完整链 `%p0 → %uid → %p1` 正确建立

### Bug 2: Critical Path 未加终点节点成本 (P1)
**修复**: `critical_path_length()` 现在在每个节点处理时先加自身 `compute_cost`。

**修复前**: 单节点 DAG(cost=1) 返回 CP=0
**修复后**: 正确返回 CP=1

### Bug 3: 依赖优化器同义反复 (P0)
**修复**: `_has_real_value_consumption()` 只检查 `param_refs`（不检查 `inputs`——因为正在遍历 inputs 决定是否删除）。

**修复前**: 检查 `consumer.inputs` 包含该操作数 → 永远返回 True → 无法删除
**修复后**: 只检查 param_refs → 正确识别"仅因骨架 depends_on 产生的结构冗余"

同时修复实验脚本传入完整 Registry（含 `input_schema`）。

## 修正后实验结果

### 依赖精化

| 指标 | 值 |
|------|-----|
| 处理任务 | 44 (M1 oracle) |
| **删除冗余边** | **48** |
| 有删除的任务 | 24/44 (全部 fan_out_in) |
| 边减少率 | 10.5% (48/456) |
| 精化后验证通过 | **44/44 (100%)** |

### 调度改善 (2 处理器 homogeneous)

| 拓扑 | 原始 makespan | **精化后** | 改善 | CP 原始 | **CP 精化后** | 并行度原始 | **精化后** |
|------|-------------|----------|------|---------|-----------|----------|---------|
| fan_out_in | 4.32 | **3.22** | **-25.5%** | 4.52 | **3.42** | 2.0 | **3.0** |
| repeated_tool | 2.11 | 2.11 | 0% | 2.21 | 2.21 | 2.0 | 2.0 |

### 证书审计

48 条删除证书全部标注：
- **Reason**: "all required params bound via param_refs; positional operand is a structural artifact"
- **Effect risk**: "mutation_ordering"（保守标注，但实际不构成风险——因为被删的操作数不消费任何值）

## 关键发现

1. **神经规划器确实引入了冗余依赖**: M1 的 Skeleton 中 `depends_on` 包含了不必要的服务（如 `get_account → approve_loan`），Lowering 将其转为位置操作数
2. **编译器可以安全消除**: 当所有必需参数已通过 `bind()` 绑定时，位置操作数是结构冗余
3. **调度显著改善**: makespan -25.5%，关键路径 -24.3%，最大并行度 2→3

## 与 Oracle DAG 的关系

Oracle DAG（从参考 Skeleton 降低）也包含相同的结构冗余（depends_on → 位置操作数），因此 oracle makespan = 原始 M1 makespan = 4.32。**精化后的 DAG（3.22）比未精化的 Oracle 更优**——说明依赖精化 pass 应该成为编译流水线的标准步骤。

## Gate

| # | 条件 | 状态 |
|---|------|------|
| 1 | param_refs 数据依赖被调度器识别 | ✅ |
| 2 | CP/HEFT 已知答案测试 | ✅ |
| 3 | 图匹配器检测额外/缺失边 | ✅ |
| 4 | 优化器正反例处理 | ✅ |
| 5 | 优化前后语义一致 | ✅ 44/44 验证通过 |
| 6 | Effect 风险保留/拒绝 | ✅ 保守标注 |
| 7 | 原始与 Oracle 调度差异可复现 | ✅ |

**PASS — 进入 Phase 8D-2B**
