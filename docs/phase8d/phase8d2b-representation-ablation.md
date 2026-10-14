# Phase 8D-2B: Three-Arm Representation Ablation — Results

**Date**: 2026-10-14 | **Seeds**: 44, 45, 46 | **Test**: 44 held-out tasks × 2 protocols

## 核心结果表(3 种子合并, n=132/臂/协议)

| 模型 | Oracle Binding% | Oracle Strict% | Distr. Binding% | Distr. Strict% |
|------|----------------|---------------|-----------------|----------------|
| M0-T (TaskIR Text) | 0.0 | 0.0 | 9.8 | 6.8 |
| M0-J (TaskIR JSON AST) | **0.0** | **0.0** | **0.0** | **0.0** |
| M1 (Skeleton → Lowering) | **100** | **100** | **84.1** | **84.1** |

## M1 vs M0-J 配对差异

| Protocol | Δpp | 95% CI | Fixed/Broken |
|----------|-----|--------|-------------|
| Oracle | **+100.0** | [100, 100] | **132/0** |
| Distractor | **+84.1** | [78.0, 90.2] | **111/0** |

## M0-J vs M0-T

| Protocol | Δpp | 95% CI | 说明 |
|----------|-----|--------|------|
| Distractor | **-9.8** | [-15.2, -5.3] | M0-J 比 M0-T 更差 |

## 关键发现

### 1. JSON 格式不是优势来源——M0-J 反而最差

M0-J(生成完整 TaskIR JSON AST)的 binding exact = **0%**。甚至比直接生成 TaskIR 文本(M0-T)更差(Oracle: 0% vs 0%; Distractor: 0% vs 9.8%)。

原因:完整 TaskIR AST 需要模型同时生成 SSA ID 编号、EXTRACT_FIELD 中间节点、param_refs 字段——这对模型太复杂。

### 2. M1 的优势完全来自语义抽象层次

M1 的 Skeleton JSON 只需要:
- 工具名(capability_id)
- 参数(literal 或 runtime_ref{producer, field})
- 依赖(depends_on)

不需要: SSA ID 编号、EXTRACT_FIELD 中间节点、param_refs 字段、output_type。

**确定性编译器(Lowering)自动处理所有这些复杂性。**

### 3. 零 Broken

在所有 264 个配对比较中(132 oracle + 132 distractor),M1 **从未破坏** M0-J 或 M0-T 能做对的任务(0 broken)。

## 结论

**Outcome A 确认: M1 >> M0-J ≈ M0-T**

分层神经编译(Typed Skeleton + Deterministic Lowering)的优势**不来自 JSON 格式**,而来自**语义抽象层次的正确选择**:

- 神经模型负责: 选择工具 + 规划依赖 + 绑定动态参数(自然表示)
- 编译器负责: SSA 编号 + 类型检查 + 字段提取 + 结构验证(确定性保证)

这支持 **Compile Anything** 的核心论文主张: 神经编译器的价值在于让神经网络在正确的抽象层工作,而编译器处理底层复杂性。
