# Phase 9A-0: Real Runtime Dataflow Feasibility Audit

**Date**: 2026-10-14 (git commit date) | **CPU-only, no training, no synthetic data**

## Audit Results

| Source | Total | Multi-tool | Runtime Value Dep (L3) |
|--------|-------|-----------|----------------------|
| BFCL V4 | 3,621 | **0** | **0** |
| ToolBench (in) | 1,588 | 50 | **0** |
| ToolBench (out) | 781 | 27 | **0** |
| τ³-bench | ~2,546 | many | UNKNOWN (args not stored) |
| xLAM | ~2,800 | few | **0** |
| **Total** | **~11,300** | — | **0 confirmed** |

## Key Finding

**No existing tool-use benchmark evaluates runtime value dependencies.** BFCL tests single-tool selection + parameter extraction. ToolBench's multi-tool tasks are independent parallel calls. This is a structural gap, not a rarity.

## Q1-Q3 Answers

Q1: No confirmed runtime_ref in existing benchmarks.
Q2: TaskIR v0.2 sufficient — no IR extension needed.
Q3: Yes — building runtime dataflow benchmark is NECESSARY.


## 勘误(Phase 9B-2.5 T7,2026-10-16)

本 survey 只审计了 BFCL / ToolBench / xLAM / τ³ 四个来源,
"**No existing tool-use benchmark evaluates runtime value
dependencies**" 与"概念空白"的表述**撤回**。已知的先行工作:

| Benchmark | 嵌套调用(上游输出→下游输入) | 可执行 | 有状态副作用 |
|---|---|---|---|
| NESTFUL (arXiv 2409.03797) | ✅ `$var_2.result$` 引用语法 | ✅ | ❌ |
| NesTools (COLING 2025, arXiv 2410.11805) | ✅ | ❌(工具不可执行) | ❌ |
| ToolHop (arXiv 2501.02506) | ✅ 多跳查询 | 部分 | ❌ |
| Seal-Tools | ✅ | ❌ | ❌ |
| LLMCompiler / ReWOO | 占位变量规划 | — | — |

本项目可守住的差异(须以此为限):**有状态、带副作用的官方模拟器上,
经重放逐参数验证的来源标注 + 类型化嵌套路径 + 确定性 lowering +
终态 DB 检查**。差异是"验证深度与类型系统",不是"概念首创"。
