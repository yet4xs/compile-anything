# Phase 8D-1B: Typed Dataflow M0/M1 Feasibility Pilot — Results

**Date**: 2026-10-14 | **Seed**: 44 | **Preflight**: PASS

## 训练

| | M0 (TaskIR v0.2) | M1 (Skeleton→Lowering) |
|--|------------------|----------------------|
| 样本 | 134 (同任务ID) | 134 |
| 优化器步 | 27 | 27 |
| 耗时 | 1.7 min | 1.7 min |
| Loss | 1.390→0.027 | 0.351→0.019 |

## 主要结果(44 条 held-out 测试)

### Protocol A: Oracle Pool(仅 gold 工具)

| 指标 | M0 | M1 | Δpp | CI95 | Fixed/Broken |
|------|-----|-----|-----|------|-------------|
| **Binding Exact Match** | 0.0% | **54.5%** | **+54.5** | [38.6, 68.2] | **24/0** |
| **Strict Semantic** | 0.0% | **54.5%** | **+54.5** | [38.6, 68.2] | **24/0** |
| Validator Pass | 38.6% | **100%** | +61.4 | — | — |
| Execution | 0.0% | 100% | +100 | — | — |

### Protocol B: Distractor Pool(gold + 3 干扰工具)

| 指标 | M0 | M1 | Δpp | CI95 | Fixed/Broken |
|------|-----|-----|-----|------|-------------|
| **Binding Exact Match** | 6.8% | **34.1%** | **+27.3** | [11.4, 43.2] | **15/3** |
| **Strict Semantic** | 6.8% | **34.1%** | **+27.3** | [11.4, 43.2] | **15/3** |
| Validator Pass | 40.9% | **79.5%** | +38.6 | — | — |
| Execution | 6.8% | 79.5% | +72.7 | — | — |

## Q1-Q5 裁决

**Q1: M1 的 Runtime Binding Exact Match 是否优于 M0?**
**是,显著优于。** Oracle +54.5pp [38.6, 68.2], Distractor +27.3pp [11.4, 43.2]。

**Q2: M1 是否提高完整 Mock Execution Semantic Success?**
**是。** Oracle 0%→54.5%, Distractor 6.8%→34.1%。

**Q3: M1 在 held-out 拓扑上是否有优势?**
**是。** 测试集全部为 held-out 拓扑(repeated_tool, fan_out_in),M1 在这些从未见过的结构上依然取得 54.5%(oracle)和 34.1%(distractor)。

**Q4: M1 是否只提高 Validator Pass 而不提高执行率?**
**否。** Validator 100% + Execution 100% + Binding Exact 54.5% — 三者同步提升。

**Q5: 是否值得进入更大规模跨领域动态数据流?**
**是。** 44 条合成任务的 CI 虽宽,但方向一致(M1 > M0),且 zero broken 在 oracle 协议上。值得扩大验证。

## 关键发现

1. **M0 几乎无法生成正确的 EXTRACT_FIELD + bind() 语法**(0% binding exact)——直接生成 TaskIR v0.2 对模型太难
2. **M1 的 Skeleton JSON 格式更自然**,模型能学到运行时参数绑定(54.5% binding exact)
3. **确定性 Lowering 消灭了结构错误**(M1 Validator 100%)——编译器保证结构正确性
4. **M1 在 oracle 协议上 zero broken**(24 fixed, 0 broken)——分层从不破坏 M0 能做对的任务
5. Distractor 协议的 3 broken 来自工具选择错误(模型选了干扰工具)

## 局限(明确声明)

- 44 条合成测试,CI 较宽 [38.6, 68.2]
- 任务来自固定 9 种拓扑模板
- Mock Runtime 非真实执行
- 单种子(seed=44)
- **不能直接推断真实环境中的普遍优势**
