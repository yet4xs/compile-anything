# BFCL V4 外部评测结果 — E1 Neural Compiler (3B QLoRA)

> 评测时间：2026-10-01（~10 小时推理）
> 模型：Qwen 3B + QLoRA（E1，28,093 条训练，98.94% 语义审计通过）
> 基准：BFCL V4 全量 4,696 条（训练时**从未见过**此基准）

## 最终结果

| 指标 | BFCL V4（外部） | 内部测试集 | Δ |
|---|---|---|---|
| **Parse Rate** | **97.51%** | 99.55% | -2.0pp |
| **Validator Pass** | **92.29%** (4,334/4,696) | 99.10% | -6.8pp |

## 按 Oracle Representability 分类

| Representability | n | Validator Pass | Valid% | 说明 |
|---|---|---|---|---|
| **Full**（完全可表达） | 1,931 | 1,828 | **94.67%** | TaskIR oracle 判定完全可表达的样本 |
| **Partial**（部分可表达） | 2,610 | 2,353 | **90.15%** | multi-turn 状态、未知工具等 |
| **None**（不可表达） | 155 | 153 | **98.71%** | memory 类（ISA 缺口） |
| **TOTAL** | **4,696** | **4,334** | **92.29%** | |

## 关键发现

1. **跨基准泛化确认**：训练数据（xLAM/Spider/VerilogEval）与 BFCL V4 零重叠，
   模型仍产出 92.3% 合法 TaskIR——**编译能力可迁移，不是数据集特化**

2. **"None" 类别 validator pass 最高（98.71%）**：这些是 memory 类任务，
   oracle 判定 TaskIR 无法完整表达其语义，但模型输出的 TaskIR 结构上合法
   （用 GENERATE-only 等替代方案）。**validator 检查的是结构合法性，不是语义完备性**

3. **与内部测试集的差距合理**：-6.8pp 的 validator pass 降幅主要来自
   multi-turn 和 web_search 类别（训练数据中缺少此类模式）

4. **Parse rate 仅降 2pp**：模型学到的 TaskIR 语法是通用的，
   不依赖特定数据分布

## 论文表格（Table 2: External Benchmark）

| Benchmark | Samples | Parse% | Valid% | Notes |
|---|---|---|---|---|
| BFCL V4 (all) | 4,696 | 97.51 | 92.29 | Zero-shot transfer |
| BFCL V4 (full rep.) | 1,931 | — | 94.67 | Oracle-representable subset |
| BFCL V4 (partial rep.) | 2,610 | — | 90.15 | Multi-turn/unknown tools |
| Internal test (E1) | 1,561 | 99.55 | 99.10 | Same-distribution reference |
