# E0 实验结果 — Qwen 3B Zero-Shot Baseline

> 实验时间：2026-10-01 08:58 完成
> 服务器：BitHub RTX 4090 24GB

## 配置

| 参数 | 值 |
|---|---|
| 模型 | Qwen2.5-3B-Instruct（**未训练，zero-shot**） |
| Prompt | Neural Compiler system prompt（同 E1） |
| 测试集 | corpus v3.1 test split（1,561 条） |
| max_new_tokens | 1024 |

## 结果

| Source | n | Parse% | Valid% | Exec% | OpSeq% | F1 |
|---|---|---|---|---|---|---|
| xLAM | 994 | 67.91 | 0.00 | 0.00 | 0.00 | 0.000 |
| Spider | 555 | 62.52 | 0.00 | 0.00 | 0.00 | 0.000 |
| ToolBench-Static | 9 | 66.67 | 0.00 | 0.00 | 0.00 | 0.000 |
| HumanEval | 1 | 100.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| MBPP | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| **TOTAL** | **1,561** | **65.92** | **0.00** | **0.00** | **0.00** | **0.000** |

## 分析

- **Parse Rate 65.92%**：模型能模仿 TaskIR 文本格式的大致样子（有 %id = OP(...) 的行），但 34% 连文本形式都不对
- **Validator Pass 0.00%**：**没有任何一条**生成的 TaskIR 通过 V1–V6 验证。失败原因包括：
  - 未定义的值引用（%id 不存在）
  - 类型不匹配（把 Str 传给 List[T] 参数）
  - 不存在的 Skill 指令名
  - def-use 顺序错误
- **结论**：zero-shot 模型能"看起来像"编译器输出，但语义全部错误

## E0 vs E1 对比表（论文主表 T7）

| 指标 | E0 (Zero-Shot) | E1 (QLoRA) | Δ |
|---|---|---|---|
| Parse Rate | 65.92% | 99.55% | +33.63 |
| Validator Pass | **0.00%** | **99.10%** | **+99.10** |
| Execution | 0.00% | 99.10% | +99.10 |
| Op-Seq Exact | 0.00% | 88.54% | +88.54 |
| Skill F1 | 0.000 | 0.92 | +0.92 |
| Unseen Comp. OpSeq | — | 61.0% (n=41) | — |

## Go/No-Go Gate 判定

| 条件 | 要求 | 实际 | 判定 |
|---|---|---|---|
| Parse rate 提升 | ✅ | 65.9→99.6 | **PASS** |
| Validator rate 提升 | ✅ | 0.0→99.1 | **PASS** |
| Execution 不降 | ✅ | 0.0→99.1 | **PASS** |
| Skill F1 提升 | ✅ | 0.0→0.92 | **PASS** |
| Seen comp. 提升 | ✅ | 0.0→~89 | **PASS** |
| Unseen 不崩塌 | ✅ | —→61.0 | **PASS** |

**GATE：全部通过，建议进入 7B 实验（E2/E3）**
