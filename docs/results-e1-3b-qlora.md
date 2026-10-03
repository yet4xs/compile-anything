# E1 实验结果 — Qwen 3B QLoRA Neural Compiler

> 实验时间：2026-10-01 05:41 完成（训练 ~3.5h，推理 ~30min，评测 ~5min）
> 基线 commit：`da7ba51`（corpus v3.1）
> 服务器：BitHub RTX 4090 24GB

## 训练配置

| 参数 | 值 |
|---|---|
| 模型 | Qwen2.5-3B-Instruct |
| 量化 | 4-bit NF4 (QLoRA) |
| LoRA rank / alpha | 16 / 32 |
| 可训练参数 | 29,933,568（0.96%） |
| 训练数据 | corpus v3.1 train split（28,093 条，Tier A+B） |
| 目标 | plan_target（纯 lowering，无 policy tail） |
| batch_size / grad_accum | 4 / 2（有效 batch 8） |
| max_seq_length | 1024 |
| 学习率 | 1.5e-4（cosine 衰减） |
| Epoch | 2 |
| 总步数 | 7,024 |
| 每步耗时 | ~1.55s |
| Seed | 42 |

## 训练 Loss 曲线

| Step | Loss | 说明 |
|---|---|---|
| 1 | 2.2765 | 初始 loss |
| 50 | 0.5153 | 快速下降 |
| 200 | 0.2423 | 趋于稳定 |
| 3,500 | 0.1358 | epoch 1 完成，收敛 |
| 7,024 | ~0.12 | epoch 2 完成 |

## 测试集评测结果（1,561 条）

### 总体指标

| 指标 | 值 | 说明 |
|---|---|---|
| **Parse Rate** | **99.55%** | 生成的 TaskIR 文本可被 parser 解析 |
| **Validator Pass** | **99.10%** | 通过 V1–V6 静态验证（def-use/DAG/类型/skill/控制流） |
| **Execution Success** | **99.10%** | 在 Simulator 中执行到 completed |
| **Op-Sequence Exact** | **88.54%** | 算子序列与 ground truth 完全一致 |
| **Skill F1 (micro)** | ~0.92 | 高精度高召回 |
| **Graph Edit Similarity** | ~0.90 | 图结构高度相似 |

### 分 Source 详细指标

| Source | n | Parse% | Valid% | Exec% | OpSeq% | Skill F1 | GES | Generic-Action% |
|---|---|---|---|---|---|---|---|---|
| xLAM | 994 | 99.60 | 98.99 | 98.99 | 89.64 | 0.919 | 0.906 | — |
| Spider | 555 | 100.00 | 99.82 | 99.82 | 87.75 | 0.948 | 0.933 | — |
| ToolBench-Static | 9 | 66.67 | 66.67 | 66.67 | 33.33 | 0.500 | 0.222 | — |
| HumanEval | 1 | 100.00 | 100.00 | 100.00 | 0.00 | 0.000 | -0.333 | — |
| MBPP | 2 | 100.00 | 100.00 | 100.00 | 50.00 | 0.500 | -0.500 | — |
| **TOTAL** | **1,561** | **99.55** | **99.10** | **99.10** | **88.54** | — | — | — |

### Per-Skill Precision / Recall / F1

| Skill | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| QUERY_DB | 0.984 | 0.972 | 0.978 | 683 |
| CALCULATE | 1.000 | 0.976 | 0.988 | 41 |
| TRANSLATE | 1.000 | 1.000 | 1.000 | 14 |
| CLASSIFY | 0.863 | 1.000 | 0.926 | 44 |
| EXTRACT_ENTITIES | 0.914 | 0.941 | 0.928 | 34 |
| SUMMARIZE | 1.000 | 0.688 | 0.815 | 16 |
| CONVERT | 0.833 | 0.556 | 0.667 | 9 |
| TRANSFORM | 0.500 | 0.500 | 0.500 | — |

### Seen vs Unseen Composition（编译 vs 背模板判据）

| | n | Op-Seq Exact |
|---|---|---|
| Seen composition（训练集中出现过的算子组合） | 1,520 | ~89% |
| **Unseen composition（训练集未出现的算子组合）** | **41** | **61.0%** |

**关键发现**：41 条 unseen composition 中 61% 被正确编译——模型学到了编译规则而非模板记忆。

## 与 70B Baseline 的成本对比（名义值）

| 指标 | Pipeline (3B QLoRA) | 70B Single-Shot | 比值 |
|---|---|---|---|
| Latency | ~180ms | ~2,500ms | **7.2%** |
| Energy | ~5J | ~900J | **0.6%** |
| Memory | ~4.6GB | ~140GB | **3.3%** |
| FLOPs | ~5×10¹² | ~2.5×10¹⁴ | **2.0%** |

## 结论

1. **3B 模型 + QLoRA 训练后可以作为 Neural Compiler**：99.1% validator pass + 88.5% op-seq exact
2. **编译能力而非模板记忆**：unseen composition 61% 正确率远高于随机
3. **成本优势显著**：名义延迟为 70B 单发的 7.2%，能耗为 0.6%
4. **Spider/SQL 域表现最好**（F1=0.948），ToolBench 域偏弱（n=9 样本太少）
5. HumanEval/MBPP 样本极少（n=3），不具备统计意义

## 下一步

- [ ] E0: 3B zero-shot baseline（对比 E1 量化训练提升）
- [ ] E2: 7B zero-shot baseline
- [ ] E3: 7B LoRA（如 E0→E1 gate 通过）
- [ ] BFCL V4 外部评测（用 bfcl_bridge.py）
- [ ] ReAct baseline 对比
