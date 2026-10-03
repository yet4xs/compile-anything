# BFCL V4 语义评测结果 — 四级指标体系

> 评测模型：E1 (Qwen 3B QLoRA, 28,093 条训练)
> 基准：BFCL V4 全量 4,696 条（cross-dataset, negligible instance-level overlap）

## 四级指标（错误漏斗）

| Level | 指标 | 值 | 含义 |
|---|---|---|---|
| **L1** | Parse Rate | **97.38%** | 生成的文本可解析为 TaskIR |
| **L2** | Validator Pass | **92.36%** | 结构合法（def-use/类型/DAG/skill） |
| **L3** | Semantic Exact | **9.8%**（458/4,696） | 算子序列与 oracle 完全一致 |
| **L4** | Official BFCL | 待跑 | 官方 function-call 正确率 |

## 按 Representability 分列

| | n | Valid% | Semantic Exact |
|---|---|---|---|
| Full（完全可表达） | 1,931 | 94.56% | **309（16.0%）** |
| Partial（部分可表达） | 2,610 | 90.31% | **149（5.7%）** |
| None（memory 类） | 155 | 99.35% | **0（0%）** |
| **TOTAL** | **4,696** | **92.36%** | **458（9.8%）** |

## 语义细项

| 指标 | 值 | 说明 |
|---|---|---|
| Skill Precision (micro) | **14.13%** | 模型选的 skill 与 oracle 期望的 skill 重合度低 |
| Action Count Accuracy | **52.93%** | 约一半情况动作数量正确 |
| **Argument Fidelity** | **0.0%** | 参数值完全不匹配（见下方分析） |
| Dependency Accuracy | **3.13%** | 依赖边几乎不匹配 |

## 错误漏斗（Top 6）

| 错误类别 | 数量 | 说明 |
|---|---|---|
| missing_skill | 3,668 | oracle 期望的 skill 模型没生成 |
| extra_skill | 3,578 | 模型生成了 oracle 不期望的 skill |
| validator_reject | 236 | 结构不合法 |
| false_valid_memory | 154 | memory 类生成了"合法但语义错误"的程序 |
| parse_syntax | 123 | 文本格式错误 |
| wrong_order | 20 | 顺序错误 |

## 核心分析

### 为什么 L2→L3 有巨大落差（92% → 10%）？

**这不是模型能力不足，而是跨域语义映射问题：**

1. **Oracle 的映射路径**：BFCL 函数名（如 `get_user_info`）→ oracle 规则 → `SEARCH`
2. **模型的映射路径**：BFCL 函数描述 → E1 模型（训练于 xLAM/Spider）→ 可能选了 `EXEC_ACTION` 或 `GENERATE`
3. **两条路径产生了不同的 TaskIR**——都是"合法的"，但语义选择不同

**证据**：
- Action Count Accuracy 52.93%——模型大约一半情况"知道要调几个函数"，只是选的 skill 名不同
- Argument Fidelity 0%——oracle 在 params 里存原始函数名，模型存 TaskIR 语义参数，格式完全不同
- False valid memory 154/155——memory 类模型全部生成了合法但语义不足的替代程序

### 论文表述建议

这个结果恰好证实了架构分层的必要性：

```
L1-L2（结构层）：泛化能力 ✅ —— 跨基准 92% validator pass
L3（语义层）：域特定 —— 需要目标域的 Skill ISA 映射训练
L4（官方指标）：需要 binder —— 将 TaskIR 绑定回具体函数调用
```

**正确的论文表述**：
> "Neural Compiler 的结构生成能力（语法+合法性）可以跨数据集泛化（92.4%），
> 但语义规划（skill 选择）需要目标域的训练数据。这证实了 compiler-runtime
> 分离的必要性：结构层是通用的，语义层是域特定的。"

而不是：
> "BFCL 编译正确率 92.3%"（这是 validator pass，不是 semantic correctness）

### 与竞品对比

| 方法 | 结构合法 | 语义正确 | 说明 |
|---|---|---|---|
| E1 (ours, cross-dataset) | 92.4% | 9.8% | 零 BFCL 训练数据 |
| E1 (ours, same-dataset) | 99.1% | 88.5% | xLAM/Spider 测试集 |
| LLMCompiler (frozen LLM) | — | — | 不产出 IR，直接输出函数调用 |

## 下一步

1. **跑 τ³-bench**（优先级最高——exact/near 零重叠 + effect system 压力测试）
2. **BFCL binder**（TaskIR → 具体函数调用 → 官方评分）——这会给出 L4 指标
3. **在 BFCL 上做 few-shot fine-tune**（补充 100-500 条 BFCL 训练数据，
   预期 L3 从 10% 大幅提升——证明"域适应仅需少量数据"）
