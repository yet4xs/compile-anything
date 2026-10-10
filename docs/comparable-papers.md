# 可对比论文分析 — Compile Anything 竞品定位

> 更新：2026-10-10
> 目的：找到论文中可以直接对比数字的竞品，以及在 Related Work 中如何定位。

---

## 一、最直接可比的论文（按相似度排序）

### 1. WorfBench（ICLR 2025）— 最相似的 benchmark

**论文**：[Benchmarking Agentic Workflow Generation](https://arxiv.org/abs/2410.07869)（Qiao et al.）

**做什么**：评测 LLM 从自然语言生成可执行的 DAG 工作流的能力（linear/branching/DAG 结构）

**与我们重叠**：
- 评测 LLM 生成结构化计算图 ← → 我们评测 TaskIR 生成
- 他们的 "graph F1" ← → 我们的 "OpSeq exact match"
- 他们的 "success rate" ← → 我们的 "Forge execution rate"

**他们报的数字**（GPT-4）：
| 指标 | GPT-4 | 说明 |
|---|---:|---|
| F1 (chain) | 67.32% | 简单线性工作流 |
| F1 (graph) | **52.47%** | DAG 结构工作流 |
| 成功率差距 | ~15% | 简单 vs 复杂结构 |

**我们的对应数字**：
| 指标 | 我们 (3B Forge) | 对比 |
|---|---:|---|
| OpSeq exact match | **95.88%** | vs GPT-4 graph F1 52.47% |
| Forge execution rate | **97.0%** | 他们不报告此指标 |

**关键差异**：我们训练了专用编译器（3B），他们用 prompted GPT-4；我们的 IR 有类型系统和验证器。

**可比性**：★★★★★（最直接的数字对比）

---

### 2. LLMCompiler（ICML 2024）— 最相似的架构

**论文**：[An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511)（Kim et al., ~293 引用）

**做什么**：LLM 规划器生成函数调用 DAG，Task Fetching Unit 并行执行

**与我们重叠**：
- DAG 结构的函数调用计划 ← → 我们的 TaskIR DAG
- 并行执行 ← → 我们的 Forge/LangGraph 执行
- 编译器类比 ← → 我们的核心叙事

**关键架构差异**：
| 维度 | LLMCompiler | Compile Anything |
|---|---|---|
| IR | 无（JSON DAG，无类型） | TaskIR v0.1（SSA + 类型 + guard/VERIFY/retry） |
| 验证器 | 无 | V1-V6（10k fuzz 零失败） |
| Skill ISA | 无 | 31 技能 + 5 维成本 |
| 前端 | Prompted GPT | 训练的神经编译器 |
| 验证 | 无 | 编译期静态检查 |
| 成本模型 | Token 计数 | 5 维指令级 |
| 调度 | 依赖驱动 | 关键路径优先 + 资源约束 |

**他们报的数字**：
| 指标 | 值 |
|---|---|
| 延迟加速 | up to 3.7× vs ReAct |
| 成本节省 | up to 6.7× |
| 准确率提升 | ~9% |

**可比性**：★★★★☆（架构对比，非直接数字对比）

---

### 3. WorkflowLLM（arXiv 2407.07859）— 最相似的训练方法

**论文**：[WorkflowLLM: Empowering Automatic Workflow Generation](https://arxiv.org/abs/2407.07859)

**做什么**：SFT 训练 8B 模型生成 workflow JSON（105K 指令-workflow 对）

**与我们重叠**：
- Fine-tune 小模型生成结构化输出 ← → 我们的神经编译器
- 大规模指令-workflow 数据集 ← → 我们的 31K TaskIR 语料

**关键差异**：
- 他们生成 JSON 格式 workflow，无类型/无验证/无执行引擎
- 我们生成类型化 SSA IR + 验证 + DAG 执行

**可比性**：★★★☆☆（训练方法对比）

---

### 4. xLAM（Salesforce）— 相同训练数据

**论文**：[xLAM: A Family of Large Action Models](https://arxiv.org/abs/2409.03215)（~169 引用）

**与我们重叠**：**完全相同的训练数据**（xlam-function-calling-60k）

**他们报的数字**：
| 基准 | xLAM-7B | 我们 7B MAX |
|---|---:|---:|
| BFCL | ~85% | **89.2%** |
| ToolBench Pass@1 | ~53-59% | 未测 |

**关键差异**：xLAM 输出函数调用 JSON（无 IR），我们输出 TaskIR + Forge 执行

**可比性**：★★★★★（同数据 BFCL 直接对比）

---

### 5. Microsoft Workflow DAG（arXiv 2608.30250）

**论文**：[Generating Workflow DAGs from NL with Non-Reasoning LLMs](https://arxiv.org/abs/2608.30250)

**他们报的数字**：
- ~89% LLM-judge validity
- ~90% exact-match condition accuracy

**关键差异**：单域（contact-center），vendor JSON 格式，无类型/无验证/无执行引擎

**可比性**：★★★☆☆（validity 概念相似）

---

## 二、论文中的定位策略

### Related Work 定位句

> Prior systems use the compiler metaphor; we supply the compiler's actual obligations —
> a specification, a verifier, an ordering model for side effects, a cost-annotated
> instruction set, and a scheduler/runtime that consume them.

### 对比表（论文 Table 1）

| 系统 | IR 规范 | 验证器 | 类型系统 | Skill ISA | 成本模型 | 训练前端 | 执行引擎 | 效果系统 |
|---|---|---|---|---|---|---|---|---|
| LLMCompiler | ✗ | ✗ | ✗ | ✗ | Token 计数 | Prompt | DAG 并行 | ✗ |
| WorkflowLLM | JSON | ✗ | ✗ | ✗ | ✗ | SFT | ✗ | ✗ |
| xLAM | ✗ | ✗ | ✗ | ✗ | ✗ | SFT | ✗ | ✗ |
| MS Workflow | JSON | ✗ | ✗ | ✗ | ✗ | SFT | ✗ | ✗ |
| **Ours** | **TaskIR SSA** | **V1-V6** | **✓** | **31 技能** | **5 维** | **三段式** | **Forge** | **提案** |

### 主结果对比表（论文 Table 2）

| 系统 | 前端 | BFCL% | 执行成功率 | 跨域 EA |
|---|---|---:|---:|---:|
| GPT-4 + ReAct | Prompted | — | — | — |
| GPT-4 + LLMCompiler | Prompted | — | — | — |
| GPT-4 (WorfBench) | Prompted | — | — | — |
| xLAM-7B | SFT | ~85% | — | — |
| **Ours (E5C-S)** | **Modular** | 89.2% | **97.0%** | — |
| **Ours (modular τ³)** | **Modular** | — | — | **60.2%** |

---

## 三、建议的对比实验（如果要跑）

| 实验 | 对比对象 | 需要什么 | 时间 |
|---|---|---|---|
| 跑 WorfBench | GPT-4/ToolLLaMA graph F1 | 下载 WorfBench 数据 | ~1 天 |
| 跑 ToolBench ToolEval | xLAM/ToolLLaMA pass rate | 接入 ToolEval | ~1 天 |
| 跑 StableToolBench | StableToolBench leaderboard | 虚拟 API 服务器 | ~2 天 |
| 延迟/成本对比 | LLMCompiler 3.7× | 我们的 scheduler | ~半天 |

**优先级**：WorfBench > ToolBench > 延迟对比 > StableToolBench

---

## 四、总结

**最直接可比的数字**：

| 我们 | 他们 | 差距 |
|---|---|---|
| BFCL 89.2% | xLAM ~85% | **+4pp** |
| 执行成功率 97.0% | 无直接对应 | 独有指标 |
| OpSeq 95.88% | WorfBench graph F1 ~52% (GPT-4) | **+43pp**（但不同 benchmark） |
| 跨域 EA 60.2% | 无直接对应 | 独有指标 |

**核心论点**：不是"我们比 xLAM 高 4pp"，而是：
> 我们是唯一有 (1) 类型化 SSA IR (2) 六类静态验证器 (3) 五维成本模型 (4) 训练的神经前端 (5) DAG 执行引擎 的完整编译器系统。
