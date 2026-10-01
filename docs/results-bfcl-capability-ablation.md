# BFCL Capability-Context Ablation 结果

> 实验时间：2026-10-02
> 模型：E1 (Qwen 3B QLoRA) — 冻结
> 样本：500 条分层 BFCL V4（simple/parallel/multiple/irrelevance/multi_turn/memory/web_search）

## A/B 对比结果

| 指标 | Protocol A（instruction only） | Protocol B（含 capability 描述） | Δ |
|---|---|---|---|
| Parse | 97.8%（489/500） | 99.2%（496/500） | +1.4pp |
| Validator | 94.4%（472/500） | 95.2%（476/500） | +0.8pp |
| **Semantic Exact** | **8.6%（43/500）** | **8.8%（44/500）** | **+0.2pp** |

## 结论

> **Capability context is NOT the primary cause of semantic mismatch.**

加入函数描述只小幅改善格式合法性（+1.4pp parse），但语义正确率几乎不变（+0.2pp）。

## 分析

模型训练时从未学过如何利用 capability context（训练数据是 instruction-only），
所以在推理时即使提供了函数描述，模型也无法利用它们来改善语义选择。

真正的瓶颈是**训练数据与 BFCL oracle 之间的语义本体论不匹配**：
模型学到的 tool→skill 映射规则（来自 xLAM/Spider）与 BFCL oracle 的映射规则不同。

## 论文价值

这是一个有信息量的**负结果**：
> "Simply providing target capabilities at inference does not resolve
> canonical semantic grounding — the compiler must be trained to use them."

这进一步支持 compiler 类比：正如传统编译器需要正确的 target ISA 描述，
Neural Compiler 需要在训练时就见过目标域的 capability mapping。
