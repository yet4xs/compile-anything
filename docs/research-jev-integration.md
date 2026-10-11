# Jev (TypeSafe AI) 与 Compile Anything 的结合

> 调研时间：2026-10-11
> Jev 发布：2026-09-15
> 论文：[Just Ask Jev](https://arxiv.org/abs/2609.29429)（Guo et al., 2026）

---

## 一、Jev 是什么

**非自回归的"System One 决策模型"**——不生成任何文本，直接输出类型化的决策结果。

```
传统 LLM:  文本输入 → 逐 token 自回归生成 → 解析 → 可能出错 → 校准困难
Jev:       状态 + 类型化问题 → 单次前向传播 → 类型化决策 + 校准置信度
```

| 特性 | 传统 LLM | Jev |
|---|---|---|
| 输出方式 | 逐 token 生成文本 | 单次前向传播，零输出 token |
| 输出类型 | 自由文本（需后处理） | 类型化决策（schema 约束） |
| 幻觉 | 可能 | 结构上不可能 |
| 速度 | 秒级 | 70-500ms |
| 成本 | ~$0.01/决策 | ~$0.0001/决策 |
| 置信度 | 不可靠的 logprob | RLCD 训练的校准置信度 |

**训练方法：RLCD（Reinforcement Learning for Calibrated Decisions）**
- 奖励不仅基于决策正确性，还基于校准度（置信度与实际准确率一致）
- 区别于 DPO（偏好排序）和 GRPO（组相对策略优化）

---

## 二、与 Compile Anything 的天然对应

我们的三段式前端已经隐式地分离了「决策」和「生成」：

| 组件 | 功能 | 本质 | 对应 Jev 概念 |
|---|---|---|---|
| **Canonicalizer** | schema → canonical skill (31 类) | 分类决策 | System One |
| **Resolver** | (task, capability) → relevant/irrelevant | 二分类决策 | System One |
| **Composer** | task + capabilities → TaskIR 文本 | 序列生成 | System Two |
| **Validator V1-V6** | 结构/类型/DAG 检查 | 确定性验证 | 已是 System One |

**关键发现**：我们的编译器架构天然验证了 Jev 的 System One/Two 分离！
- 决策层（Canonicalizer + Resolver）：F1 0.917 / 0.995，可以用非自回归模型
- 生成层（Composer）：OpSeq 95.9%，需要自回归的序列建模能力

---

## 三、结合方案

### 方案 A：给现有管线加校准（零训练，立即可实现）

利用已有的序列似然分数（logP(relevant) - logP(irrelevant)）：

```python
# Resolver 校准
score = logP("relevant") - logP("irrelevant")
if score > CALIBRATION_THRESHOLD:    # 高置信 relevant
    selected.add(capability)
elif score < -CALIBRATION_THRESHOLD: # 高置信 irrelevant
    pass
else:                                 # 低置信 → 降级到 VERIFY
    taskir.add_verify_node(capability)
```

这与我们的 VERIFY/RETRY 语义天然匹配：
- 低置信度的能力选择 → 自动添加 VERIFY 节点
- VERIFY 失败 → 触发 RETRY 或降级
- 编译器架构提供了 LLM agent 通常缺少的运行时安全机制

### 方案 B：训练非自回归决策头（需要训练，效果好）

用 BERT/ELECTRA 类模型替换 Qwen 3B 做 Canonicalizer + Resolver：
- 输入：task text + capability schema text → BERT encoding → 分类头
- 优势：100x 推理加速，天然校准概率输出
- 实现：~500 行代码 + 2h 训练

### 方案 C：论文定位升级（引用级结合）

在 Related Work 中引用 Jev：
> "Jev (TypeSafe AI, 2026) introduced System One decision models — non-autoregressive
> models that produce typed decisions with calibrated confidence. Our compiler
> architecture validates this separation empirically: the decision layers
> (Canonicalizer: 0.917 F1, Resolver: 0.995 F1) achieve near-perfect accuracy
> and can potentially be replaced with non-autoregressive models for 100x speedup,
> while the generation layer (Composer) requires autoregressive sequence modeling."

---

## 四、论文叙事增强

### 原来的叙事：
> "我们建了一个三段式编译器前端"

### 加入 Jev 框架后的叙事：
> "我们的编译器架构天然分离了 System One 决策（类型分类、能力选择——可以用
> 非自回归模型实现 100x 加速和校准置信度）和 System Two 生成（TaskIR 组合
> ——需要自回归序列建模），这与最近的 System One 模型（Jev）的架构哲学一致。
> 关键是，编译器的 VERIFY/RETRY 机制为低置信度决策提供了运行时安全网——
> 这是纯 LLM 系统无法做到的。"

---

## 五、与 RLCD 的关系

Jev 的 RLCD 训练方法与我们提出的 V-DPO 有相似之处：

| | V-DPO（我们提出的） | RLCD（Jev） |
|---|---|---|
| 奖励来源 | 编译器验证器（确定性） | 决策正确性 + 校准度 |
| 输出 | TaskIR 文本（自回归） | 类型化决策（非自回归） |
| 校准 | 隐式（验证通过/失败） | 显式（RLCD 训练目标） |
| 可借鉴 | — | 把校准加入 V-DPO 的奖励 |

**可以合并**：V-DPO + 校准奖励 = Compiler-Guided Calibrated DPO

---

## 六、具体行动项

| 行动 | 难度 | 时间 | 效果 |
|---|---|---|---|
| 引用 Jev + System One/Two 框架 | 零 | 0 | 论文定位升级 |
| 给 Resolver 加校准阈值 → VERIFY | 低 | 2h | 与 VERIFY 语义联动 |
| 给 Canonicalizer 加置信度 → 降级 | 低 | 1h | 安全默认 |
| 训练非自回归决策头 | 中 | 1 天 | 100x 加速 |
| V-DPO + RLCD 校准奖励 | 中 | 2 天 | 新训练方法 |
