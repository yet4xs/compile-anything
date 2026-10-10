# 创新方向：用编译器作为 RL 奖励函数

> 日期：2026-10-10
> 核心洞察：我们的 V1-V6 验证器和 Forge 执行引擎是天然的确定性 RL 奖励信号

---

## 一、为什么这有创新性

传统 RL 奖励信号的问题：
| 方法 | 问题 |
|---|---|
| RLHF（人类反馈） | 昂贵、主观、不可重复 |
| LLM-as-judge | 不可靠、有偏差 |
| 最终答案对错 | 稀疏信号、只看结果 |
| 过程奖励模型（PRM） | 需要额外训练、不透明 |

我们独有的优势（没有其他系统同时有这些）：
- **V1-V6 验证器**：确定性、即时、可分解（6 类不同错误）
- **Forge 执行引擎**：确定性、可验证（成功/失败 + trace）
- **调度器**：可量化（makespan、成本、并行度）

---

## 二、三个新方法

### 方法 1: V-DPO（Validator-Guided Direct Preference Optimization）

```
生成多个 TaskIR 候选（beam search 或采样）
     ↓
V1-V6 验证 → 偏好排序
     ↓
DPO 训练：偏好验证通过的程序
```

**创新点**：用确定性编译器验证作为 DPO 偏好信号（现有 DPO 用人类偏好或 LLM 评判）

**与现有工作的区别**：
- [DiaTool-DPO](https://arxiv.org)（2025）：用对话质量做偏好，我们用编译器验证
- 传统 DPO：需要人工标注偏好对，我们自动生成

### 方法 2: CG-RLVR（Compiler-Guided RL with Verifiable Rewards）

```
生成 TaskIR → V1-V6 验证 → Forge 调度 → Forge 执行
     ↓                                          ↓
奖励 = α·验证通过 + β·执行成功 + γ·语义正确
```

用 GRPO（[DeepSeek-R1](https://arxiv.org/abs/2402.03300) 的方法）训练。

**创新点**：编译管线作为多级可验证奖励函数
- 现有 RLVR 只有最终答案对错（二值）
- 我们有中间反馈（哪一步验证失败、哪个节点执行出错）

### 方法 3: EF-SC（Error-Feedback Self-Correction）

```
生成 TaskIR
     ↓
V1-V6 验证
     ↓ 失败
反馈错误类型和位置（如 "TYPE_MISMATCH at node %c2"）
     ↓
模型重新生成（自我修正）
     ↓
再次验证 → 最多 N 轮
```

**创新点**：编译器错误驱动的自我修正循环
- 现有方法的重试是盲目重试（同样的 prompt）
- 我们的重试有结构化错误信息指导

---

## 三、论文叙事升级

### 旧叙事（只有 SFT）
> "我们建了一个编译器系统，用 SFT 训练了前端"
> → 工程贡献

### 新叙事（加入 V-DPO + CG-RLVR + EF-SC）
> "我们发现编译器的验证和执行引擎可以作为确定性奖励信号，
> 提出 V-DPO 和 CG-RLVR 两种新训练方法，
> 证明编译器引导的 RL 显著优于纯 SFT，
> 并通过 EF-SC 实现验证驱动的自我修正"
> → **方法贡献 + 理论洞察**

---

## 四、需要的实验

| 实验 | 对比 | 预期结果 | 时间 |
|---|---|---|---|
| SFT vs V-DPO | 验证通过率提升 | V-DPO > SFT 3-5pp | ~2h |
| SFT vs CG-RLVR | 执行成功率提升 | RLVR > SFT 5-10pp | ~4h |
| EF-SC | 自我修正成功率 | >50% 错误可修正 | ~1h |
| 多级奖励消融 | 只验证 vs 验证+执行 | 多级 > 单级 | ~2h |
| 跨域改善 | RL 后 EXEC_ACTION | RL > SFT | ~2h |

总计：~11h GPU 时间

---

## 五、与最新文献的关系

| 技术 | 来源 | 我们怎么用 | 区别 |
|---|---|---|---|
| DPO | [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290) | 验证器偏好对 | 用编译器而非人类 |
| GRPO | [DeepSeek-R1 2025](https://arxiv.org/abs/2402.03300) | 多级编译奖励 | 中间反馈而非只有结果 |
| RLVR | [2025 趋势](https://github.com/RLVR) | 验证+执行+语义 | 三级奖励 |
| MCTS | [ToolTree ICLR 2026](https://arxiv.org) | 可选：搜索最优 TaskIR | 编译器指导搜索 |
| 约束解码 | [XGrammar 等](https://arxiv.org) | 可选：语法约束生成 | 与验证器互补 |

---

## 六、实施计划

1. **V-DPO**（最简单，先做）：
   - 生成偏好对：同一任务生成 4 个候选，按验证质量排序
   - 用 TRL 的 DPOTrainer 训练
   - 评测：验证通过率、OpSeq、执行成功率

2. **CG-RLVR**（中等难度）：
   - 用 TRL 的 GRPOTrainer 或 trl 的 RLOOTrainer
   - 奖励函数：连接我们的验证器和 Forge
   - 评测：同上 + 跨域指标

3. **EF-SC**（最简单，可以零训练）：
   - 推理时：验证失败 → 错误信息拼入 prompt → 重新生成
   - 评测：自我修正成功率
