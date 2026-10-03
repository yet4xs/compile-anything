# 摘要（Abstract）— Phase 5D 终版

> 论文：*Compile Anything: A Neural Compiler and Runtime Architecture for General AI Task Execution*（目标会议：ASPLOS / DAC）

大语言模型智能体以串行循环逐步解释执行任务：计划是无类型的临时文本，依赖与副作用顺序无法在执行前静态检查，成本事后结算。本文提出 Compile Anything，引入 compiler–runtime 分离架构：类型化扁平 SSA 中间表示 TaskIR（guard/SELECT 谓词分支、VERIFY 与有界重试回滚为一等 IR 结构）、携带五维名义成本模型的 Skill ISA、V1–V6 六类静态不变量验证器，以及训练于 31,215 条经确定性语义审计语料（98.94% verified）的 3B 神经编译器前端。实验表明：3B 模型可学会 TaskIR 的结构编译规则——域内 validator 通过率 99%，且在 BFCL、τ³-bench、AgentBoard 三个从未参与训练的基准家族上保持高结构有效性（92–99%）；但语义接地在域外急剧退化（BFCL 端到端功能语义 15.5%，τ³ 语义召回 0.6%），揭示**结构编译、语义技能选择与规划深度是三个可分离的能力维度**。Schema 条件化训练小幅改善 BFCL 端到端语义正确性并显著改变规划长度先验；朴素的深度再平衡只增加计划长度而不改善语义正确性。显式 IR 分层使这些失败模式可分别度量：前端语义接地是瓶颈，而后端技能→工具绑定 oracle 上界为 99.95%。

---

# §12 结论（Conclusion）— Phase 5D 终版

本文提出并实现了面向通用 AI 任务执行的神经编译器与运行时架构 Compile Anything。TaskIR v0.1 与 Skill ISA 为 AI 任务给出带书面规范的类型化 SSA 中间表示，验证与有界重试回滚成为一等 IR 结构；71,855 条真实基准样本经 adapter–lifter–validator–simulator 管线转换为 31,215 条训练语料，逐条回链真值的确定性语义审计达 98.94% verified；13 个外部基准经污染防火墙冻结为双指标评测体系。

本文的实验发现有三。

**发现一：小模型的结构编译可学习且可迁移。** 28k 条审计监督使 3B 模型从零 validator 通过率（zero-shot 0%）达到域内 99.10%；在 BFCL（92.4%）、τ³（98.9%）、AgentBoard（91.7%，untouched 首测）三个零污染基准家族上，结构有效性不需任何域内数据即成立。TaskIR 的扁平 SSA 与谓词分支设计因此得到跨域验证。

**发现二：结构合法性不蕴含语义正确性。** 同一批模型在结构指标 92–99% 的同时，BFCL 端到端功能语义仅 15.5%、τ³ 语义召回仅 0.6%——两个数量级的落差。主导的跨域失败经 op 分布分析定位在 retrieval–action 边界的语义接地（τ³ 96% 参考动作为 EXEC_ACTION 而模型产出率近零；BFCL multi_turn 同构失败），而非语法或工具绑定；后端 binder 的 oracle 上界为 99.95%，瓶颈在前端技能选择。

**发现三：语义接地与规划深度是可分离的维度。** Phase 5C 2×2 消融显示：schema 条件化小幅改善跨域语义（BFCL E2E +0.35pp）并强烈改变规划长度先验（pred/T 1.00→1.75）；朴素深度再平衡使长度增加（+0.65）而语义不改善（召回持平甚至微降），且以域内 OpSeq（−4.77pp）与跨域结构鲁棒性（τ³ valid −12pp）为代价。**长度可训练，正确性并非由深度再平衡获得。**

架构含义：这些失败模式之所以能被分别度量，正是因为前端 IR、validator 与后端绑定是显式分层——失败定位（而非仅失败报告）是 compiler 分层架构区别于 prompt 栈的直接评测收益。

局限与未来工作如实披露：成本模型全部为名义常量（nominal analytical model，非实测硬件能耗/延迟，实测 profiling 未接入）；Skill ISA 缺少 LOOP 与 STRING_OP（代码域覆盖 7.3%）；effect system v0.2（含跨类事务 region 与 V7）为设计提案未实现，τ³ 的 2,443 个多动作任务仅作 workload characterization；官方交互环境 AgentBoard 指标 UNSCORED-OFFLINE；E3 7B LoRA 未运行。下一步（Phase 6A）为 EXEC_ACTION 边界的受控 skill grounding probe，进而构成 capability grounding → skill selection → program composition 的两阶段神经前端。

---

*写作纪律：名义成本一律标注 nominal；未执行的实验一律标注 NOT RUN / UNSCORED-OFFLINE；E2E 与条件口径不得混用（见 paper-evaluation.md §4.1）。*
