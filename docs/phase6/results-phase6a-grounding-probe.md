# Phase 6A 最终报告 — Skill Grounding Probe

> 数据齐备时间：2026-10-04。协议：`experiments/phase6/phase6a_manifest.json`（base `d6b42ef`，评测前冻结）。
> 工件：`results/phase6/{phase6a_metrics, phase6a_external, phase6a_followups, phase6a_probe_c_g1g2}.json`
> 过程修正（全部保留原始产物）：G1-only 降级为 label-tier ablation；外部诊断首轮因共享 base PEFT 污染作废（`_INVALID_sharedbase.json`），顺序重载重跑；Probe C 首轮误绑 G1-only 模型，已补 G1+G2 版。
> B0 toolmap 内部 100% 为 label-generator self-agreement（tautology），归入 dataset audit，不进模型榜。

## 论文级主表 1 — 内部 grounding（Probe A，test_ood = 142 个未见工具家族）

| Model | 输入 | OOD-family Macro F1 | ACTION-family Recall | **EXEC_ACTION Recall** |
|---|---|---:|---:|---:|
| Qwen 3B zero-shot | schema | 0.477 | 0.000 | 0.000 |
| E1-A as classifier* | schema | 0.676 | 0.229 | 0.000 |
| E5C-S as classifier | schema | 0.782 | 0.412 | 0.000 |
| Grounder G1-only（3seed） | schema | 0.517 ± 0.012 | — | 0.652 ± 0.040 |
| **Grounder G1+G2（3seed）** | schema | **0.917**（0.847/0.952/0.951） | 0.657 | **0.681**（三 seed 一致，32/47） |
| Grounder G1+G2 | task+schema (B) | 0.849（s42） | — | — |

\* E1-A 行为 post-hoc 归因诊断（非预注册主结果）。G1-only 为 label-tier ablation（FETCH/SEND 监督为 0）。

## 主表 2 — 反事实（name-shortcut 检验，F1：full-test_ood / description-corroborated 切片）

| Model | full | name-masked | desc-masked | name-perturbed |
|---|---|---|---|---|
| E1-A | 0.676 / 0.699 | 0.715 / 0.828 | 0.814 / 0.985 | 0.616 / 0.608 |
| E5C-S | 0.782 / **0.9997** | 0.577 / 0.699 | 0.889 / 0.983 | 0.488 / 0.499 |
| G1+G2 s42 | 0.847 / 0.858 | **0.849 / 0.952** | 0.847 / 0.858 | 0.622 / — |
| G1+G2 s43 | 0.952 / — | **0.880 / 0.988** | — | 0.864 / — |
| G1+G2 s44 | 0.951 / — | **0.868 / 0.983** | — | 0.855 / — |

- **Grounder 遮名不降反升、遮描述也不降、换名部分下降**（dc 切片 0.858→0.952/0.988/0.983；name_perturbed 0.62-0.86）——学到的是**多通道能力表征**（multi-channel：词法名字 + schema/描述语义），既非神经版名字正则，也非纯 description grounding。
- **E5C-S 重度依赖名字**（遮名 0.78→0.58、换名 →0.49），且**遮描述反而涨**（0.89，dc 0.9997）——读名字、样板描述是噪声。
- 限定：description-corroborated 切片无 EXEC_ACTION gold（xLAM 样板描述不含动作动词），该切片只检验 retrieval 侧；独立证据仍看外部 oracle。

## 主表 3 — 外部诊断（全部为 diagnostic，三基准均已非 untouched；τ³ 为 name-only 输入）

| Model | τ³ F1 / EXR | BFCL F1 / EXR | AgentBoard EXEC_ACTION 产出率 |
|---|---|---|---|
| B0 toolmap | **0.940 / 0.997** | 0.565 / 0.542 | — |
| 3B zero-shot | 0.462 / 0.477 | 0.462 / 0.466 | todo 2/15 |
| **E5C-S** | **0.788 / 0.650** | 0.367 / 0.173 | todo 2/15 |
| Grounder G1+G2 s42 | 0.402 / 0.255 | 0.334 / 0.168 | todo 0/15 |

（B0 τ³ 0.94：确定性正则与 τ³ oracle 高度一致——映射规则本身"知道"答案，Phase 5C 模型没学会应用它。）

## 主表 4 — Probe C 与 NO_CALL（G1+G2，3seed）

| 指标 | s42 | s43 | s44 | 备注 |
|---|---:|---:|---:|---|
| capability selection acc | 0.224 | 0.188 | 0.212 | **≈ 五选一随机（0.20）** |
| skill acc（选中项分类） | 0.642 | 0.656 | 0.662 | |
| joint exact | 0.144 | 0.126 | 0.148 | |
| NO_CALL rate（合成负例） | 0.0 | 0.0 | 0.0 | **从不拒绝调用** |

## 口径冻结（Phase 6A.5，审稿人裁定版）

- Grounding 结论允许表述：**"capability-to-skill grounding is independently learnable"**；禁止 "description-only grounding solved"（模型同时利用 name/description/schema）。
- 反事实结论允许表述：**"the dedicated grounder is substantially more robust to name masking than E5C-S, indicating semantic signals beyond tool-name patterns"**；禁止 "name masking proves pure semantic understanding"。机制定名 **multi-channel grounding**。
- E5C-S−E1-A 的差值（+0.11）**不得再归因于 schema semantics**（capability audit：xLAM 条件是常量标记 bug）。
- τ³ whole-compiler EXEC_ACTION≈0 的结论**降级为**："E5C-S 内部存在可访问的 EXEC_ACTION grounding 知识，但未在现有 whole-compiler 执行路径中表现出来"——bare-instruction（compiler）与 name-visible（classifier）的信息输入不同，composition 归因需 Phase 6B-0 matched-information 诊断控制变量后方可强化。
- Probe C selection ≈ 0.20 是**独立失败**（训练含 20% Probe C 样本仍为随机水平）；NO_CALL=0 的正确表述：**"无显式 NO_CALL 监督时拒绝行为不涌现（non-emergence），而非已训练目标的失败"**。

## 六问裁决

**Q1 边界（retrieval vs action）学会了吗？——是。**
G1+G2 内部未见家族 F1 0.917±0.05；E5C-S 分类器 0.782（内部，干净复现逐位一致）；零样本基线 0.477（类不平衡猜测水平）。

**Q2 EXEC_ACTION canonical 标签学会了吗？——部分，且路线依赖。**
G1+G2 内部 EXR 0.681（三 seed 完全一致）；E5C-S 在 τ³ 上 EXR 0.650 但内部 0.0（内部 ACTION 样本被它发给 SEND：SEND recall 0.643/precision 1.0）；E1-A/B1 内部 0.0。BFCL 上所有神经模型 ≤0.48。

**Q3 学的是名字正则还是描述语义？——分模型。**
**Grounder = 语义**（遮名不降、dc 切片升至 0.95+，双通道聚合）；**E5C-S = 名字模式**（遮名/换名大跌、遮描述反升）。E1-A 居中。

**Q4 TaskIR 训练本身产生 latent grounding 吗？——是，且是主要来源。**
E1-A 分类器 0.676 vs 零样本 0.477（+0.20 来自纯 TaskIR 程序监督）；E5C-S 再 +0.11（据 capability audit，该增量来自常量标记条件化而非语义）。

**Q5 专用 grounder 比 whole compiler 更跨域吗？——否，方向相反且互补。**
内部未见家族：grounder 0.92 > E5C-S 0.78；τ³（name-only）：E5C-S 0.79 ≫ grounder 0.40。
机制自洽：grounder 靠 description（τ³ 无描述→失灵）；E5C-S 靠 name（τ³ 有名字→生效，book/cancel 类动词正中其模式）。
**两模型在不同通道上互补，而非一个取代另一个。**

**Q6 失败归因（grounding / canonicalization / composition）？——分域分层。**
- τ³ whole-compiler EXEC_ACTION≈0：**composition 层**（同模型分类模式 EXR 0.65）
- 编译器内部不发 EXEC_ACTION 而 grounder 能：canonicalization/标签频率层
- BFCL：grounding 本身（所有模型 F1 ≤0.57，含 toolmap 0.565；task+first-schema 协议含噪声下限）
- 新隔离的两个失败：**capability selection ≈ 随机**（Probe C 0.20）、**irrelevance 拒绝为零**（NO_CALL 0.0，与 BFCL irrelevance 失败同构）

## 决策结果：混合型（非纯 G1）

- 内部 grounding 成立 + 反事实通过（G1 要素满足）
- 但 grounder 外部迁移失败、E5C-S 外部靠名字通道（G2 要素出现）
- **Phase 6B 方向修正**：不是单纯 Grounder→Composer 两阶段，而是**多通道 grounding（name + description）与 composition 的集成**——候选形态：(a) multi-task 共享前端（分类 + 程序生成联合训练）；(b) two-stage 但 grounder 需 name-only 鲁棒性训练（τ³ 式输入增广）。深计划合成仍不是主靶。
- 另两个新靶心（Phase 6B+ 候选）：capability selection（选对工具）与 NO_CALL（拒绝不该做的）——两者目前都在随机/零水平。

## 证据链完整版（论文 Discussion 素材）

```
tool-name regex lifter
    → ① 参数保全 bug（16.7%→98.9%，5B 修复）
    → ② EXEC_ACTION 子串污染（539/1649 contradictory exact）
    → ③ capabilities 字符串迭代 bug（xLAM 全部 → 常量 EXEC_ACTION）
    → 带噪语义监督 → 神经编译器 → 跨域接地失败
```

> The compiler-style decomposition made supervision itself auditable: three
> seemingly model-level issues were traced to deterministic data-front-end
> defects rather than model capacity. 披露 bug、修正解释（Phase 5C 机制改写为
> constant-marker conditioning）、并展示分层审计如何定位它们。

## 遗留与限制

- τ³ 无工具描述 → 外部 τ³ 仅为 name-only grounding diagnostic（措辞限定）
- BFCL gold 用 first-listed function（噪声下限已披露，single-op 子集另列）
- 反事实 dc 切片无 EXEC_ACTION gold（样板描述不含动作动词）
- G1-only 三 seed 保留为负消融（over-filtering destroys label-space coverage）
- E3 7B 取消（未预注册）；如需 scale 对照 → Phase 6C 重新预注册
