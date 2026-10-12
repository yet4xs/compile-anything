# Phase 8A: TaskIR Semantic Correctness & Error Attribution

**日期**: 2026-10-13
**基线**: `6f84340`
**范围**: 不训练、不改权重、不执行真实写操作;G0/G1/G2 冻结基线

---

## Task 1 — 工具注册表与评测隔离修复

- 新增 `src/eval/tau3_tool_registry.py`:**纯 AST 解析** τ³ 官方工具定义(`@is_tool`/`@is_discoverable_tool`,含 ToolType、mutates_state、docstring 描述、类型化参数 schema),不触碰任何参考动作/模型预测
- `scripts/audit_tau3_tool_registry.py` 审计结论(results/phase8a/tool_registry_audit.json):
  - 四域共 145 个工具定义;**agent 可见运行时池 = 67**(airline 14 / banking 24 / retail 16 / telecom 13);banking 有 48 个 discoverable(须经知识库解锁,不入池)
  - 描述与参数文档 100% 覆盖(agent 池)
  - **12 个 telecom 参考动作是用户侧工具**(`user_tools.py`,agent 只能请求用户执行)——这些任务对 agent-only 管线部分不可行,评测器标记 `partial_infeasible` 而非记为模型失败
  - 旧 `extract_domain_tools()` 的 ref-names-union 污染已废除;NONE 决策保留(移除 top-1 强制回退),本轮全量运行 NONE=0

## Task 2 — 语义技能 Oracle(人工核验)

`src/eval/tau3_semantic_oracle_data.py`:67 个 agent 池工具逐条标注 semantic_skill / effect_class(read_only | state_update_reversible | state_creation | state_transition_irreversible | communication | agent_internal)/ policy_guarded / evidence / confidence。

- **全部变异型工具(25 个)逐一读取实现核验(verified)**:包括 airline 六个(book/cancel/update×3/send_certificate 的退票不可逆与"Be careful"策略)、banking 五个(submit_transaction 的卡类型校验与奖励计算等)、retail 七个(cancel 的 reason 枚举与 pending 守卫等)、telecom 六个(suspend 的 6 个月上限等)
- 读取型工具按文档规则标注(heuristic),其中 search_direct_flight 等已读实现(verified)
- Oracle 构建未读取模型预测或任务参考答案;未知映射 confidence=unknown,**不计为正确**
- `tau3_skill_oracle.py` 新增访问器:`get_tool_oracle()`(合并 AST 注册表事实)、`oracle_semantic_plan()`

## Task 3 — 五层语义评测器

`src/eval/taskir_semantic_eval.py`(L0 结构 / L1 工具选择 / L2 技能语义 exact+coarse 双口径 / L3 参数值与有序规划 / L4 效果与绑定;不可表达项记 unsupported;executable success 不用 L0-L4 代替)。

**关键口径修正(发现于对拍)**:τ³ 参考是**多回合执行轨迹**(如 airline-44 含连续 10 次 search、5 次 get_reservation),TaskIR 是**计划**——参考轨迹按"折叠连续重复、保留首现顺序"规范化(原始计数保留),否则序级指标不公平。

## Task 6 — 回归测试(CPU,12 项全过)

`tests/test_tau3_semantic_grounding.py`:get→只读 FETCH、update→变异 EXEC_ACTION、cancel→不可逆、update 不得映射 FETCH/TRANSLATE、同 ISA 不同工具不塌缩、单合法节点≠多步正确、变异降级为只读被标记、不安全 retry 被标记、用户侧参考→partial_infeasible、未知工具不自动算对、参数值错误率、有序/多重集双报告。

## Task 5 — 探索性重评(7C-2.1 存量 100 任务,标记 exploratory)

results/phase8a/exploratory_rescore.json(规范化参考后):

| 臂 | Valid% | L1 工具 F1 | L2 exact F1 | L4 effect | 备注 |
|----|--------|-----------|-------------|-----------|------|
| G0 | 92.4 | **0.0** | **0.014** | 0.035 | 静态合法 ≈ 语义塌缩 |
| G1 | 93.4 | 0.0 | 0.013 | 0.034 | 约束未改变语义 |
| G2 | 95.5 | 0.0 | 0.014 | 0.034 | 同上 |

**诊断**(F1=0 的 192/198 为 banking 任务):banking 参考几乎全为 EXEC_ACTION 变异(submit_transaction 等),而学习管线(含 learned canonicalizer 标签)把它们映射为 FETCH/SEARCH 只读技能——**跨域语义接地错误导致整体语义塌缩,而 Validator 全然不知**。airline 等域单样本 F1 可达 0.86(L2),证明评测器分辨率正常。

{{FULL_RUN_RESULTS}}

## Task 4/5 全量结果(120 任务,域平衡 30×4,排除 7C-2.1 重叠;规范化参考;空计划记 0)

| 条件 | Valid% | L1 工具 F1 | L2 exact F1 | L2 序列% | L3 ordRec | L4 effect | L4 bind | 步数 pred/ref |
|------|--------|-----------|-------------|---------|-----------|-----------|---------|---------------|
| A 全学习 G0 | 42.5 | 0.004 | 0.058 | 1.7 | 0.18 | 0.15 | 0.28 | 0.92 / 3.31 |
| A + G1 | 50.0 | 0.041 | 0.093 | 0.8 | **0.41** | **0.42** | **0.53** | **2.72** / 3.31 |
| A + G2 | 51.7 | 0.041 | 0.093 | 0.8 | 0.41 | 0.42 | 0.53 | 2.72 / 3.31 |
| B Canonicalizer Oracle | **15.0** | 0.0 | 0.065 | 6.7 | 0.49 | 0.27 | 0.49 | 0.35 / 3.31 |
| C Gold 工具集 | 40.8 | 0.005 | **0.344** | **29.2** | **0.82** | 0.51 | 0.44 | 0.68 / 3.31 |
| D 参考计划上界 | 96.7 | 0.60 | 0.74* | 39.2* | 0.94 | **1.00** | **1.00** | 5.58 / 3.31 |

*D 按原始轨迹逐动作构造、相对规范化参考偏长,其 L2/seq 是**下界**;绑定指标(argAcc 0.85 / bindability 1.0 / effect 1.0)才是上界信号。

### 归因(多因保留,120 任务中 101 个多因)

| 错误类型 | 计数 | 说明 |
|----------|------|------|
| **Composer 规划错误** | **86** | 主导:欠规划(0.9 步 vs 需 3.3;给对工具仍只 0.7 步)|
| 绑定上界缺口 | 74 | 参考计划自身也无法完美对齐规范化参考(含 oracle 未知/用户侧引用)|
| Validator-only 错误 | 50 | 合法但语义错误 |
| 工具绑定错误 | 34 | 语义对但 IR 不携带工具身份(L1≈0;D 证明通道本身可用)|
| Resolver 错误 | 33 | 选择集不对 |
| Canonicalizer 单独错误 | 1 | 但见下——B 条件的意外发现 |

### 三个关键发现

1. **B 条件(Canonicalizer Oracle)使 Valid 从 42.5% 崩到 15%**:给 Composer "正确"的 EXEC_ACTION 标签反而灾难性变差——Composer 的训练分布与 Canonicalizer 的标签分布**耦合**。不能通过只修标签修语义;标签与规划必须联合解决。
2. **G1/G2 约束提升部分语义指标但不产生完全正确**:ordRec 0.18→0.41、effect 0.15→0.42、步数 ×3(EOS 门控阻止过早收尾),但 semantic_fixed=0——约束改善计划完整性,正确性仍需语义能力。
3. **C 条件(给对工具集)语义大幅恢复**(L2 ×6、序列匹配 1.7→29.2%)但规划深度仍不足(0.68 步)——**欠规划是最大单点瓶颈**。

## Go/No-Go 裁决(Task 7 三问)

1. **静态合法 ≠ 语义正确**:平衡任务集上 A 条件 Valid 42.5%,其中语义序列匹配仅 1.7%;50/120 任务"合法但错误"。
2. **最大语义损失 = Composer 规划深度**(86/120),且与 Canonicalizer 标签分布耦合(B 崩溃证据);Resolver 次之(33);绑定通道在 IR 层可用(D:bindability 1.0、argAcc 0.85),缺的是让模型**表达**工具身份。
3. **G2 相对 G1 在 L2-L4 无额外语义增益**(此集上完全相同);G1 相对 G0 提升部分语义指标(计划完整性)但不产生语义正确的程序。

**下一阶段方向(按预冻结规则)**:主要问题是操作序列过短 → **优先研究分层图规划与多步生成**(例如:计划草稿-展开两阶段、迭代式节点扩展、CapabilityIR→ISA 的工具身份保留);禁止在明确瓶颈前直接联合训练/RL——本阶段证据表明单纯联合训练会撞上分布耦合问题。

## 交付物清单

```
src/eval/taskir_semantic_eval.py        五层评测器
src/eval/tau3_tool_registry.py          AST 运行时注册表
src/eval/tau3_semantic_oracle_data.py   67 工具人工核验 oracle(25 变异全 verified)
src/eval/tau3_skill_oracle.py           oracle 访问器(扩展)
scripts/audit_tau3_tool_registry.py     注册表审计
scripts/eval_phase8a_semantics.py       四条件评测
scripts/rescore_phase8a.py / fixup_phase8a_tables.py  规范化重评
tests/test_tau3_semantic_grounding.py   12 项回归(全过)
results/phase8a/{tool_registry_audit, semantic_layers, error_attribution,
                 semantic_details_rescored.jsonl, exploratory_rescore}.json
```

## 核心结论(初步,待全量 120 任务确认)

1. **回答 Go/No-Go 问题 1**:τ³ 静态 90-96% 合法率中,语义正确率(此样本)≈ 1-4%——"合法但错误"是常态而非个例
2. **问题 2 初步指向**:主要损失在 **Canonicalizer 跨域标签**(把变异工具标成只读技能),而非 Composer 结构
3. **问题 3**:G1/G2 对语义层零改善(约束只让错误程序合法化,此样本上未提高真实工具选择/规划质量)——与 7C-2 的"结构可靠性"贡献正交,须分层陈述
