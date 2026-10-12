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

## 核心结论(初步,待全量 120 任务确认)

1. **回答 Go/No-Go 问题 1**:τ³ 静态 90-96% 合法率中,语义正确率(此样本)≈ 1-4%——"合法但错误"是常态而非个例
2. **问题 2 初步指向**:主要损失在 **Canonicalizer 跨域标签**(把变异工具标成只读技能),而非 Composer 结构
3. **问题 3**:G1/G2 对语义层零改善(约束只让错误程序合法化,此样本上未提高真实工具选择/规划质量)——与 7C-2 的"结构可靠性"贡献正交,须分层陈述
