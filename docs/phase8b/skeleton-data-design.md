# Phase 8B-1A: Semantic Skeleton Supervision Dataset — 设计与审计

**日期**: 2026-10-13 | **基线**: `e417e02` | **不训练、不改权重**

## Task 1 — P1 失败归因(71.7% 之外的 28.3%)

`results/phase8b/p1_failure_audit.json`(120 任务逐条):

| 类别 | 数量 | 说明 |
|------|------|------|
| ok | 86→**88**(Task 2 修复后) | 编译链成功 |
| user_side_ref | 30 | 参考动作含**用户侧工具**(telecom 用户设备操作,agent 不可调用)——参考可行性问题,非编译器失败 |
| runtime_unknown_input | 2 | 参考首操作需要会话中间值(如 CALCULATE 的数值),TaskIR v0.1 无法从任务文本构造 |

**关键成功率**:全集 73.3%;**表达能力覆盖内 97.8%**(90 任务);完整参数/无用户侧同口径。骨架校验/缺参/Parser 失败 = **0**。

## Task 2 — 骨架语义不变量(已实现并有测试)

- `depends_on`/`order_on` 只准引用**更早**定义的节点(前向引用 = 错误,不再静默)
- `order_on` 与 `depends_on` 分离(数据依赖 vs 顺序依赖);变异操作保守串行
- EXEC_ACTION 不再无条件绕过兼容性:读工具走 EXEC_ACTION 时 oracle 技能不匹配会**记 warning**(保留身份但显式标注);变异工具必须 EXEC_ACTION
- Lowering 类型感知:数据依赖输出类型与技能槽位不兼容时**强制转顺序依赖并报告**(`coerced_ordering_deps`),修掉了 P1 的 4 个类型失败中的 2 个;其余 2 个属 runtime_unknown_input
- **绑定契约统一**:评测器 v2 现接受非变异工具经 `hints(tool=...)` 的身份绑定(变异仍必须 `EXEC_ACTION(action=...)`)
- corpus 模式:非 τ³ 工具以伪注册表验证,效果未知显式 warning 不猜测

## Task 3+4 — 语料构建(`data/semantic_skeleton/`)

来源(**τ³ 全程未触碰,防火墙保持**):
- `compiler_corpus_v4/{train,val}` 28,093 条 → 转换 21,822(ambiguous 7,832 单独列出:工具身份无法对齐,主要是 lowering 缺失或多工具同技能歧义);身份按 lowering 调用序对齐(修正:corpus 的 `EXEC_ACTION(action=...)` 存的是规则词而非工具名)
- 合成深度补齐 1,200 条(同工具家族顺序组合,`order_on` 序依赖,显式 synthetic,仅入 train 且不跨家族)

每条记录:`instruction / available_capabilities(原语料自带金+负例) / target_skeleton / provenance{dataset, evidence, split_group}`。

**切分(工具家族不相交)**:train 17,702 / dev 1,972 / test 2,226;家族交集 = ∅。

深度(节点数,封顶 8):train {1:9753, 2:5988, 3:1284, 4:460, 5:138, 6:54, 7:10, 8:15};DAG 形态:single/linear/independent_branches(7+ 深度经合成覆盖)。

## Task 5 — 训练前审计(`results/phase8b/skeleton_data_audit.json`)

全链路(skeleton→validator→lowering→parser→validator):
- train 85.0% / dev 91.6% / test 97.8%(失败 2,765 条全部保存于 `skeleton_data_failures.json`,主因 CONVERT 类技能槽位与 @task:Str 类型不兼容——与 P1 runtime_unknown_input 同类)
- **SFT-ready 过滤集**:`{train,dev,test}_sft.jsonl` = 15,047 / 1,807 / 2,176(仅链路通过者)
- 近重复(test vs train,8-gram Jaccard>0.8):5/300
- τ³ id:0;家族隔离:三对交集均空
- 7 项数据集完整性测试全过(`tests/test_skeleton_dataset.py`)

## Task 6 — M0/M1/M2 公平对照协议(冻结)

| 模型 | 结构 | 输出 |
|------|------|------|
| M0 Direct TaskIR | Qwen2.5-3B + LoRA | 直接生成 TaskIR 文本 |
| M1 Semantic Skeleton | 同底座同预算 LoRA | 生成骨架 JSON → 确定性 lowering |
| M2 Hierarchical Expansion | 同底座 | 先骨架后分阶段补参数/依赖 → lowering |

共同:同一训练集(SFT-ready train)、同 LoRA 配置与训练预算、同一 available_capabilities 输入、同一评测划分与评分器(v2);零训练 P2 保留为参照。分层收益若存在,必须来自表示与编译过程而非额外 oracle 信息(M0 的训练目标含相同任务/工具信息)。

## Task 7 — 独立测试集冻结

- 主测:`data/semantic_skeleton/test_sft.jsonl`(2,176 条,家族与 train/dev 不相交,未参与任何构建调参)
- 外部:τ³ 120 开发任务仅作 diagnostic(已参与方法开发,不算 untouched);**τ³ 其余未用任务**(排除 120 开发 + 7C-2.1 的 100)留作跨源确认,评测时经 τ³ 注册表 + oracle
- P0-P4 的 120 条不用于模型选择

评测指标(冻结):Valid∧Ordered Plan F1、Tool Identity F1、Argument Value Accuracy、DAG Correctness(依赖边精确率)、Effect Correctness、Exact Joint Match(工具+顺序+必要参数全对)、Validator Pass、Latency。
