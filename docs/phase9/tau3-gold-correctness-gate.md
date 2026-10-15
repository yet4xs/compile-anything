# τ³ Gold Correctness Gate — Phase 9B-2.2 报告

**Baseline**: 8eda615 | **本阶段提交**: Phase 9B-2.2 | **日期**: 2026-10-16
**约束遵守**: 无 GPU 训练、未运行 AgentBoard、未修改任何旧结果文件
(9B-2.1 的 `tau3_evaluation_rescored.json` 保留为审计痕迹,已被
`tau3_m1_failure_reclassified.json` 取代)。

## Task 0 — 复现性修复(已提交)

| 项 | 修复 |
|----|------|
| `a["n_correct"]` 契约 | `scripts/train_three_arm.py` 改用 `n_correct_bindings` |
| 回归测试 | `tests/test_canonical_matcher_contract.py`(4 例,锁定 matcher 返回 schema;`n_correct` 键不存在性也被锁定) |
| 完整预测保存 | `raw` 不再截断;新增 `raw_preview`(2000 字符)、`raw_log_truncated`、`hit_generation_cap` 两个独立状态位 |
| 阶段分离评分 | gen_and_eval 记录 `json_syntax` / `skeleton_valid` / `lowering_ok` / `parse`(=TaskIR valid) |
| 嵌套分类修复 | 新评分器按**字段路径语法**(parse_path)+ 真实 schema 根字段判定,不再看错误字符串中的 `[` |
| CPU 评分入口 | `scripts/score_tau3_offline.py`(确定性、无 GPU、可重复;9B-2.1 的 rescore 脚本已删除避免误用) |

证据缺口声明: 旧预测的 `raw` 在生成时已被截断至 2000 字符
(M0-J 11/14、M0-T 4/14 命中)。离线重评的 `TASKIR_SYNTAX`/`JSON_SYNTAX`
判定因此受日志截断污染,与 700-token 生成上限**分开标注**
(`hit_generation_cap` vs `raw_log_truncated`),不混为一谈。修复只对
未来生成生效;旧结果不改写。

## Task 3 — 官方 Replay 语义(证据: tau2-bench `docs/evaluation.md`)

`evaluation_criteria.actions` 的官方语义(原文):

1. **它是一条参考轨迹,不是对 agent 的动作要求**。airline/retail/telecom
   的 `reward_basis = [DB, COMMUNICATE]`,`ACTION` 从不出现(仅少数
   banking_knowledge 任务使用)。
2. 它的官方用途: 在干净 gold 环境上重放以推导**目标 DB 终态 hash**;
   agent 走任何产生等价终态的路径都得分。文档明确警告不要把它读成
   "agent 必须执行的动作清单"(issue #224 / RFC #129)。
3. 官方 evaluator 对 gold 动作异常**静默容错**(catch-and-continue)。

**对本项目含义**:

- 我们的 benchmark 把参考轨迹用作"编译目标程序"是一个**自选语义**,
  必须在论文里明示 — 它在 τ³ 官方语义中不是唯一正确解。
- 9B-2.1 报告的 "6 个坏任务" 措辞修正为: **6 个任务的参考动作列表在
  官方数据上重放抛异常**(3×Product not found, 2×User not found,
  1×Non-delivered exchange)。官方 reward 对这些任务仍可能有定义
  (部分重放推导的 DB hash),我们不宣称 τ³ 官方 benchmark 有 6 个坏任务。
- 分层报告(按 advisor 要求):

| 检查 | 结果 |
|------|------|
| Action Replay Without Exception | 47/53(不称 Task Success) |
| Official Goal / Reward Check | **N/A** — 无 agent 轨迹可评;53 任务 env_assertions 全空,reward_basis=DB+COMMUNICATE 需完整模拟 |
| Real Output Schema Match | 见下表,手写 schema 14/18 与真实返回不符 |
| Full Binding Provenance Match | 233 deps 中 45 LABEL_EXACT(24 字段路径 + 21 裸标量根) |

### 手写 Schema vs 真实返回(Task 3,重放观测)

结构化遍历真实返回(pydantic 模型字段)对照 `convert_tau3_l3.py` 的
手写 output_schema:

| 工具 | 手写(错) | 真实顶层字段 |
|------|-----------|--------------|
| retail.get_order_details | `payment_method_id` | 无此顶层字段;真实在 `payment_history[i].payment_method_id`(嵌套) |
| retail.get_product_details | `price, sku, in_stock` | 真实顶层仅 `name, product_id, variants`;价格等嵌套在 `variants[i]` |
| retail.get_user_details | `default_address` | 真实为 `address` |
| airline.get_user_details | `membership_level` | 真实为 `membership` |
| retail.find_user_id_by_* | `user_id`(命名字段) | 真实返回**裸字符串**(根标量,无字段名) |
| retail.exchange/return/modify(写工具) | `{order_id, success}` | 真实返回完整更新后 `Order`(无 `success` 字段) |
| airline.get_reservation_details 等 | 少列大量字段 | 真实返回完整 Reservation/User |

共 14/18 工具不符;4 个工具未被 gold 覆盖(无证据)。
**直接影响**: M1 在 retail_48 上输出 `default_address` 被拒 — 该字段名
正是我们的 registry 教给模型的错误名称。评测器用错误 schema 拒绝了
"按教材犯错"的模型。

## Task 2 — Gold Dependency Repair(233 deps 全量)

方法: 官方模拟器重放,直接调用底层工具函数获得**原始 pydantic 对象**
(非 repr 字符串),结构化遍历(pydantic 字段/dict 键/list 索引)建立
值→(step, path) 索引;对每条消费者参数寻找真实来源。

| 判定 | 数量 | 含义 |
|------|------|------|
| LABEL_EXACT | 24 | 原标注 (producer_step, field) 真实携带该值 |
| LABEL_EXACT_ROOT_SCALAR | 21 | 裸标量工具(find_user_*),标注语义成立 |
| REPAIRED_UNIQUE | 92 | 唯一真实来源,标注已修正(附新 step+path) |
| AMBIGUOUS | 75 | 值出现于多个先前输出 — 不自动选择,候选全记录 |
| USER_LATENT | 18 | 任何工具输出均无、指令文本亦无 → 用户交互值 |
| UNEXECUTABLE_CONTEXT | 3 | 上游 gold 动作抛错,证据不完整 |

逐条 diff 与 provenance 见 `results/phase9/tau3_gold_repaired.json`
(含每步 `outputs_flat` 真实路径值与 `real_schema_observations`)。
未为任何模型指标做选择性调整;原标注一律保留在新判定旁。

## Task 1 — 三级 Gold 质量等级(冻结)

`results/phase9/tau3_gold_verified_manifest.json`:

| Tier | 数量 | 判据 |
|------|------|------|
| **BINDING_EXACT** | **9**(全 retail;train 8 + dev 1;22 deps) | 可执行 + 全部 dep 唯一证据来源(LABEL_EXACT/ROOT_SCALAR/REPAIRED_UNIQUE)+ 全部字面量可回答 |
| VALUE_OBSERVABLE | 17 | 可执行 + 全部 dep 值真实存在(含 AMBIGUOUS)+ 字面量可回答 |
| REPLAY_EXECUTABLE | 21 | 仅可执行(含全部 12 个 airline — 均含 USER_LATENT 依赖) |
| FAILED_REPLAY | 6 | gold 动作抛异常 |

9B-2.1 的 26 条 "VERIFIED" 更名 **OBSERVABLE_CANDIDATE**(被本 manifest
取代): 其中仅 9 条达到 BINDING_EXACT;其余 17 条携带 AMBIGUOUS 依赖。
advisor 独立复核的 33/99 精确吻合与本报告结构化口径(45/233 全量)一致
数量级 — 差异来自根标量工具与逐元素列表处理,均已显式分类。

**探索/冻结协议**: BINDING_EXACT 的 dev 1 条(train 8)不足以构成独立
测试集。Phase 9B-3 若启动,必须先扩充 BINDING_EXACT 规模(修复 AMBIGUOUS
需要任务级消歧规则或人工裁决),并冻结新测试划分;现有 dev/test 已被
错误分析污染,不得再用作冻结测试。

## Task 5 — M1 十四条预测重分类(修正嵌套分类后)

`results/phase9/tau3_m1_failure_reclassified.json`,对真实输入 schema
与修复后 gold:

| M1 首个失败 | 数量 | 说明 |
|-------------|------|------|
| FORWARD_REF | 3 | 引用未定义生产者(含 retail_6/8/58) |
| NONEXISTENT_FIELD | 4 | 字段不存在(含 retail_48 的 `default_address` — schema 教错的名称) |
| GENUINE_NESTED_ACCESS | 2 | 真嵌套路径(含 retail_11 的 `orders[0].order_id`)— IR 表达力缺口 |
| SKELETON_SCHEMA | 3 | 骨架结构错误 |
| LOWERING_OTHER | 2 | 其他 lowering 失败 |

深指标(全部 14 条): JSON syntax 通过 14/14;骨架有效 11/14;
runtime_ref 共 43 条(其中有效生产者 22、前向引用 10、生产者不存在 11、
真嵌套访问 3);**与修复后 gold 的输出字段吻合 0/43**;Strict Binding
Exact 0/14。

**结论修正**(回应 advisor): "runtime binding 能力跨域迁移"不成立 —
成立的是**绑定语法/结构迁移**(模型会写 runtime_ref),但生产者选择、
字段路径选择均未迁移。"M1 失败全部可通过扩展 IR 解决"也不成立 —
只有 2/14(GENUINE_NESTED_ACCESS)是 IR 缺口;FORWARD_REF(3)与
NONEXISTENT_FIELD(4)是模型规划错误,应被编译器拒绝。

## Task 6 — Phase 9B-3 GO / NO-GO

| # | GO 条件 | 状态 |
|---|---------|------|
| 1 | 足够数量、来源可靠的 BINDING_EXACT Gold | **NO** — 仅 9 任务 22 deps,且 dev/test 已被探索污染 |
| 2 | 修复后的评测器通过全部负面测试 | **YES** — matcher 契约 4 例 + field_path 负例 8 例全过 |
| 3 | Typed Nested Access 正负例正确 | **YES(原型)** — 13/13;未接入生产 validator(待设计批准) |
| 4 | Oracle 在真实返回值上完成绑定验证 | **YES** — 修复 pipeline 全部基于官方模拟器真实对象 |
| 5 | 评测拆分/测试集使用记录/原始输出可复现 | **PARTIAL** — CPU 评分入口可复现;但历史 raw 被截断、旧划分已污染 |

**裁决: NO-GO(阻塞于条件 1 与 5)**。

阻塞原因与解锁路径(按优先级):
1. 扩充 BINDING_EXACT: 为 75 条 AMBIGUOUS 依赖制定消歧规则(同一值
   多来源时,按"消费者需要的语义角色"或任务作者意图裁决),或对高价值
   任务人工裁决 — 目标 ≥30 任务、≥80 deps,并冻结新 test 划分。
2. registry schema 修正: 用 `real_schema_observations` 重建
   capability_registry(嵌套结构 + 裸标量),这是模型输入的一部分,
   错误 schema 直接误导模型。
3. Typed Nested Access 设计批准后接入 validator/lowering(设计文档:
   docs/phase9/typed-nested-access-design.md)。
4. 重跑生成(需服务器 LoRA): 保存完整 raw + 阶段分离字段(本轮已实现)。

在这些完成之前启动 GPU 训练,只会让模型更好地拟合有缺陷的 Gold。

## 交付物清单

- `results/phase9/tau3_gold_repaired.json` — 233 deps 逐条 diff + 真实输出
- `results/phase9/tau3_gold_verified_manifest.json` — 三级冻结 manifest
- `results/phase9/tau3_m1_failure_reclassified.json` — M1 重分类 + 深指标
- `docs/phase9/typed-nested-access-design.md` — 设计文档
- `src/compiler/field_path.py` + `tests/test_field_path.py` — 原型 + 13 测试
- `tests/test_canonical_matcher_contract.py` — matcher 契约回归
- `scripts/repair_tau3_gold.py` / `build_tau3_gold_manifest.py` /
  `score_tau3_offline.py` — 可重复 CPU pipeline
- `scripts/train_three_arm.py` — n_correct 契约修复 + 完整 raw + 阶段分离
