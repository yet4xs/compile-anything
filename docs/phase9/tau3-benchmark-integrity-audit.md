# τ³ Benchmark Integrity Audit

**Phase**: 9B-2.1 | **Date**: 2026-10-15 | **Tasks 4-6 replay evidence added 2026-10-16**

## 核心发现

### Gold Literal Observability (Task 2, static)

| 来源 | 数量 | 说明 |
|------|------|------|
| USER_TEXT | 123 | 值在用户指令中可见 |
| runtime_ref | 233 | 标注为运行时依赖 |
| **ENV_STATE_OR_UNKNOWN** | **245** | **值不在用户指令中,模型无法从可见信息推导** |

| 严重度 | 数量 | 含义 |
|--------|------|------|
| **HIGH** | 37 | 第一步就使用模型不可知的值 |
| MEDIUM | 157 | 后续步骤,可能来自工具但未验证 |
| LOW | 51 | 可能从上下文推导 |

### Runtime Dependency Verification (Task 3, static)

| 状态 | 数量 | 含义 |
|------|------|------|
| VERIFIED | 68 | 字段在生产者 output_schema 中声明 |
| UNKNOWN | 165 | 回显字段(需要真实运行时验证) |
| REFUTED | 0 | 无明确错误标签 |

## Task 4: 真实 τ³ 模拟器 Oracle Replay(53 任务全量)

用 tau2-bench 官方模拟器(tau3 conda env, Python 3.12,无 LLM、未运行 AgentBoard)
逐条重放 gold 动作序列,并捕获每一步真实工具返回值。

**数据库一致性**: 本地 db.json 与官方仓库 md5 一致;抽检 5 个代表任务
gold actions 与官方当前版本逐字节一致(仅 category 元数据差异)。

### Oracle 可执行性

| 指标 | 值 |
|------|-----|
| **Oracle 可执行任务** | **47 / 53 (88.7%)** |
| 总动作数 | 326 |
| 动作错误 | 6 |

6 个任务的 gold 在官方数据上无法执行:

| 任务 | 失败动作 | 错误 |
|------|----------|------|
| tau3_retail_2/3/4 | get_product_details | `Product not found`(gold 引用的商品不在 db.json) |
| tau3_retail_35/37 | find_user_id_by_email | `User not found` |
| tau3_retail_64 | exchange_delivered_order_items | `Non-delivered order cannot be exchanged`(状态机违约) |

注: τ³ 官方 evaluator 对 gold 错误静默容错(catch-and-continue),目标 DB hash
由部分失败的 gold 重放推导 — 这些任务的 reward 基准本身可疑。

### Runtime Dep 真实回放验证(233 deps)

| 判定 | 数量 | 含义 |
|------|------|------|
| **VERIFIED_BY_REPLAY** | **59 (25.3%)** | 生产者步骤真实输出中该字段=该值 |
| WRONG_PRODUCER | 27 (11.6%) | 值在生产者输出中但非该字段 |
| VALUE_IN_EARLIER_ONLY | 126 (54.1%) | 值在某更早工具输出中(转换器选错生产者) |
| **NOT_IN_ANY_OUTPUT** | **21 (9.0%)** | 任何工具输出都未返回该值 → 值来自用户对话或转换器伪标注 |

**结论**: 91% 的 dep 值确实是环境派生的(运行时依赖真实存在),但只有 25%
的 (producer, field) 标注完全正确。静态审计的 UNKNOWN 165 条中,大部分
的值可以在真实输出中找到 — 依赖存在性成立,生产者归属不精确。

## Task 5: Verified Subset 冻结

三级分类(`results/phase9/tau3_verified_subset_manifest.json`):

| Tier | 数量 | 判据 |
|------|------|------|
| **VERIFIED** | **26** | gold 可执行 + 全部 dep 值环境可推导 + 全部字面量可回答(用户文本 ∪ 前驱工具输出) |
| DEGRADED | 21 | 可执行但含不可回答字面量或伪 dep |
| EXCLUDED | 6 | gold 本身不可执行 |

**26 个 VERIFIED 全部是 retail**。原因(结构性,非抽样偶然):

- **airline 12/12 全部 DEGRADED**: 第一步 `get_user_details(user_id=...)` /
  `get_reservation_details(reservation_id=...)` 的 ID 值既不在用户文本、也不在
  任何工具输出中 — 它们是 **user-latent 值**: τ³ 交互式用户模拟器会在被问及时
  提供,但单轮编译式 benchmark 无法获得。
- **retail 的 9 个 DEGRADED**: 6 个是压缩 gold(gold 只含写操作,跳过所有
  lookup 步骤,全部字面量不可推导);3 个单值缺失(如"最近订单"需交互解析)。
- retail 通过 `find_user_id_by_name_zip(name, zip)` 从文本派生 ID — 这正是
  retail 26 个任务可验证的结构原因。

列表值按元素逐一判定(item_ids 列表内每个 id 单独检查)。

## Task 6: 现有预测的阶段级失败归因

对 `tau3_real_evaluation.json` 的 42 条预测(14 任务 × 3 臂,seed 44)离线重放
全部编译阶段并归因首个失败阶段(`results/phase9/tau3_evaluation_rescored.json`)。

**关键事实: 14 个已评估任务中只有 5 个属于 VERIFIED 子集**;9 个为
DEGRADED/EXCLUDED(gold 损坏或含不可知值)。

### VERIFIED 任务上的归因(5 任务 × 3 臂)

| 阶段失败 | M0-T | M0-J | M1 |
|----------|------|------|----|
| SYNTAX | **5** | **5** | 0 |
| FORWARD_REF | 0 | 0 | **3** |
| UNREPRESENTABLE_NESTED_FIELD | 0 | 0 | **2** |
| SKELETON_SCHEMA | 0 | 0 | 0 |
| 全部通过 | 0 | 0 | 0 |

### 归因解读

1. **M0-T/M0-J 的失败是生成层面的**: M0-T 产出退化 IR(发明不存在的 op 与
   工具名、训练分布残留),且 4/14 撞上 700-token 截断;M0-J 12/14 SYNTAX
   (11 条截断于 token 上限,JSON 残缺)。
2. **M1 的失败全部发生在编译器类型阶段,不是生成阶段**: 骨架 JSON 均可解析、
   结构校验通过、且**全部 5 条都携带 runtime_ref 绑定** — 语义骨架的运行时
   绑定能力跨域迁移成立。
   - FORWARD_REF ×3: runtime_ref 指向尚未定义的生产者节点(顺序错误)。
   - UNREPRESENTABLE_NESTED_FIELD ×2: 模型输出 `orders[0].order_id` 这类
     **嵌套字段路径** — 语义上正确,但 TaskIR v0.2 的 EXTRACT_FIELD 只支持
     顶层字段,typed validation 拒绝,lowering fail-closed。
3. **"全模型 0%" 的构成**: gold/schema 问题占 9/14 任务(64%),模型真实
   跨域缺陷集中在 M0 系语法坍塌与 M1 的字段路径表达;IR 表达力(nested field
   access)是可修复的编译器侧缺口,不是模型能力边界。

## 结论(修订)

| 指标 | 值 |
|------|-----|
| 53 任务中 VERIFIED(论文级) | **26**(全 retail,99 deps) |
| DEGRADED(user-latent / 压缩 gold) | 21 |
| EXCLUDED(gold 不可执行) | 6 |
| dep 标注完全正确 | 59 / 233 |
| dep 值环境可推导 | 212 / 233 (91%) |

## 建议

1. **论文评估只用 26 个 VERIFIED 任务**;DEGRADED 需交互阶段(benchmark 格式
   扩展)才能合法使用;EXCLUDED 上报 τ³ 数据缺陷。
2. **IR 扩展(Phase 9B 前置)**: EXTRACT_FIELD 支持嵌套路径
   (`orders[0].order_id`)— M1 在真实域上的最大单一失败原因(6/14 all-tasks),
   且是编译器侧可修复的。
3. **重新评估需要服务器 LoRA 权重**: 现有预测只覆盖 5 个 VERIFIED 任务;
   26 任务全量三臂评估待服务器开机(本地无 adapter,重训不可复现)。
4. Phase 9B-3 跨域训练值得做的条件: 先落地 (2) 的 IR 扩展并在 26 VERIFIED
   上重测 — M1 已能在真实域产出合法骨架+绑定,当前瓶颈在编译器表达力与
   M0 的语法坍塌,不是"小模型完全不能泛化"。
