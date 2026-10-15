# 可执行 Gold 重建与验证 — Phase 9B-2.3 报告

**Baseline**: d3e84ca | **日期**: 2026-10-16 | 全程 CPU,官方 τ³ 模拟器,
未运行 AgentBoard,未修改旧结果文件。安全处置见
`docs/phase9/security-incident-remediation.md`(所有者决定: 服务器侧从简,
仓库侧已加固;GPU/服务器任务维持 NO-GO 直至所有者确认)。

## Task 1 — 逐参数定源的 Gold 重建

对 53 任务 × 全部 601 个参数元素(不再只修 233 条启发式 dep 标注),
按序分类: 文本可见 → LITERAL;先前真实输出携带 → runtime_ref(唯一)或
AMBIGUOUS(多来源);仅 DB 存在 → USER_LATENT(用户自有实体)/ENV_STATE;
均无 → UNKNOWN。列表参数先整表匹配,否则逐元素(runtime_ref_multi)。

**消歧规则(全部显式记录 selected_by,不选边站模型)**:
- `unique_candidate`: 唯一来源直接采用(含裸标量工具 path "")
- `canonical_field`: 同一生产者内,路径末端字段与消费者参数语义匹配
- `canonical_minimal_depth`: 同生产者同语义多路径时取最小访问深度
- `root_scalar_origin`: 多生产者但存在唯一裸标量起源(后续出现皆为回显)
- 自回显路径(`variants.<值>.item_id`)仅在无其他来源时作为来源保留 —
  替换商品的 item_id 确实只能从商品 variants 枚举获得(retail_0 案例)
- 以上规则均不适用 → AMBIGUOUS,候选全记录

**advisor 指名案例复核**:
- tau3_retail_25 ✅ `get_order_details.order_id` 现为
  runtime_ref(get_user_details, `orders[...]`) — 用户不知订单号,从用户
  订单列表读取(与 advisor 判断一致)
- tau3_retail_0 ✅ 严格化: item_ids→multi-ref, new_item_ids→
  variants 枚举来源;全参数有源
- tau3_retail_33 ⚠️ user_id 已定源,但 `country`("USA")在 5 个订单的
  address 中回显且不在用户文本 → AMBIGUOUS,不能严格化(诚实保留)

**参数分类总账(53 任务)**:

| 类别 | 数量 |
|------|------|
| LITERAL | 124 |
| runtime_ref(唯一来源) | 226 |
| runtime_ref_multi(逐元素) | 57 |
| AMBIGUOUS | 106 |
| ENV_STATE | 64 |
| USER_LATENT | 8 |
| UNKNOWN | 16 |

## Task 2 — 独立执行验证

双环境对照: 原始参考动作(env1)与重建程序(env2,runtime_ref 经
typed path 在**活体输出**上解析,非回放字符串)逐步骤比对 —
实际参数值、工具实例、结构化输出、每步 DB hash、终态 hash。

| 指标 | 值 |
|------|-----|
| 参考轨迹可重放 | 47/53 |
| 重建程序完整执行(零字面量回退) | **5/5**(全部 strict 任务) |
| strict 任务逐步参数+工具一致 | 5/5 |
| strict 任务最终 DB 状态一致 | 5/5 |

**等价性区分**: 参考轨迹等价(逐步一致)与最终目标等价(终态 hash)
分别记录。非 strict 任务的"最终一致"不具证明力(混入字面量回退的
Frankenstein 执行),已在输出中标注 `used_fallback_literals` 并从
分母中剔除 — 此前 9B-2.2 报告的 53/53 终态一致即属此无效口径。

## Task 3 — Gold Scorer 重建

新增 `src/eval/gold_binding_scorer.py`(28 项回归测试全过):
- **双射匹配**,预测节点下标永不与 gold 下标对齐;同工具多实例以
  (tool, 字面量签名) 区分;拓扑合法重排不影响(位置不进比较)
- Producer + Field Path + Consumer + Argument 精确比较;
  root-scalar(path "")与字段访问区分,单列 ROOT_SCALAR_EQUIV
- 未定源 gold 参数(gold_unsourced_params)显式报告,不得静默通过
- `score_tau3_offline.py` 已接入: 对 42 条既有预测按 **v1 重建 gold**
  评分(取代旧 target_skeleton 对照)

既有预测对照 v1 gold 的诚实结果: 21 条可解析中,
**calls 双射完整 0、绑定精确 0/121、ROOT_SCALAR_EQUIV 1**。

## Task 4 — Typed Access 加固(field_path.py,9 项新测试)

- `orders[0]id`(]后无名隔离)→ PARSE_ERROR ✓
- `orders.[0]`(.后直接索引)→ PARSE_ERROR ✓
- dunder/下划线前缀路径(`__class__`、`__dict__`、`_private`)→
  编译期与运行期双重拒绝 ✓
- 运行期 getattr 限制公共属性;未声明属性 → FIELD_NOT_FOUND ✓
- `orders[0].order_id` 正常;List[Any] 保留动态检查标识 ✓
- 重建 pipeline 已实际使用该解析器执行 5 个 strict 任务(活体验证)

## Task 5 — v1 冻结(`tau3_binding_exact_v1_manifest.json`)

**准入 5/53**: tau3_retail_0, 1, 25, 62, 65 — 全部满足:
无不可观察常量、参数全定源、真实路径、活体执行+终态一致、
未参与错误分析(可入未来测试划分)。

**诚实声明**:
- 5 任务全部需要 typed nested access + 列表元素引用
  (current_ir_representable = 0/5)— 当前 IR 尚不能编译它们,
  集成前置条件明确
- 测试划分冻结推迟: 目标 ≥15 任务,现有 5 全部留 TRAIN;
  9B-2/9B-2.2 错误分析涉及的 14 任务永久排除出测试划分
- 泄漏检查: task_id 唯一 ✓;(domain, source_task_id) 唯一 ✓;
  τ³ 无模板 id,source_task_id 唯一性是可得的最强保证

**其余 48 任务阻塞分布**: AMBIGUOUS-only 22、ENV_STATE-only 8、
USER_LATENT-only 4、混合 14(其中 6 个 gold 本身不可重放)。

## Task 6 — GO / NO-GO

**NO-GO 维持**。阻塞与解锁路径:
1. **Gold 规模**: 5 << 15。扩大途径(按证据强度): AMBIGUOUS 22 条中
   多为跨订单回显的通用值(country/cabin 类)— 需任务级人工裁决或
   交互阶段建模;ENV_STATE 8 条多为压缩 gold(补全 lookup 前缀可转
   runtime_ref,已在 retail_25/0 上验证可行)
2. **IR 集成**: typed path + 列表元素引用进 validator/lowering
   (设计已批准,原型 28 测试全过)
3. **Registry 修正**: 用 real_schema_observations 重建输入 schema
4. GPU 三臂重评在 1-3 完成前不启动 — 现有 0/121 的 v1 口径已是
   可用的悲观基线

## 交付物

- `results/phase9/tau3_gold_compiled_v1.json` — 逐参数定源重建程序
- `results/phase9/tau3_gold_execution_validation.json` — 双环境执行对照
- `results/phase9/tau3_binding_exact_v1_manifest.json` — v1 冻结清单
- `src/eval/gold_binding_scorer.py` + `tests/test_gold_binding_scorer.py`
- `src/compiler/field_path.py` 加固 + `tests/test_field_path.py`(22 例)
- `scripts/reconstruct_executable_gold.py` / `build_tau3_v1_manifest.py`
- `scripts/ssh_config.py`(环境变量 + 主机指纹 pinned)+ 处置报告
