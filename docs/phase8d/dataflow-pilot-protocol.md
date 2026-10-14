# Phase 8D-1A.1: Dataflow Dataset v2 — Pilot Protocol

## 语料统计(data/runtime_dataflow_v2/)

| 指标 | v1 | v2 |
|------|----|----|
| 生成 | 195 | **275** |
| 去重后 | 97 | **204** |
| 严格验证通过 | 195/195 | **275/275** |
| Runtime 执行成功 | 195/195 | **275/275** |
| 不可观察 literal | ~50(hidden) | **0** |
| DAG 保真(data+order) | 未审计 | **275/275 (100%)** |
| 多余串行化边 | 存在 | **0** |
| 工具候选集 | 仅 gold | **gold + 3 干扰工具/任务** |
| 测试隔离 | 随机 | **拓扑级 held-out** |

## 切分

| | train | dev | test |
|--|-------|-----|------|
| 数量 | 134 | 26 | 44 |
| 拓扑 | 9 种(train可见) | 同 train 子集 | **仅 held-out 2 种** |

**Held-out 测试拓扑**: `repeated_tool`(同工具多次调用不同参数)、`fan_out_in`(扇出-扇入+变异)——训练集完全不见这两种结构。

## 深度分布
3步(69) / 4步(76) / 5步(39) / 6步(20)

## 动态参数
每任务 runtime_refs: 1-7 个(全部来自前驱工具输出)

## 修复对照

| v1 缺陷 | v2 修复 |
|---------|---------|
| 随机金额不在指令中 | 所有金额在指令文本中明写($XX,XXX) |
| read_only 被串行化 | effect_class 从 registry 查找,不再退化为 None |
| Schema 降级为 Any | Str/Int/Float/Bool 全层保持 |
| 候选池=gold tools | 每任务 3 个签名相似的干扰工具 |
| 拓扑无隔离 | 2 种拓扑完全 held-out |

## Pilot 评测协议

### 双协议
**A. Oracle Pool**(只测绑定): offered = gold tools
**B. Distractor Pool**(测选择+绑定): offered = gold + distractors

### 指标
1. Validator Pass
2. Tool Identity Exact Match
3. Runtime Binding Edge F1(DATA 边精确匹配)
4. Runtime Parameter Provenance(下游参数实际来自上游输出)
5. DATA/ORDER DAG Accuracy
6. End-to-End Mock Execution Success
7. Strict Joint Semantic Match
8. Generated Tokens / Latency

### 基线
- **M0/M1 zero-shot**(现有 s43 checkpoint,不重训)
- **Oracle Lowering**(Skeleton → Lowering,测编译器上界)

## GO/NO-GO GPU 训练裁决

**条件**:
1. ✅ literal_observable = 0
2. ✅ DAG 保真 100% (275/275)
3. ✅ 严格验证+执行 100%
4. ✅ 拓扑级隔离(held-out 测试)
5. ✅ 干扰工具池(3/任务)
6. ✅ Typed Schema 全层保持
7. ✅ v0.1 回归 100/100

**裁决: GO — 可启动 M0/M1 小规模 QLoRA 可行性训练**
