# Phase 8B-1A.1: Skeleton Dataset Semantic Integrity v2

**日期**: 2026-10-13 | **基线**: `d457e3f` | **不训练**

## 顾问指正的确认与数字更正

- 全链路失败数 v1 实为 **2,870**(2,655+165+50),此前误报 2,765;合成保留实为 **1,053**(54 是家族组数);train 真实 7+ 步仅 **25 条**——远不足以独自支撑长计划监督,已如实记入 v2 报告
- 问题 A(合成丢数据依赖)、B(空 Schema 假通过)、C(家族隔离只看首工具)、审计自动补 gold 工具、过滤后位置错位、有损 coercion 混入——**六处全部修复**,均有测试

## v2 数据(`data/semantic_skeleton_v2/`,v1 保留未动)

构建(`build_skeleton_corpus_v2.py`):
- **Task 1**:原始 SSA id → 骨架 id 显式映射;经策略节点的依赖标 `unsupported`(降级 semantic_loss),不再静默重指
- **Task 2**:合成保留子图全部数据依赖(重映射 id),仅子任务间加单条 `order_on`;同源组件不跨 split
- **Task 3**:注册表仅来自记录自身 `available_capabilities`(含 description/required/properties);无 Schema 标 `schema_unknown`(不入首轮 SFT);gold 工具不在注册表 → registry 失败
- **Task 4**:`lower_skeleton(strict=True)`——类型不兼容数据依赖 = 编译失败(依赖输出类型从**骨架技能规格**推导,与域无关);缺运行时输入 = unresolved;coercion 仅诊断模式,永不入 strict 数据

分级(tiers):**strict_verified 15,867 / schema_unknown 1,108 / semantic_loss 4,847 / ambiguous 7,832**(全部可归因可复现,失败清单保存)

## 切分与泄漏审计(`audit_skeleton_semantics.py` + `purge_near_dup_v2.py`)

**诚实发现**:工具共现图存在 11,837 条的巨型连通分量——严格 unseen-family 隔离下 dev/test 只能有小规模。

| 切分 | n | 严格通过率 | 说明 |
|------|-----|-----------|------|
| train | 19,933(合成 1,000) | 85.9%(17,123) | 含巨型分量 |
| dev | 285 | 99.3% | unseen-family |
| test | 379 | ~98% | unseen-family |
| test_samefamily | 1,154 | 100% | 巨型分量内留出,**明示非 unseen**(manifest 声明) |

三层泄漏审计(全量,非抽样):
- 源谱系:跨 split 组件 = **0**
- 全工具家族(strict 三切分两两重叠)= **0**;samefamily|train 共享 288 工具(设计内,已声明)
- 全量 8-gram 近重复(J>0.8):检出 127 对 → **清除 124 条**(优先删 samefamily/train 侧)→ 复审 = **0**

## 深度分布(诚实呈现)

train {1:10767, 2:7108, 3:1475, 4:444, 5:92, 6:24, **7+:23**};真实数据最深约 5-8 步极少。**结论:当前监督数据仍以 1-3 步为主,不足以单独证明解决 τ³ 长计划欠规划;8B-1B 训练目标定位于工具身份/依赖/参数的表示收益(短中程),长程规划能力需要后续数据工程(真实多步轨迹)——不在本轮虚增。**

## Task 7 — M0/M1 公平输入冻结

- 同一 Qwen2.5-3B 基座;**同一任务 ID 集合**(train 19,933 的 strict-pass 子集,两模型逐条相同)
- 同一运行时可见 CapabilityIR(记录的 available_capabilities,无 oracle 标签差异);相同 LoRA 配置/优化器;预算按**有效训练 token** 对齐(M0 序列=instruction+TaskIR,M1=instruction+骨架 JSON;分别统计样本数/序列长/token,报告于训练日志)
- 推理成本分别记录:生成 token、延迟、编译(M1 lowering)耗时
- M2 本轮不训练;零训练 P2 保留参照

Go/No-Go:Strict 数据已过语义不变量 + 三层隔离清零 + M0/M1 输入一致 → **满足进入 8B-1B 微调的门槛**。
