# Phase 7C-2.1: Paper Evidence Hardening

**日期**: 2026-10-12
**基线**: `87673ad`(7C-2)→ 本阶段提交
**范围**: 不改变已冻结的 G0/G1/G2 逐样本结果;修正统计推断、测试执行、声称口径,新增跨来源验证与解码开销优化

---

## Task 1 — 测试修复(`4aade47`)

- `__main__` 运行器移到文件末尾:**此前定义在块后的 4 个测试从未被执行**;现在 24 个全部执行
- 无效断言 `assert x is not False or True` 替换为真实检查(permissive feed + 掩码空集)
- **新执行的测试抓到一个真 bug**:G3 类型 trie 未应用于 `S_INPUT_COLON`(类型首字符状态)——已修复(仅 level≥4 路径,G0/G1/G2 不受影响)
- pytest 验证:phase7 三个测试文件 **39 项全部通过**

## Task 2 — 统计推断修正(`f6b3200`)

原始逐样本结果未改动;重算输出 `results/phase7/clustered_statistics.json`:

### 预注册对比(manifest 中预设;Holm 校正基于这 4 个)

| 对比 | 分 seed w/l(s42/s43/s44) | pooled w/l | 原始 p | **Holm p** | 聚类 bootstrap |
|------|--------------------------|-----------|--------|-----------|----------------|
| G1−G0 | 0/0, 22/0, 2/0 | **24/0** | 1.2e-07 | **4.8e-07** | +5.3pp CI95=[3.6, 7.3] |
| G2a−G1 | 0/0, 2/0, 0/0 | 2/0 | 0.50 | 1.00 | +0.4pp [0.0, 1.1] |
| G2−G2a | 0/0, 3/0, 2/0 | 5/0 | 0.0625 | 0.1875 | +1.1pp [0.0, 2.7] |
| G3−G2 | 0/0, 1/0, 0/0 | 1/0 | 1.00 | 1.00 | +0.2pp [0.0, 0.7] |

### 事后合成对比(明示标签,不参与校正)

| 对比 | pooled w/l | p | 聚类 bootstrap |
|------|-----------|-----|----------------|
| G2−G1 | 7/0 | 0.0156 | +1.6pp CI95=[0.44, 3.11] |

**聚类 bootstrap**:重采样任务 id、保留其全部 3 个种子记录(同一任务跨种子相关)。450 条 seed-task 记录共享 150 个任务,普通 bootstrap 会低估方差——聚类版 CI 更宽,以上均已采用聚类口径。

### 零退化分母(替代此前不严谨的"2412 对")

精确定义:对每个约束臂 X ∈ {G1, G2a, G2, G3},在 (种子, 任务) 上与 G0 同时有记录者计为一对,退化为"G0 正确且 X 错误"。分母 = 4 臂 × 3 种子 × 150 任务 = **1800 对,退化 0**。

**结论口径**:G1−G0 经多重比较校正与聚类校正后依然稳固;G2−G2a / G2−G1 为积极信号但未达常规显著水平,表述为 "additional positive but modest improvement beyond grammar-constrained decoding, motivating further evaluation"。

## Task 3 — 结果标注冻结(`f6b3200`)

- 确认集标注:`held-out-from-decoder-development, oracle-capability-conditioned`
- 声明范围:已知正确工具集时在线约束改善 Composer 的结构与操作序列准确率;**不主张**跨域泛化、全管线准确率、任务执行成功率
- 种子表述:三个被测种子上离散程度明显降低(非普适方差主张)
- 强种子:zero accuracy regression,**非** zero runtime overhead(P50 仍 1.8×)

## Task 5 — 解码开销优化

三项优化,语义完全等价(自由区改用禁集掩码、状态键不变复用张量缓存、`allowed()`/键构造去重):

- **字节一致性:12/12 样本 G1/G2 输出与冻结结果逐字节相同**
- 端到端延迟(P50,s43 确认集 12 样本):G1 16.04→14.63s,G2 14.28→12.95s(×1.10)
- **约束相对开销:1.8× → 1.38×**(G0 基线 9.35s);吞吐 13.1 tok/s(G0 ~24 tok/s)
- 剩余开销主要在每步 `argmax().item()` 的 GPU 同步与 Python 逐步调度;进一步压缩需向量化批内解码,超出本轮范围

## Task 4 — 跨来源验证(τ³,运行中)

设计:τ³(约束器开发中从未使用)× 两协议 × G0/G1/G2:
- **Protocol A(oracle capabilities)**:能力块 = 该 case 的 gold 参考动作工具(canonicalizer 标注)
- **Protocol B(learned)**:完整冻结模块化链(canonicalizer→resolver→composer,同 6B-3,无任务级 gold)
- 指标:Parse / Valid / EXEC_ACTION 输出率 / EXEC_ACTION 参考召回(与 6B-3 的 60.23% 基线同口径)
- 抽样:100 任务按域分层(airline/banking/retail/telecom),seed=20261012

{{CROSS_SOURCE_RESULTS}}

## 论文定位(按 advisor 分层)

1. **系统架构层**:TaskIR + Skill ISA + Validator + Runtime
2. **方法层**:动态 SSA/操作数/类型约束进入在线生成——初步证据优于纯文法,幅度较小(本文件 Task 2 口径)
3. **实验发现层(最强)**:不重训练,在线编译器约束改善弱种子 E2E 并缩小被测种子间离散;跨来源验证见 Task 4

Task 6 裁决(跨来源结果出来后):若 G1 明显提升、G2 相对 G1 可重复、learned 管线无严重退化、延迟可接受 → Typed SSA 升级算法贡献;否则定位为可靠性增强模块,核心创新转向可靠性/成本感知的联合编译优化。
