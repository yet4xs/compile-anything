# 架构有效性验证实验设计

> 核心问题：Compile Anything 分层编译器架构是否优于直接 Function Calling？
> 当前缺陷：Track A 89.20% 是直接 SFT → 函数调用，不经过 TaskIR/Validator/Resolver/Composer。

---

## 一、缺失的对照矩阵

| # | 系统 | 路径 | 证明什么 | 状态 |
|---|---|---|---|---|
| A1 | Qwen-7B zero-shot | instruction → 函数调用 | 基座能力下限 | ❌ 未测 |
| A2 | **7B Direct FC SFT** | instruction → 函数调用 | 强非编译器基线 | ✅ **89.20%** |
| A3 | **Compile Anything Modular** | instruction → Canonicalizer → Resolver → Composer → TaskIR → Bridge → 函数调用 | 分层架构有效性 | ❌ 未测 |
| A4 | Oracle CapIR + Composer | instruction + oracle标注 + oracle选择 → Composer → TaskIR → Bridge | 编译器上界 | ❌ 未测 |

**关键对比**：
- A2 vs A3：直接 SFT vs 分层编译器（同基座、同数据量）
- A3 vs A4：学习前端 vs oracle 前端（前端误差代价）
- A3 在 τ³ vs A2 在 τ³：跨域时分层是否优于直接（已知：直接=0.25%，分层=60.23%）

---

## 二、A3 实现方案

```
输入：BFCL instruction + function schemas
  ↓
[1] Canonicalizer（冻结 s42）
    将每个 function schema → canonical_skill
  ↓
[2] Resolver（冻结 R2 s42）
    对每个 (task, capability) 对打分
    score > 0 → selected
  ↓
[3] Composer（冻结 C2 s42）
    输入 task + selected capabilities
    输出 TaskIR
  ↓
[4] Validator（V1-V6）
    静态验证 TaskIR
  ↓
[5] Bridge（已有 bfcl_bridge.py, 1170 行）
    TaskIR → function call 格式
  ↓
输出：函数调用（与 A2 相同格式，用同一 evaluator 评分）
```

---

## 三、A4 实现方案

```
输入：BFCL instruction + function schemas + ground truth selection
  ↓
[1] Oracle Canonicalizer
    使用冻结 BFCL oracle 的 canonical 标签
  ↓
[2] Oracle Resolver
    使用 ground truth 的函数选择
  ↓
[3] Composer（同 A3）
  ↓
[4] Bridge（同 A3）
  ↓
输出：函数调用
```

---

## 四、评测协议

### 内部评测（当前方案）
- 同一 held-out 20% split（seed=999）
- 同一简化 parser
- 同一类别分解
- **可比性**：A1-A4 全部用同一 evaluator

### 官方 BFCL 提交（可选升级）
- 使用 gorilla 官方 eval harness
- AST 精确匹配
- 包含 executable accuracy
- 官方类别加权
- **限制**：需要模型可通过 API 访问或本地运行官方代码

---

## 五、预期结果与论文论证

### 预期分数

| 系统 | BFCL held-out | τ³ SemRecall | 分析 |
|---|---:|---:|---|
| A1 zero-shot | ~30-40% | ~0% | 基座有一些 FC 能力但无跨域 |
| A2 Direct SFT | **89.20%** | **~0.25%** | 域内极强但跨域崩溃 |
| A3 Modular | ~60-75%? | **23.02%** | 域内可能有损但跨域大幅恢复 |
| A4 Oracle+Comp | ~75-85%? | — | 架构上界 |

### 论文论证策略

A3 的 BFCL 分数**可能低于** A2（因为管线开销），但这不是失败：

```
直接 SFT (A2)：域内 89.20% + 跨域 0.25% = 高方差 specialist
分层编译器 (A3)：域内 ~70%? + 跨域 60.23% = 低方差 generalist
```

**价值主张**：
1. **跨域鲁棒性**：A3 在 τ³ 恢复 240×，A2 无法做到
2. **静态验证**：A3 的每一步输出可验证（V1-V6），A2 无法验证
3. **可审计性**：三个 bug 通过分层审计发现，直接 SFT 无法做到
4. **成本模型**：A3 支持 scheduling/cost accounting，A2 不支持
5. **独立改进**：各层可独立优化（如升级 Canonicalizer 不影响 Composer）

---

## 六、执行计划

| 实验 | 时间 | 优先级 |
|---|---|---|
| A1 zero-shot 7B on BFCL | ~30 min | 高（便宜） |
| A3 Modular on BFCL | ~2h | **最高（核心对照）** |
| A4 Oracle+Composer on BFCL | ~1h | 高 |
| 官方 BFCL 提交 | 需额外工程 | 中（可选） |

总计：~4h GPU 时间

---

## 七、官方 BFCL V4 评测补充

当前简化评测的局限：

| 方面 | 影响 | 严重性 |
|---|---|---|
| AST vs parser | 我们的宽松匹配可能高估 2-5pp | 中 |
| 无执行评测 | 不测真实 API 调用结果 | 低（多数论文也不测） |
| multi-turn 简化 | 缺少完整对话状态 | 中 |
| agentic 未覆盖 | web_search 等未评 | 低（可选类别） |
| 类别加权 | 简单平均 vs 官方权重 | 低 |

**建议**：先用内部评测完成 A1-A4 对照，然后如果时间允许，用 gorilla 官方代码重新评测 A2 和 A3。
