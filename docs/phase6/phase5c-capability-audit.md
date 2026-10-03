# Phase 5C Capability 条件化审计 — 字符串迭代 Bug 与证据链修正

> 审计时间：2026-10-04（Phase 6A 期间，CPU audit）
> 产物：`results/phase6/phase5c_capability_audit.json`
> `paper_snapshot_v1` 保持 immutable；本审计为 Phase 6 新增事实，投稿前并入 paper_snapshot_v2。

## 结论先行（三级证据链）

Phase 5C 的 "schema-conditioned training" 实际条件信号是：

| 来源 | n (train) | capabilities 实际内容 | 性质 |
|---|---:|---|---|
| xLAM | 19,700（70%） | **常量 `["EXEC_ACTION"]`** | **bug 产物，零语义信息** |
| toolbench_static | 1,074 | 语义技能视图（\|C\|≈2.71，T⊆C 64%） | 合法的 available-tool 语义 |
| Spider / code / RTL | 7,319 | 空（裸指令） | 无条件 |

**不是 target leak（Case B 排除）**：xLAM 上 C==T 仅 4.82%、Jaccard 0.057——常量与目标
不匹配（19,700 条中只有 950 条目标真的含 EXEC_ACTION）。

## 根因：`semantic_capabilities` 对未解析 JSON 字符串做迭代

`src/dataset/adapters/tooluse.py`：

```python
metadata={"capabilities": semantic_capabilities(rec.get("tools"))}
```

xLAM 原始记录的 `tools` 是 **JSON 编码字符串**（`answers` 得到了 `json.loads` 处理，
`tools` 没有）。`semantic_capabilities` 的 `for t in tool_defs` 因此迭代的是**字符**：

```text
'[' → map_tool fallback → EXEC_ACTION
'n' → fallback → EXEC_ACTION
...（26 个不同字符全部落入 fallback）
dedup → ["EXEC_ACTION"]
```

复现验证（本仓库可跑）：

```text
tools = [{"name": "get_account_info"}, {"name": "search_ticker"}]（JSON 字符串）
字符串迭代 → ['EXEC_ACTION']          # bug 路径，即 corpus v3.1 实况
正确解析   → ['FETCH', 'QUERY_DB']    # 本应是这个
```

## 对 Phase 5C 结果的再解释（不改数字，改解释）

1. **pred/T 1.00→1.75 的机制改写**：训练时"存在 capabilities 块（哪怕是常量
   EXEC_ACTION）↔ 更长计划"的**格式关联**改变了无条件长度先验。不是 capability
   语义信息驱动。——这解释了为什么 schema-only（S）与 depth（D）同样提升长度。
2. **BFCL E2E +0.35pp**：推理时给的是真实多函数 schema（训练中只有 toolbench
   1,074 条近似格式），效果为正但微小——现在看已经是意外之喜。
3. **E5C-S 分类器 boundary F1 0.782 / EXEC_ACTION recall 0.0**：70% 训练数据的
   常量 EXEC_ACTION 上下文并未教会模型输出 EXEC_ACTION——进一步佐证 EXEC_ACTION
   失败不在"没见过标签字符串"。
4. **internal matched OpSeq 94.11 仍成立**：matched 协议下 xLAM 测试样本的
   prompt 同样带常量块（与训练一致），无泄漏，数字有效——但其"条件"的含义是
   格式而非语义。

## 命名修正（paper_snapshot_v2 时执行）

```text
旧：schema-conditioned training / capability-schema conditioning
新：constant-marker + partial semantic-capability conditioning
    （或按审稿人建议 semantic-capability conditioning，并附本审计）
```

论文 §8 必须披露：语料 capabilities 字段的 xLAM 部分为 builder bug 产生的常量；
Phase 5C 的正效应（BFCL +0.35pp、长度先验变化）在剔除语义解释后依然成立，
但其机制解释改为格式/长度先验通道。

## 数据质量教训（论文 Discussion 素材）

这是项目第二次发现"上游 lifter 级 bug 改写下游结论解释"（第一次：参数保全
16.7%→98.9%）。共同模式：**确定性审计可以定位监督信号的真实内容**——正是
compiler 式分层（lifter 可单独审计）使这类发现成为可能，而非灾难。
