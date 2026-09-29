# Semantic Label Audit — 训练前最后一道质量关（Phase 5B-0.46）

> 基线 `073ba99`；corpus v3 冻结不动。审计是**确定性**的：每条 v3 记录
> 经 sample_id 回链 `data/raw/` 的原始 ground truth，逐源比对
> **"TaskIR 标签是否忠实于原始任务"**——不是 parser/validator 合法性。
> 审计期间抓到并修复了 checker 自身的两个 bug（flag 被默认状态吞掉、
> code def-use 硬编码 @task），修复后的数字才是真实数字。

## 总览（31,215/31,215 全覆盖）

| 分类 | 数量 | 占比 |
|---|---:|---:|
| **verified** | 11,056 | 35.4% |
| **suspect** | 20,159 | 64.6% |
| unverifiable | 0 | 0% |

| 层 | n | suspect% |
|---|---:|---:|
| Tier A | 15,755 | **43.9%** |
| Tier B | 15,460 | **85.7%** |

| source | n | suspect% | 主因 |
|---|---:|---:|---|
| xlam | 21,736 | **87.2%** | ARGUMENT_LOSS 18,949 |
| toolbench_static | 1,089 | 53.2% | ARGUMENT_LOSS 579 |
| humaneval | 12 | 33.3% | op 级差异（小样本） |
| mbpp | 32 | 25.0% | 语句式 GT（annotate 为主） |
| spider | 8,034 | **7.6%** | MISSING_CONSTRAINT 638 |
| verilogeval | 312 | **4.5%** | MISSING_CONSTRAINT 14 |

**Top 错误模式**：ARGUMENT_LOSS 19,528 ／ GROUND_TRUTH_AMBIGUOUS 1,016
（注解）／ MISSING_CONSTRAINT 662 ／ WRONG_ORDER 2 ／ EXTRA_ACTION 1。

## 核心发现（真实、可定位、可修）

### 1. ARGUMENT_LOSS 是压倒性主因（19,528 条，全部在 tool-use 源）

xLAM 的参数保全率 **mean 16.7%、median 0.0**——原始 tool call 的参数
值（origin/date/amount/type...）大多没有进入 TaskIR 节点 params。
根因单一且明确：`src/lifter/toolmap.py` 每个 mk() 只保留 2-3 个键字段，
其余丢弃。**这不是 IR 表达力问题，是 toolmap 参数映射覆盖问题**
——schema 层（SkillSpec params 已是开放 dict）完全装得下。

### 2. Spider 7.6% suspect 全部是 MISSING_CONSTRAINT

"how many/cheapest" 等指令关键词在保守 lowering 的 SQL payload 里
没有对应特征（count(/min(）。SQL 本身 byte-faithful（payload 保全
审计通过率 92.4% 的 complement 正是这 638 条）。

### 3. RTL 4.5% + 模板注解

SEARCH(known_bug_db) 与 VERIFY 属 POLICY_DERIVED/TEMPLATE_BIAS（已
注解不计 suspect）；4.5% suspect 是指令含 count/maximum 而模板无
对应 op——真实的模板化局限提示。

## Gate 判定（Task 13）

| gate 阈值 | 实际 | 判定 |
|---|---|---|
| Tier A < 1% suspect | 43.9% | **NO-GO（全量训练）** |
| Tier B < 5% suspect | 85.7% | **NO-GO（全量训练）** |

但按"只修对应 lifter，不重做 corpus"原则，问题定位极其集中：
**单修 toolmap 参数保全即可消除 ~97% 的 suspect**（ARGUMENT_LOSS
19,528 / 20,159）。spider 的 638 条需检查保守 lowering 的 SQL 特征
透传（次优先）。两项都只动 lifter/映射层，不动 IR、不动 split、
不动 external 基准——修复后重跑本审计即得 v3.1。

## 三条可选路径（供 reviewer 决策）

1. **修 toolmap 参数保全 → corpus v3.1 → 重跑审计 → 训练**（预计
   Tier A suspect 降到 ~5% 以下：剩 spider 638 + 少量注解边缘）
2. **先训练再修**：用现 v3 训练一版 baseline，参数保全作为 5B-2
   消融变量（对比"丢参数 vs 全参数"对编译质量的影响）——学术上
   也说得通，但第一轮 baseline 语义噪声偏高
3. **sanity_overfit 先行**（200 条 verified-only 样本已就绪，
   `data/sanity_overfit/samples.jsonl`）：无论选哪条路，先在服务器
   跑 256 条 overfit 验证训练管线本身

## 复现

```bash
python scripts/audit_semantic_labels.py          # 全量审计 (~10 min)
python scripts/run_semantic_judge.py             # 需 --judge，否则 skip
python scripts/analyze_semantic_judge.py         # judge 跑过后分析
```

审计 manifest：`experiments/semantic_audit/manifest.json`（base commit
`073ba99`、corpus 三 split SHA256、audit 代码哈希、judge 样本 ID、
seed=13）。**未删除任何样本；未用 LLM 改写任何标签。**
