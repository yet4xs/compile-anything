# BFCL V4 本体论不匹配分析 — 三级语义指标

> 前提：Capability ablation 已确认 capability context 不是主因（8.6% → 8.8%）

## 核心结果表（论文 Table）

| Metric | Full (1,931) | Partial (2,610) | None (155) | All (4,696) |
|---|---:|---:|---:|---:|
| Parse | ~97% | ~97% | ~99% | **97.4%** |
| Validator | 94.6% | 90.3% | 99.4% | **92.4%** |
| **Strict Semantic** | **16.0%** | **5.7%** | 0% | **10.4%** |
| **Ontology-Equivalent** | **23.1%** | **5.7%** | 0% | **13.5%** |
| **Functional Semantic** | **28.3%** | **6.1%** | 0% | **16.0%** |

## 判定：Outcome B

> **模型具有结构泛化能力（92.4% valid），但 semantic planning 没有跨域泛化（仅 16.0% functional）。**

Ontology 等价性只解释了 **3.1pp**（10.4% → 13.5%）的提升——
92%→10% 落差的主要成分不是命名/本体差异，而是**真正的语义规划失败**。

## 关系分类统计

| 关系类型 | 数量 | 说明 |
|---|---|---|
| **INCOMPATIBLE** | **8,008** | 完全不同的 skill 选择（主要来源） |
| EQUIVALENT | 296 | 语义等价（SEARCH↔FETCH 等） |
| SAME_FAMILY_DIFFERENT_OP | 239 | 同族不同操作（如同是 retrieval 但选了不同 skill） |
| EXACT | 218 | 完全匹配 |
| MODEL_MORE_SPECIFIC | 22 | 模型更具体 |

## 模型 Op 分布 vs Oracle 期望（按 BFCL 类别）

| BFCL 类别 | Oracle 期望 | 模型实际 Top-3 | 问题 |
|---|---|---|---|
| live_simple | SEARCH | SEARCH(36%), FETCH(32%), EXEC_ACTION(11%) | 部分选了 FETCH（等价） |
| live_irrelevance | GENERATE（不调用） | **FETCH(46%), SEARCH(27%)** | **模型不知道"不该调用"** |
| memory | GENERATE | **QUERY_DB(55%), FETCH(33%)** | **模型不知道 memory 不可表达** |
| multi_turn | EXEC_ACTION | FETCH(28%), QUERY_DB(24%), SEARCH(22%) | **shell 操作应映射到 EXEC_ACTION** |
| web_search | SEARCH | **FETCH(58%), QUERY_DB(29%)** | 模型选了错误的 retrieval skill |
| parallel | SEARCH | **CALCULATE(39%), FETCH(25%)** | **模型把并行函数调用当作计算任务** |

## 四大失败模式（按影响大小排序）

### 1. Irrelevance detection failure（最大差距）
- 1,124 条 irrelevance 案例，oracle 期望"不调用任何函数"（GENERATE-only）
- 模型 73% 的情况下仍然调用了 FETCH/SEARCH/QUERY_DB
- **模型学会了"总要做什么"，没学会"识别不需要调用"**

### 2. Category-to-skill mapping mismatch
- multi_turn（shell 操作）oracle 期望 EXEC_ACTION，模型选 retrieval 类
- parallel（多函数调用）oracle 期望 SEARCH，模型选 CALCULATE
- **训练数据中没有对应的任务类型模式**

### 3. Memory inexpressibility
- 155 条 memory 案例，oracle 期望 GENERATE（因为 TaskIR 无 memory ops）
- 模型 88% 选了 QUERY_DB/FETCH（试图"执行"但用了错误的 skill）
- **模型不理解 TaskIR 的表达能力边界**

### 4. Retrieval skill 选择混乱
- 在应该用 SEARCH 的场景，模型 32% 选 FETCH
- SEARCH 和 FETCH 在 Skill ISA 中是等价的（都是 retrieval）
- 这部分贡献了 EQUIVALENT 296 次匹配

## 对论文的结论

这比"92% 全成功"更有价值的研究发现：

> **Neural Compiler 学会了 TaskIR 的结构规则（语法+SSA+类型+DAG），这些规则
> 跨数据集泛化（92.4% validator pass）。但语义规划——选择哪个 skill 来表达
> 任务意图——严格绑定于训练数据的任务分布。**
>
> 这证实了 compiler-runtime 分离的价值：结构层是可泛化的通用组件，
> 语义层需要目标域的适配。

## 下一步路线（GPT 框架）

判定为 **Outcome B** → 下一步需要训练 semantic grounding：
1. 扩大训练数据的 skill 多样性（更多不同 tool→skill 映射模式）
2. 加入 irrelevance/negative examples（训练"不调用"的能力）
3. 对目标域做 few-shot adaptation（使用独立的 adaptation split，不污染 test）
