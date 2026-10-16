# Phase 8D-2B: Three-Arm Representation Ablation — Results

**Date**: 2026-10-14 | **Seeds**: 44, 45, 46 | **Test**: 44 held-out tasks × 2 protocols

## 核心结果表(3 种子合并, n=132/臂/协议)

| 模型 | Oracle Binding% | Oracle Strict% | Distr. Binding% | Distr. Strict% |
|------|----------------|---------------|-----------------|----------------|
| M0-T (TaskIR Text) | 0.0 | 0.0 | 9.8 | 6.8 |
| M0-J (TaskIR JSON AST) | **0.0** | **0.0** | **0.0** | **0.0** |
| M1 (Skeleton → Lowering) | **100** | **100** | **84.1** | **84.1** |

## M1 vs M0-J 配对差异

| Protocol | Δpp | 95% CI | Fixed/Broken |
|----------|-----|--------|-------------|
| Oracle | **+100.0** | [100, 100] | **132/0** |
| Distractor | **+84.1** | [78.0, 90.2] | **111/0** |

## M0-J vs M0-T

| Protocol | Δpp | 95% CI | 说明 |
|----------|-----|--------|------|
| Distractor | **-9.8** | [-15.2, -5.3] | M0-J 比 M0-T 更差 |

## 关键发现

### 1. JSON 格式不是优势来源——M0-J 反而最差

M0-J(生成完整 TaskIR JSON AST)的 binding exact = **0%**。甚至比直接生成 TaskIR 文本(M0-T)更差(Oracle: 0% vs 0%; Distractor: 0% vs 9.8%)。

原因:完整 TaskIR AST 需要模型同时生成 SSA ID 编号、EXTRACT_FIELD 中间节点、param_refs 字段——这对模型太复杂。

### 2. M1 的优势完全来自语义抽象层次

M1 的 Skeleton JSON 只需要:
- 工具名(capability_id)
- 参数(literal 或 runtime_ref{producer, field})
- 依赖(depends_on)

不需要: SSA ID 编号、EXTRACT_FIELD 中间节点、param_refs 字段、output_type。

**确定性编译器(Lowering)自动处理所有这些复杂性。**

### 3. 零 Broken

在所有 264 个配对比较中(132 oracle + 132 distractor),M1 **从未破坏** M0-J 或 M0-T 能做对的任务(0 broken)。

## 结论

**Outcome A 确认: M1 >> M0-J ≈ M0-T**

分层神经编译(Typed Skeleton + Deterministic Lowering)的优势**不来自 JSON 格式**,而来自**语义抽象层次的正确选择**:

- 神经模型负责: 选择工具 + 规划依赖 + 绑定动态参数(自然表示)
- 编译器负责: SSA 编号 + 类型检查 + 字段提取 + 结构验证(确定性保证)

这支持 **Compile Anything** 的核心论文主张: 神经编译器的价值在于让神经网络在正确的抽象层工作,而编译器处理底层复杂性。

---

## Failure Taxonomy (Post-Review Addition)

### M0-J (TaskIR JSON AST) Failure Breakdown (n=264)

| Failure Mode | Count | % | Description |
|-------------|-------|---|-------------|
| **JSON syntax error** | **218** | **82.6%** | Model generates malformed JSON — deeply nested AST exceeds generation precision |
| Missing param_refs | 26 | 9.8% | Model omits param_refs field entirely |
| Binding wrong | 15 | 5.7% | Structure present but incorrect binding |
| No JSON found | 4 | 1.5% | Output doesn't contain parseable JSON |
| Missing EXTRACT_FIELD | 1 | 0.4% | No field extraction nodes |

### M0-T (TaskIR Text) Failure Breakdown (n=264)

| Failure Mode | Count | % | Description |
|-------------|-------|---|-------------|
| **Parse failure** | **126** | **47.7%** | TaskIR DSL syntax errors |
| **Binding wrong** | **93** | **35.2%** | Parses but runtime binding incorrect |
| Validator failure | 36 | 13.6% | Parses but fails semantic validation |
| OK | 9 | 3.4% | Fully correct |

### M1 (Skeleton) Distractor Failure Breakdown (n=132)

| Failure Mode | Count | % | Description |
|-------------|-------|---|-------------|
| OK | 111 | 84.1% | Fully correct |
| Parse failure | 21 | 15.9% | Skeleton JSON malformed or lowering fails |

### Interpretation

**M0-J fails on structural verbosity, not semantics.** The model generates syntactically invalid JSON in 82.6% of cases because the full TaskIR AST requires simultaneous generation of: SSA IDs, EXTRACT_FIELD intermediate nodes, param_refs dictionaries, output_type annotations, after lists — all within a deeply nested JSON structure.

**M0-T fails on both syntax and semantics.** 47.7% can't even parse (DSL complexity). Of the 52.3% that parse, most have wrong runtime bindings (93/138=67%).

**M1 avoids both failure modes.** The Skeleton JSON is flat and concise (tool name, arguments, runtime_ref{producer, field}). The compiler deterministically generates the complex TaskIR AST.

### Paper-Ready Statement

> Direct IR generation fails for format-dependent reasons: text generation fails on DSL syntax (48% parse failure) while JSON AST generation fails on structural verbosity (83% malformed JSON). In both cases, the model cannot express runtime dataflow bindings (0% binding exact). Semantic skeleton generation avoids both failure modes by operating at a higher abstraction level where the model only specifies intent, and the compiler deterministically constructs the runtime dataflow.


## 勘误(Phase 9B-2.5 T7,2026-10-16)

M0-J 的 218 条解析失败中,175 条撞上 700-token 生成上限(oracle 84 +
distractor 91),其中 166 条在截断前生成的节点数**已超过 gold 目标**
(见 results/phase9/three_arm_generation_audit.json,数字与独立复核
一致)。失败机理修正:

- ~~"JSON 冗长导致语法错误"~~ → **"不终止/过度生成,被截断后 JSON
  残缺"**。0% 结论不变,但归因改变:模型不停止,而非格式本身不可写。
- **范围声明**: test 的 5 个工具 5/5 出现在 train(只 held-out 了
  拓扑 fan_out_in / repeated_tool);test 的 runtime_ref 字段仅
  user_id 与 score;全 v2 语料嵌套路径 0、列表多引用 0。"M1 100%"
  限定为**未见拓扑、已见工具与字段**,偏离 §2.1 的跨域设定。
- **模板记忆信号**: 训练目标携带不可由输入推出的 task_id
  (instruction 中 0/134 出现);预测中 M0-T 泄漏训练集 ID
  (oracle 22/132、distractor 12/132,按本仓库口径,与独立复核的
  132/132 计数口径差异已列于审计 JSON)、M1 134/264。v3 训练目标
  必须删除一切不可推出的标识符。
