# Compiler Corpus Dataset Design — Phase 4

## 1. 目标

回答 Phase 4 的核心问题：**TaskIR 是否足以作为真实 AI 任务的 machine
language？** 用四个开源 benchmark 形态的数据源做 lifting，统计覆盖率——
lift 不出来的进 `docs/missing-skills.md`，lift 出来但 validator 拒绝的
暴露语义问题，能执行的进 corpus 供 Phase 5 训练 compiler。

## 2. 数据来源与 provenance（重要）

**当前 corpus 的样本是 schema-faithful synthetic**：按各 benchmark 的
真实记录格式生成（字段、形状、内容分布模仿真实数据），用于离线端到端
验证 lift 管线。它们**不是**真实数据集的抽样。原因：离线环境 + 版权
谨慎。换真实数据零代码改动：

```bash
python scripts/build_compiler_corpus.py --raw-dir data/raw/real
# data/raw/real/{toolbench,code,sql,rtl}.jsonl，schema 见 §3
```

每个 corpus record 的 `raw_trace` 保留完整原始样本，`source.dataset`
标记来源——真实数据混入后 provenance 可区分（synthetic 记录的
`raw_trace` 含生成器特征字段如 `pattern`/`shape`/`kind`）。

## 3. 各源 schema

| source | 记录 schema | lifter |
|---|---|---|
| toolbench (4000) | `{task_id, instruction, trajectory: [{tool, args}]}`（API-Bank/xLAM 步式） | `benchmark/toolbench.py` → toolmap 语义化 |
| code (3000) | `{task_id, prompt, code, pattern}`（HumanEval/MBPP 式） | `benchmark/humaneval.py` → ast 模式抽取（SORT/ARGMIN/SUM/COUNT/FILTER/TRANSFORM/JOIN） |
| sql (2000) | `{task_id, db_id, question, query}`（Spider/BIRD 式） | `benchmark/spider.py` → QUERY_DB(+EXTRACT 分解) |
| rtl (1000) | `{task_id, prompt, rtl, signals[, timing]}`（EDA 任务式） | `benchmark/rtl.py` → LOAD→EXTRACT→SEARCH→CODEGEN+VERIFY |

## 4. Corpus record schema

见 `data/schema/compiler_sample.json`（构建时自动写入的真实首条记录）：

```json
{
  "source":  {"dataset": "toolbench", "id": "tb-00001", "lifter": "toolbench"},
  "input":   {"text": "Find flights from ..."},
  "raw_trace": { ...原始样本... },
  "taskir":  { ...canonical JSON module... },
  "taskir_text": "; TaskIR v0.1 ...",        // 供 text 解析评测与 SFT
  "validation": {"passed": true, "errors": [], "warnings": []},
  "execution": {
    "trace":  {"status": "completed", "events": [{node, op, status, attempt, latency_ms}]},
    "cost":   {"latency_sequential_ms": ..., "critical_path_ms": ..., "energy_j": ...,
               "memory_peak_mb": ..., "lm_calls": ..., "api_calls": ..., "retries": ...}
  }
}
```

约束：**只收录 lift 成功且 validator 通过的记录**（coverage 损失在
stats.json 里如实计数，不进 corpus 污染训练）。split：5% val（seeded）。

## 5. 覆盖率统计（stats.json）

每源：`generated / lifted_and_valid / coverage_pct / unsupported_reasons`
直方图。当前基线（10k 构建）：

- toolbench 100%（toolmap + EXTRACT 桥接覆盖全部链型）
- code 83.4%（拒绝项 = 有意混入的 loop/if-else 模式 → IR 无 LOOP，见
  missing-skills.md #1）
- sql 100%（GROUP/HAVING/JOIN 不强行映射，保守落在 QUERY_DB 内，
  `decomposed` 标记哪些做了算子级分解）
- rtl 100%

## 6. 评测协议（benchmark/compiler_eval.py）

三级闸门，**不用 exact match**：

1. **syntax accuracy**：taskir_text 能否被 `src/ir/parser.py` 解析；
2. **validator pass rate**：解析结果是合法程序（V1–V6）；
3. **semantic execution**：simulator 跑到 completed。

`--predictions` 模式即 Phase 5 模型接口（`{input_text, output_text}` →
同一三级闸门），训练/评测协议今天就已对齐。

## 7. 已知限制

- synthetic 样本的 NL 多样性受模板限制（比真实 ToolBench/Spider 窄）；
- code lifter 只覆盖单 return 表达式的纯函数模式（HumanEval 真实分布
  更难，预计真实覆盖率会显著低于 83%——这本身就是 Phase 4 想测的数）；
- SQL 的保守 lowering 意味着关系算子大多不出现在 IR 层（对比实验里
  "decomposed vs not" 是一个可控变量）；
- corpus 记录含完整 execution trace（compact 事件），文件体积 ~MB 级；
  后续可只保留 cost。
