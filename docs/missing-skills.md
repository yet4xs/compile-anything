# Missing Skills — ISA gaps exposed by benchmark lifting (Phase 4)

> 规则：benchmark 需求暴露的 ISA 缺口**记录在此**，不硬塞进 Skill ISA。
> 每项含证据（corpus 覆盖率损失）、建议签名、优先级。v0.2 ISA 扩容
> 时按优先级逐项评审。

## #1 LOOP / 迭代（优先级：最高）

- **证据**：code 源覆盖率 83.4%，全部损失来自 `for`/`while`/递归/分支
  赋值模式（stats.json `unsupported_reasons`: "loop (no LOOP op in
  TaskIR v0.1)"）。真实 HumanEval 大概率更高比例不可表达。
- **需要**：fixpoint 迭代。建议（taskir-spec v0.2 计划的 region 方案）：
  ```
  LOOP %iter (max=K | until=%cond) { region: [nodes] }
  ```
  区域内节点可引用 `%iter` 值；`until` 引用 region 内 VERIFY。语义上
  是受控回边（与 retry 的时间回边不同：retry 重执行同一定义，LOOP 是
  值的版本链）。
- **影响面**：validator（V3 环检测需豁免回边）、runtime（版本化 memo，
  Phase 2 已有 superseded 机制可复用）、scheduler（循环展开/流水）。

## #2 SQL 关系算子（优先级：高，DAC 相关）

- **证据**：sql 源中 GROUP BY/HAVING/JOIN/聚合的样本 `decomposed=false`
  （SQL 整体留在 QUERY_DB 内，IR 层看不到关系算子）。
- **需要**：
  | skill | 建议签名 | 说明 |
  |---|---|---|
  | `GROUP` | `(List[T], keys) -> List[List[T]]` 或 `-> Map[K, List[T]]` | GROUP BY |
  | `TABLE_SCAN` | `(Table) -> List[T]` | 让 QUERY_DB 结果进入算子层（当前 Table 与 List 断裂） |
  | `HAVING` | `(Map[K,V], predicate) -> Map[K,V]` | 分组后过滤 |
- **立场**：不是所有 SQL 都该分解到 IR 层（DB 执行本来就快）——需要
  的是**表达能力存在**，让 optimizer 决定下推还是上提。TABLE_SCAN 是
  关键解锁（它打通 Table→List，FILTER/SORT/AGG 才能作用其上）。

## #3 字符串操作（优先级：中）

- **证据**：code 模式 `loop_str`（拼接/重复）拒绝；HumanEval 高频的
  字符串重建类任务无 op 可映射。
- **需要**：`SPLIT/CONCAT/REPLACE/REGEX_MATCH`（python 类，成本低）。
  是否值得进 ISA 取决于 Phase 5 训练分布，暂列观察。

## #4 EDA 工具链（优先级：中，论文差异化）

- **证据**：rtl 源的 VERIFY 目前是泛型 lm verify（check 字符串描述
  "syntax_and_lint_and_assertion_pass"），没有真实工具语义。
- **需要**（benchmark/README.md 已有草案）：
  `LINT(rtl)->Report`、`SYNTH(rtl,target)->Netlist`、
  `SIMULATE(netlist,vectors)->Waveform`、`FORMAL_CHECK(prop,rtl)->Bool`；
  executors: yosys/verilator/sby。这让 RTL workload 的成本模型真实化
  （tool latency 实测可查）。

## #5 粒度问题（不新增 skill，记录观察）

- **证据**：toolbench 源 `EXEC_ACTION` 兜底占比高（book/create_event/
  playlist 等 12+ 种动作全部塌缩到一个 EXEC_ACTION(action=...)）。
- **风险**：语义粒度过粗会削弱 optimizer 的调度/融合机会。
- **立场**：v0.1 有意保守（宁粗勿错）。当 corpus 中某 action 动词频率
  足够高（stats 可查）时再考虑提升为一等 skill（如 BOOK、SCHEDULE）。

## 统计接口

覆盖率和原因直方图在 `data/compiler_corpus/stats.json`
（`unsupported_reasons` 字段）；换真实数据后此文件即 ISA 缺口的量化
证据，直接支撑 v0.2 ISA 评审。
