# Skill ISA Specification — v0.1

> Skill ISA 是"目标机的指令集"：描述系统**能做什么**（能力层），
> 不描述某个具体任务怎么做（那是 TaskIR 的职责）。

---

## 1. 定位与两层 lowering

```
xLAM / tool-use 数据中的具体工具调用          （如 google_flight_api.search）
        ↓  lifter: tool → semantic skill      【第一层 lowering】
TaskIR 节点引用 semantic skill                （如 SEARCH domain=flight）
        ↓  scheduler / runtime: skill → executor 【第二层 lowering，绑定】
执行单元                                      （tool / small LM / python / db）
```

约定：

- **工具名永不进入 TaskIR 语义部分**，只保留在 `meta.provenance`；
- 一个 skill 可绑定多个 executor，绑定决策属于 scheduler/runtime（v0.1 只做模拟）；
- Skill ISA 的变更（增删 skill）必须同步更新本文件与 `src/isa/registry.py`。

---

## 2. SkillSpec 字段

| 字段 | 说明 |
|---|---|
| `name` | 唯一指令名（大写） |
| `class` | 分类：`io` / `retrieval` / `transform` / `compute` / `lm` / `action` / `control` |
| `input_types` | 位置参数类型列表（配合 `min_inputs` / `max_inputs` 支持变参） |
| `output_rule` | 输出类型规则：`FIXED:t`、`SAME_AS:i`（同第 i 个输入）、`ITEM_OF:i`（List 的元素类型） |
| `resource_class` | 名义资源类别：`api` / `python` / `lm` / `db` / `runtime` |
| `cost` | 名义成本：`latency_ms`、`tokens_in`、`tokens_out`、`flops`、`calls` |
| `executors` | 候选执行单元（示例性，v0.1 不做真实绑定） |

代码实现：`src/isa/registry.py`（唯一权威来源，本文件与其同步）。

---

## 3. 指令表 v0.1

### 3.1 io

| 指令 | 签名 | 资源 | 名义延迟 |
|---|---|---|---|
| `LOAD` | `(Str) -> Any` | python | 20ms |
| `SAVE` | `(Any) -> Any` | python | 20ms |

### 3.2 retrieval

| 指令 | 签名 | 资源 | 名义延迟 |
|---|---|---|---|
| `SEARCH` | `(query: Str, corpus?: Any) -> List[Any]`，`params.domain` 指定领域 | api | 120ms |
| `FETCH` | `(url: Str) -> Str` | api | 150ms |
| `QUERY_DB` | `(query: Str) -> Table`，`params.table` | db | 40ms |

### 3.3 transform

| 指令 | 签名 | 资源 | 延迟 |
|---|---|---|---|
| `FILTER` | `(List[T]) -> List[T]`，`params.predicate` | python | 5ms |
| `TRANSFORM` | `(List[T]) -> List[T]`，`params.op` | python | 10ms |
| `EXTRACT` | `(Any) -> Any`，`params.fields` | python | 8ms |
| `DEDUP` | `(List[T]) -> List[T]` | python | 3ms |
| `SORT` | `(List[T]) -> List[T]`，`params.key, order` | python | 4ms |
| `JOIN` | `(List, List) -> List[Any]` | python | 8ms |
| `MERGE` | `(T, T) -> T`（并行分支汇聚，同类型） | python | 2ms |

### 3.4 compute

| 指令 | 签名 | 资源 | 延迟 |
|---|---|---|---|
| `ARGMIN` / `ARGMAX` | `(List[T]) -> T`，`params.key` | python | 1ms |
| `MIN` / `MAX` | `(List[T]) -> T`，`params.key` | python | 1ms |
| `SUM` / `AVG` | `(List[Float]) -> Float` | python | 1ms |
| `COUNT` | `(List[T]) -> Int` | python | 1ms |
| `CALCULATE` | `(*Float) -> Float`，`params.expr` | python | 1ms |
| `COMPARE` | `(Any, Any) -> Bool`，`params.op` | python | 1ms |
| `CONVERT` | `(Float) -> Float`，`params.from, to`（汇率等） | api | 80ms |

### 3.5 lm（小模型执行单元）

| 指令 | 签名 | tokens(in/out) | 延迟 |
|---|---|---|---|
| `GENERATE` | `(*Any) -> Str`，`params.role` | 800/300 | 600ms |
| `SUMMARIZE` | `(Str) -> Str` | 2000/300 | 400ms |
| `TRANSLATE` | `(Str) -> Str` | 1000/1000 | 500ms |
| `CLASSIFY` | `(Any) -> Str` | 600/5 | 200ms |
| `EXTRACT_ENTITIES` | `(Str/Any) -> List[Entity]` | 1000/200 | 300ms |
| `CODEGEN` | `(*Any) -> Str` | 1000/600 | 900ms |
| `PLAN` | `(Str) -> Json` | 600/400 | 500ms |

### 3.6 action

| 指令 | 签名 | 资源 | 延迟 |
|---|---|---|---|
| `SEND` | `(Str) -> Any`，`params.channel` | api | 100ms |
| `EXEC_ACTION` | `(*Any) -> Any`，`params.action`（兜底动作指令） | api | 150ms |

### 3.7 控制指令（IR 级，不属于 Skill ISA 商店，但共用 SkillSpec 结构）

| 指令 | 签名 | 资源 | 延迟 |
|---|---|---|---|
| `VERIFY` | `(*Any) -> Bool`，`params.check` | lm | 50ms（400/1 tokens） |
| `SELECT` | `(Bool, T, T) -> T` | runtime | ~0 |

---

## 4. 成本模型（名义值）

- `flops_per_token = 2 × n_params`：
  - 2B 小模型执行器：`4e9` FLOPs/token；
  - 70B 单次直答 baseline：`1.4e11` FLOPs/token。
- lm 类指令的 flops = `(tokens_in + tokens_out) × 4e9`；
- python 类指令名义 `1e6` FLOPs；api/db 类 flops 记 0（延迟为主）。
- **所有成本是名义常量**，供 simulator 做 workload 对比与 cost report，不是实测值。
  真实 profiling 接入是后续工作（scheduler / real runtime 阶段）。

---

## 5. Lifting 映射约定（工具域 → semantic skill）

lifter（`src/lifter/toolmap.py`）按模式匹配工具名与参数键，映射到语义指令：

| 工具特征（正则） | semantic skill | params |
|---|---|---|
| `flight.*(search|find)` 等 | `SEARCH` | `domain=flight` |
| `hotel` | `SEARCH` | `domain=hotel` |
| `weather` | `SEARCH` | `domain=weather` |
| `news` | `SEARCH` | `domain=news` |
| `map/route/direction` | `SEARCH` | `domain=maps` |
| `stock/price/market` | `QUERY_DB` | `table=market` |
| `currency/convert/exchange` | `CONVERT` | `from/to/amount` |
| `calculat/math/evaluate` | `CALCULATE` | `expr` |
| `send.*(mail/email)/email` | `SEND` | `channel=email` |
| `sms/text/message` | `SEND` | `channel=sms` |
| `calendar/event/meeting` | `EXEC_ACTION` | `action=create_event` |
| `book/reserve/order/purchase` | `EXEC_ACTION` | `action=book` |
| `translate` | `TRANSLATE` | `lang` |
| `summar` | `SUMMARIZE` | — |
| 未识别（兜底） | `EXEC_ACTION` | `action=<动词短语>`，原名记入 provenance |

规则：映射输出**只含语义**；原始工具名、原始参数保留在 `meta.provenance.tools`，
供后续审计与反向数据增强。
