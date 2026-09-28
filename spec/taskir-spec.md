# TaskIR Specification — v0.1

> Compile Anything: AI 任务编译器的中间表示。
> 设计目标：**简洁、可序列化、可静态验证、适合训练数据生成**。
> v0.1 不追求完整编译器语法，只定义最小闭环（compile → validate → execute）所需的语义。

---

## 1. 定位与分层

系统分三层，职责严格分离：

| 层 | 表达内容 | 类比 |
|---|---|---|
| **TaskIR** | 任务语义：做什么、数据流、控制流 | LLVM IR |
| **Skill ISA** | 执行能力：操作签名 + 类型 + 成本（见 skill-isa.md） | target ISA |
| **Executor** | 具体执行单元：small LM / Python / API / DB | 硬件功能单元 |

硬性规则：

- TaskIR 节点只能引用 **semantic skill**（如 `SEARCH`、`ARGMIN`）。
- **具体工具名（如 `google_flight_api.search`）禁止出现在语义部分**，只允许记录在 `meta.provenance`。
- 工具 → skill 的映射发生在 lifter（两层 lowering：`CALL_TOOL → SEARCH`）。

---

## 2. 程序结构

一个 TaskIR **module** = 版本号 + 元数据 + 一个 **program**（v0.1 为单 program 模块）。

program 由三部分组成：

1. `inputs`：外部输入（全局值）
2. `nodes`：SSA 风格的扁平节点列表（**规范序**：按拓扑序排列）
3. `output`：终端值（某个节点的 id）

### 2.1 标识符

| 形式 | 含义 | 类比 |
|---|---|---|
| `%name` | 节点定义的 SSA 值（node id 即 value id，一个节点恰好产生一个值） | SSA 寄存器 |
| `@name` | 外部输入，在 `program.inputs` 中声明类型 | global |

合法字符：`[A-Za-z0-9_.-]`。`%1`、`%search_0` 均合法。节点 id 在 program 内唯一。

### 2.2 节点字段

| 字段 | 类型 | 必填 | 语义 |
|---|---|---|---|
| `id` | str | ✓ | SSA 值名，如 `%1` |
| `op` | str | ✓ | semantic skill 名，或控制指令（`VERIFY` / `SELECT`） |
| `inputs` | list[str] | | **数据依赖**：值引用，必须先定义后使用 |
| `params` | dict | | 编译期常量（`predicate`、`key`、`domain`、`expr`…），不参与类型检查 |
| `after` | list[str] | | **控制依赖**：仅顺序约束，无数据流 |
| `guard` | `{cond, expect}` | | **谓词执行**：`cond` 指向 Bool 节点，值 `== expect` 才执行，否则 skip |
| `retry` | `{max_attempts, on}` | | **重试**：`on="error"` 或指向一个 VERIFY 节点 id |
| `output_type` | str | | 可选：显式声明/收窄输出类型（必须与 ISA 推断兼容） |
| `hints` | dict | | 调度提示（`resource_class`、`model_class`…），非语义，可被忽略 |

---

## 3. 类型系统

- **原子类型**：`Str` `Int` `Float` `Bool` `Json` `Table` `Any`（`Unknown` 等同 `Any`）
- **参数化类型**：`List[T]`、`Set[T]`、`Map[K, V]`
- **领域类型**（开放集合）：任意大写开头的类型名，如 `Flight`、`Report`、`Entity`，
  由 skill 签名或节点显式 `output_type` 引入。领域类型只做名义匹配，不做结构推导。

**兼容规则**（保守，用于 validator）：

1. `Any` 与任何类型**双向**兼容；
2. 参数化类型递归比较：`List[Any]` 与 `List[Flight]` 兼容；
3. 其余情况必须字面相等（canonical 化空白后）。

v0.1 不做跨节点类型推断：类型来自 ISA 签名规则 + 显式 `output_type`。

---

## 4. 数据流与控制流

### 4.1 数据依赖（SSA def-use）

- `inputs` 中每个引用必须在当前节点**之前**定义（flat SSA 的规范序不变量）；
- 一个节点恰好产生一个值；没有 void 节点（副作用操作的输出类型为 `Any`）。

### 4.2 控制依赖

- `after: [%x]` 表示"必须在 `%x` 完成后执行"，不传递数据。
- **guard 谓词执行**——分支不用跳转表达，用谓词 + select（类比 GPU predication / VLIW guard）：

```
%c   = VERIFY(%ans)                              ; Bool
%ok  = GENERATE(%ans, @task)  guard(%c == true)  ; 主路径
%fb  = GENERATE(@task)       guard(%c == false)  ; 备路径
%out = SELECT(%c, %ok, %fb)                      ; 分支汇聚（数据流 φ）
```

### 4.3 控制指令（IR 级内建，不属于 Skill ISA）

| 指令 | 签名 | 语义 |
|---|---|---|
| `VERIFY` | `(value: Any, ...) -> Bool` | 验证节点，产出 Bool；`params.check` 描述检查内容 |
| `SELECT` | `(cond: Bool, a: T, b: T) -> T` | 数据流 select/φ，分支汇聚 |

### 4.4 retry（运行时回滚语义）

```json
"retry": {"max_attempts": 3, "on": "%5"}
```

- `on: "error"`：节点执行出错时重试；
- `on: "%v"`（VERIFY 节点）：该 VERIFY 判定为 false 时，运行时**使本节点的值及其所有
  传递下游的记忆失效并重执行**（rollback-lite），直到 VERIFY 通过或耗尽 `max_attempts`；
- validator 要求：`on` 引用的 VERIFY 必须**传递地依赖本节点**（否则重试无意义）。

### 4.5 明确不在 v0.1 的内容

- 循环 / fixpoint 迭代（v0.2 计划：region + back-edge，或 fixpoint 属性）；
- 多 program 链接、函数抽象与调用；
- 文本格式的**解析器**（text 形式由 printer 单向输出；JSON 是唯一 canonical 输入格式）。

---

## 5. 序列化

### 5.1 JSON（canonical 机器格式）

```json
{
  "taskir_version": "0.1",
  "meta": {
    "name": "find_cheapest_flight",
    "provenance": {"source": "handwritten"}
  },
  "program": {
    "name": "find_cheapest_flight",
    "description": "Find the cheapest flight arriving before 8 PM",
    "inputs": [{"name": "@task", "type": "Str"}],
    "nodes": [
      {"id": "%1", "op": "SEARCH", "inputs": ["@task"],
       "params": {"domain": "flight"}, "output_type": "List[Flight]",
       "hints": {"resource_class": "api"}},
      {"id": "%2", "op": "FILTER", "inputs": ["%1"],
       "params": {"predicate": "arrive_time < '20:00'"}},
      {"id": "%3", "op": "ARGMIN", "inputs": ["%2"],
       "params": {"key": "price"}}
    ],
    "output": "%3"
  }
}
```

缺省字段按 §2.2 默认值处理（`inputs/after/params/hints` 为空，`guard/retry/output_type` 为 null）。

### 5.2 文本形式（human / debug / 训练目标格式）

由 printer 单向生成（v0.1 不提供 parser）：

```
; TaskIR v0.1  module=find_cheapest_flight
; task: Find the cheapest flight arriving before 8 PM
inputs: @task: Str

%1 = SEARCH(@task, domain="flight")                    -> List[Flight]
%2 = FILTER(%1, predicate="arrive_time < '20:00'")     -> List[Flight]
%3 = ARGMIN(%2, key="price")                           -> Flight

return %3
```

注解语法：`guard(%c == true/false)`、`retry(on=%v, max=3)`、`after(%x)`、`hint: k=v`。

---

## 6. 规范不变量（validator 强制执行）

| 编号 | 检查 | 说明 |
|---|---|---|
| V1 | 结构 | 字段齐全、id 匹配 `^[%@][\w.-]+$` 且唯一、op 非空 |
| V2 | def-use | `inputs` / `after` / `guard.cond` / `retry.on` 引用均已定义；数据引用先定义后使用 |
| V3 | DAG | data + after + guard 边无环（retry 边是时间回边，**不参与**环判定） |
| V4 | 类型 | 输入与 ISA 签名兼容；显式 `output_type` 与推断兼容；`guard.cond` 为 Bool；SELECT 两分支类型兼容 |
| V5 | skill 可用性 | `op` ∈ Skill ISA registry ∪ {`VERIFY`, `SELECT`} |
| V6 | 控制流 | arity 检查；`retry.on` 指向 VERIFY 且传递依赖本节点；`max_attempts ≥ 1`；`output` 引用已定义节点 |
| W  | 警告 | 死节点（不可达 `output`）、未使用的外部输入、版本不匹配 |

违反 V1–V6 任意一条即 invalid；数据进入 TaskIR dataset 前必须通过全部检查。

---

## 7. 版本与扩展约定

- `taskir_version` 字段声明规范版本，本文档描述 `0.1`。
- **前向兼容约定**：`params`、`hints`、`meta` 是开放字典；新增顶层/节点字段必须可被旧
  validator 安全忽略（未知字段 → 警告，不报错）。
