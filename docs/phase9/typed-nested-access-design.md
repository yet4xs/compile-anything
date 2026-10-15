# Typed Nested Access — Design (Phase 9B-2.2 Task 4)

**Status**: design + prototype (`src/compiler/field_path.py`) + tests
(`tests/test_field_path.py`, 13 cases, all pass). **Not** wired into the
validator/lowering — production EXTRACT_FIELD semantics unchanged until
this design is accepted.

## 1. Problem

TaskIR v0.2 的 EXTRACT_FIELD 只支持生产者输出的顶层字段。真实 τ³ 数据流
需要嵌套访问:

- `get_user_details` 返回 `User{orders: List[Order], ...}`
- 消费者需要 `orders[1].order_id`(Phase 9B-2.2 重放实证的真实路径)

M1 模型在 `tau3_retail_11` 上自发产出了 `orders[0].order_id` — 语义正确、
IR 无法表达,typed validation 拒绝(fail-closed)。这是编译器表达力缺口。

**同时必须拒绝的**:`find_user_id_by_name_zip.default_address`
(tau3_retail_48)— 生产者返回裸字符串,该字段根本不存在。这是模型规划
错误,不能靠放宽 IR 规则放行。

## 2. 方案比较

### A. 链式类型操作(每个访问一个 IR 节点)

```
%u  = EXEC_ACTION(action=get_user_details) bind(user_id=%ext0)
%o  = GET_FIELD(%u, field="orders")          ; List[Order]
%e  = GET_INDEX(%o, index=0)                 ; Order
%id = GET_FIELD(%e, field="order_id")        ; Str
%p2 = EXEC_ACTION(action=get_order_details) bind(order_id=%id)
```

优点: 每步独立类型检查;与现有 EXTRACT_FIELD 单字段哲学一致;
中间值有 SSA 名字(可复用、可调度感知)。
缺点: 节点数膨胀(每层 × 每索引一个节点);模型要生成正确的多节点链,
错误面更大;调度器/DAG 匹配器都要认识新 op。

### B. 编译期解析的结构化 Path(EXTRACT_FIELD 的 field 参数升级为路径)

```
%ext = EXTRACT_FIELD(%u, path="orders[0].order_id")   ; Str
```

优点: 骨架层(skeleton)不膨胀 — runtime_ref 的 field 直接写
`orders[0].order_id`;lowering 展开为一个内部链或单个路径节点;
对 M1 模型最友好(它已经自发产出这种路径)。
缺点: 单个 token 携带复合语义,解析错误必须在编译期全捕获;
类型检查变成路径遍历而非节点遍历。

### 裁决: B 为骨架接口,A 为 IR 内部表示(hybrid)

- **Skeleton/runtime_ref 层**: 用结构化路径(B)— `field: "orders[0].order_id"`
- **TaskIR 层**: lowering 把路径编译为链式 GET_FIELD/GET_INDEX/EXTRACT_FIELD(A),
  每步携带类型证据,validator 逐节点检查
- 两层之间是**纯机械展开**,无语义判断 — 编译期全部可判定部分在
  parse_path + check_path 完成

这使模型只需表达"我要哪个值"(B),编译器负责证明"这个值如何取到"(A)。

## 3. 类型检查语义(原型已实现)

类型语言沿用 registry 现有原语(Str/Int/Float/Bool/Any)+ 复合:

```
ListT(elem) | DictT(fields) | StructT(name, fields)
```

`check_path(declared_type, steps)` 产出:

- `check="static"` + `result_type`:全路径静态可判定
- `check="dynamic"` + `dynamic_from_step=k`:第 k 步起越过 `Any`,
  **显式动态检查边界**

错误码(编译期): `PARSE_ERROR` / `FIELD_NOT_FOUND` / `FIELD_ON_PRIMITIVE` /
`INDEX_ON_NON_LIST`;运行期(resolve_path): `INDEX_OUT_OF_RANGE` /
`FIELD_NOT_FOUND` on 实际值。

## 4. List[Any] 边界 — 不伪造静态证明

现行手写 registry 把 `orders` 声明为 `List[Any]`(14/18 工具 schema 与
真实返回不符,见 gold-correctness-gate 报告)。此时:

- `check_path` 对 `orders[0].order_id` 返回 `check="dynamic"`,
  `dynamic_from_step=2`(到 `orders[0]` 为静态,`.order_id` 越界进入 Any)
- 编译**成功但降级**: TaskIR 中该链路标注 `dynamic`,runtime 在 resolve 时
  用 resolve_path 重验实际值,失败码与静态失败同构
- **明确禁止**: 把 `Any` 当作"任何字段都存在"的静态通行证

若 registry schema 修正为 `List[StructT(Order,...)]`(Task 3 的建议),
同一路径立即升级为全静态 — IR 不变,schema 质量决定证明强度。

## 5. 测试覆盖(tests/test_field_path.py, 13 例全过)

正例:
- `orders[0].order_id` 在 `User{orders: List[Order]}` 下静态检查通过 → Str
- 顶层字段、运行期解析真实 dict/list 值
- `List[Any]` 显式标记 dynamic 边界(dynamic_from_step=2)

负例:
- `find_user_id_by_name_zip.default_address` → `FIELD_ON_PRIMITIVE`
  (retail_48 的实际失败,必须拒绝)
- 结构体不存在字段、嵌套不存在字段 → `FIELD_NOT_FOUND`
- 原始值上取字段 / 非列表上取索引 → `FIELD_ON_PRIMITIVE` / `INDEX_ON_NON_LIST`
- 畸形路径(`orders[x]`、`orders[0`、`.orders`、`orders..id`、尾点、空)→ `PARSE_ERROR`
- 运行期越界索引 → `INDEX_OUT_OF_RANGE`;实际值缺字段 → `FIELD_NOT_FOUND`

## 6. effect/order 约束不变性

路径访问是纯函数式读取(GET_FIELD/GET_INDEX 无副作用),不引入新的 effect
类。dataflow 依赖仍由 EXTRACT_FIELD 链的 SSA 边表达;ORDER/EFFECT 边语义
与现有 validator 完全一致 — 本扩展不触碰 effect 分类与调度约束。

## 7. 落地步骤(待设计批准后)

1. parser: EXTRACT_FIELD 的 `field="..."` 接受路径语法(经 parse_path 校验)
2. validator: FIELD_NOT_FOUND 检查从"顶层字段集合"升级为 check_path
3. skeleton_lowering: runtime_ref.field 含路径时展开为链式节点
4. registry schema 修正(独立 PR,见 gold-correctness-gate Task 3 结论)
5. 负面回归: 上述负例进入 CI,防止未来放宽时回归
