# Validator V7 Proposal — Effect Ordering

> 状态：**提案**。前置：`docs/effect-system-proposal.md`（IR 表示）。
> 目标：把"action 必须成链"从约定升级为可静态检查的不变量。

## 1. V7 检查集

前提：`registry` 中每个 SkillSpec 增加 `effect_class ∈ {None, "world",
"state"}`；`Node.effect = {"class", "in", "out"}`（可选字段）。

| 编号 | 检查 | 级别 | 说明 |
|---|---|---|---|
| V7a | effect class 合法性 | error `EFFECT_CLASS` | `node.op` 的 ISA `effect_class` 为 `None` 时，节点不得携带 `effect` 字段（纯 skill 挂 token = 表示错误） |
| V7b | effect_in 已定义 | error `EFFECT_UNDEF` | `effect.in` 必须是 ENTRY token（`%e0<class>`）或某个**先定义**节点的 `effect.out`（复用 V2 的先定义后使用规则） |
| V7c | effect_out 唯一 | error `EFFECT_DUP` | `effect.out` 不得与任何节点 id / 其他 effect.out 冲突 |
| V7d | 链线性（无分叉） | error `EFFECT_FORK` | 每个 `effect.out` **至多被一个** `effect.in` 消费。多消费者 = 链分叉 = 顺序歧义（v0.2 不提供 effect-φ，见提案 §4.4） |
| V7e | 链无断点 | error `EFFECT_GAP` | 每个 class 若存在 action，链必须从 ENTRY 开始连续到某个终点；不允许悬空段 |
| V7f | effect 边入 DAG | error `CYCLE` | `effect.in` 边并入 V3 环检测 |
| V7g | world 类禁止 verify-retry | error `EFFECT_RETRY` | `effect_class=="world"` 且 `retry.on != "error"` → 拒绝（不可逆动作不可重放；幂等例外 `hints.idempotent=true` → 降为警告） |
| V7h | token 不进数据流 | error `EFFECT_TYPE` | effect token 不得出现在任何节点的 `inputs` / `guard.cond` / SELECT 中 |

## 2. 判定示例

非法（两个 SEND 无链）：

```
%a = SEND(%msg1)            ; 无 effect 字段，且 class 里有 action
%b = SEND(%msg2)            ; 同上 → EFFECT_GAP（链不存在）
```

非法（分叉）：

```
%a = SEND(%m1)   effect_in=%e0world effect_out=%e1
%b = SEND(%m2)   effect_in=%e0world        ; 链头被二次消费 → EFFECT_FORK
%c = SEND(%m3)   effect_in=%e0world        ; 同上
```

合法：

```
%e0world（ENTRY）
%a = SEND(%m1)   effect_in=%e0world effect_out=%e1
%b = SEND(%m2)   effect_in=%e1        effect_out=%e2
%d = SAVE(%r)    effect_in=%e0state   effect_out=%e1s   ; 另一条链，并行合法
```

## 3. 伪代码

```python
def check_effects(prog, isa, report):
    chain = {}                       # class -> 当前链尾 token
    consumers = defaultdict(int)     # effect.out -> 消费次数
    for node in prog.nodes:          # 规范序 ⇒ 链序即程序序的前缀约束
        spec = isa.get(node.op)
        ec = spec.effect_class if spec else None
        eff = node.effect
        if ec is None:
            if eff: report.err("EFFECT_CLASS", ...); continue
            continue
        if eff is None:                              # action 不成链
            report.err("EFFECT_GAP", node); continue
        if eff.class != ec:  report.err("EFFECT_CLASS", ...)
        head = f"%e0{ec}"
        if eff.in == head:
            chain.setdefault(ec, head)
        elif eff.in in defined_tokens:               # V7b 已定义
            pass
        else:
            report.err("EFFECT_UNDEF", node); continue
        if eff.in != chain.get(ec, head):
            report.err("EFFECT_GAP", "链断点/乱序", node)
        consumers[eff.in] += 1
        defined_tokens.add(eff.out)
        chain[ec] = eff.out
        # V7g / V7h 在主 pass 中顺带检查
    for tok, n in consumers.items():
        if n > 1: report.err("EFFECT_FORK", tok)
```

（V7b 的"先定义后使用"并入现有 V2 循环零成本；V7f 并入现有 DFS。）

## 4. 向后兼容与迁移

| taskir_version | 行为 |
|---|---|
| `0.1`（存量 1105 条数据） | action 无链 → **警告** `EFFECT_UNSPECIFIED`（不破坏现有数据集） |
| `0.2`（effect 落地后） | action 无链 → **错误**；数据需经 lifter/生成器再生成补链 |

迁移步骤：IR 加字段 → validator 双模式 → `gen_synthetic`/`xlam lifter`
给 action 节点自动补链（每个 action 顺序消费链尾）→ 数据再生成 →
fuzzer 增加 effect 链拓扑不变量（线性、无分叉、ENTRY 连续）→ 升版本。

## 5. 测试计划（实施时）

1. 合法链 / 无链（0.1 警告、0.2 错误）/ 分叉 / 断点 / 乱序 / 纯 skill 挂
   token / world-retry / token 进 SELECT —— 8 个单测；
2. fuzzer：builder 以 p≈0.5 注入"忘补链"或"重复消费"变异，validator
   必须全部抓住（这一条是 V7 的 fuzz 验收线，对齐 Phase 2 的
   10000/10000 标准）。

## 6. 已知取舍

- **线性链最保守**：强制全序牺牲部分可证明安全的并行（如两个发往不同
  收件人的邮件）。v0.2 宁可保守——细粒度 effect key（per-recipient、
  per-path）是 v0.3 的 `effect_key` 参数，链按 key 分裂，表示法不变。
- **与 guard 的组合**：guarded action skip 时 token 直通（提案 §4.1），
  V7 静态检查不区分运行时是否 skip——链是静态全序，运行时按需直通。
