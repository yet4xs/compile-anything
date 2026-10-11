# Phase 7C-0.1: Parser-Faithful Constrained Generation (v2)

**日期**: 2026-10-09
**状态**: 完成(冒烟 n=6 + 全量 n=30 × 3 seeds)
**产物**: `results/phase7/constrained_generation_v2.json`, `tests/test_constrained_decoder.py`, `src/compiler/constrained_assembly.py`
**前作**: v1 结果(全臂 13.3%)作废 —— 解析器 bug,不是约束效果

---

## 1. 修复内容(对应 advisor 六项审查意见)

| # | 问题 | 修复 |
|---|------|------|
| 1 | v1 约束引擎 SSA ID 正则只认 `%name`,拒绝合法的 `%1` | 统一使用 `taskir.ID_RE`(`^[%@][A-Za-z0-9_.\-]+$`) |
| 2 | v1 自写节点 regex 解析,与官方 parser 不一致 | `src/ir/parser.py::parse_text()` 为**唯一**解析入口;约束在 AST(Node/Program)层面操作 |
| 3 | v1 丢弃原始 `inputs:`/`return`,换装默认值 | 深拷贝原 module,只做最小修复,原对象不可变 |
| 4 | v1 用字符串替换 `output_rule` 推断类型 | 调用 `spec.output_type(input_types)`;检查用 `types.is_compatible()` |
| 5 | 三个 seed 的 PEFT adapter 共享同一 base | 每个 seed **全新加载** Qwen base → 挂 adapter → 生成 → 完全卸载 |
| 6 | 声称 "decode-time 约束" 名不副实 | 明确定位:**post-hoc constrained assembly**(事后约束装配),非 token 级约束解码;v1 文件加弃用头 |

新模块 `src/compiler/constrained_assembly.py`(AST 级三级约束):
- **L1 grammar** — `ID_RE` + 重复定义检查。不改动任何节点,对合法程序必须无损
- **L2 typed SSA** — 未定义引用自动改指类型兼容的最近上游定义(`is_compatible` 选优)、arity 截断到 `spec.max_inputs`、悬空 `return` 重定向到最后节点。全部改动记入 corrections 日志
- **L3 control** — `after/guard/retry` 装配前后必须一致;变化记为 rejection,不做静默修复

## 2. D1 无损性测试(Task 3 交付物)

`tests/test_constrained_decoder.py` — **10/10 通过**(服务器 CPU 运行):

| 测试 | 内容 |
|------|------|
| `test_d1_lossless_on_valid_corpus` | **门控测试**:100 个验证合法的语料程序过 D1,AST 100% 保持 |
| `test_d1_lossless_roundtrip` | parse → to_text → parse → D1 往返无损 |
| `test_numeric_ssa_ids_accepted` | v1 回归用例:节点重命名为 `%1` 后 D1 仍合法 |
| `test_d2_fixes_undefined_ref` | 注入 `%ghost` 引用 → D2 修复 → 合法,原对象未被污染 |
| `test_d2_retargets_dangling_output` | 悬空 return → 重定向 correction |
| `test_d2_arity_truncation` | 超额输入 → 截断到 max_inputs |
| `test_d3_preserves_guard_annotations` | guard 的 cond/expect 装配后原样保留 |
| `test_d3_flags_changed_control` | guard 漂移会被 L3 检出 |
| 其余 2 项 | parse 失败路径、手写 example 全级通过 |

## 3. 主结果:D0-D3 × 三种子(n=30,valid% / opseq%)

| seed | D0 | D1 | D2 | D3 | D1 无损 |
|------|-----|-----|-----|-----|---------|
| s42 | 90.0 / 86.7 | 90.0 / 86.7 | 90.0 / 86.7 | 90.0 / 86.7 | 27/27 |
| s43 | 60.0 / 53.3 | 60.0 / 53.3 | **63.3 / 56.7** | 63.3 / 56.7 | 18/18 |
| s44 | 90.0 / 83.3 | 90.0 / 83.3 | 90.0 / 83.3 | 90.0 / 83.3 | 27/27 |
| 均值 | 80.0 / 77.8 | 80.0 / 77.8 | **81.1 / 78.9** | 81.1 / 78.9 | **72/72 (100%)** |

- **D1 无损 72/72**:所有 D0 合法程序过 D1 后 AST 完全一致(v1 此处全崩,这是 v1→v2 的本质差别)
- **D2 真实收益**:s43 上 60.0→63.3(3 次 arity 截断救回 1 个程序),opseq 同步 +3.3pp;s42/s44 无 SSA 级错误可修,收益为 0
- **种子隔离后不稳定性仍在**:s43=60% vs s42/s44=90%,与隔离前一致 → 排除了 adapter 污染假说,这是真实的训练种子方差
- D3 相对 D2 无额外损耗(装配过程不触碰控制注解)

## 4. s43 错误分类(n=30,为何约束只救回 1 个)

| 数量 | 错误类型 | post-hoc SSA 装配能否修复 |
|------|----------|--------------------------|
| 4 | parse 失败("no nodes found" 等) | ✗ 语法层,需要 token 级约束解码或重生成 |
| 4 | 含 UNDEFINED_REF(多与 STRUCT 叠加) | 部分 — 引用可改指,但叠加的 STRUCT 错误仍在 |
| 4 | 含 ARITY / TYPE_MISMATCH | 部分 — arity 可截断;TYPE_MISMATCH 不可自动定夺 |
| 1 | 纯 ARITY | ✓ 实际被 D2 救回的就是这类 |

**结论**:s43 的 12 个错误中,post-hoc 装配只能修 SSA 层的 1 个;4 个语法失败 + 7 个含 STRUCT 叠加的错误都在其能力之外。

## 5. 诚实结论(回应核心研究问题)

**"编译器静态约束能否降低神经程序生成对训练种子的敏感性?"** — 基于 v2 数据的答案:

1. **正确实现的 post-hoc SSA 约束装配是无损的**(72/72),并带来**小幅但真实**的修复收益(弱种子 +3.3pp,均值 +1.1pp)。这一收益在强种子上为零,因为强种子的错误几乎全在语法层。
2. **它不改变种子方差的量级**(90 vs 63.3)。种子差距的主导错误模式是 parse 失败与结构错误,二者都发生在 token/语法层 —— 这正是 post-hoc 装配(在完整生成之后才介入)原理上无法触及的。
3. **方向性推论**:若要用编译器约束真正攻击种子敏感性,需要在**解码时**强制 TaskIR 语法(token 级 grammar-constrained decoding,如 PICARD/SynCode 思路挂我们自己的 TaskIR 文法)。v2 结果把"哪些错误可被哪层约束修复"做了干净归因,为该方向提供了直接动机。

## 6. 与 v1 的对照

| | v1(作废) | v2 |
|---|---------|-----|
| 解析 | 自写 regex | 官方 `parse_text()` |
| D1 对合法程序 | 破坏(全臂 13.3%) | 100% 无损 |
| 类型推断 | 字符串替换 | `spec.output_type()` |
| 种子隔离 | 共享 base | 每 seed 全新加载 |
| 表述 | "decode-time 约束" | post-hoc constrained assembly |

## 7. 复现

```bash
# 单测(CPU)
/opt/conda/bin/python tests/test_constrained_decoder.py
# 评估(GPU,~15 min)
/opt/conda/bin/python scripts/eval_constrained_v2.py --n 30
```

生成是确定性的(do_sample=False);三 seed 共享同一批原始输出,D0-D3 只是对同一输出的不同后处理管线。
