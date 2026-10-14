# Phase 8D-0.1: Typed SSA Dataflow Semantic Hardening — Results

**Date**: 2026-10-14 | **Baseline**: `44e7957`

## 已修复缺陷清单

| # | 缺陷 | 修复 | 回归测试 |
|---|------|------|----------|
| 1 | JSON 序列化丢失 param_refs | `_node_to_dict/_node_from_dict` 增加 param_refs 字段 | `test_json_roundtrip_preserves_param_refs` |
| 2 | EXTRACT_FIELD 缺 Schema 退化为 Any | `validate(mod, capability_registry=...)` 严格模式检查 FIELD_NOT_FOUND + 类型推断 | `test_strict_field_not_found` |
| 3 | 推断类型被 ISA 默认 Any 覆盖 | EXTRACT_FIELD 节点跳过通用 ISA 推断，保留 Schema 派生类型 | `test_extract_field_type_not_overridden` |
| 4 | param_refs 不在依赖图中 | `_dep_edges()` 增加 `node.param_refs.values()` | `test_param_refs_in_dep_edges` |
| 5 | Runtime 不检查输入输出类型 | `_check_type()` 映射 Str/Int/Float/Bool 到 Python 类型并执行 isinstance 检查 | `test_runtime_output_type_mismatch` |
| 6 | EXTRACT_FIELD scalar passthrough | 改为 RuntimeError 拒绝非 dict 结果 | (通过 runtime 测试覆盖) |
| 7 | Guard/Retry 静默忽略 | DataflowRuntime fail-closed：遇到 guard/retry 抛 RuntimeError | `test_guard_fail_closed` / `test_retry_fail_closed` |
| 8 | Skeleton 无 runtime_ref 支持 | Lowering 识别 `{"kind": "runtime_ref"}` 参数，自动生成 EXTRACT_FIELD + bind() | `test_skeleton_runtime_ref_generates_extract_field` |

## 严格模式 vs Legacy 模式

- **严格模式**: `validate(mod, capability_registry=CAPS)` — 检查字段存在 + 类型兼容 + param_refs 输入 Schema
- **Legacy 模式**: `validate(mod)` — 不检查字段存在性（v0.1 兼容），但 param_refs 结构检查仍执行
- Legacy 通过率 **不计入** strict typed-valid 指标

## 测试统计

- **25 个 v0.2 语义测试**: 25/25 通过
- **全部回归**: 77/77 通过(含 v0.1 IR/Validator/Runtime/约束解码器/骨架/数据集/τ³)
- 30 条运行时数据流批量执行: 30/30 正确(值传播可追溯到执行 Trace)

## Go/No-Go (Task 8)

| # | 条件 | 状态 |
|---|------|------|
| 1 | JSON/Text/AST Roundtrip 保留 param_refs | ✅ |
| 2 | Strict Validator 验证字段存在 + 类型 | ✅ |
| 3 | param_refs 在所有依赖分析中识别 | ✅ |
| 4 | Mock Runtime 输入输出类型检查 | ✅ |
| 5 | 非法引用/不支持控制语义拒绝 | ✅ |
| 6 | Mutation 不意外重复执行 | ✅ |
| 7 | Skeleton 自动生成 EXTRACT_FIELD + bind() | ✅ |
| 8 | 合法案例值传播正确 | ✅ |
| 9 | v0.1 回归不退化 | ✅ (77/77) |

**GATE: PASS — 允许进入 Phase 8D-1 构建动态参数训练集**
