# Phase 8D-2A.2: Effect-Safe Dependency Refinement — Safety Audit

**Date**: 2026-10-14 | **Baseline**: `867d620`

## 核心修复

### Communication ≠ NON_STATE

**修复前**: `communication` 被归入 NON_STATE → `send_alert` 视为无副作用 → 允许删除 `approve_loan → send_alert` 顺序约束

**修复后**: `_is_safe_to_reorder()` 只有 `read_only` 返回 True。`communication`、`state_update_reversible`、`state_creation`、`state_transition_irreversible` 全部视为 UNSAFE

### 安全删除三条件（缺一不可）

1. 消费者不通过 param_refs 使用生产者值
2. **双端都是纯 read_only**（无 mutation/communication）
3. 所有必需参数已通过命名机制绑定

## 重新验证 48 条候选证书

| 分类 | 数量 | 原因 |
|------|------|------|
| **SAFE** | **0** | 所有候选都涉及至少一个非 read_only 端点 |
| **UNSAFE** | **48** | `approve_loan`(state_creation) 或 `send_alert`(communication) 参与 |

**具体案例 (v2_fan_out_in_000)**:
- `%p1(get_account) → %p3(approve_loan)`: approve_loan 是 state_creation → UNSAFE
- `%p3(approve_loan) → %p4(send_alert)`: send_alert 是 communication → UNSAFE（**绝不允许在贷款审批前发送通知**）

## 修正后结论

| 指标 | 不安全版本(修复前) | **安全版本(修复后)** |
|------|------------------|-------------------|
| 删除边数 | 48 | **0** |
| makespan 改善 | 4.32→3.22 (-25.5%) | **无改善** |
| 安全删除数 | — | **0** |

**此前报告的 25.5% makespan 改善基于 UNSAFE 删除，不构成有效的编译器优化证据。**

## 对 Phase 8D-2A Gate 的影响

在当前数据流语料中，所有冗余位置操作数都涉及副作用端点（mutation 或 communication）。在这些端点之间删除顺序约束可能改变可观察的业务语义。

**未来安全的优化需要**:
1. 更丰富的 Effect 类型系统（区分"可选通知"与"必要顺序通知"）
2. 用户显式声明的顺序约束（skeleton order_on 而非仅 depends_on）
3. 运行时等价性验证（执行轨迹比较）

## Gate 状态

| # | 条件 | 状态 |
|---|------|------|
| 1 | 无 UNSAFE 删除 | ✅ (0 删除 = 0 UNSAFE) |
| 2 | 不能证明安全的默认保留 | ✅ |
| 3 | Schedule Verifier | 待实现 |
| 4 | 执行轨迹等价 | 待实现 |
| 5 | 公平对照 | 待修正 |

**结论：编译器依赖优化在当前语料上暂无安全的正收益。25.5% 改善不可作为论文结果。**

**8D-2A.2 Gate: PASS for Safety（保守正确），FAIL for Optimization Benefit（无可证明的调度改善）**

**允许进入 8D-2B 三臂消融**（不依赖调度优化收益）。
