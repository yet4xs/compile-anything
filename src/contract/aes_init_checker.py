"""AES 初始化操作 IR 的 CPU 契约检查器(主实验组件,Phase 2026-10-24)。

输入: 操作序列(步骤列表)+ 场景(回答读操作的确定性环境)。
依据: results/raw_contract_intake_v1/aes_contract_table_v1.json
(13/13 条引用经机验,OpenTitan commit 3a7f6947)。

操作 IR(模型产物被解析成这个形状):
  {"step": i, "op": "read",  "reg": "STATUS", "field": "IDLE"}
  {"step": i, "op": "write", "reg": "CTRL_SHADOWED", "value": "0x.."}
  {"step": i, "op": "write", "reg": "KEY_SHARE0_3", "value": "0x.."}
允许寄存器集合来自 registers.md;未知寄存器记 VIOLATION(UNKNOWN_REG)。

场景 = 决定每次读返回什么的环境(代替 RTL):
  {"initial_idle": True, "ctrl_commit_busy_polls": k, "key_busy_polls": k2}
k/k2 为"IDLE 何时回到 1"的轮询数;never 场景用 budget=∞ 区分。

四类判定(与 raw-document-contract-next-stage 第2条一致):
  SUCCESS / VIOLATION(contract_id) / INSUFFICIENT_EVIDENCE /
  BUDGET_INFEASIBLE(≠ 资料不足;单列)
判定附可重放轨迹(逐步状态快照),供独立核验。
"""

from dataclasses import dataclass, field
from typing import List, Optional

KEY_REGS = [f"KEY_SHARE{s}_{i}" for s in (0, 1) for i in range(8)]
CTRL = "CTRL_SHADOWED"
STATUS = "STATUS"
IV_REGS = [f"IV_{i}" for i in range(4)]

INITIALIZED_GATED = {CTRL, *KEY_REGS, *IV_REGS}   # C2 范围


@dataclass
class Verdict:
    judgment: str                      # SUCCESS / VIOLATION / INSUFFICIENT_EVIDENCE / BUDGET_INFEASIBLE
    contract: Optional[str] = None     # C1..C13
    at_step: Optional[int] = None
    reason: str = ""
    trace: List[dict] = field(default_factory=list)


class Scenario:
    """确定性读环境:k 次 STATUS 轮询返回 IDLE=0,之后 1。"""

    def __init__(self, initial_idle=True, ctrl_commit_busy_polls=0,
                 key_busy_polls=0, never_idle_after_ctrl=False):
        self.initial_idle = initial_idle
        self.ctrl_busy = ctrl_commit_busy_polls
        self.key_busy = key_busy_polls
        self.never = never_idle_after_ctrl
        self._polls_since_ctrl_commit = 0
        self._ctrl_committed = False
        self._key_started = False

    def on_ctrl_committed(self):
        self._ctrl_committed = True
        self._polls_since_ctrl_commit = 0

    def on_first_key_write(self):
        self._key_started = True

    def read_status_idle(self):
        if not self._ctrl_committed:
            return self.initial_idle          # C1 阶段
        if self.never:
            return False                       # C13 无界忙(预算不可行场景)
        if self._key_started:
            return True                        # 密钥阶段忙窗口极短,文档未定义;
                                               # 保持 True,不假设额外门槛(unknown 不补猜)
        if self._polls_since_ctrl_commit < self.ctrl_busy:
            self._polls_since_ctrl_commit += 1
            return False
        return True


class AESInitChecker:
    """把契约表编译成状态机并对操作序列逐步判定。"""

    def __init__(self, scenario: Scenario, max_steps: int = 200):
        self.sc = scenario
        self.max_steps = max_steps

    def check(self, steps: List[dict]) -> Verdict:
        v = Verdict(judgment="SUCCESS")
        idle = True                       # 模型已证实的 idle 状态
        idle_evidenced = False            # C1 是否已满足(读过 IDLE=1)
        ctrl_phase = 0                    # shadow phase: 0/1 (C5/C6)
        ctrl_shadow_val = None            # 第一写暂存
        ctrl_committed = False
        idle_rechecked_after_ctrl = False # C4
        key_written = {r: 0 for r in KEY_REGS}
        key_started = False
        pending_no_effect = []            # C2 静默忽略的写(记录,不判死)

        def snap(**kw):
            v.trace.append({"step": kw.get("i"), "idle": idle,
                            "ctrl_phase": ctrl_phase,
                            "ctrl_committed": ctrl_committed,
                            "c4_recheck": idle_rechecked_after_ctrl,
                            "key_regs_touched": sum(1 for c in
                                                    key_written.values() if c),
                            **{k: w for k, w in kw.items() if k != "i"}})

        for i, st in enumerate(steps):
            if i >= self.max_steps:
                v.judgment = "BUDGET_INFEASIBLE"
                v.at_step = i
                v.reason = "step budget exhausted (distinct from unknown)"
                return v
            op, reg = st.get("op"), st.get("reg", "")
            if op == "read":
                if reg == STATUS:
                    idle = self.sc.read_status_idle()
                    if idle:
                        if not ctrl_committed:
                            idle_evidenced = True          # C1 证据
                        else:
                            idle_rechecked_after_ctrl = True  # C4 证据
                    snap(i=i, event="read_status", value=int(idle))
                elif reg == CTRL:
                    # C6: 读 shadowed 寄存器把 phase 复位回 1
                    ctrl_phase = 0
                    ctrl_shadow_val = None
                    snap(i=i, event="read_ctrl_phase_reset")
                else:
                    snap(i=i, event="read", reg=reg)
            elif op == "write":
                val = st.get("value")
                if reg not in INITIALIZED_GATED and reg not in ("TRIGGER",):
                    v.judgment = "VIOLATION"
                    v.contract = "UNKNOWN_REG"
                    v.at_step = i
                    v.reason = f"write to non-contract register {reg}"
                    snap(i=i, event="unknown_reg")
                    return v
                if reg == CTRL:
                    if not idle_evidenced:
                        v.judgment = "VIOLATION"
                        v.contract = "C1"
                        v.at_step = i
                        v.reason = "write before any IDLE=1 observation"
                        snap(i=i, event="violation_C1")
                        return v
                    if not idle:
                        pending_no_effect.append((i, reg))   # C2
                        snap(i=i, event="no_effect_write", reg=reg)
                        continue
                    if ctrl_phase == 0:
                        ctrl_phase = 1
                        ctrl_shadow_val = val
                        snap(i=i, event="ctrl_first_write")
                    else:
                        if val == ctrl_shadow_val:
                            ctrl_committed = True
                            self.sc.on_ctrl_committed()
                            snap(i=i, event="ctrl_committed")
                        else:
                            # C7 update error: shadow 回退 committed
                            ctrl_phase = 0
                            ctrl_shadow_val = None
                            snap(i=i, event="ctrl_update_error")
                elif reg in KEY_REGS:
                    if not ctrl_committed:
                        v.judgment = "VIOLATION"
                        v.contract = "C3"
                        v.at_step = i
                        v.reason = "key write before CTRL configuration"
                        snap(i=i, event="violation_C3")
                        return v
                    if not idle_rechecked_after_ctrl:
                        v.judgment = "VIOLATION"
                        v.contract = "C4"
                        v.at_step = i
                        v.reason = ("key write without IDLE=1 re-check "
                                    "after CTRL commit")
                        snap(i=i, event="violation_C4")
                        return v
                    if not idle:
                        pending_no_effect.append((i, reg))
                        snap(i=i, event="no_effect_write", reg=reg)
                        continue
                    if not key_started:
                        key_started = True
                        self.sc.on_first_key_write()
                    key_written[reg] += 1
                    snap(i=i, event="key_write", reg=reg)
                else:
                    snap(i=i, event="write", reg=reg)
        # 终局判定
        if not ctrl_committed:
            v.judgment = "INSUFFICIENT_EVIDENCE"
            v.reason = "CTRL_SHADOWED never committed (two matching writes)"
            return v
        missing = [r for r, c in key_written.items() if c == 0]
        if missing:
            v.judgment = ("VIOLATION" if any(key_written.values())
                          else "INSUFFICIENT_EVIDENCE")
            v.contract = "C8"
            v.reason = f"key registers not written: {missing[:4]}..."
            return v
        v.reason = ("complete: C1 idle, C5 two-write commit, C4 recheck, "
                    "C8 all 16 key regs; no-effect writes: "
                    f"{pending_no_effect}")
        return v
