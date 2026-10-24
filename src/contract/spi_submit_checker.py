"""SPI_HOST 多 segment 提交检查器(polling 口径,契约 S1-S9)。

操作 IR(与 AES 同形): read STATUS{field}/write COMMAND/TXDATA。
场景: ready 返回窗口(提交后第 k 次轮询起 READY=1)+ txempty 置位窗。

判定:
  S1 写 COMMAND 前无 READY=1 观测(最近一次 STATUS.READY 读值为 1)
  S3 已排 2 个后、第 3 个 COMMAND 前无新的 READY=1 观测
  S7 宣称完成(计划终点)前未同时观测 READY=1 与 TXEMPTY=1
  UNKNOWN_REG / BUDGET_INFEASIBLE / INSUFFICIENT_EVIDENCE(未提交
  全部 segment)/ SUCCESS
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Verdict:
    judgment: str
    contract: Optional[str] = None
    at_step: Optional[int] = None
    reason: str = ""
    trace: List[dict] = field(default_factory=list)


class SPIScenario:
    """ready_n: 第 i 个 COMMAND 提交后,前 ready_n 次 READY 轮询返回 0,
    其后 1。txempty_n: 最后 COMMAND 后第 txempty_n 次轮询起 TXEMPTY=1
    (同一次读同时返回两个字段)。"""

    def __init__(self, segments=3, ready_windows=(0, 1, 2), txempty_after=2,
                 never_ready=False):
        self.ready_windows = list(ready_windows)
        self.txempty_after = txempty_after
        self.never = never_ready
        self.cmd_count = 0
        self._polls_since_cmd = 0
        self.total_cmds_target = segments

    def on_command(self):
        self.cmd_count += 1
        self._polls_since_cmd = 0

    def read(self):
        """返回 (READY, TXEMPTY)。"""
        if self.cmd_count == 0:
            return True, True          # 初始空闲
        if self.never:
            return False, False
        w = self.ready_windows[min(self.cmd_count - 1,
                                   len(self.ready_windows) - 1)]
        ready = self._polls_since_cmd >= w
        txempty = (self.cmd_count >= self.total_cmds_target
                   and self._polls_since_cmd >= self.txempty_after)
        self._polls_since_cmd += 1
        return ready, txempty


def _observed_not_ready(trace):
    return any(t.get("event", "").startswith("status") and
               t.get("ready") is False for t in trace)


class SPISubmitChecker:
    def __init__(self, scenario: SPIScenario, max_steps: int = 200):
        self.sc = scenario
        self.max_steps = max_steps

    def check(self, steps: List[dict]) -> Verdict:
        v = Verdict("SUCCESS")
        last_ready = False            # 最近一次观测的 READY
        last_txempty = False
        cmds = 0
        ready_since_queue2 = False    # 排满 2 后的新 READY=1 观测

        def snap(i, ev):
            v.trace.append({"step": i, "cmds": cmds, "ready": last_ready,
                            "txempty": last_txempty, "event": ev})

        for i, st in enumerate(steps):
            if i >= self.max_steps:
                v.judgment = "BUDGET_INFEASIBLE"
                v.at_step = i
                return v
            op = st.get("op")
            if op == "read" and st.get("reg") == "STATUS":
                r, t = self.sc.read()
                prev_ready = last_ready
                last_ready, last_txempty = r, t
                if r and cmds >= 2:
                    ready_since_queue2 = True
                snap(i, f"status R={int(r)} T={int(t)}")
            elif op == "write" and st.get("reg") == "COMMAND":
                if cmds == 0:
                    # S1: 首个 COMMAND 需要就绪证据
                    if not last_ready:
                        v.judgment = "VIOLATION"
                        v.contract = "S1"
                        v.at_step = i
                        v.reason = ("first COMMAND without READY=1 "
                                    "evidence")
                        snap(i, "violation_S1")
                        return v
                elif cmds == 1:
                    # S2 文档例外: 处理中允许再排恰一个;仅当最近观测
                    # 明确为 0 且未刷新时才违反(带着已知未就绪前进)
                    if last_ready is False:
                        v.judgment = "VIOLATION"
                        v.contract = "S1"
                        v.at_step = i
                        v.reason = ("2nd COMMAND after observing READY=0 "
                                    "without refresh")
                        snap(i, "violation_S1_stale0")
                        return v
                else:
                    # S3: 满 2 后,第 3+ 个需新的 READY=1 观测
                    if not ready_since_queue2:
                        v.judgment = "VIOLATION"
                        v.contract = "S3"
                        v.at_step = i
                        v.reason = ("3rd+ COMMAND after queue full "
                                    "without a fresh READY=1 observation")
                        snap(i, "violation_S3")
                        return v
                self.sc.on_command()
                cmds += 1
                ready_since_queue2 = False
                snap(i, "command")
            elif op in ("read", "write"):
                snap(i, f"{op}:{st.get('reg')}")
            else:
                v.judgment = "VIOLATION"
                v.contract = "UNKNOWN_OP"
                v.at_step = i
                v.reason = f"unrecognized step {st}"
                return v

        if cmds < self.sc.total_cmds_target:
            v.judgment = "INSUFFICIENT_EVIDENCE"
            v.reason = (f"only {cmds}/{self.sc.total_cmds_target} "
                        "segments submitted")
            return v
        if not (last_ready and last_txempty):
            v.judgment = "VIOLATION"
            v.contract = "S7"
            v.reason = ("completion claimed without READY=1 AND TXEMPTY=1 "
                        "as the final observations")
        return v
