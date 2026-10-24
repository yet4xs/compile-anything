"""AES init checker 契约级测试:每条可机械判定的契约至少一正一负。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contract.aes_init_checker import (
    AESInitChecker, Scenario, KEY_REGS, CTRL, STATUS)


def rd(i, reg=STATUS, field="IDLE"):
    return {"step": i, "op": "read", "reg": reg, "field": field}


def wr(i, reg, val="0x0"):
    return {"step": i, "op": "write", "reg": reg, "value": val}


def full_flow(poll_after_ctrl=1, read_between_shadow=False):
    s = []
    i = 0
    s.append(rd(i)); i += 1                                   # C1
    s.append(wr(i, CTRL, "0x1C1")); i += 1                    # C5 first
    if read_between_shadow:
        s.append(rd(i, CTRL)); i += 1                         # C6 phase reset
    s.append(wr(i, CTRL, "0x1C1")); i += 1                    # C5 commit
    for _ in range(poll_after_ctrl):
        s.append(rd(i)); i += 1                               # C4
    for r in KEY_REGS:
        s.append(wr(i, r, "0xA5A5A5A5")); i += 1              # C8
    return s


class Positive(unittest.TestCase):
    def test_full_flow_success(self):
        v = AESInitChecker(Scenario(ctrl_commit_busy_polls=0)).check(
            full_flow())
        self.assertEqual(v.judgment, "SUCCESS", v.reason)
        self.assertTrue(v.trace)

    def test_full_flow_with_reseed_busy_polls(self):
        v = AESInitChecker(Scenario(ctrl_commit_busy_polls=3)).check(
            full_flow(poll_after_ctrl=4))
        self.assertEqual(v.judgment, "SUCCESS", v.reason)

    def test_insufficient_polling_is_evidence_gate(self):
        s = full_flow(poll_after_ctrl=1)
        v = AESInitChecker(Scenario(ctrl_commit_busy_polls=2)).check(s)
        # 单次轮询返回 IDLE=0:计划缺少继续的证据——C4 门先于 no-effect
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "C4"))

    def test_no_effect_writes_recorded_not_fatal(self):
        # 场景:初始即忙(C1 阶段 IDLE=0)——写被忽略但流程可继续
        s = [rd(0), rd(1), wr(2, CTRL, "0x1"), wr(3, CTRL, "0x1"),
             rd(4)] + [wr(i, r) for i, r in enumerate(KEY_REGS, 5)]
        v = AESInitChecker(Scenario(initial_idle=False)).check(s)
        # 初始非 idle:C1 的 IDLE=1 从未出现 -> 第一笔 CTRL 写即 VIOLATION C1
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "C1"))


class Negative(unittest.TestCase):
    def test_C1_write_before_idle_observation(self):
        s = [wr(0, CTRL), wr(1, CTRL)]
        v = AESInitChecker(Scenario()).check(s)
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "C1"))

    def test_C3_key_before_ctrl(self):
        s = [rd(0), wr(1, KEY_REGS[0])]
        v = AESInitChecker(Scenario()).check(s)
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "C3"))

    def test_C4_missing_recheck_after_ctrl(self):
        s = [rd(0), wr(1, CTRL, "0x1"), wr(2, CTRL, "0x1")]
        # scenario: ctrl commit makes busy for 2 polls; flow has NO poll
        s += [wr(i, r) for i, r in enumerate(KEY_REGS, start=3)]
        v = AESInitChecker(Scenario(ctrl_commit_busy_polls=2)).check(s)
        # idle flag still True from C1 read? — no: commit flips env busy;
        # checker's `idle` stays True (stale) but C4 gate is the evidence,
        # and the key write arrives without any post-commit read.
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "C4"))

    def test_C5_update_error_then_key_write_is_C3(self):
        s = [rd(0), wr(1, CTRL, "0x1"), wr(2, CTRL, "0x2"),
             rd(3)] + [wr(i, r) for i, r in enumerate(KEY_REGS, 4)]
        v = AESInitChecker(Scenario()).check(s)
        # 两写不一致未提交即写 key:具体违反(C3)优先于笼统证据不足
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "C3"))

    def test_C5_update_error_then_matching_repair_commits(self):
        s = [rd(0), wr(1, CTRL, "0x1"), wr(2, CTRL, "0x2"),  # error
             wr(3, CTRL, "0x9"), wr(4, CTRL, "0x9"),          # fresh pair
             rd(5)] + [wr(i, r) for i, r in enumerate(KEY_REGS, 6)]
        v = AESInitChecker(Scenario()).check(s)
        self.assertEqual(v.judgment, "SUCCESS", v.reason)

    def test_C6_read_between_shadow_writes_requires_two_fresh(self):
        # 两写之间读 CTRL:phase 复位,原第二写变成新第一写;序列止于此
        s = [rd(0), wr(1, CTRL, "0x1"), rd(2, CTRL), wr(3, CTRL, "0x1")]
        v = AESInitChecker(Scenario()).check(s)
        self.assertEqual(v.judgment, "INSUFFICIENT_EVIDENCE")
        self.assertIn("never committed", v.reason)

    def test_C6_recovery_two_writes_after_reset(self):
        # 读复位后再写两次相同值 -> 提交成功(文档允许的恢复路径)
        s = [rd(0), wr(1, CTRL, "0x7"), rd(2, CTRL),
             wr(3, CTRL, "0x7"), wr(4, CTRL, "0x7"), rd(5)]
        s += [wr(i, r) for i, r in enumerate(KEY_REGS, 6)]
        v = AESInitChecker(Scenario()).check(s)
        self.assertEqual(v.judgment, "SUCCESS", v.reason)

    def test_C8_missing_key_registers(self):
        s = [rd(0), wr(1, CTRL, "0x1"), wr(2, CTRL, "0x1"), rd(3)]
        s += [wr(i, r) for i, r in enumerate(KEY_REGS[:10], 4)]
        v = AESInitChecker(Scenario()).check(s)
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "C8"))

    def test_budget_infeasible_distinct_from_unknown(self):
        s = [rd(0), wr(1, CTRL, "0x1"), wr(2, CTRL, "0x1")]
        s += [rd(i) for i in range(3, 203)]                    # 200 polls
        v = AESInitChecker(Scenario(never_idle_after_ctrl=True),
                           max_steps=100).check(s)
        self.assertEqual(v.judgment, "BUDGET_INFEASIBLE")

    def test_unknown_register_rejected(self):
        s = [rd(0), wr(1, "MYSTERY_REG")]
        v = AESInitChecker(Scenario()).check(s)
        self.assertEqual(v.judgment, "VIOLATION")
        self.assertEqual(v.contract, "UNKNOWN_REG")


if __name__ == "__main__":
    unittest.main()
