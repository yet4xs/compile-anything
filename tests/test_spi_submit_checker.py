"""SPI 提交检查器契约级测试。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contract.spi_submit_checker import (
    SPISubmitChecker, SPIScenario)


def rd(i, ready=True, txempty=True):
    return {"step": i, "op": "read", "reg": "STATUS"}


def cmd(i):
    return {"step": i, "op": "write", "reg": "COMMAND", "value": "0x3"}


def good(windows, extra_polls=3):
    """READY 门 + 满 2 后新观测 + 完成双证据的规范计划。"""
    s = [rd(0)]                       # 初始 READY=1
    maxw = max(list(windows) + [2])
    for k in range(3):
        s.append(cmd(len(s)))
        if k >= 1:
            for _ in range(maxw + 1):  # 覆盖窗口 + 1 次确认
                s.append(rd(len(s)))
    for _ in range(extra_polls):
        s.append(rd(len(s)))
    return s


class Test(unittest.TestCase):
    def test_success(self):
        v = SPISubmitChecker(SPIScenario(ready_windows=(0, 1, 2))).check(
            good((0, 1, 2)))
        self.assertEqual(v.judgment, "SUCCESS", v.reason)

    def test_S1_first_cmd_without_ready(self):
        s = [cmd(0), cmd(1)]
        v = SPISubmitChecker(SPIScenario()).check(s)
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "S1"))

    def test_S3_third_cmd_without_fresh_ready(self):
        s = [rd(0), cmd(1), cmd(2), cmd(3)]     # 排满2后无新观测
        v = SPISubmitChecker(SPIScenario()).check(s)
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "S3"))

    def test_S7_completion_without_txempty(self):
        # 3 段都提交,最后只观测 READY(TXEMPTY 仍 0)
        sc = SPIScenario(ready_windows=(0, 0, 0), txempty_after=99)
        s = [rd(0), cmd(1), cmd(2)]
        for _ in range(5):
            s.append(rd(len(s)))
        s.append(cmd(len(s)))
        for _ in range(5):                       # ready=1 但 txempty=0
            s.append(rd(len(s)))
        v = SPISubmitChecker(sc).check(s)
        self.assertEqual((v.judgment, v.contract), ("VIOLATION", "S7"))

    def test_insufficient_segments(self):
        s = [rd(0), cmd(1)]
        v = SPISubmitChecker(SPIScenario()).check(s)
        self.assertEqual(v.judgment, "INSUFFICIENT_EVIDENCE")

    def test_budget_infeasible(self):
        sc = SPIScenario(never_ready=True)
        s = [rd(0), cmd(1)] + [rd(i) for i in range(2, 120)]
        v = SPISubmitChecker(sc, max_steps=50).check(s)
        self.assertEqual(v.judgment, "BUDGET_INFEASIBLE")


if __name__ == "__main__":
    unittest.main()
