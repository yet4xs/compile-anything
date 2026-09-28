import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import load_module
from src.runtime.simulator import Simulator

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "taskir" / "examples"


class TestSimulator(unittest.TestCase):
    def test_flight_cheapest_totals(self):
        mod = load_module(EXAMPLES / "flight_cheapest.json")
        res = Simulator(mod, seed="t", jitter=0).run()
        self.assertEqual(res.status, "completed")
        ops = [e.op for e in res.events]
        self.assertEqual(ops, ["SEARCH", "FILTER", "ARGMIN"])
        # nominal costs 120 + 5 + 1, no jitter
        self.assertAlmostEqual(res.seq_latency_ms, 126.0, places=3)
        self.assertAlmostEqual(res.critical_path_ms, 126.0, places=3)
        self.assertEqual(res.retries, 0)
        # ARGMIN really picked the min-price flight among the FILTERED ones
        filtered = res.values["%2"]
        best = min(filtered, key=lambda x: x["price"])
        self.assertEqual(res.values["%3"], best)

    def test_verify_retry_rollback(self):
        mod = load_module(EXAMPLES / "flight_verified.json")
        res = Simulator(mod, seed="t", jitter=0,
                        fail_plan={"%5": ["verify_false"]}).run()
        self.assertEqual(res.status, "completed")
        gen_events = [e for e in res.events if e.node == "%4"]
        self.assertEqual(len(gen_events), 2)          # retried once
        self.assertEqual(gen_events[-1].attempt, 2)
        self.assertEqual(res.retries, 1)
        self.assertEqual(res.values["%5"], True)      # verdict after retry

    def test_retry_exhausted(self):
        mod = load_module(EXAMPLES / "flight_verified.json")
        res = Simulator(mod, seed="t", jitter=0,
                        fail_plan={"%5": ["verify_false", "verify_false",
                                          "verify_false"]}).run()
        self.assertEqual(res.status, "completed")     # degrades, not crashes
        gen_events = [e for e in res.events if e.node == "%4"]
        self.assertEqual(len(gen_events), 3)          # max_attempts = 3
        self.assertEqual(res.retries, 2)
        self.assertTrue(any(e.status == "retry_exhausted" for e in res.events))

    def test_guard_branch_select(self):
        mod = load_module(EXAMPLES / "guarded_answer.json")
        # VERIFY %3 判 False：polished 被跳过，fallback 执行，SELECT 选 fallback
        res = Simulator(mod, seed="t", jitter=0,
                        fail_plan={"%4": ["verify_false"]}).run()
        self.assertEqual(res.status, "completed")
        by_node = {e.node: e for e in res.events}
        self.assertEqual(by_node["%5"].status, "skipped")
        self.assertEqual(by_node["%6"].status, "ok")
        self.assertEqual(res.skipped, 1)
        self.assertEqual(res.values["%7"], res.values["%6"])

    def test_error_retry(self):
        from src.ir.taskir import Module, Node, Program, Retry
        mod = Module(program=Program(
            name="t", inputs=[{"name": "@task", "type": "Str"}],
            nodes=[Node(id="%1", op="SEARCH", inputs=["@task"]),
                   Node(id="%2", op="FILTER", inputs=["%1"],
                        retry=Retry(max_attempts=2, on="error")),
                   Node(id="%3", op="ARGMIN", inputs=["%2"],
                        params={"key": "price"})],
            output="%3"))
        res = Simulator(mod, seed="t", jitter=0,
                        fail_plan={"%2": ["error"]}).run()
        self.assertEqual(res.status, "completed")
        flt = [e for e in res.events if e.node == "%2"]
        self.assertEqual(len(flt), 2)
        self.assertEqual(flt[0].status, "error")
        self.assertEqual(flt[1].status, "ok")

    def test_determinism(self):
        mod = load_module(EXAMPLES / "flight_verified.json")
        a = Simulator(mod, seed="same").run()
        b = Simulator(mod, seed="same").run()
        self.assertEqual([e.value_digest for e in a.events],
                         [e.value_digest for e in b.events])


if __name__ == "__main__":
    unittest.main()
