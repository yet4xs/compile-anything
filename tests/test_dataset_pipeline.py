"""Phase 5A tests: dataset registry/schema/adapters (offline, fixtures only)."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dataset.registry import REGISTRY, available, unavailable
from src.dataset.schema import make_sample, to_lifter_input
from src.dataset.adapters.tooluse import ToolBenchAdapter, _find_tool_calls
from src.dataset.adapters.code import HumanEvalAdapter, MBPPAdapter
from src.dataset.adapters.sql import SpiderAdapter
from src.dataset.adapters.rtl import VerilogEvalAdapter, _signals_from_verilog
from src.dataset.pipeline import coverage_stats


class TestRegistry(unittest.TestCase):
    def test_registered_names(self):
        for n in ("toolbench", "apibank", "agentbench", "humaneval", "mbpp",
                  "spider", "bird", "verilogeval", "hdlbits"):
            self.assertIn(n, REGISTRY)
        self.assertIn("bird", unavailable())
        self.assertIn("spider", available())

    def test_available_have_plans(self):
        for n in available():
            self.assertTrue(REGISTRY[n].plan, n)
        for n in unavailable():
            self.assertEqual(REGISTRY[n].plan, [])


class TestSchema(unittest.TestCase):
    def test_to_lifter_input_mappings(self):
        cases = [
            (make_sample(id="t1", source="toolbench", input_text="q",
                         trajectory=[{"tool": "A.b", "args": {"x": 1}}]),
             ("instruction", "trajectory")),
            (make_sample(id="c1", source="humaneval", input_text="p",
                         code="def f(x): return x"),
             ("prompt", "code")),
            (make_sample(id="s1", source="spider", input_text="q",
                         sql="SELECT 1",
                         metadata={"db_id": "d"}),
             ("question", "query")),
            (make_sample(id="r1", source="verilogeval", input_text="p",
                         rtl="module m(); endmodule",
                         metadata={"signals": ["a"]}),
             ("prompt", "rtl")),
        ]
        for sample, keys in cases:
            mapped = to_lifter_input(sample)
            for k in keys:
                self.assertIn(k, mapped, sample["source"])

    def test_raw_payload_preserved(self):
        s = make_sample(id="x", source="spider", sql="SELECT 1",
                        raw_payload={"question": "q", "query": "SELECT 1"})
        self.assertEqual(s["raw_payload"]["query"], "SELECT 1")


class TestAdapters(unittest.TestCase):
    def test_toolbench_tool_call_extraction(self):
        rec = {"answer_generation": {
            "query": "find customs agency",
            "train_messages": [
                [{"role": "user", "content": "..."}],
                [{"role": "assistant",
                  "function_call": {"name": "transitaires_for_x",
                                    "arguments": "{\"k\": 1}"}}],
            ]}}
        s = ToolBenchAdapter().normalize(rec, "G1/10")
        self.assertEqual(s["input_text"], "find customs agency")
        self.assertEqual(s["trajectory"][0]["tool"], "transitaires_for_x")
        self.assertEqual(s["trajectory"][0]["args"], {"k": 1})

    def test_find_tool_calls_skips_results(self):
        calls = []
        _find_tool_calls({"role": "function", "name": "f", "content": "ok"},
                         calls)
        self.assertEqual(calls, [])          # results are not calls

    def test_humaneval_normalize_concat(self):
        s = HumanEvalAdapter().normalize(
            {"task_id": "HumanEval/3", "prompt": "def f(x):\n    \"\"\"d\"\"\"\n",
             "canonical_solution": "    return x\n", "entry_point": "f"})
        self.assertTrue(s["code"].startswith("def f(x):"))
        self.assertIn("return x", s["code"])
        self.assertEqual(s["id"], "HumanEval_3")

    def test_mbpp_wrap_body(self):
        s = MBPPAdapter().normalize(
            {"task_id": "7", "prompt": "sum a list",
             "code": "total = 0\nfor i in x:\n    total += i\nreturn total"})
        self.assertIn("def f(", s["code"])      # wrapped for ast parsing

    def test_spider_normalize(self):
        s = SpiderAdapter().normalize(
            {"question": "how many?", "query": "SELECT count(*) FROM t",
             "db_id": "db"}, "train-0", "train")
        self.assertEqual(s["sql"], "SELECT count(*) FROM t")
        self.assertEqual(s["metadata"]["db_id"], "db")

    def test_verilogeval_signals(self):
        sigs = _signals_from_verilog(
            "module top_module(\n  input clk,\n  input rst_n,\n"
            "  input [7:0] din,\n  output [7:0] dout);")
        self.assertIn("din", sigs)
        self.assertIn("dout", sigs)
        s = VerilogEvalAdapter().normalize("build a counter",
                                           "module t(); endmodule",
                                           "code-complete/Prob001_x")
        self.assertEqual(s["source"], "verilogeval")
        self.assertEqual(s["metadata"]["track"], "code-complete")


class TestCoverageStats(unittest.TestCase):
    def test_buckets(self):
        from src.ir.taskir import Module, Node, Program
        mod = Module(program=Program(
            name="t", inputs=[{"name": "@task", "type": "Str"}],
            nodes=[Node(id="%1", op="SEARCH", inputs=["@task"]),
                   Node(id="%2", op="FILTER", inputs=["%1"])], output="%2"))
        lifted = [
            {"sample": {"source": "x"}, "module": None, "reason": "loop (…)"},
            {"sample": {"source": "x"}, "module": mod, "reason": None,
             "valid": True, "warnings": [],
             "metrics": {"nodes": 2, "depth": 2, "width": 1,
                         "branch_ratio": 0.0, "parallel_ratio": 0.0,
                         "skills": {"SEARCH": 1, "FILTER": 1}},
             "result": None},
        ]
        stats = coverage_stats(lifted)
        self.assertEqual(stats["x"]["samples"], 2)
        self.assertEqual(stats["x"]["reject_reasons"], {"LOOP": 1})
        self.assertEqual(stats["x"]["skill_coverage"]["SEARCH"], 1)


if __name__ == "__main__":
    unittest.main()
