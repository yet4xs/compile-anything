"""Phase 4 tests: benchmark lifters + TaskIR text parser roundtrip."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import load_module, to_text, module_to_dict
from src.ir.parser import parse_text, TaskIRSyntaxError
from src.validator.validator import validate
from src.lifter.benchmark import lift_sample

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestToolBenchLifter(unittest.TestCase):
    SAMPLE = {
        "task_id": "tb-1",
        "instruction": "Find flights SFO to Tokyo and convert the price to JPY",
        "trajectory": [
            {"tool": "google_flight.search",
             "args": {"origin": "SFO", "destination": "Tokyo"}},
            {"tool": "CurrencyConverter.convert",
             "args": {"amount": 540, "from_currency": "USD",
                      "to_currency": "JPY"}},
        ],
    }

    def test_semantic_only_ops(self):
        mod, reason, lifter = lift_sample(self.SAMPLE)
        self.assertEqual(lifter, "toolbench")
        self.assertIsNotNone(mod, reason)
        ops = [n.op for n in mod.program.nodes]
        self.assertIn("SEARCH", ops)
        self.assertIn("CONVERT", ops)
        for n in mod.program.nodes:
            self.assertNotIn(".", n.op)            # no google_flight.search
        prov = mod.meta["provenance"]
        self.assertIn("google_flight.search", prov["tools"])

    def test_lifted_program_valid(self):
        mod, _, _ = lift_sample(self.SAMPLE)
        rep = validate(mod)
        self.assertTrue(rep.valid, rep.summary())


class TestHumanEvalLifter(unittest.TestCase):
    def _ops(self, code, prompt="task"):
        mod, reason, _ = lift_sample({"task_id": "he", "prompt": prompt,
                                      "code": code})
        if mod is None:
            return None, reason
        return [n.op for n in mod.program.nodes], None

    def test_patterns(self):
        cases = [
            ('def f(items):\n    return sorted(items, key=lambda x: x["p"])',
             ["SORT"]),
            ('def f(items):\n    return min(items, key=lambda x: x.p)',
             ["ARGMIN"]),
            ('def f(items):\n    return max(items)', ["MAX"]),
            ('def f(items):\n    return sum(items)', ["SUM"]),
            ('def f(items):\n    return len(items)', ["COUNT"]),
            ('def f(items):\n    return [x for x in items if x > 0]',
             ["FILTER"]),
            ('def f(items):\n    return [x * 2 for x in items]',
             ["TRANSFORM"]),
            ('def f(a, b):\n    return a + b', ["JOIN"]),
            ('def f(items):\n    return sorted([x for x in items if x["ok"]],'
             ' key=lambda x: x["k"])', ["FILTER", "SORT"]),
        ]
        for code, expected in cases:
            ops, reason = self._ops(code)
            self.assertIsNotNone(ops, f"{code}: {reason}")
            self.assertEqual(ops, expected, code)
            mod, _, _ = lift_sample({"task_id": "he", "prompt": "t",
                                     "code": code})
            self.assertTrue(validate(mod).valid, code)

    def test_unsupported_loop_reason(self):
        ops, reason = self._ops(
            'def f(n):\n    t = 0\n    for i in range(n):\n        t += i\n'
            '    return t')
        self.assertIsNone(ops)
        self.assertIn("loop", reason)

    def test_sort_desc(self):
        ops, _ = self._ops('def f(items):\n    return sorted(items, '
                           'key=lambda x: x["p"], reverse=True)')
        self.assertEqual(ops, ["SORT"])
        mod, _, _ = lift_sample({"task_id": "he", "prompt": "t",
                                 "code": 'def f(items):\n    return sorted('
                                 'items, key=lambda x: x["p"], reverse=True)'})
        self.assertEqual(mod.program.nodes[0].params["order"], "desc")


class TestSpiderLifter(unittest.TestCase):
    def test_group_by_stays_in_query_db(self):
        s = {"task_id": "s1", "db_id": "flights",
             "question": "avg price by country",
             "query": "SELECT country, AVG(price) FROM flights "
                      "GROUP BY country"}
        mod, reason, lifter = lift_sample(s)
        self.assertEqual(lifter, "spider")
        self.assertIsNotNone(mod, reason)
        self.assertTrue(validate(mod).valid)
        prov = mod.meta["provenance"]
        self.assertFalse(prov["decomposed"])           # GROUP BY: no EXTRACT
        self.assertTrue(prov["sql_features"]["group_by"])

    def test_simple_select_decomposes(self):
        s = {"task_id": "s2", "db_id": "flights",
             "question": "origin and price",
             "query": "SELECT origin, price FROM flights WHERE price > 100"}
        mod, _, _ = lift_sample(s)
        ops = [n.op for n in mod.program.nodes]
        self.assertEqual(ops, ["QUERY_DB", "EXTRACT", "GENERATE", "VERIFY"])
        self.assertTrue(mod.meta["provenance"]["decomposed"])


class TestRtlLifter(unittest.TestCase):
    def test_shape_and_retry(self):
        s = {"task_id": "r1", "prompt": "Fix the FIFO overflow assertion",
             "rtl": "module fifo(); endmodule",
             "signals": ["fifo_count", "depth"]}
        mod, reason, lifter = lift_sample(s)
        self.assertEqual(lifter, "rtl")
        self.assertIsNotNone(mod, reason)
        self.assertTrue(validate(mod).valid)
        ops = [n.op for n in mod.program.nodes]
        self.assertIn("CODEGEN", ops)
        fix = next(n for n in mod.program.nodes if n.op == "CODEGEN")
        self.assertEqual(fix.retry.max_attempts, 3)
        self.assertIn("VERIFY", ops)


class TestTextParser(unittest.TestCase):
    def test_roundtrip_all_examples(self):
        for f in sorted((ROOT / "data" / "taskir" / "examples").glob("*.json")):
            m = load_module(f)
            m2 = parse_text(to_text(m))
            self.assertEqual(module_to_dict(m2), module_to_dict(m), f)

    def test_roundtrip_annotations(self):
        m = load_module(ROOT / "data" / "taskir" / "examples"
                        / "flight_verified.json")
        m2 = parse_text(to_text(m))
        n1 = {n.id: n for n in m.program.nodes}["%4"]
        n2 = {n.id: n for n in m2.program.nodes}["%4"]
        self.assertEqual(n1.retry, n2.retry)
        self.assertEqual(module_to_dict(m2), module_to_dict(m))

    def test_roundtrip_multiline_description(self):
        from src.ir.taskir import Module, Program, Node
        m = Module(program=Program(
            name="t", description="line1\nline2 with ; semi\ntail",
            inputs=[{"name": "@task", "type": "Str"}],
            nodes=[Node(id="%1", op="SEARCH", inputs=["@task"])],
            output="%1"),
            meta={"name": "t", "provenance": {"source": "x"}})
        m2 = parse_text(to_text(m))
        self.assertEqual(module_to_dict(m2), module_to_dict(m))

    def test_syntax_error(self):
        with self.assertRaises(TaskIRSyntaxError):
            parse_text("not a taskir program at all")
        with self.assertRaises(TaskIRSyntaxError):
            parse_text("%1 = SEARCH(@task\nreturn %1")   # unbalanced parens

    def test_missing_return(self):
        with self.assertRaises(TaskIRSyntaxError):
            parse_text("%1 = SEARCH(@task)")


if __name__ == "__main__":
    unittest.main()
