import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import Guard, Module, Node, Program
from src.stats import aggregate, program_metrics


def m(nodes, output):
    return Module(program=Program(name="t", inputs=[{"name": "@task",
                                                     "type": "Str"}],
                                  nodes=nodes, output=output))


class TestStats(unittest.TestCase):
    def test_chain_metrics(self):
        mod = m([Node(id="%1", op="SEARCH", inputs=["@task"]),
                 Node(id="%2", op="FILTER", inputs=["%1"]),
                 Node(id="%3", op="ARGMIN", inputs=["%2"])], "%3")
        s = program_metrics(mod)
        self.assertEqual(s["nodes"], 3)
        self.assertEqual(s["depth"], 3)
        self.assertEqual(s["width"], 1)
        self.assertEqual(s["parallel_ratio"], 0.0)
        self.assertFalse(s["has_parallelism"])

    def test_parallel_metrics(self):
        mod = m([Node(id="%1", op="SEARCH", inputs=["@task"]),
                 Node(id="%2", op="SEARCH", inputs=["@task"]),
                 Node(id="%3", op="MERGE", inputs=["%1", "%2"])], "%3")
        s = program_metrics(mod)
        self.assertEqual(s["nodes"], 3)
        self.assertEqual(s["depth"], 2)
        self.assertEqual(s["width"], 2)
        self.assertAlmostEqual(s["parallel_ratio"], round(1 - 2 / 3, 4),
                               places=4)
        self.assertTrue(s["has_parallelism"])

    def test_branch_ratio(self):
        mod = m([Node(id="%1", op="SEARCH", inputs=["@task"]),
                 Node(id="%2", op="VERIFY", inputs=["%1"]),
                 Node(id="%3", op="GENERATE", inputs=["%1"],
                      guard=Guard(cond="%2", expect=True)),
                 Node(id="%4", op="SELECT", inputs=["%2", "%3", "%3"])], "%4")
        s = program_metrics(mod)
        self.assertAlmostEqual(s["branch_ratio"], 2 / 4)
        self.assertEqual(s["selects"], 1)
        self.assertEqual(s["guarded"], 1)

    def test_aggregate(self):
        mods = [m([Node(id="%1", op="SEARCH", inputs=["@task"])], "%1"),
                m([Node(id="%1", op="SEARCH", inputs=["@task"]),
                   Node(id="%2", op="FILTER", inputs=["%1"])], "%2")]
        agg = aggregate([program_metrics(x) for x in mods], subset="x")
        self.assertEqual(agg["programs"], 2)
        self.assertEqual(agg["total_nodes"], 3)
        self.assertEqual(agg["skill_frequency"]["SEARCH"], 2)


if __name__ == "__main__":
    unittest.main()
