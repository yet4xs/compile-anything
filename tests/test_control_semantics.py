"""Control-flow semantics: guard + VERIFY + SELECT (predicated execution).

Case A normal predicate execution; Case B nested guards (incl. the sharp
edge: VERIFY consuming a skipped value is a defined runtime failure);
Case C illegal guard (non-Bool cond) must be rejected by V4; Case D SELECT
with a skipped chosen branch / skipped cond (predication propagation).
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import Guard, Module, Node, Program
from src.validator.validator import validate
from src.runtime.simulator import Simulator


def prog(nodes, output, name="t"):
    return Module(program=Program(name=name, description=name,
                                  inputs=[{"name": "@task", "type": "Str"}],
                                  nodes=nodes, output=output))


class TestCaseA(unittest.TestCase):
    def test_predicate_execution(self):
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"], params={"domain": "flight"}),
            Node(id="%2", op="GENERATE", inputs=["%1", "@task"], params={"role": "draft"}),
            Node(id="%c", op="VERIFY", inputs=["%2"], params={"check": "facts"}),
            Node(id="%x", op="GENERATE", inputs=["%2"], params={"role": "polished"},
                 guard=Guard(cond="%c", expect=True)),
            Node(id="%y", op="GENERATE", inputs=["@task"], params={"role": "fallback"},
                 guard=Guard(cond="%c", expect=False)),
            Node(id="%out", op="SELECT", inputs=["%c", "%x", "%y"]),
        ], "%out")
        rep = validate(m)
        self.assertTrue(rep.valid, rep.summary())
        res = Simulator(m, seed="ca", jitter=0).run()
        self.assertEqual(res.status, "completed")
        by = {e.node: e for e in res.events}
        self.assertEqual(by["%x"].status, "ok")
        self.assertEqual(by["%y"].status, "skipped")
        self.assertEqual(res.skipped, 1)
        # SELECT picks the executed branch's value, not the skipped one
        self.assertEqual(res.values["%out"], res.values["%x"])
        self.assertNotEqual(res.values["%out"], "<skipped>")

    def test_predicate_execution_inverted(self):
        """cond False: the other branch runs, SELECT still correct."""
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%c", op="VERIFY", inputs=["%1"]),
            Node(id="%x", op="GENERATE", inputs=["%1"], guard=Guard(cond="%c", expect=True)),
            Node(id="%y", op="GENERATE", inputs=["@task"], guard=Guard(cond="%c", expect=False)),
            Node(id="%out", op="SELECT", inputs=["%c", "%x", "%y"]),
        ], "%out")
        res = Simulator(m, seed="ca", jitter=0, fail_plan={"%c": ["verify_false"]}).run()
        self.assertEqual(res.status, "completed")
        by = {e.node: e for e in res.events}
        self.assertEqual(by["%x"].status, "skipped")
        self.assertEqual(by["%y"].status, "ok")
        self.assertEqual(res.values["%out"], res.values["%y"])


class TestCaseB(unittest.TestCase):
    def test_nested_guard_happy_path(self):
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%c1", op="VERIFY", inputs=["%1"]),
            Node(id="%x", op="GENERATE", inputs=["%1"], guard=Guard(cond="%c1", expect=True)),
            Node(id="%c2", op="VERIFY", inputs=["%x"]),
            Node(id="%y", op="GENERATE", inputs=["%x"], guard=Guard(cond="%c2", expect=True)),
        ], "%y")
        rep = validate(m)
        self.assertTrue(rep.valid, rep.summary())       # guard DAG order correct
        res = Simulator(m, seed="cb", jitter=0).run()
        self.assertEqual(res.status, "completed")
        # guard edges are real dependencies: all 5 nodes on the critical path
        self.assertAlmostEqual(res.critical_path_ms, res.seq_latency_ms, places=3)

    def test_nested_guard_skipped_upstream_is_defined_failure(self):
        """%c1 false -> %x skipped -> %c2 consumes skipped -> defined failure.

        Documented semantics (docs/runtime-review.md): an unguarded consumer
        of a skipped value is a runtime error, not silent execution."""
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%c1", op="VERIFY", inputs=["%1"]),
            Node(id="%x", op="GENERATE", inputs=["%1"], guard=Guard(cond="%c1", expect=True)),
            Node(id="%c2", op="VERIFY", inputs=["%x"]),
            Node(id="%y", op="GENERATE", inputs=["%x"], guard=Guard(cond="%c2", expect=True)),
        ], "%y")
        res = Simulator(m, seed="cb", jitter=0,
                        fail_plan={"%c1": ["verify_false"]}).run()
        self.assertEqual(res.status, "failed")
        self.assertIn("consumes skipped value", res.output_digest)


class TestCaseC(unittest.TestCase):
    def test_guard_on_non_bool_rejected(self):
        m = prog([
            Node(id="%s", op="GENERATE", inputs=["@task"]),        # -> Str
            Node(id="%x", op="GENERATE", inputs=["@task"], guard=Guard(cond="%s", expect=True)),
        ], "%x")
        rep = validate(m)
        self.assertFalse(rep.valid)
        self.assertTrue(any(i.code == "TYPE_MISMATCH" for i in rep.errors))

    def test_guard_on_undeclared_ref_rejected(self):
        m = prog([
            Node(id="%x", op="GENERATE", inputs=["@task"], guard=Guard(cond="%nope", expect=True)),
        ], "%x")
        self.assertFalse(validate(m).valid)


class TestCaseD(unittest.TestCase):
    def _select_program(self):
        # cond always True (no injection): %sk is guarded OFF, %real runs
        return prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%c", op="VERIFY", inputs=["%1"]),
            Node(id="%sk", op="GENERATE", inputs=["@task"],
                 guard=Guard(cond="%c", expect=False)),
            Node(id="%real", op="GENERATE", inputs=["%1"]),
            Node(id="%out", op="SELECT", inputs=["%c", "%sk", "%real"]),
        ], "%out")

    def test_select_chosen_branch_skipped_is_defined_failure(self):
        """SELECT(cond=True, skipped, real): chosen branch was skipped ->
        inconsistent program -> defined runtime failure."""
        res = Simulator(self._select_program(), seed="cd", jitter=0).run()
        self.assertEqual(res.status, "failed")
        self.assertIn("chosen branch was skipped", res.output_digest)

    def test_select_nonchosen_branch_skipped_is_fine(self):
        """SELECT(cond=True, real, skipped): the skipped branch is the
        non-taken one -> legal, completes."""
        m = self._select_program()
        m.program.nodes[-1].inputs = ["%c", "%real", "%sk"]
        res = Simulator(m, seed="cd", jitter=0).run()
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.values["%out"], res.values["%real"])

    def test_select_cond_skipped_propagates(self):
        """If the SELECT's own cond is skipped (unknown), SELECT is skipped
        too (predication propagation), not silently treated as True."""
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%m", op="VERIFY", inputs=["%1"]),                    # True
            Node(id="%cond", op="VERIFY", inputs=["%1"],
                 guard=Guard(cond="%m", expect=False)),                   # skipped
            Node(id="%a", op="GENERATE", inputs=["%1"]),
            Node(id="%b", op="GENERATE", inputs=["@task"]),
            Node(id="%out", op="SELECT", inputs=["%cond", "%a", "%b"]),
        ], "%out")
        rep = validate(m)
        self.assertTrue(rep.valid, rep.summary())
        res = Simulator(m, seed="cd", jitter=0).run()
        self.assertEqual(res.status, "completed")
        by = {e.node: e for e in res.events}
        self.assertEqual(by["%out"].status, "skipped")
        self.assertIn("cond skipped", by["%out"].note)

    def test_guard_cond_skipped_propagates(self):
        """Guard whose cond value is skipped -> node skipped (not executed
        with bool(skipped)==True). Regression test for the propagation fix."""
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%m", op="VERIFY", inputs=["%1"]),                    # True
            Node(id="%cond", op="VERIFY", inputs=["%1"],
                 guard=Guard(cond="%m", expect=False)),                   # skipped
            Node(id="%x", op="GENERATE", inputs=["%1"],
                 guard=Guard(cond="%cond", expect=True)),
        ], "%x")
        res = Simulator(m, seed="cd", jitter=0).run()
        self.assertEqual(res.status, "completed")
        by = {e.node: e for e in res.events}
        self.assertEqual(by["%x"].status, "skipped")
        self.assertIn("predication propagation", by["%x"].note)


if __name__ == "__main__":
    unittest.main()
