"""Retry semantics: error retry, VERIFY-driven rollback (memo invalidation),
and validator rejection of illegal retry constructs."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import Module, Node, Program, Retry
from src.validator.validator import validate
from src.runtime.simulator import Simulator


def prog(nodes, output, name="t"):
    return Module(program=Program(name=name, description=name,
                                  inputs=[{"name": "@task", "type": "Str"}],
                                  nodes=nodes, output=output))


class TestCaseA(unittest.TestCase):
    def test_error_retry_event_sequence(self):
        m = prog([
            Node(id="%a", op="SEARCH", inputs=["@task"],
                 retry=Retry(max_attempts=3, on="error")),
            Node(id="%b", op="FILTER", inputs=["%a"]),
        ], "%b")
        res = Simulator(m, seed="ra", jitter=0,
                        fail_plan={"%a": ["error"]}).run()
        self.assertEqual(res.status, "completed")
        ev = [e for e in res.events if e.node == "%a"]
        self.assertEqual([e.status for e in ev], ["error", "ok"])
        self.assertEqual([e.attempt for e in ev], [1, 2])
        self.assertEqual(res.retries, 1)
        self.assertEqual(len([e for e in res.events if e.node == "%b"]), 1)

    def test_error_retry_exhaustion_fails(self):
        m = prog([
            Node(id="%a", op="SEARCH", inputs=["@task"],
                 retry=Retry(max_attempts=2, on="error")),
            Node(id="%b", op="FILTER", inputs=["%a"]),
        ], "%b")
        res = Simulator(m, seed="ra", jitter=0,
                        fail_plan={"%a": ["error", "error"]}).run()
        self.assertEqual(res.status, "failed")
        self.assertEqual(len([e for e in res.events if e.node == "%a"]), 2)


class TestCaseB(unittest.TestCase):
    def test_verify_rollback_reexecutes_chain_without_stale_memo(self):
        """%A -> %B -> %C(VERIFY); %A carries retry(on=%C). VERIFY returns
        False on its first evaluation: A and B must both re-execute (their
        memos invalidated), and C must re-evaluate against the NEW values."""
        m = prog([
            Node(id="%a", op="SEARCH", inputs=["@task"],
                 retry=Retry(max_attempts=2, on="%c")),
            Node(id="%b", op="GENERATE", inputs=["%a"], params={"role": "mid"}),
            Node(id="%c", op="VERIFY", inputs=["%b"], params={"check": "ok"}),
        ], "%b")
        res = Simulator(m, seed="rb", jitter=0,
                        fail_plan={"%c": ["verify_false"]}).run()
        self.assertEqual(res.status, "completed")

        a_ev = [e for e in res.events if e.node == "%a"]
        b_ev = [e for e in res.events if e.node == "%b"]
        c_ev = [e for e in res.events if e.node == "%c"]
        # both A and B re-executed (memo invalidation reached B, not just A)
        self.assertEqual(len(a_ev), 2)
        self.assertEqual(len(b_ev), 2)
        self.assertEqual(len(c_ev), 2)
        # B really re-executed (attempt counter advanced), not stale memo reuse
        self.assertEqual([e.attempt for e in b_ev], [1, 2])
        # final verdict flipped to True after rollback + retry
        self.assertEqual(res.values["%c"], True)
        self.assertEqual(res.retries, 1)

    def test_rollback_ordering_invariant(self):
        """After rollback, every dependency's last-ok event precedes its
        consumer's last-ok event (execution order respects the DAG)."""
        m = prog([
            Node(id="%a", op="SEARCH", inputs=["@task"],
                 retry=Retry(max_attempts=3, on="%c")),
            Node(id="%b", op="GENERATE", inputs=["%a"]),
            Node(id="%c", op="VERIFY", inputs=["%b"]),
        ], "%b")
        res = Simulator(m, seed="rb2", jitter=0,
                        fail_plan={"%c": ["verify_false", "verify_false"]}).run()
        self.assertEqual(res.status, "completed")
        last_ok = {}
        for e in res.events:
            if e.status in ("ok", "retry_exhausted") and not e.superseded:
                last_ok[e.node] = e.seq
        self.assertLess(last_ok["%a"], last_ok["%b"])
        self.assertLess(last_ok["%b"], last_ok["%c"])


class TestCaseC(unittest.TestCase):
    def test_retry_on_non_verify_rejected(self):
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%2", op="GENERATE", inputs=["%1"],
                 retry=Retry(max_attempts=2, on="%1")),      # %1 is SEARCH
        ], "%2")
        rep = validate(m)
        self.assertFalse(rep.valid)
        self.assertTrue(any(i.code == "CONTROL" for i in rep.errors))

    def test_retry_on_undefined_rejected(self):
        m = prog([
            Node(id="%2", op="GENERATE", inputs=["@task"],
                 retry=Retry(max_attempts=2, on="%ghost")),
        ], "%2")
        rep = validate(m)
        self.assertFalse(rep.valid)
        self.assertTrue(any(i.code == "UNDEFINED_REF" for i in rep.errors))

    def test_retry_verify_must_depend_on_node(self):
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%2", op="SEARCH", inputs=["@task"]),
            Node(id="%v", op="VERIFY", inputs=["%2"]),       # depends on %2
            Node(id="%3", op="GENERATE", inputs=["%1"],
                 retry=Retry(max_attempts=2, on="%v")),      # ...not on %3
        ], "%3")
        rep = validate(m)
        self.assertFalse(rep.valid)
        self.assertTrue(any(i.code == "CONTROL" for i in rep.errors))

    def test_retry_max_attempts_zero_rejected(self):
        m = prog([
            Node(id="%2", op="GENERATE", inputs=["@task"],
                 retry=Retry(max_attempts=0, on="error")),
        ], "%2")
        rep = validate(m)
        self.assertFalse(rep.valid)
        self.assertTrue(any(i.code == "CONTROL" for i in rep.errors))


if __name__ == "__main__":
    unittest.main()
