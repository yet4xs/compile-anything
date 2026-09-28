"""Scheduler cost validation (Phase 3 Task 4).

Plan A: one 70B model answers directly (BIG_MODEL_BASELINE reference).
Plan B: compiled pipeline SEARCH -> FILTER -> VERIFY(2B) -> GENERATE(2B).

The scheduler must produce, for plan B: feasible schedule (dep-respecting),
makespan, energy and peak memory — and the pipeline must dominate the 70B
single-shot baseline on every cost axis with nominal numbers.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import Module, Node, Program
from src.validator.validator import validate
from src.optimizer.scheduler import ListScheduler
from src.runtime.simulator import Simulator
from src.isa.registry import BIG_MODEL_BASELINE, ALL_OPS
from src.cost.model import baseline_flops


def pipeline_program() -> Module:
    return Module(program=Program(
        name="plan_b_pipeline",
        description="Plan B: compiled pipeline",
        inputs=[{"name": "@task", "type": "Str"}],
        nodes=[
            Node(id="%1", op="SEARCH", inputs=["@task"],
                 params={"domain": "flight"}),
            Node(id="%2", op="FILTER", inputs=["%1"], params={"predicate": "p"}),
            Node(id="%3", op="VERIFY", inputs=["%2"]),
            Node(id="%4", op="GENERATE", inputs=["%3", "@task"]),
        ],
        output="%4"))


class TestScheduleCorrectness(unittest.TestCase):
    def test_pipeline_metrics(self):
        mod = pipeline_program()
        self.assertTrue(validate(mod).valid)
        sched = ListScheduler(mod).schedule()
        self.assertTrue(sched.ok, sched.notes)

        c = {k: ALL_OPS[k].cost for k in
             ("SEARCH", "FILTER", "VERIFY", "GENERATE")}
        seq = sum(x.latency_ms for x in c.values())
        # chain: makespan == sequential latency == critical path
        self.assertAlmostEqual(sched.makespan_ms, seq, places=3)
        res = Simulator(mod, seed="sc", jitter=0).run()
        self.assertAlmostEqual(res.critical_path_ms, seq, places=3)

    def test_dependency_respect_and_order(self):
        mod = pipeline_program()
        sched = ListScheduler(mod).schedule()
        end_of = {i.node: i.end_ms for i in sched.items}
        start_of = {i.node: i.start_ms for i in sched.items}
        self.assertLessEqual(end_of["%1"], start_of["%2"])
        self.assertLessEqual(end_of["%2"], start_of["%3"])
        self.assertLessEqual(end_of["%3"], start_of["%4"])
        self.assertEqual(sched.order, ["%1", "%2", "%3", "%4"])

    def test_resource_grouping_serializes_same_class(self):
        """Two independent GENERATEs on one lm executor serialize."""
        mod = Module(program=Program(
            name="two_lm", inputs=[{"name": "@task", "type": "Str"}],
            nodes=[Node(id="%a", op="GENERATE", inputs=["@task"]),
                   Node(id="%b", op="GENERATE", inputs=["@task"])],
            output="%a"))
        one = ListScheduler(mod, pools={"lm": 1}).schedule()
        two = ListScheduler(mod, pools={"lm": 2}).schedule()
        gen = ALL_OPS["GENERATE"].cost.latency_ms
        self.assertAlmostEqual(one.makespan_ms, 2 * gen, places=3)
        self.assertAlmostEqual(two.makespan_ms, gen, places=3)

    def test_parallel_searches_overlap(self):
        """flight + hotel SEARCH run in parallel when the api pool allows."""
        mod = Module(program=Program(
            name="par", inputs=[{"name": "@task", "type": "Str"}],
            nodes=[Node(id="%f", op="SEARCH", inputs=["@task"],
                        params={"domain": "flight"}),
                   Node(id="%h", op="SEARCH", inputs=["@task"],
                        params={"domain": "hotel"}),
                   Node(id="%m", op="MERGE", inputs=["%f", "%h"])],
            output="%m"))
        pooled = ListScheduler(mod, pools={"api": 2}).schedule()
        serial = ListScheduler(mod, pools={"api": 1}).schedule()
        search = ALL_OPS["SEARCH"].cost.latency_ms
        merge = ALL_OPS["MERGE"].cost.latency_ms
        self.assertAlmostEqual(pooled.makespan_ms, search + merge, places=3)
        self.assertAlmostEqual(serial.makespan_ms, 2 * search + merge, places=3)
        # scheduled makespan respects the simulator's critical-path lower bound
        res = Simulator(mod, seed="sp", jitter=0).run()
        self.assertGreaterEqual(pooled.makespan_ms + 1e-6, res.critical_path_ms)

    def test_effect_ordering_via_after_edges(self):
        """Ordering rules (today: `after`; v0.2: effect tokens) are respected:
        SEND(confirmation) must not run before SAVE(result) completes."""
        mod = Module(program=Program(
            name="ordered_effects", inputs=[{"name": "@task", "type": "Str"}],
            nodes=[Node(id="%1", op="SEARCH", inputs=["@task"]),
                   Node(id="%2", op="SAVE", inputs=["%1"]),
                   Node(id="%3", op="SEND", inputs=["%1"]),
                   Node(id="%4", op="GENERATE", inputs=["%2"])],
            output="%4"))
        mod.program.nodes[2].after = ["%2"]      # SEND after SAVE
        sched = ListScheduler(mod).schedule()
        self.assertTrue(sched.ok, sched.notes)
        end_of = {i.node: i.end_ms for i in sched.items}
        start_of = {i.node: i.start_ms for i in sched.items}
        self.assertLessEqual(end_of["%2"], start_of["%3"])


class TestCostComparisonVs70B(unittest.TestCase):
    """Plan A (70B single-shot) vs Plan B (compiled pipeline)."""

    def setUp(self):
        self.mod = pipeline_program()
        self.sched = ListScheduler(self.mod).schedule()
        self.res = Simulator(self.mod, seed="cmp", jitter=0).run()
        self.assertEqual(self.res.status, "completed")

    def test_latency(self):
        self.assertLess(self.sched.makespan_ms,
                        BIG_MODEL_BASELINE["latency_ms"])

    def test_energy(self):
        self.assertLess(self.res.energy_j, BIG_MODEL_BASELINE["energy_j"])
        ratio = self.res.energy_j / BIG_MODEL_BASELINE["energy_j"]
        self.assertLess(ratio, 0.10)          # nominal: <10% of 70B energy

    def test_memory(self):
        self.assertLess(self.res.peak_memory_mb,
                        BIG_MODEL_BASELINE["memory_mb"])

    def test_flops(self):
        self.assertLess(self.res.flops, baseline_flops())

    def test_report_fields_present(self):
        """All protocol fields computable from scheduler + simulator."""
        self.assertGreater(self.sched.makespan_ms, 0)
        self.assertGreater(self.res.critical_path_ms, 0)
        self.assertGreaterEqual(self.res.energy_j, 0)
        self.assertGreaterEqual(self.res.peak_memory_mb, 0)
        self.assertGreaterEqual(self.res.lm_calls, 1)
        self.assertGreaterEqual(self.res.api_calls, 1)
        self.assertEqual(len(self.sched.items), 4)


if __name__ == "__main__":
    unittest.main()
