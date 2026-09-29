"""Phase 5B-0.45 tests: eval schema purity, adapter counts (raw ==
normalized), BFCL input/ground-truth separation, oracle representability,
suite freezing."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]

from src.eval.schema import EvalSample                      # noqa: E402
from src.eval.adapters import (load_bfcl, load_tau3,        # noqa: E402
                               load_agentboard, load_rtl_repo, load_bird)
from src.eval.oracle_bfcl import oracle, map_function       # noqa: E402


class TestEvalSchemaPurity(unittest.TestCase):
    def test_no_taskir_target_field(self):
        s = EvalSample(benchmark="x", case_id="1")
        self.assertFalse(hasattr(s, "taskir_target"))
        with self.assertRaises(ValueError):
            EvalSample(benchmark="x", case_id="1",
                       metadata={"taskir_target": "; TaskIR"})

    def test_eval_package_does_not_import_dataset_pipeline(self):
        import src.eval.adapters.bfcl as b
        src_text = pathlib.Path(b.__file__).read_text(encoding="utf-8")
        self.assertNotIn("src.dataset", src_text.replace(
            "src.dataset.external_guard", ""))


class TestAdapterCounts(unittest.TestCase):
    """raw == normalized (audit-verified raw counts from Phase 5B-0.4)."""

    def test_bfcl(self):
        s = load_bfcl()
        self.assertEqual(len(s), 4696)      # +1 format_sensitivity doc
        cats = {x.metadata["category"] for x in s}
        self.assertIn("multi_turn_base", cats)
        self.assertIn("memory", cats)

    def test_tau3(self):
        s = load_tau3()
        self.assertEqual(len(s), 2546)
        self.assertEqual({x.metadata["domain"] for x in s},
                         {"airline", "retail", "telecom",
                          "banking_knowledge"})

    def test_agentboard(self):
        s = load_agentboard(tasks=["tool-query", "tool-operation",
                                   "webshop"])
        self.assertEqual(len(s), 351)

    def test_rtl_repo(self):
        self.assertEqual(len(load_rtl_repo("test")), 1174)
        self.assertEqual(len(load_rtl_repo("train")), 2924)

    def test_bird(self):
        self.assertEqual(len(load_bird()), 500)


class TestBFCLSeparation(unittest.TestCase):
    """Ground truth must NEVER appear in the model input."""

    def setUp(self):
        self.samples = load_bfcl()

    def test_gt_not_in_instruction(self):
        for s in self.samples[:500]:
            if not s.reference_actions:
                continue
            gt_str = json.dumps(s.reference_actions, default=str)
            self.assertNotIn(gt_str[:40], s.instruction)

    def test_irrelevance_has_empty_actions(self):
        irr = [s for s in self.samples
               if s.metadata["kind"] == "irrelevance"]
        self.assertTrue(irr)
        # irrelevance ground truth = no call (or empty list)
        for s in irr[:100]:
            if s.reference_actions is not None:
                self.assertEqual(s.reference_actions, [])

    def test_capability_defs_present_for_simple(self):
        s = next(x for x in self.samples
                 if x.metadata["category"] == "live_simple")
        self.assertTrue(s.capabilities)     # function defs available
        # OpenAI-style function schema: name + parameters
        self.assertIn("name", s.capabilities[0])
        self.assertIn("parameters", s.capabilities[0])


class TestBFCLOracle(unittest.TestCase):
    def test_map_function_deterministic(self):
        self.assertEqual(map_function("get_user_info")[0], "SEARCH")
        self.assertEqual(map_function("cd")[0], "EXEC_ACTION")
        self.assertEqual(map_function("send_email")[0], "SEND")
        self.assertEqual(map_function("book_reservation")[0], "EXEC_ACTION")

    def test_oracle_irrelevance_is_generate_only(self):
        s = next(x for x in load_bfcl()
                 if x.metadata["kind"] == "irrelevance")
        r = oracle(s)
        self.assertEqual(r["status"], "full")
        ops = [n.op for n in r["module"].program.nodes]
        self.assertEqual(ops, ["GENERATE"])

    def test_oracle_memory_unsupported(self):
        s = next(x for x in load_bfcl() if x.metadata["kind"] == "memory")
        r = oracle(s)
        self.assertEqual(r["status"], "none")
        self.assertIn("memory", r["reasons"][0])

    def test_oracle_parallel_not_fabricated_dataflow(self):
        s = next(x for x in load_bfcl()
                 if x.metadata["kind"] == "parallel"
                 and x.reference_actions)
        r = oracle(s)
        mod = r["module"]
        # parallel calls are independent: no node consumes another node
        for n in mod.program.nodes:
            for inp in n.inputs:
                self.assertNotIn("%", inp)   # only @task, no chaining

    def test_representability_counts_match_report(self):
        rep = json.loads((ROOT / "data" / "external_benchmarks" /
                          "derived_oracle" /
                          "bfcl_v4_representability.json"
                          ).read_text(encoding="utf-8"))
        self.assertEqual(sum(rep["counts"].values()), 4696)
        self.assertIn("full", rep["counts"])


class TestEvalSuitesFrozen(unittest.TestCase):
    def test_suites_exist_with_ids_only(self):
        for name in ("bfcl_v4", "tau3_bench", "agentboard_tools_tool_query",
                     "agentboard_tools_tool_operation", "agentboard_webshop",
                     "rtl_repo", "bird_mini_dev", "toolbench_full"):
            p = ROOT / "data" / "external_benchmarks" / "eval_suites" / \
                f"{name}.json"
            self.assertTrue(p.exists(), name)
            suite = json.loads(p.read_text(encoding="utf-8"))
            self.assertTrue(suite["cases"])
            for c in suite["cases"][:20]:
                self.assertNotIn("instruction", c)   # ids + hashes only
                self.assertIn("case_id", c)
                self.assertIn("sha", c)


if __name__ == "__main__":
    unittest.main()
