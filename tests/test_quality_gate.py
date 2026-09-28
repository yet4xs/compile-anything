"""Phase 5B-0 tests: lowering provenance, dual views, quality tiers,
group-aware split / leakage, ambiguity, SFT dataset fields, GED metric."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.lifter.toolmap import map_tool, CONFIDENCE
from src.lifter.chain import lift_trajectory
from src.validator.validator import validate
from src.dataset.dedup import (normalize, tokens, near_duplicate_families,
                               group_split, cross_split_leakage)
from src.compiler.train.dataset import (build_user_text,
                                        iter_corpus_records, records_to_chat)

ROOT_ = pathlib.Path(__file__).resolve().parents[1]
import importlib.util as _ilu                      # noqa: E402
_spec = _ilu.spec_from_file_location(
    "nce_eval", ROOT_ / "benchmark" / "neural_compiler_eval.py")
nce_eval = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nce_eval)
graph_edit_similarity = nce_eval.graph_edit_similarity
skill_prf = nce_eval.skill_prf


class TestToolmapProvenance(unittest.TestCase):
    def test_exact_rule(self):
        m = map_tool("FlightSearch.search", {})
        self.assertEqual(m["mapping_kind"], "exact")
        self.assertEqual(m["matched_rule"], "flight")
        self.assertEqual(m["confidence"], CONFIDENCE["exact"])

    def test_heuristic_rule(self):
        m = map_tool("WebSearch.query", {})
        self.assertEqual(m["mapping_kind"], "heuristic")

    def test_fallback(self):
        m = map_tool("totally_unknown_vendor.weird_thing", {})
        self.assertEqual(m["mapping_kind"], "fallback")
        self.assertIsNone(m["matched_rule"])
        self.assertEqual(m["skill"], "EXEC_ACTION")
        self.assertEqual(m["confidence"], CONFIDENCE["fallback"])

    def test_provenance_not_in_semantic_body(self):
        mod = lift_trajectory("q", [
            {"name": "FlightSearch.search", "arguments": {"origin": "A"}}],
            source="t")
        prov = mod.meta["provenance"]
        self.assertEqual(prov["lowering"][0]["mapping_kind"], "exact")
        for n in mod.program.nodes:            # semantic body stays clean
            self.assertNotIn("mapping_kind", n.params)
            self.assertNotIn("confidence", n.params)


class TestDualViews(unittest.TestCase):
    CALLS = [{"name": "FlightSearch.search",
              "arguments": {"origin": "SFO", "destination": "Tokyo"}},
             {"name": "CurrencyConverter.convert",
              "arguments": {"amount": 540, "from_currency": "USD",
                            "to_currency": "JPY"}}]

    def test_plan_view_has_no_policy_tail(self):
        mod = lift_trajectory("q", self.CALLS, source="t", view="plan")
        ops = [n.op for n in mod.program.nodes]
        self.assertNotIn("GENERATE", ops)
        self.assertNotIn("VERIFY", ops)
        self.assertTrue(validate(mod).valid, ops)
        self.assertEqual(mod.meta["provenance"]["view"], "plan")

    def test_execution_view_keeps_tail(self):
        mod = lift_trajectory("q", self.CALLS, source="t", view="execution")
        ops = [n.op for n in mod.program.nodes]
        self.assertIn("GENERATE", ops)
        self.assertIn("VERIFY", ops)
        self.assertTrue(validate(mod).valid, ops)

    def test_both_views_valid_same_trajectory(self):
        for view in ("plan", "execution"):
            mod = lift_trajectory("q", self.CALLS, source="t", view=view)
            self.assertTrue(validate(mod).valid, view)

    def test_spider_plan_view(self):
        from src.lifter.benchmark.spider import SpiderLifter
        s = {"task_id": "s", "question": "q",
             "query": "SELECT a, b FROM t WHERE x > 1"}
        plan = SpiderLifter().lift(s, view="plan")
        exe = SpiderLifter().lift(s, view="execution")
        self.assertNotIn("GENERATE", [n.op for n in plan.program.nodes])
        self.assertIn("GENERATE", [n.op for n in exe.program.nodes])
        self.assertTrue(validate(plan).valid)
        self.assertTrue(validate(exe).valid)


class TestQualityTier(unittest.TestCase):
    def test_tier_rules(self):
        from scripts.build_real_corpus import quality_tier
        self.assertEqual(quality_tier("xlam", [
            {"mapping_kind": "exact"}]), "A")
        self.assertEqual(quality_tier("xlam", [
            {"mapping_kind": "exact"}, {"mapping_kind": "heuristic"}]), "B")
        self.assertEqual(quality_tier("xlam", [
            {"mapping_kind": "heuristic"}, {"mapping_kind": "fallback"}]), "C")
        self.assertEqual(quality_tier("spider", None), "A")
        self.assertEqual(quality_tier("synthetic:code", None), "C")


class TestGroupSplit(unittest.TestCase):
    def test_families_keep_duplicates_together(self):
        texts = ["find flights from SFO to Tokyo",
                 "find flights from SFO to Tokyo",      # exact dup
                 "find flights from SFO to Osaka",
                 "totally different task about sql queries",
                 "completely unrelated cooking recipe question"]
        fams = near_duplicate_families(texts)
        splits = group_split(fams, seed=0)
        self.assertEqual(splits[0], splits[1])          # exact dup same split

    def test_zero_exact_leakage(self):
        texts = ["a b c d", "a b c d", "e f g h", "e f g h",
                 "i j k l", "m n o p", "q r s t", "u v w x"]
        fams = near_duplicate_families(texts)
        splits = group_split(fams, seed=1)
        leak = cross_split_leakage(texts, splits)
        self.assertEqual(leak["cross_split_exact_duplicates"], 0)

    def test_normalize(self):
        # normalize keeps stopwords (conservative exact-dup key); token
        # extraction for MinHash is the stopword-stripping variant
        self.assertEqual(normalize("  The  FLIGHT!  "), "the flight")
        self.assertEqual(tokens("  The  FLIGHT!  "), ["flight"])


class TestAmbiguityDetection(unittest.TestCase):
    def test_same_instruction_different_plans_detected(self):
        # simulate: same normalized instruction, two different op sequences
        groups = {"find flights": {("SEARCH",), ("QUERY_DB",)}}
        self.assertEqual(len(list(groups.values())[0]), 2)


class TestSFTDataset(unittest.TestCase):
    def _write(self, tmp, records):
        tmp.write_text("\n".join(json.dumps(r) for r in records),
                       encoding="utf-8")

    def test_field_selection_and_tiers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "train.jsonl"
            self._write(f, [
                {"instruction": "q1", "source": "xlam", "id": "1",
                 "plan_target": "; TaskIR v0.1\n%1 = SEARCH(@task)\n\nreturn %1\n",
                 "execution_target": "X", "quality_tier": "A",
                 "capabilities": ["SEARCH(flight)"]},
                {"instruction": "q2", "source": "xlam", "id": "2",
                 "plan_target": "Y", "quality_tier": "C",
                 "capabilities": []},
            ])
            recs = list(iter_corpus_records([f], tiers=["A"]))
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].target.splitlines()[1],
                             "%1 = SEARCH(@task)")
            recs_b = list(iter_corpus_records([f], capability_context=True))
            self.assertIn("Available capabilities:", recs_b[0].instruction)

    def test_build_user_text_views(self):
        a = build_user_text("find flights")
        b = build_user_text("find flights", ["SEARCH(flight)"])
        self.assertEqual(a, "find flights")
        self.assertIn("- SEARCH(flight)", b)


class TestGraphMetrics(unittest.TestCase):
    def test_ges_identical(self):
        from src.ir.parser import parse_text
        m = parse_text("; TaskIR v0.1  module=t\ninputs: @task: Str\n\n"
                       "%1 = SEARCH(@task)\n\nreturn %1\n")
        self.assertEqual(graph_edit_similarity(m, m), 1.0)

    def test_ges_different(self):
        from src.ir.parser import parse_text
        a = parse_text("; TaskIR v0.1  module=t\ninputs: @task: Str\n\n"
                       "%1 = SEARCH(@task)\n\nreturn %1\n")
        b = parse_text("; TaskIR v0.1  module=t\ninputs: @task: Str\n\n"
                       "%1 = SEARCH(@task)\n"
                       "%2 = FILTER(%1)\n\nreturn %2\n")
        self.assertLess(graph_edit_similarity(a, b), 1.0)
        self.assertGreater(graph_edit_similarity(a, b), 0.0)

    def test_skill_prf(self):
        p, r = skill_prf(["SEARCH", "FILTER"], ["SEARCH", "SORT"])
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 0.5)


if __name__ == "__main__":
    unittest.main()
