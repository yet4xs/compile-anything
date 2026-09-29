"""Phase 5B-0.46 tests: semantic audit schema, checkers (argument loss,
coverage with bridges, SQL preservation, AST ops, RTL policy marking),
keyword detector modes."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]

from src.audit.semantic_label import (SemanticAuditResult, flag,    # noqa
                                      annotate,
                                      instruction_plan_consistency)
from src.audit.checkers import (audit_trajectory, audit_spider,     # noqa
                                audit_code, audit_rtl,
                                expected_ops_from_code)


def v3rec(plan_ops, lowering=None, instruction="", params=None,
          source="xlam", inputs=None, output=None):
    prog_inputs = inputs or [{"name": "@task", "type": "Str"}]
    first_ref = prog_inputs[0].get("name", "@task")
    nodes = [{"id": f"%c{i}", "op": op,
              "inputs": ([f"%c{i - 1}"] if i else [first_ref]),
              "params": (params or {}).get(i, {})}
             for i, op in enumerate(plan_ops)]
    if output is None and nodes:
        output = nodes[-1]["id"]
    return {"id": "t1", "source": source, "quality_tier": "A",
            "instruction": instruction, "plan_target": "",
            "plan_json": {"program": {"nodes": nodes, "output": output,
                                      "inputs": prog_inputs}},
            "lowering": lowering}


class TestSchemaAndFlag(unittest.TestCase):
    def test_flag_escalates_to_suspect(self):
        r = SemanticAuditResult(sample_id="x", source="s", tier="A")
        flag(r, "ARGUMENT_LOSS", "test")
        self.assertEqual(r.status, "suspect")

    def test_annotate_does_not_escalate(self):
        r = SemanticAuditResult(sample_id="x", source="s", tier="A",
                                status="verified")
        annotate(r, "POLICY_DERIVED", "SEARCH")
        self.assertEqual(r.status, "verified")
        self.assertIn("POLICY_DERIVED:SEARCH", r.reasons)


class TestTrajectoryChecker(unittest.TestCase):
    CALLS = [{"tool": "FlightSearch.search",
              "args": {"origin": "SFO", "destination": "Tokyo",
                       "date": "2026-10-01"}}]

    def test_faithful_case_verified(self):
        rec = v3rec(["SEARCH"], instruction="find flights",
                    params={0: {"domain": "flight",
                                "query": "SFO Tokyo 2026-10-01"}},
                    lowering=[{"skill": "SEARCH", "mapping_kind": "exact",
                               "tool": "FlightSearch.search"}])
        r = audit_trajectory(rec, self.CALLS)
        self.assertEqual(r.status, "verified", r.reasons)
        self.assertEqual(r.checks["argument_preservation_rate"], 1.0)

    def test_argument_loss_detected(self):
        rec = v3rec(["SEARCH"], instruction="find flights",
                    params={0: {"domain": "flight", "query": "SFO"}},
                    lowering=[{"skill": "SEARCH", "mapping_kind": "exact",
                               "tool": "FlightSearch.search"}])
        r = audit_trajectory(rec, self.CALLS)
        self.assertEqual(r.status, "suspect")
        self.assertTrue(any(x.startswith("ARGUMENT_LOSS")
                            for x in r.reasons))
        self.assertLess(r.checks["argument_preservation_rate"], 1.0)

    def test_coverage_counts_exclude_policy_ops(self):
        # EXTRACT bridge + policy GENERATE/VERIFY must not inflate actions
        rec = v3rec(["EXTRACT", "SEND", "GENERATE", "VERIFY"],
                    instruction="send it",
                    params={1: {"channel": "email", "query": "bob"}},
                    lowering=[{"skill": "SEND", "mapping_kind": "heuristic",
                               "tool": "EmailClient.send"}])
        r = audit_trajectory(rec, [{"tool": "EmailClient.send",
                                    "args": {"to": "bob@example.com"}}])
        # 1 raw call vs 1 action node (SEND) -> coverage ok; email arg lost
        self.assertFalse(any(x.startswith(("MISSING_ACTION", "EXTRA_ACTION"))
                             for x in r.reasons), r.reasons)

    def test_wrong_skill_detected(self):
        rec = v3rec(["SEARCH"], lowering=[{"skill": "SEND",
                                           "mapping_kind": "heuristic"}],
                    instruction="x")
        r = audit_trajectory(rec, self.CALLS)
        self.assertTrue(any(x.startswith("WRONG_SKILL") for x in r.reasons))

    def test_keyword_gap_is_annotation_not_suspect(self):
        rec = v3rec(["SEARCH"], instruction="find the cheapest flights",
                    params={0: {"domain": "flight",
                                "query": "SFO Tokyo 2026-10-01"}},
                    lowering=[{"skill": "SEARCH", "mapping_kind": "exact"}])
        r = audit_trajectory(rec, self.CALLS)
        self.assertEqual(r.status, "verified", r.reasons)
        self.assertTrue(any(x.startswith("GROUND_TRUTH_AMBIGUOUS")
                            for x in r.reasons))


class TestSpiderChecker(unittest.TestCase):
    def test_payload_preserved(self):
        rec = v3rec(["QUERY_DB"],
                    params={0: {"table": "t",
                                "query": "SELECT count(*) FROM t "
                                         "WHERE age > 56"}},
                    source="spider",
                    instruction="How many heads are older than 56?")
        r = audit_spider(rec, "SELECT count(*) FROM t WHERE age > 56")
        self.assertEqual(r.status, "verified", r.reasons)

    def test_clause_loss_detected(self):
        rec = v3rec(["QUERY_DB"], params={0: {"table": "t",
                                              "query": "SELECT a FROM t"}},
                    source="spider", instruction="list")
        r = audit_spider(rec, "SELECT a FROM t WHERE x > 1 LIMIT 5")
        self.assertEqual(r.status, "suspect")
        self.assertTrue(any("WHERE" in x or "clause" in x
                            for x in r.reasons))

    def test_count_keyword_satisfied_by_sql(self):
        rec = v3rec(["QUERY_DB"],
                    params={0: {"table": "t",
                                "query": "SELECT count(*) FROM t"}},
                    source="spider", instruction="How many rows?")
        r = audit_spider(rec, "SELECT count(*) FROM t")
        self.assertEqual(r.status, "verified", r.reasons)


class TestCodeChecker(unittest.TestCase):
    def test_expected_ops(self):
        self.assertEqual(
            expected_ops_from_code(
                "def f(x):\n    return sorted(x, key=lambda a: a['p'])"),
            ["SORT"])
        self.assertEqual(
            expected_ops_from_code(
                "def f(x):\n    return min(x, key=lambda a: a.p)"),
            ["ARGMIN"])

    def test_matching_ops_verified(self):
        rec = v3rec(["SORT"], source="humaneval",
                    inputs=[{"name": "@items", "type": "List[Any]"}],
                    instruction="sort by price")
        r = audit_code(rec, "def f(items):\n"
                       "    return sorted(items, key=lambda a: a['p'])")
        self.assertEqual(r.status, "verified", r.reasons)

    def test_missing_op_detected(self):
        rec = v3rec(["COUNT"], source="humaneval",
                    inputs=[{"name": "@items", "type": "List[Any]"}])
        r = audit_code(rec, "def f(items):\n    return sorted(items)")
        self.assertEqual(r.status, "suspect")
        self.assertTrue(any(x.startswith("MISSING_ACTION")
                            for x in r.reasons))

    def test_global_inputs_accepted(self):
        # code programs declare @items — def-use must accept it
        rec = v3rec(["SORT"], source="humaneval",
                    inputs=[{"name": "@items", "type": "List[Any]"}])
        r = audit_code(rec, "def f(items):\n    return sorted(items)")
        self.assertFalse(any(x.startswith("MISSING_DEPENDENCY")
                             for x in r.reasons))

    def test_mbpp_non_strict_is_annotation(self):
        rec = v3rec(["TRANSFORM"], source="mbpp",
                    inputs=[{"name": "@items", "type": "List[Any]"}])
        r = audit_code(rec, "def f(items):\n    x = 1\n    return x",
                       strict_ops=False)
        self.assertEqual(r.status, "verified", r.reasons)
        self.assertTrue(any(x.startswith("GROUND_TRUTH_AMBIGUOUS")
                            for x in r.reasons))


class TestRTLChecker(unittest.TestCase):
    def test_policy_ops_annotated_not_suspect(self):
        rec = v3rec(["LOAD", "EXTRACT", "SEARCH", "CODEGEN", "VERIFY"],
                    source="verilogeval", instruction="fix the FIFO")
        r = audit_rtl(rec, "fix the FIFO overflow assertion")
        self.assertEqual(r.status, "verified", r.reasons)
        self.assertTrue(any("POLICY_DERIVED:VERIFY" == x
                            for x in r.reasons))
        self.assertTrue(any(x.startswith("TEMPLATE_BIAS")
                            for x in r.reasons))
        ev = r.checks["op_evidence"]
        self.assertEqual(ev["mandatory_from_ground_truth"], 3)

    def test_missing_codegen_suspect(self):
        rec = v3rec(["LOAD", "EXTRACT"], source="verilogeval",
                    instruction="fix")
        r = audit_rtl(rec, "fix it")
        self.assertEqual(r.status, "suspect")


class TestAuditArtifacts(unittest.TestCase):
    def test_full_audit_report_exists_and_complete(self):
        p = ROOT / "data" / "reports" / "semantic_label_audit.json"
        if not p.exists():
            self.skipTest("audit not run")
        d = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(d["summary"]["total"], 31215)
        self.assertEqual(len(d["results"]), 31215)
        statuses = {r["status"] for r in d["results"]}
        self.assertTrue(statuses <= {"verified", "suspect", "unverifiable"})

    def test_sanity_overfit_256_verified(self):
        p = ROOT / "data" / "sanity_overfit" / "samples.jsonl"
        if not p.exists():
            self.skipTest("sanity set not built")
        rows = [json.loads(l) for l in
                p.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(rows), 256)

    def test_judge_sample_has_ground_truth(self):
        p = ROOT / "data" / "reports" / "semantic_audit_sample.jsonl"
        if not p.exists():
            self.skipTest("sample not built")
        rows = [json.loads(l) for l in
                p.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertTrue(rows)
        for r in rows[:20]:
            self.assertIn("raw_ground_truth_summary", r)
            self.assertIn("taskir", r)
            self.assertIn("instruction", r)


if __name__ == "__main__":
    unittest.main()
