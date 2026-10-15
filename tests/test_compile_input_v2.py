"""T1 unit tests: field order, missing fields, description exclusion,
non-ASCII, retail equivalence with the old dict-flatten."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.compile_input_v2 import render_compile_input

SCENARIO = {
    "instructions": {
        "task_instructions": "You are polite.",
        "reason_for_call": "You want to change your flight.",
        "known_info": "You are Yusuf Rossi, zip 19122. Your user id is "
                      "yusuf_rossi_9620.",
        "unknown_info": "You do not remember your email.",
    },
    "persona": None,
}


class CompileInputV2Test(unittest.TestCase):
    def test_fixed_field_order(self):
        out = render_compile_input(SCENARIO)
        idx = [out.index(f + ":") for f in
               ("reason_for_call", "known_info", "unknown_info",
                "task_instructions")]
        self.assertEqual(idx, sorted(idx), out)

    def test_missing_fields_omitted(self):
        out = render_compile_input(
            {"instructions": {"reason_for_call": "x"}})
        self.assertEqual(out, "reason_for_call: x")

    def test_description_never_rendered(self):
        task = {"user_scenario": {"instructions":
                                    {"reason_for_call": "r"}},
                "description": {"purpose": "Test agent capacity",
                                 "notes": "grader hint"}}
        out = render_compile_input(task["user_scenario"])
        self.assertNotIn("Test agent capacity", out)
        self.assertNotIn("grader", out)

    def test_non_ascii_preserved(self):
        out = render_compile_input(
            {"instructions": {"known_info": "名字 张伟, 城市 北京"}})
        self.assertIn("张伟", out)

    def test_retail_equivalence(self):
        """Every retail v1 instruction value must still appear in the v2
        rendering (field-labeled), so retail LITERAL verdicts carry over."""
        import json
        repo = Path(__file__).resolve().parents[1]
        checked = 0
        for split in ("train", "dev", "test"):
            with open(repo / "data" / "tau3_runtime_benchmark" /
                      f"{split}.jsonl", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["task_id"].startswith("tau3_retail"):
                        src = int(rec["provenance"]["source_task_id"])
                        tasks = json.load(open(
                            repo / "data" / "external_benchmarks" /
                            "tau3_bench" / "domains" / "retail" /
                            "tasks.json", encoding="utf-8"))
                        t = next(x for x in tasks if int(x["id"]) == src)
                        v2 = " ".join(render_compile_input(
                            t.get("user_scenario")).split())
                        for val in rec["instruction"].values():
                            s = " ".join(str(val).strip().split())
                            if s and s not in ("None",):
                                self.assertIn(s[:60], v2)
                                checked += 1
        self.assertGreater(checked, 50)

    def test_bare_string_instructions(self):
        self.assertEqual(render_compile_input(
            {"instructions": "do the thing"}),
            "task_instructions: do the thing")


if __name__ == "__main__":
    unittest.main()
