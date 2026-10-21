"""Phase 9B-3 v3.2: mix the REAL gold-shaped A-train partition into training.

condition_A_train tasks (freeze manifest, leak-free vs A-test by group
closure) are converted into v3.1-format records: instruction via
compile_input_v2, skeleton nodes from the gold v2.2 program (same
mapping as the frozen-eval builder), registry = gold tools + 3
distractors. VALUE-ISOLATION GUARD: any A-train task whose acting user
also appears in an A-test group is EXCLUDED (different-template groups
may share users; sharing users across train/test would leak values).

Output: data/runtime_dataflow_v32/{train,dev}.jsonl + manifest
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.eval.compile_input_v2 import render_compile_input

TAU2_TASKS = Path("D:/ccfa-2026/tau3-bench-src/data/tau2/domains")
V31 = REPO / "data/runtime_dataflow_v31"
OUT = REPO / "data/runtime_dataflow_v32"
GOLD = REPO / "results/phase9/tau3_gold_compiled_v2_2.json"
FREEZE = REPO / "results/phase9/tau3_gold_v2_2_freeze_manifest.json"
REG = REPO / "results/phase9/verified_registry_manifest.json"


def skeleton_nodes(t):
    nodes = []
    for n in t["program"]:
        args = {}
        for p, a in n["arguments"].items():
            c = a["class"]
            if c == "LITERAL":
                args[p] = a["value"]
            elif c == "runtime_ref_multi":
                args[p] = {"kind": "runtime_ref_multi",
                           "elements": a["elements"]}
            else:
                cc = (a.get("candidates") or a.get("acceptable")
                      or [{"producer_step": a.get("producer_step"),
                           "path": a.get("path", "")}])[0]
                args[p] = {"kind": "runtime_ref",
                           "producer": f"p{cc['producer_step']}",
                           "field": cc.get("path", "")}
        nodes.append({"id": f"p{n['step']}", "capability_id": n["tool"],
                      "skill": "EXEC_ACTION", "depends_on": [],
                      "order_on": [], "arguments": args})
    return nodes


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    reg = json.loads(REG.read_text(encoding="utf-8"))
    testA = set(freeze["summary"]["condition_A_test"])
    trainA = freeze["summary"]["condition_A_train"]
    g_by = {t["task_id"]: t for t in gold["tasks"]}

    # value-isolation: acting users of A-test groups
    test_users = set()
    for g, ids in freeze["groups"].items():
        if any(i in testA for i in ids):
            m = re.match(r"\('retail', '([^']+)'", g)
            if m:
                test_users.add(m.group(1))

    retail_reg = {k.split(".", 1)[1]: v for k, v in reg["tools"].items()
                  if k.startswith("retail.")}
    raw_tasks = json.loads(
        (TAU2_TASKS / "retail/tasks.json").read_text(encoding="utf-8"))
    raw_by_id = {str(t["id"]): t for t in raw_tasks}

    added, excluded = [], []
    for tid in trainA:
        t = g_by.get(tid)
        if not t:
            continue
        if t.get("acting_user") in test_users:
            excluded.append(tid)
            continue
        sid = str(t.get("source_task_id") or tid.rsplit("_", 1)[1])
        raw = raw_by_id[sid]
        nodes = skeleton_nodes(t)
        gold_tools = sorted({n["capability_id"] for n in nodes})
        distractors = [x for x in retail_reg if x not in gold_tools][:3]
        offered = gold_tools + distractors
        added.append({
            "task_id": f"v32g_{tid}",
            "instruction": render_compile_input(raw.get("user_scenario")),
            "domain": "retail",
            "capability_registry": {
                x: {"input_schema": retail_reg[x]["input_schema"],
                    "output_schema": retail_reg[x]["output_schema"],
                    "effect_class": retail_reg[x]["effect_class"]}
                for x in offered},
            "target_skeleton": {"nodes": nodes},
            "template": "gold",
        })

    v31_train = [json.loads(l) for l in
                 open(V31 / "train.jsonl", encoding="utf-8") if l.strip()]
    v31_dev = [json.loads(l) for l in
               open(V31 / "dev.jsonl", encoding="utf-8") if l.strip()]
    train = v31_train + added
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in train),
        encoding="utf-8")
    (OUT / "dev.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in v31_dev),
        encoding="utf-8")
    man = {"n_train": len(train), "n_dev": len(v31_dev),
           "gold_shaped_added": len(added),
           "excluded_user_overlap_with_test": excluded,
           "test_users_guarded": len(test_users),
           "provenance": "v3.1 synthesis + condition_A_train gold programs "
                         "(freeze manifest, leak-free partition)"}
    (OUT / "manifest.json").write_text(
        json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
