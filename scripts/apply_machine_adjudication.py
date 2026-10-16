"""Apply the owner-directed MACHINE adjudication to gold v2 -> v2.1.

Reads: adjudication/v2/items.jsonl + machine_adjudication.csv +
results/phase9/tau3_gold_compiled_v2.json
Writes: results/phase9/tau3_gold_compiled_v2_1.json

Every accepted param becomes runtime_ref with the accepted candidate as
canonical source (selected_by=machine_adjudication_R*, the full
candidate set kept in acceptable_sources). NONE keeps the param
non-strict. The output records the adjudication provenance in meta.
"""

import csv
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D = REPO / "adjudication" / "v2"
SRC = REPO / "results" / "phase9" / "tau3_gold_compiled_v2.json"
OUT = REPO / "results" / "phase9" / "tau3_gold_compiled_v2_1.json"

STRICT_CLASSES = {"LITERAL", "DUAL_SOURCE", "runtime_ref",
                  "runtime_ref_multi", "SAME_ENTITY_ECHO"}


def main():
    items = {json.loads(l)["item_id"]: json.loads(l)
             for l in open(D / "items.jsonl", encoding="utf-8")}
    verdicts = {}
    with open(D / "machine_adjudication.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verdicts[row["item_id"]] = row

    g = json.loads(SRC.read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=REPO, capture_output=True,
                            text=True).stdout.strip()

    applied = 0
    none_ct = 0
    for t in g["tasks"]:
        for it in [i for i in items.values() if i["task_id"] == t["task_id"]]:
            v = verdicts.get(it["item_id"])
            if not v:
                continue
            j = v["judgment"]
            node = t["program"][it["step"]]
            param = it["consumer"].split(".", 1)[1]
            a = node["arguments"].get(param)
            if not a or it["consumer"].split(".")[0] != node["tool"]:
                continue
            if j.startswith("ACCEPT_"):
                n = int(j.split("_")[1])
                cand = it["candidates"][n - 1]
                rule = v["rule_reason"].split(":")[0]
                node["arguments"][param] = {
                    "class": "runtime_ref",
                    "value": a.get("value"),
                    "producer_step": cand["producer_step"],
                    "path": cand["path"],
                    "selected_by": f"machine_adjudication_{rule}",
                    "acceptable": [
                        {"producer_step": c["producer_step"],
                         "path": c["path"]}
                        for c in it["candidates"]],
                    "adjudication_item": it["item_id"],
                }
                applied += 1
            else:
                a["adjudication_verdict"] = "NONE (machine)"
                none_ct += 1
        # recompute strict v2.1
        pc = {}
        for node in t["program"]:
            for a in node["arguments"].values():
                pc[a["class"]] = pc.get(a["class"], 0) + 1
        t["param_class_counts"] = pc
        strict = (t["replay_ok"] and bool(t["program"])
                  and "POLICY_DEPENDENT" not in t["flags"]
                  and "EXPECTED_ERROR_STEP" not in t["flags"]
                  and set(pc) <= STRICT_CLASSES)
        for node in t["program"]:
            for a in node["arguments"].values():
                steps = [a.get("producer_step")] + [
                    e.get("producer_step") for e in a.get("elements", [])]
                if a["class"] in ("runtime_ref", "runtime_ref_multi") \
                        and any(s is not None and s >= node["step"]
                                for s in steps):
                    strict = False
        t["strict_v2_1"] = strict

    old_strict = sum(t["strict_v2"] for t in g["tasks"])
    new_strict = sum(t["strict_v2_1"] for t in g["tasks"])
    g["summary"] = {
        "meta": {
            "generated_commit": commit,
            "base": "tau3_gold_compiled_v2.json",
            "preregistration": "3d78f7a + amendment (machine adjudication)",
            "adjudication_provenance": (
                "MACHINE-ADJUDICATED: AI assistant under repository "
                "owner instruction 2026-10-16; the pre-registered "
                "dual-human kappa>=0.8 protocol was NOT executed; no "
                "human-independence claim is made"),
            "input_sha256": g["summary"]["meta"]["input_sha256"],
            "tau2_bench_commit": "5ade54e",
        },
        "n_candidates": g["summary"]["n_candidates"],
        "adjudication": {"items": len(verdicts), "accepted": applied,
                         "none": none_ct},
        "strict_v2_before": old_strict,
        "strict_v2_1": new_strict,
        "strict_by_domain": {
            d: sum(t["strict_v2_1"] for t in g["tasks"] if t["domain"] == d)
            for d in ("retail", "airline")},
        "strict_in_official_test": sum(
            t["strict_v2_1"] and t.get("official_test_split")
            for t in g["tasks"]),
    }
    OUT.write_text(json.dumps(g, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in g["summary"].items()
                      if k != "meta"}, indent=1))
    print("strict ids:", [t["task_id"] for t in g["tasks"] if t["strict_v2_1"]])


if __name__ == "__main__":
    main()
