"""Phase 9B-3: score frozen-test predictions with scorer v3 (local, CPU).

Reads results/phase9/v3_frozen_predictions_s<seed>.jsonl (full raws from
the server), scores each prediction against the machine-adjudicated gold
v2.2 program (equivalence classes), splits by condition A (in-domain
retail) / B (cross-tool-family airline), and reports per arm x protocol:
value_match / equivalence_class_exact / strict_origin_exact with
Clopper-Pearson 95% CIs, plus cap-hit rates (>5% invalidates).

Usage: python scripts/score_frozen_v3.py --seed 44
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_from_dict
from src.eval.gold_binding_scorer import (
    score_prediction_v3, calls_from_skeleton_nodes,
    bindings_from_skeleton_nodes)
from src.eval.stats_ci import clopper_pearson
from scripts.score_tau3_offline import extract_pred_from_taskir

MAX_NEW = {"M0-T": 864, "M0-J": 2396, "M1": 1070}


def pred_side(arm, raw):
    try:
        if arm == "M1":
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return None
            nodes = json.loads(m.group(0)).get("nodes", [])
            return (calls_from_skeleton_nodes(nodes),
                    bindings_from_skeleton_nodes(nodes))
        if arm == "M0-J":
            ast = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            return extract_pred_from_taskir(
                parse_text(to_text(module_from_dict(ast))))
        return extract_pred_from_taskir(parse_text(raw))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    pred_file = REPO / "results/phase9" / \
        f"v3_frozen_predictions_s{args.seed}.jsonl"
    rows = [json.loads(l) for l in
            open(pred_file, encoding="utf-8") if l.strip()]
    gold = json.loads((REPO / "results/phase9/tau3_gold_compiled_v2_2.json")
                      .read_text(encoding="utf-8"))
    g_by = {t["task_id"]: t for t in gold["tasks"]}

    details = []
    for r in rows:
        t = g_by.get(r["task_id"])
        if not t:
            continue
        ps = pred_side(r["arm"], r["raw"])
        if ps is None:
            m = {"task_id": r["task_id"], "condition": r["condition"],
                 "arm": r["arm"], "protocol": r["protocol"],
                 "parse_ok": False, "value_match": False,
                 "equivalence_class_exact": False,
                 "strict_origin_exact": False}
        else:
            s = score_prediction_v3(ps[0], ps[1], t["program"])
            m = {"task_id": r["task_id"], "condition": r["condition"],
                 "arm": r["arm"], "protocol": r["protocol"],
                 "parse_ok": True, **{k: s[k] for k in
                                      ("value_match",
                                       "equivalence_class_exact",
                                       "strict_origin_exact",
                                       "n_bindings_canonical",
                                       "n_bindings_in_class",
                                       "n_binding_slots")},
                 "hit_generation_cap": r.get("hit_generation_cap", False),
                 "gen_tokens": r.get("gen_tokens", 0)}
        details.append(m)

    agg = defaultdict(list)
    for d in details:
        agg[(d["condition"], d["arm"], d["protocol"])].append(d)
        agg[(d["condition"], d["arm"], "both")].append(d)

    table = {}
    for (cond, arm, proto), ds in sorted(agg.items()):
        n = len(ds)
        def rate(k):
            c = sum(1 for d in ds if d.get(k))
            lo, hi = clopper_pearson(c, n) if n else (0, 0)
            return {"n": n, "k": c, "rate": round(c / n, 4) if n else None,
                    "cp95": [round(lo, 4), round(hi, 4)]}
        table[f"{cond}|{arm}|{proto}"] = {
            "value_match": rate("value_match"),
            "equivalence_class_exact": rate("equivalence_class_exact"),
            "strict_origin_exact": rate("strict_origin_exact"),
            "parse_ok_rate": round(sum(1 for d in ds if d["parse_ok"]) / n, 3),
            "cap_hit_rate": round(
                sum(1 for d in ds if d.get("hit_generation_cap")) / n, 3)}

    out = {"seed": args.seed, "n_predictions": len(details), "table": table,
           "cap_rule": "any arm x protocol cap-hit > 5% invalidates"}
    outf = REPO / "results/phase9" / f"v3_frozen_scores_s{args.seed}.json"
    outf.write_text(json.dumps(out, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
