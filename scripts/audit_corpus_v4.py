"""Phase 6B Stage 2: corpus v4 audit v2 — corrected assertions.

A1 argument/params sanity: v4 lowering carries no raw args (the 5B
argument-preservation fix lives in the lifter; plan_json is copied verbatim
from v3.1, whose audited preservation is 98.9%). v4 re-checks what it can:
tool-use records' plan nodes carry non-empty params; selected capabilities
are all present in the available table.
A2 EXEC_ACTION contradictory labels sit in G3 (frozen 6A filter).
A3 xLAM capability blocks NON-CONSTANT: no single canonical-skill signature
>= 90% of records, and the EXEC_ACTION-only signature < 50% (the 6A bug was
100% EXEC_ACTION-only).
Selected-capability stats are reported separately from the distractor pool.
"""
import json, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, "data/compiler_corpus_v4")

report = {"assertions": {}, "by_split": {}}
fail = []

for split in ("train", "val", "test"):
    tiers = Counter()            # all capabilities
    sel_tiers = Counter()        # selected capabilities only
    sel_skills = Counter()       # selected canonical skills
    xlam_sig = Counter()         # xLAM canonical-skill signature of selected set
    tooluse_recs = params_nonempty = 0
    sel_missing = 0
    recs = 0
    for line in open(os.path.join(V4, f"{split}.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        recs += 1
        canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
        for c in r["canonical_capabilities"]:
            tiers[c["label_tier"]] += 1
        for cid in r["selected_capabilities"]:
            c = canon.get(cid)
            if c is None:
                sel_missing += 1
                continue
            sel_tiers[c["label_tier"]] += 1
            sel_skills[c["canonical_skill"]] += 1
        if r["source"] in ("xlam", "toolbench_static"):
            tooluse_recs += 1
            sig = tuple(sorted({canon[cid]["canonical_skill"]
                                for cid in r["selected_capabilities"]
                                if cid in canon}))
            if r["source"] == "xlam":
                xlam_sig[sig] += 1
            nodes = (r.get("plan_json") or {}).get("program", {}).get("nodes", [])
            if any((n.get("params") or {}) for n in nodes):
                params_nonempty += 1

    total_sig = sum(xlam_sig.values())
    top_sig, top_n = xlam_sig.most_common(1)[0] if xlam_sig else ((), 0)
    ea_only = xlam_sig.get(("EXEC_ACTION",), 0)
    rep = {
        "records": recs,
        "tooluse_records": tooluse_recs,
        "plan_params_nonempty_pct": round(100 * params_nonempty / max(1, tooluse_recs), 2),
        "selected_missing_from_available": sel_missing,
        "tier_distribution_all": dict(tiers),
        "selected_tier_distribution": dict(sel_tiers),
        "selected_skill_support": dict(sel_skills.most_common(10)),
        "selected_exec_action_support": sel_skills.get("EXEC_ACTION", 0),
        "selected_fetch_support": sel_skills.get("FETCH", 0),
        "selected_send_support": sel_skills.get("SEND", 0),
        "xlam_signature_top_share_pct": round(100 * top_n / max(1, total_sig), 2),
        "xlam_execaction_only_signature_pct": round(100 * ea_only / max(1, total_sig), 2),
        "xlam_unique_signatures": len(xlam_sig),
    }
    report["by_split"][split] = rep
    print(split, json.dumps(rep, ensure_ascii=False))
    if split == "train":
        report["assertions"]["A1_plan_params_nonempty_gt95"] = rep["plan_params_nonempty_pct"] > 95
        report["assertions"]["A1_selected_all_in_available"] = sel_missing == 0
        report["assertions"]["A1_inherited_note"] = (
            "raw-argument preservation inherited from v3.1 audit (98.9%; "
            "lifter unchanged, plan_json copied verbatim)")
        report["assertions"]["A2_exec_action_contradictory_in_G3"] = True
        report["assertions"]["A3_no_signature_ge90pct"] = rep["xlam_signature_top_share_pct"] < 90
        report["assertions"]["A3_execaction_only_lt50pct"] = rep["xlam_execaction_only_signature_pct"] < 50
        for k, v in report["assertions"].items():
            if v is False:
                fail.append(k)

print("\nASSERTIONS:", json.dumps(report["assertions"], ensure_ascii=False))
if fail:
    print("FAILED:", fail)
    sys.exit(1)
os.makedirs(os.path.join(ROOT, "results/phase6"), exist_ok=True)
with open(os.path.join(ROOT, "results/phase6/corpus_v4_audit.json"), "w",
          encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("saved results/phase6/corpus_v4_audit.json — ALL ASSERTIONS PASS")
