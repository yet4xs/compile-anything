"""Phase 8A Task 1: audit the tau3 runtime tool registry.

Outputs per domain: tool counts by type/mutability/discoverability,
description & param-doc coverage, and (for audit purposes only) whether
every reference action used by the eval tasks exists in the registry.
Reference names are used ONLY for this coverage check — never to build
the candidate pool.
"""
import json, os, sys
from collections import Counter

ROOT = "/ccfa2026/compile-anything"
if not os.path.isdir(ROOT):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.eval.tau3_tool_registry import build_registry, runtime_pool

reg = build_registry()
pool = runtime_pool(reg)

audit = {"domains": {}}
print(f"{'domain':<20}{'total':>6}{'pool':>6}{'read':>6}{'write':>6}"
      f"{'gen':>5}{'think':>6}{'mut':>5}{'disc':>6}{'no-desc':>8}"
      f"{'no-param-doc%':>14}")
for d, ts in reg.items():
    p = pool[d]
    c = Counter(t["tool_type"] for t in ts)
    mut = sum(1 for t in ts if t["mutates_state"])
    disc = sum(1 for t in ts if t["discoverable"])
    nodesc = sum(1 for t in ts if not t["description"])
    nparams = sum(len(t["params"]) for t in ts)
    npdoc = sum(1 for t in ts for pp in t["params"] if not pp["doc"])
    audit["domains"][d] = {
        "total": len(ts), "runtime_pool": len(p),
        "by_type": dict(c), "mutating": mut, "discoverable": disc,
        "missing_description": nodesc,
        "params_total": nparams, "params_missing_doc": npdoc,
        "tools": ts,
    }
    pct = round(100 * npdoc / nparams, 1) if nparams else 0
    print(f"{d:<20}{len(ts):>6}{len(p):>6}{c['read']:>6}{c['write']:>6}"
          f"{c.get('generic', 0):>5}{c.get('think', 0):>6}{mut:>5}"
          f"{disc:>6}{nodesc:>8}{pct:>13}%")

# reference-action coverage (audit only)
ref_names = set()
with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        ref_names.update(r.get("ref_action_names") or [])
all_pool = {t["name"] for ts in pool.values() for t in ts}
all_reg = {t["name"] for ts in reg.values() for t in ts}
missing_from_pool = sorted(ref_names - all_pool)
missing_from_registry = sorted(ref_names - all_reg)
disc_refs = sorted(n for n in (ref_names & all_reg) if n not in all_pool)

audit["ref_coverage"] = {
    "n_ref_action_names": len(ref_names),
    "in_runtime_pool": len(ref_names & all_pool),
    "discoverable_only": disc_refs,
    "missing_from_registry": missing_from_registry,
}
print(f"\nreference actions: {len(ref_names)} distinct")
print(f"  in runtime pool: {len(ref_names & all_pool)}")
print(f"  discoverable-only: {disc_refs}")
print(f"  missing from registry entirely: {missing_from_registry}")
user_side = sorted(n for n in ref_names
                   if n not in all_pool and n in all_reg)
audit["ref_coverage"]["user_side_only"] = user_side
print(f"  user-side tools (agent cannot call; recorded, refs partial): "
      f"{user_side}")

os.makedirs("results/phase8a", exist_ok=True)
with open("results/phase8a/tool_registry_audit.json", "w") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
print("\nsaved results/phase8a/tool_registry_audit.json")
