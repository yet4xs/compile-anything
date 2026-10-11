"""Phase 7C-2 Task 5: draw the 150-sample held-out confirmation set.

Stratified by source (id prefix) x reference op-sequence depth, from the
994 matched tasks minus the 30 development samples. Deterministic
(random.seed(20261009)).

Error-type stratification is NOT available a priori for unrun samples; it is
approximated by source, which carried the error profile in the dev-30
(tbs -> comment/parse failures, xlam -> input naming/arity/types). This
limitation is recorded in the manifest.
"""
import json, os, sys, random
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.ir.parser import parse_text

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
SEED = 20261009
N_CONFIRM = 150


def source_of(task_id):
    if task_id.startswith("tbs-in_domain"):
        return "tbs-in"
    if task_id.startswith("tbs-out_of_domain"):
        return "tbs-out"
    return task_id.split("-")[0]


def depth_bucket(text):
    try:
        mod = parse_text(text)
        n = len([x for x in mod.program.nodes if x.op not in POLICY_OPS])
    except Exception:
        return "0"
    return str(min(n, 3))


def main():
    refs = {}
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        refs[r["id"]] = r.get("plan_text") or r.get("plan_target") or ""

    all_tasks = []
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        gold = [c for c in t.get("candidates", []) if c.get("is_gold")]
        if not gold:
            continue
        all_tasks.append(t["id"])
    print(f"matched tasks with gold: {len(all_tasks)}")

    dev30 = all_tasks[:30]
    pool = [i for i in all_tasks if i not in set(dev30)]
    print(f"pool after dev-30 exclusion: {len(pool)}")

    strata = defaultdict(list)
    for tid in pool:
        ref = refs.get(tid, "")
        if not ref:
            continue                      # no reference -> cannot score opseq
        strata[(source_of(tid), depth_bucket(ref))].append(tid)
    total = sum(len(v) for v in strata.values())
    print(f"stratifiable pool (with ref): {total}")
    for k in sorted(strata):
        print(f"  {k}: {len(strata[k])}")

    rng = random.Random(SEED)
    picked = []
    for k in sorted(strata):
        members = sorted(strata[k])
        rng.shuffle(members)
        quota = round(N_CONFIRM * len(members) / total)
        picked.extend(members[:quota])
    # top up to exactly N_CONFIRM from the largest remaining pool
    picked = sorted(set(picked))
    rng2 = random.Random(SEED + 1)
    remaining = sorted(set(m for v in strata.values() for m in v)
                       - set(picked))
    rng2.shuffle(remaining)
    while len(picked) < N_CONFIRM and remaining:
        picked.append(remaining.pop())
    picked = sorted(picked[:N_CONFIRM])

    print(f"\nconfirmation set: {len(picked)}")
    print("by source:", dict(Counter(source_of(i) for i in picked)))
    print("by depth:", dict(Counter(depth_bucket(refs[i]) for i in picked)))

    os.makedirs("data/phase7", exist_ok=True)
    with open("data/phase7/confirmation_ids.json", "w") as f:
        json.dump({"seed": SEED, "n": len(picked),
                   "dev30_ids": dev30,
                   "confirmation_ids": picked,
                   "strata_report": {
                       f"{k[0]}|depth{k[1]}": len(v)
                       for k, v in sorted(strata.items())}},
                  f, indent=2)
    print("saved data/phase7/confirmation_ids.json")


if __name__ == "__main__":
    main()
