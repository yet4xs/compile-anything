"""Phase 8B-1C Task 1: CORRECT paired statistics for M1 vs M0.

Fixes the 8B-1B bootstrap bug (it resampled M1's rate, not the paired
delta). Consumes the frozen per-task records from
results/phase8b/m0_m1_comparison.json (untouched) and produces:
  - task-level deltas delta_i = M1_i - M0_i (0/1 outcomes)
  - paired percentile bootstrap 95% CI of the mean delta
  - exact McNemar (two-sided binomial on discordant pairs)
  - Fixed / Broken / BothCorrect / BothWrong fourfold table
  - same for Plan F1 differences (continuous, paired bootstrap)
  - explicit denominators + Holm note over the 3 split-level tests

Usage:
  python scripts/rescore_m0_m1_v3.py            # from comparison.json
  python scripts/rescore_m0_m1_v3.py --outputs results/phase8b/m0_m1_outputs.jsonl
      (later: recompute v3 metrics from saved raw outputs instead)
"""
import json, math, random, argparse, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = "results/phase8b/m0_m1_comparison.json"
OUT = "results/phase8b/m0_m1_paired_statistics_v3.json"


def paired_bootstrap(deltas, iters=10000, seed=13):
    rng = random.Random(seed)
    n = len(deltas)
    if not n:
        return None, None, None
    base = sum(deltas) / n
    means = []
    for _ in range(iters):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return (round(100 * base, 2),
            (round(100 * means[int(0.025 * iters)], 2),
             round(100 * means[int(0.975 * iters)], 2)),
            n)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_low = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p_low), 5)


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i])
        adj[i] = max(v, prev)
        prev = adj[i]
    return adj


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    details = d["details"]
    by = {}
    for r in details:
        by[(r["arm"], r["split"], r["task_id"])] = r
    splits = sorted({r["split"] for r in details})
    out = {"source": SRC, "note": "task-level paired deltas; bootstrap "
            "resamples delta_i, not a single arm's rate", "splits": {}}
    mcn_p = []
    for sp in splits:
        ids = sorted({t for (a, s, t) in by if a == "M0" and s == sp})
        four = {"both_correct": 0, "both_wrong": 0,
                "m1_only": 0, "m0_only": 0}
        deltas_joint, deltas_plan = [], []
        for t in ids:
            r0 = by.get(("M0", sp, t))
            r1 = by.get(("M1", sp, t))
            if not (r0 and r1):
                continue
            deltas_joint.append(1 if r1["exact_joint"] else 0)
            deltas_joint[-1] -= 1 if r0["exact_joint"] else 0
            if r1["exact_joint"] and r0["exact_joint"]:
                four["both_correct"] += 1
            elif r1["exact_joint"] and not r0["exact_joint"]:
                four["m1_only"] += 1
            elif r0["exact_joint"] and not r1["exact_joint"]:
                four["m0_only"] += 1
            else:
                four["both_wrong"] += 1
            deltas_plan.append(r1["plan_f1"] - r0["plan_f1"])
        dj, cij, n = paired_bootstrap(deltas_joint)
        dp, cip, _ = paired_bootstrap(deltas_plan)
        mc = mcnemar_exact(four["m1_only"], four["m0_only"])
        mcn_p.append(mc)
        out["splits"][sp] = {
            "n_paired": n,
            "exact_joint": {
                "M0_pct": round(100 * sum(
                    1 for t in ids if by[("M0", sp, t)]["exact_joint"])
                    / n, 1),
                "M1_pct": round(100 * sum(
                    1 for t in ids if by[("M1", sp, t)]["exact_joint"])
                    / n, 1),
                "delta_pp": dj,
                "ci95_pp": cij,
                "mcnemar_p": mc},
            "plan_f1": {"delta_mean": dp, "ci95": cip},
            "fourfold": four,
            "denominators": {"exact_joint": "all paired task ids in split",
                             "fixed": "M1 correct AND M0 wrong",
                             "broken": "M0 correct AND M1 wrong"},
        }
    adj = holm(mcn_p)
    for sp, a in zip(splits, adj):
        out["splits"][sp]["exact_joint"]["mcnemar_holm_p"] = a
    out["multiple_comparison"] = {
        "family": "3 split-level exact-joint McNemar tests",
        "adjustment": "Holm-Bonferroni",
        "raw_p": dict(zip(splits, mcn_p)),
        "holm_p": dict(zip(splits, adj))}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for sp in splits:
        s = out["splits"][sp]
        j = s["exact_joint"]
        print(f"{sp}: n={s['n_paired']} "
              f"M0={j['M0_pct']}% M1={j['M1_pct']}% "
              f"delta={j['delta_pp']}pp CI95={j['ci95_pp']} "
              f"McNemar p={j['mcnemar_p']} (Holm {j['mcnemar_holm_p']}) "
              f"4fold={s['fourfold']}")
        print(f"    planF1 delta={s['plan_f1']['delta_mean']} "
              f"CI95={s['plan_f1']['ci95']}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
