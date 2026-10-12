"""Phase 7C-2.1 Task 2: clustered statistics for the confirmation set.

Handles the advisor's statistical objections:
  - per-seed wins/losses (not pooled only)
  - task-clustered bootstrap: resample TASK ids, keep all 3 seeds of each
    resampled task -> preserves the cross-seed correlation structure
    (the same 150 tasks repeat across seeds)
  - exact McNemar per seed and pooled, with Holm adjustment over the
    PRE-REGISTERED contrasts; G2-G1 explicitly labeled post-hoc composite
  - precise denominator for the zero-regression claim
Reads the unmodified runs jsonl; writes results/phase7/clustered_statistics.json.
"""
import json, math, os, sys, random
from collections import defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

RUNS = "results/phase7/online_large_runs.jsonl"
OUT = "results/phase7/clustered_statistics.json"

PREREGISTERED = [("G1", "G0"), ("G2a", "G1"), ("G2", "G2a"), ("G3", "G2")]
POSTHOC = [("G2", "G1")]          # composite of two consecutive increments
SEEDS = ["s42", "s43", "s44"]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_low = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p_low)


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values in original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        val = max(val, prev)          # enforce monotonicity
        adj[i] = val
        prev = val
    return adj


def cluster_bootstrap(tasks, seed_diffs, iters=10000, seed=13):
    """tasks: list of task ids; seed_diffs[task] = {seed: diff(0/1/-1)}.
    Statistic: mean over all (seed, resampled task) pairs."""
    rng = random.Random(seed)
    tlist = list(tasks)
    n = len(tlist)
    if not n:
        return None, None
    flat = []
    for t in tlist:
        for s in SEEDS:
            d = seed_diffs[t].get(s)
            if d is not None:
                flat.append(d)
    if not flat:
        return None, None
    base = sum(flat) / len(flat)
    means = []
    for _ in range(iters):
        acc = []
        for _ in range(n):
            t = tlist[rng.randrange(n)]
            for s in SEEDS:
                d = seed_diffs[t].get(s)
                if d is not None:
                    acc.append(d)
        means.append(sum(acc) / len(acc))
    means.sort()
    return (round(100 * base, 2),
            (round(100 * means[int(0.025 * iters)], 2),
             round(100 * means[int(0.975 * iters)], 2)))


def main():
    recs = [json.loads(l) for l in open(RUNS, encoding="utf-8") if l.strip()]
    e2e = {}                      # (seed, task, arm) -> bool
    for r in recs:
        if r["set"] == "confirmation":
            e2e[(r["seed"], r["id"], r["arm"])] = r["judge"]["e2e"]
    tasks = sorted({k[1] for k in e2e})
    print(f"confirmation tasks: {len(tasks)}, seeds: {SEEDS}")

    out = {"n_tasks": len(tasks), "seeds": SEEDS,
           "note": "cluster bootstrap resamples task ids and keeps all "
                   "seeds of each resampled task (cross-seed correlation "
                   "preserved); Holm adjustment applied over the four "
                   "pre-registered contrasts only",
           "contrasts": {}}

    pvals_prereg = []
    for (A, B) in PREREGISTERED + POSTHOC:
        per_seed = {}
        seed_diffs = defaultdict(dict)
        pooled_w = pooled_l = 0
        for seed in SEEDS:
            w = l = 0
            for t in tasks:
                va = e2e.get((seed, t, A))
                vb = e2e.get((seed, t, B))
                if va is None or vb is None:
                    continue
                d = (1 if va else 0) - (1 if vb else 0)
                seed_diffs[t][seed] = d
                if d > 0:
                    w += 1
                elif d < 0:
                    l += 1
            per_seed[seed] = {"wins": w, "losses": l,
                               "mcnemar_p": mcnemar_exact(w, l)}
            pooled_w += w
            pooled_l += l
        base, ci = cluster_bootstrap(tasks, seed_diffs)
        rec = {
            "preregistered": (A, B) in PREREGISTERED,
            "role": ("pre-registered in manifest" if (A, B) in PREREGISTERED
                     else "POST-HOC composite of G2a-G1 + G2-G2a"),
            "per_seed": per_seed,
            "pooled": {"wins": pooled_w, "losses": pooled_l,
                        "mcnemar_p": mcnemar_exact(pooled_w, pooled_l)},
            "cluster_bootstrap_mean_pp": base,
            "cluster_bootstrap_ci95_pp": ci,
            "n_pairs_per_seed": len(tasks),
        }
        out["contrasts"][f"{A}-{B}"] = rec
        if (A, B) in PREREGISTERED:
            pvals_prereg.append(rec["pooled"]["mcnemar_p"])

    adj = holm(pvals_prereg)
    for i, (A, B) in enumerate(PREREGISTERED):
        out["contrasts"][f"{A}-{B}"]["holm_p"] = adj[i]

    # exact zero-regression denominators (vs G0 and vs previous arm)
    denom = {}
    for X in ["G1", "G2a", "G2", "G3"]:
        ref = "G0"
        pairs = broken = 0
        for seed in SEEDS:
            for t in tasks:
                va = e2e.get((seed, t, X))
                vb = e2e.get((seed, t, ref))
                if va is None or vb is None:
                    continue
                pairs += 1
                if vb and not va:
                    broken += 1
        denom[f"{X}_vs_G0"] = {"pairs": pairs, "regressions": broken}
    total_pairs = sum(v["pairs"] for v in denom.values())
    total_broken = sum(v["regressions"] for v in denom.values())
    out["zero_regression_audit"] = {
        "definition": "for each constrained arm X vs G0, over all "
                      "(seed, task) with both records: pairs counted, "
                      "regressions = G0 correct AND X incorrect",
        "per_arm": denom,
        "total_pairs": total_pairs,
        "total_regressions": total_broken,
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== contrasts ===")
    for name, r in out["contrasts"].items():
        ps = " ".join(
            f"{s}:{v['wins']}/{v['losses']}" for s, v in r["per_seed"].items())
        extra = (f" holm_p={r.get('holm_p')}" if "holm_p" in r
                 else " (post-hoc, no adjustment)")
        print(f"{name:<8} {ps} pooled={r['pooled']['wins']}/"
              f"{r['pooled']['losses']} p={r['pooled']['mcnemar_p']}{extra}")
        print(f"         cluster-bootstrap {r['cluster_bootstrap_mean_pp']:+.1f}pp "
              f"CI95={r['cluster_bootstrap_ci95_pp']}")
    print(f"\nzero-regression audit: {total_pairs} pairs, "
          f"{total_broken} regressions")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
