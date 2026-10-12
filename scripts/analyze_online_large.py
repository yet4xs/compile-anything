"""Phase 7C-2 Task 6: aggregate + paired statistics for the large eval.

Reads results/phase7/online_large_runs.jsonl, reports per seed/arm metric
tables, per-task Fixed/Broken transitions, constraint-safety rates, and
paired contrasts (bootstrap 95% CI + exact McNemar) for:
  G1-G0, G2a-G1, G2-G2a, G3-G2
Writes results/phase7/online_constrained_large.json.
"""
import json, math, os, sys, random
from collections import defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

RUNS = "results/phase7/online_large_runs.jsonl"
OUT = "results/phase7/online_constrained_large.json"
ARMS = ["G0", "G1", "G2a", "G2", "G3"]
CONTRASTS = [("G1", "G0"), ("G2a", "G1"), ("G2", "G2a"), ("G3", "G2")]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def pxx(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar via binomial tail sums."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_low = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p_low), 5)


def bootstrap_ci(diffs, iters=10000, seed=13):
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (None, None)
    means = []
    for _ in range(iters):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (round(means[int(0.025 * iters)], 4),
            round(means[int(0.975 * iters)], 4))


def main():
    recs = []
    for line in open(RUNS, encoding="utf-8"):
        if line.strip():
            recs.append(json.loads(line))
    print(f"records: {len(recs)}")

    by_key = {(r["seed"], r["id"], r["arm"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    sets = sorted({r["set"] for r in recs})

    out = {"records": len(recs), "seeds": seeds, "sets": sets,
           "per_set": {}}

    for st in sets:
        print(f"\n{'='*70}\nSET: {st}\n{'='*70}")
        set_summary = {}
        for seed in seeds:
            ids = sorted({r["id"] for r in recs
                          if r["seed"] == seed and r["set"] == st})
            if not ids:
                continue
            n = len(ids)
            arm_rows = {}
            print(f"\n--- seed {seed} (n={n}) ---")
            hdr = (f"{'arm':<5}{'parse%':>8}{'valid%':>8}{'E2E%':>7}"
                   f"{'SkillF1':>9}{'ExecRec':>9}{'ParamE2E':>10}"
                   f"{'DLock%':>8}{'Viol%':>7}{'Forced':>7}{'P50':>7}{'P95':>7}")
            print(hdr)
            for arm in ARMS:
                rows = [by_key[(seed, i, arm)] for i in ids
                        if (seed, i, arm) in by_key]
                if not rows:
                    continue
                js = [r["judge"] for r in rows]
                dlock = [bool(r.get("constraint_stats", {}).get("deadlocks"))
                         for r in rows]
                viol = [bool(r.get("constraint_stats", {}).get("fsm_violation"))
                        for r in rows]
                forced = [r.get("constraint_stats", {}).get("forced_steps", 0)
                          for r in rows]
                arm_rows[arm] = {
                    "n": len(rows),
                    "parse_pct": round(100 * sum(j["parse_ok"] for j in js) / n, 1),
                    "valid_pct": round(100 * sum(j["valid"] for j in js) / n, 1),
                    "e2e_pct": round(100 * sum(j["e2e"] for j in js) / n, 1),
                    "skill_f1_e2e": mean([j["skill_f1_e2e"] for j in js]),
                    "exec_recall_e2e": mean(
                        [j["exec_recall_e2e"] for j in js
                         if j.get("exec_recall_defined")]),
                    "param_key_e2e": mean([j["param_key_e2e"] for j in js]),
                    "param_key_cond": mean([j["param_key_cond"] for j in js]),
                    "deadlock_pct": round(100 * sum(dlock) / n, 1),
                    "violation_pct": round(100 * sum(viol) / n, 1),
                    "forced_mean": round(mean(forced) or 0, 2),
                    "tokens_mean": mean([r["tokens"] for r in rows]),
                    "latency_p50": pxx([r["latency_s"] for r in rows], .5),
                    "latency_p95": pxx([r["latency_s"] for r in rows], .95),
                }
                a = arm_rows[arm]
                print(f"{arm:<5}{a['parse_pct']:>7.1f}%{a['valid_pct']:>7.1f}%"
                      f"{a['e2e_pct']:>6.1f}%{a['skill_f1_e2e']:>9}"
                      f"{a['exec_recall_e2e']:>9}{a['param_key_e2e']:>10}"
                      f"{a['deadlock_pct']:>7.1f}%{a['violation_pct']:>6.1f}%"
                      f"{a['forced_mean']:>7}{a['latency_p50']:>7}"
                      f"{a['latency_p95']:>7}")

            # transitions + paired stats
            transitions = {"fixed": defaultdict(int),
                           "broken": defaultdict(int)}
            for A, B in CONTRASTS:
                fixed = broken = 0
                e2e_diffs = []
                for i in ids:
                    ra, rb = by_key.get((seed, i, A)), by_key.get((seed, i, B))
                    if not (ra and rb):
                        continue
                    va, vb = ra["judge"]["e2e"], rb["judge"]["e2e"]
                    e2e_diffs.append(1 if va else 0)
                    e2e_diffs_b = 1 if vb else 0
                    if va and not vb:
                        fixed += 1
                    elif vb and not va:
                        broken += 1
                transitions["fixed"][f"{A}>{B}"] = fixed
                transitions["broken"][f"{A}>{B}"] = broken

            paired = {}
            for A, B in CONTRASTS:
                b_cnt = c_cnt = 0
                diffs = []
                for i in ids:
                    ra, rb = by_key.get((seed, i, A)), by_key.get((seed, i, B))
                    if not (ra and rb):
                        continue
                    va = 1 if ra["judge"]["e2e"] else 0
                    vb = 1 if rb["judge"]["e2e"] else 0
                    diffs.append(va - vb)
                    if va > vb:
                        b_cnt += 1
                    elif vb > va:
                        c_cnt += 1
                dbar = mean(diffs)
                lo, hi = bootstrap_ci(diffs)
                paired[f"{A}-{B}"] = {
                    "e2e_diff_pp": round((dbar or 0) * 100, 1),
                    "ci95": [lo * 100 if lo is not None else None,
                             hi * 100 if hi is not None else None],
                    "wins": b_cnt, "losses": c_cnt,
                    "mcnemar_p": mcnemar_exact(b_cnt, c_cnt),
                }
            arm_rows["transitions"] = {k: dict(v) for k, v in transitions.items()}
            arm_rows["paired_e2e"] = paired
            for k, v in paired.items():
                print(f"  {k}: {v['e2e_diff_pp']:+.1f}pp "
                      f"CI95=[{v['ci95'][0]:.1f},{v['ci95'][1]:.1f}] "
                      f"w/l={v['wins']}/{v['losses']} "
                      f"p={v['mcnemar_p']}")
            set_summary[seed] = arm_rows
        out["per_set"][st] = set_summary

    # ---------------- pooled across seeds (confirmation) ----------------
    print(f"\n{'='*70}\nPOOLED across seeds: confirmation\n{'='*70}")
    pooled = {}
    for arm in ARMS:
        rows = [r for r in recs if r["set"] == "confirmation" and r["arm"] == arm]
        if not rows:
            continue
        js = [r["judge"] for r in rows]
        per_seed_e2e = []
        for seed in seeds:
            sjs = [r["judge"] for r in rows if r["seed"] == seed]
            if sjs:
                per_seed_e2e.append(100 * sum(j["e2e"] for j in sjs) / len(sjs))
        m = sum(per_seed_e2e) / len(per_seed_e2e)
        sd = (sum((x - m) ** 2 for x in per_seed_e2e) / len(per_seed_e2e)) ** 0.5
        pooled[arm] = {
            "n": len(rows),
            "e2e_mean": round(m, 1), "e2e_std_across_seeds": round(sd, 1),
            "e2e_worst_seed": round(min(per_seed_e2e), 1),
            "e2e_best_seed": round(max(per_seed_e2e), 1),
            "seed_spread_pp": round(max(per_seed_e2e) - min(per_seed_e2e), 1),
            "skill_f1": mean([j["skill_f1_e2e"] for j in js]),
            "param_key_e2e": mean([j["param_key_e2e"] for j in js]),
        }
        p = pooled[arm]
        print(f"{arm}: E2E {p['e2e_mean']}±{p['e2e_std_across_seeds']} "
              f"(worst {p['e2e_worst_seed']}, best {p['e2e_best_seed']}, "
              f"spread {p['seed_spread_pp']}pp) SkillF1={p['skill_f1']} "
              f"Param={p['param_key_e2e']}")
    out["pooled_confirmation"] = pooled

    print("\npooled paired contrasts (all seeds combined):")
    pooled_pairs = {}
    for A, B in CONTRASTS:
        b_cnt = c_cnt = 0
        ra_keys = {(r["seed"], r["id"]) for r in recs
                   if r["set"] == "confirmation" and r["arm"] == A}
        rb_map = {(r["seed"], r["id"]): r["judge"]["e2e"] for r in recs
                  if r["set"] == "confirmation" and r["arm"] == B}
        for k in ra_keys:
            if k in rb_map:
                va = next(r["judge"]["e2e"] for r in recs
                          if r["set"] == "confirmation" and r["arm"] == A
                          and (r["seed"], r["id"]) == k)
                vb = rb_map[k]
                if va and not vb:
                    b_cnt += 1
                elif vb and not va:
                    c_cnt += 1
        pooled_pairs[f"{A}-{B}"] = {
            "wins": b_cnt, "losses": c_cnt,
            "mcnemar_p": mcnemar_exact(b_cnt, c_cnt)}
        print(f"  {A}-{B}: w/l={b_cnt}/{c_cnt} "
              f"p={pooled_pairs[f'{A}-{B}']['mcnemar_p']}")
    out["pooled_confirmation_paired"] = pooled_pairs

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {OUT}")
    print("ANALYZE-DONE")


if __name__ == "__main__":
    main()
