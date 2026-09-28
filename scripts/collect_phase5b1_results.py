"""Collect Phase 5B-1 experiment results into the final table.

    python scripts/collect_phase5b1_results.py

Reads runs/phase5b1/<exp>/eval.json for E0..E3. Missing experiments are
listed as PENDING — no numbers are fabricated. Includes the E0-vs-E1
Go/No-Go gate and the template-memorization flag (seen high + unseen ~0).
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs" / "phase5b1"
EXPS = [
    ("E0", "e0_3b_base", "3B base (zero-shot)", 0),
    ("E1", "e1_3b_qlora", "3B QLoRA", 28093),
    ("E2", "e2_7b_base", "7B base (zero-shot)", 0),
    ("E3", "e3_7b_lora", "7B LoRA/QLoRA", 28093),
]


def agg(per_source: dict) -> dict:
    """Weighted aggregate over sources (excluding per_skill)."""
    n = sum(s["n"] for s in per_source.values())
    if not n:
        return {}
    w = lambda k: sum(s[k] * s["n"] for s in per_source.values()) / n  # noqa
    tp = sum(s["tp"] for s in per_source.values())
    fp = sum(s["fp"] for s in per_source.values())
    fn = sum(s["fn"] for s in per_source.values())
    f1 = round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0.0
    seen_n = n - sum(s["unseen_composition"]["n"] for s in per_source.values())
    seen_exact = (sum(s["opseq_exact"] for s in per_source.values())
                  - sum(s["unseen_composition"]["opseq_exact"]
                        for s in per_source.values()))
    return {
        "n": n,
        "parse": round(w("parse_rate_pct"), 2),
        "valid": round(w("validator_pass_pct"), 2),
        "execute": round(w("execution_pct"), 2),
        "op_exact": round(w("opseq_exact_pct"), 2),
        "skill_f1": f1,
        "ges": round(w("graph_edit_similarity_mean"), 4),
        "seen_op_exact": round(100 * seen_exact / seen_n, 2) if seen_n else 0,
        "unseen_n": sum(s["unseen_composition"]["n"]
                        for s in per_source.values()),
        "unseen_op_exact": round(
            100 * sum(s["unseen_composition"]["opseq_exact"]
                      for s in per_source.values())
            / max(1, sum(s["unseen_composition"]["n"]
                         for s in per_source.values())), 2),
    }


def main() -> int:
    rows, per_exp_source = {}, {}
    for tag, d, label, n_train in EXPS:
        ev = RUNS / d / "eval.json"
        if ev.exists():
            r = json.loads(ev.read_text(encoding="utf-8"))
            rows[tag] = {"label": label, "train": n_train,
                         **agg(r["per_source"])}
            per_exp_source[tag] = r["per_source"]
        else:
            rows[tag] = {"label": label, "train": n_train, "pending": True}

    L = ["# Phase 5B-1 Results — Neural Compiler Baseline", "",
         "| Experiment | Model | Train | Parse | Valid | Execute | Op exact "
         "| Skill F1 | GED sim | Seen | Unseen |", "|---|---|---:|---:|---:|"
         "---:|---:|---:|---:|---:|---:|"]
    for tag, *_ in EXPS:
        r = rows[tag]
        if r.get("pending"):
            L.append(f"| {tag} | {r['label']} | {r['train']} | PENDING | "
                     f"PENDING | PENDING | PENDING | PENDING | PENDING | "
                     f"PENDING | PENDING |")
        else:
            L.append(f"| {tag} | {r['label']} | {r['train']} | {r['parse']} "
                     f"| {r['valid']} | {r['execute']} | {r['op_exact']} "
                     f"| {r['skill_f1']} | {r['ges']} | {r['seen_op_exact']} "
                     f"| {r['unseen_op_exact']} (n={r['unseen_n']}) |")

    # per-source tables
    for tag in ("E0", "E1", "E2", "E3"):
        if tag not in per_exp_source:
            continue
        L += ["", f"## {tag} per-source", "",
              "| source | n | parse% | valid% | exec% | opseq% | F1 | "
              "generic% | unseen opseq% |",
              "|---|---|---|---|---|---|---|---|---|"]
        for src, s in sorted(per_exp_source[tag].items()):
            u = s["unseen_composition"]
            L.append(f"| {src} | {s['n']} | {s['parse_rate_pct']} "
                     f"| {s['validator_pass_pct']} | {s['execution_pct']} "
                     f"| {s['opseq_exact_pct']} | {s['skill_f1_micro']} "
                     f"| {s['generic_action_rate_pct']} "
                     f"| {u['opseq_exact_pct']} (n={u['n']}) |")

    # Go/No-Go gate E0 vs E1
    L += ["", "## E0 vs E1 Go/No-Go gate", ""]
    if "E0" in per_exp_source and "E1" in per_exp_source:
        e0, e1 = rows["E0"], rows["E1"]
        checks = [
            ("parse rate up", e1["parse"] > e0["parse"]),
            ("validator rate up", e1["valid"] > e0["valid"]),
            ("execution not worse", e1["execute"] >= e0["execute"] - 1e-9),
            ("skill F1 up", e1["skill_f1"] > e0["skill_f1"]),
            ("seen composition up", e1["seen_op_exact"] > e0["seen_op_exact"]),
            ("unseen not collapsed",
             e1["unseen_op_exact"] >= 0.5 * max(e1["seen_op_exact"], 1e-9)
             or e1["unseen_op_exact"] > 0),
        ]
        for name, ok in checks:
            L.append(f"- [{'x' if ok else ' '}] {name}")
        if e1["seen_op_exact"] > 50 and e1["unseen_op_exact"] < 5:
            L.append("- **TEMPLATE MEMORIZATION SUSPECTED** "
                     "(seen high, unseen ~0) — do not fix with more epochs")
        else:
            L.append("- template-memorization check: not triggered")
    else:
        L.append("- PENDING (needs both E0 and E1 results)")

    out = ROOT / "data" / "reports" / "phase5b1_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:14]))
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
