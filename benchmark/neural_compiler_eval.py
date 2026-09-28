"""Neural Compiler evaluation (Phase 5B).

Measures compiler behavior, not training loss:

  gates     parse rate (text->Module) -> validator pass -> execution success
  semantics op-sequence exact accuracy, per-skill precision/recall,
            graph edit similarity (node+edge approximate GED),
            fallback/generic-action rate (EXEC_ACTION share of predicted ops)
  splits    per-source breakdown AND seen vs unseen semantic composition
            (unseen = reference op-sequence never appearing in train) — the
            memorization-vs-compiling discriminator

    python benchmark/neural_compiler_eval.py --predictions preds/test.jsonl \
        --train data/compiler_corpus_v3/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ir.parser import parse_text, TaskIRSyntaxError   # noqa: E402
from src.validator.validator import validate               # noqa: E402
from src.runtime.simulator import Simulator                # noqa: E402


def ops_of(module) -> list:
    return [n.op for n in module.program.nodes]


def edges_of(module) -> set:
    e = set()
    for n in module.program.nodes:
        for r in n.inputs:
            e.add((r, n.id))
    return e


def graph_edit_similarity(ref, pred) -> float:
    """Approximate GED similarity over op multiset + def-use edges:
    sim = 1 - (node_edits + edge_edits) / (max_nodes + max_edges)."""
    if ref is None or pred is None:
        return 0.0
    rn, pn = Counter(ops_of(ref)), Counter(ops_of(pred))
    node_edits = sum((rn - pn).values()) + sum((pn - rn).values())
    re_, pe = edges_of(ref), edges_of(pred)
    edge_edits = len(re_ - pe) + len(pe - re_)
    denom = max(len(ops_of(ref)), len(ops_of(pred))) + \
        max(len(re_), len(pe))
    return round(1.0 - (node_edits + edge_edits) / denom, 4) if denom else 1.0


def skill_prf(ref_ops: list, pred_ops: list):
    """Per-skill precision/recall over op multisets."""
    rc, pc = Counter(ref_ops), Counter(pred_ops)
    tp = sum((rc & pc).values())
    prec = tp / sum(pc.values()) if pc else 0.0
    rec = tp / sum(rc.values()) if rc else 0.0
    return round(prec, 4), round(rec, 4)


def evaluate(predictions: list, train_opseqs: set,
             dump_unseen: str = None) -> dict:
    per_source = defaultdict(lambda: {
        "n": 0, "parsed": 0, "valid": 0, "executed": 0, "opseq_exact": 0,
        "ges_sum": 0.0, "prec_sum": 0.0, "rec_sum": 0.0,
        "exec_action_ops": 0, "total_ops": 0,
        "unseen_n": 0, "unseen_opseq_exact": 0, "unseen_valid": 0,
        "tp": 0, "fp": 0, "fn": 0})
    skill_tp, skill_fp, skill_fn = Counter(), Counter(), Counter()
    unseen_dump = []

    for p in predictions:
        src = p.get("source", "?")
        s = per_source[src]
        s["n"] += 1
        ref_ops, ref_mod = [], None
        ref_text = p.get("reference_plan") or ""
        try:
            ref_mod = parse_text(ref_text)
            ref_ops = ops_of(ref_mod)
        except TaskIRSyntaxError:
            pass
        try:
            pred_mod = parse_text(p.get("output_text", ""))
        except TaskIRSyntaxError as e:
            if "|".join(ref_ops) not in train_opseqs and ref_ops:
                s["unseen_n"] += 1
                unseen_dump.append({
                    "id": p.get("id", ""),
                    "instruction": (p.get("input_text") or "")[:300],
                    "reference_ops": ref_ops, "predicted_ops": None,
                    "parsed": False,
                    "validator_errors": [f"syntax: {e}"],
                    "graph_edit_similarity": 0.0})
            continue
        s["parsed"] += 1
        if not validate(pred_mod).valid:
            continue
        s["valid"] += 1
        res = Simulator(pred_mod, seed=f"nce:{p.get('id', '')}",
                        jitter=0).run()
        if res.status == "completed":
            s["executed"] += 1

        pred_ops = ops_of(pred_mod)
        unseen = "|".join(ref_ops) not in train_opseqs
        if unseen:
            s["unseen_n"] += 1
            if ref_ops == pred_ops:
                s["unseen_opseq_exact"] += 1
            s["unseen_valid"] += 1
            unseen_dump.append({
                "id": p.get("id", ""),
                "instruction": (p.get("input_text") or "")[:300],
                "reference_ops": ref_ops,
                "predicted_ops": pred_ops,
                "parsed": True,
                "validator_errors": [],
                "graph_edit_similarity": graph_edit_similarity(ref_mod,
                                                               pred_mod)})
        if ref_ops == pred_ops:
            s["opseq_exact"] += 1
        s["ges_sum"] += graph_edit_similarity(ref_mod, pred_mod)
        prec, rec = skill_prf(ref_ops, pred_ops)
        s["prec_sum"] += prec
        s["rec_sum"] += rec
        s["exec_action_ops"] += pred_ops.count("EXEC_ACTION")
        s["total_ops"] += len(pred_ops)
        rc, pc = Counter(ref_ops), Counter(pred_ops)
        s["tp"] += sum((rc & pc).values())
        s["fp"] += sum((pc - rc).values())
        s["fn"] += sum((rc - pc).values())
        for op in (rc & pc):
            skill_tp[op] += (rc & pc)[op]
        for op in (pc - rc):
            skill_fp[op] += (pc - rc)[op]
        for op in (rc - pc):
            skill_fn[op] += (rc - pc)[op]

    if dump_unseen and unseen_dump:
        pathlib.Path(dump_unseen).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(dump_unseen).write_text(
            json.dumps(unseen_dump, indent=2, ensure_ascii=False),
            encoding="utf-8")

    def pct(a, b):
        return round(100 * a / b, 2) if b else 0.0

    def f1(tp, fp, fn):
        return round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) \
            else 0.0

    out = {}
    for src, s in sorted(per_source.items()):
        out[src] = {
            "n": s["n"],
            "parse_rate_pct": pct(s["parsed"], s["n"]),
            "validator_pass_pct": pct(s["valid"], s["n"]),
            "execution_pct": pct(s["executed"], s["n"]),
            "opseq_exact_pct": pct(s["opseq_exact"], s["n"]),
            "graph_edit_similarity_mean": round(
                s["ges_sum"] / s["n"], 4) if s["n"] else 0.0,
            "skill_precision_mean": round(s["prec_sum"] / s["n"], 4)
            if s["n"] else 0.0,
            "skill_recall_mean": round(s["rec_sum"] / s["n"], 4)
            if s["n"] else 0.0,
            "skill_f1_micro": f1(s["tp"], s["fp"], s["fn"]),
            "generic_action_rate_pct": pct(s["exec_action_ops"],
                                           s["total_ops"]),
            "unseen_composition": {
                "n": s["unseen_n"],
                "opseq_exact_pct": pct(s["unseen_opseq_exact"], s["unseen_n"]),
                "validator_pass_pct_of_unseen": pct(s["unseen_valid"],
                                                    s["unseen_n"])},
        }
    out["per_skill"] = {op: {
        "precision": round(skill_tp[op] / (skill_tp[op] + skill_fp[op]), 4)
        if skill_tp[op] + skill_fp[op] else 0.0,
        "recall": round(skill_tp[op] / (skill_tp[op] + skill_fn[op]), 4)
        if skill_tp[op] + skill_fn[op] else 0.0,
        "f1": f1(skill_tp[op], skill_fp[op], skill_fn[op]),
        "support": skill_tp[op] + skill_fn[op]}
        for op in set(skill_tp) | set(skill_fp) | set(skill_fn)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="jsonl {id, source, output_text, reference_plan}")
    ap.add_argument("--train", default=None,
                    help="corpus train.jsonl for seen/unseen composition")
    ap.add_argument("--dump-unseen", default=None,
                    help="write per-case unseen-composition records here")
    ap.add_argument("--out", default=str(ROOT / "data" / "reports"
                                         / "neural_compiler_eval.json"))
    args = ap.parse_args()

    preds = [json.loads(l) for l in
             open(args.predictions, encoding="utf-8").read().splitlines()
             if l.strip()]

    train_opseqs = set()
    if args.train and pathlib.Path(args.train).exists():
        for line in open(args.train, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            try:
                m = parse_text(r.get("plan_target", ""))
                train_opseqs.add("|".join(ops_of(m)))
            except TaskIRSyntaxError:
                continue

    report = {"predictions": len(preds),
              "train_opseqs": len(train_opseqs)}
    ev = evaluate(preds, train_opseqs, dump_unseen=args.dump_unseen)
    report["per_skill"] = ev.pop("per_skill", {})
    report["per_source"] = ev

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    md = out.with_suffix(".md")
    L = ["# Neural Compiler Eval", "",
         "| source | n | parse% | valid% | exec% | opseq% | GES | P | R | F1 | "
         "generic-action% | unseen n | unseen opseq% |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for src, s in report["per_source"].items():
        if src == "per_skill":
            continue
        u = s["unseen_composition"]
        L.append(f"| {src} | {s['n']} | {s['parse_rate_pct']} "
                 f"| {s['validator_pass_pct']} | {s['execution_pct']} "
                 f"| {s['opseq_exact_pct']} | {s['graph_edit_similarity_mean']} "
                 f"| {s['skill_precision_mean']} | {s['skill_recall_mean']} "
                 f"| {s['skill_f1_micro']} "
                 f"| {s['generic_action_rate_pct']} | {u['n']} "
                 f"| {u['opseq_exact_pct']} |")
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "per_skill"}, indent=2)[:2000])
    print(f"report -> {out} / {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
