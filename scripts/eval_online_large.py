"""Phase 7C-2 Task 5: large-scale online constrained decoding evaluation.

Arms (frozen): G0=0, G1=1, G2a=2, G2=3, G3=4.
One invocation covers ONE seed over an id list, checkpointing every
(seed, id, arm) record to results/phase7/online_large_runs.jsonl (append;
resume by skipping already-recorded triples).

Metrics per record (Task 6, E2E = valid AND opseq; invalid scores zero):
  parse_ok, valid, opseq_match, e2e,
  skill_f1_e2e (multiset F1 over policy-op-filtered op sequences),
  exec_recall_e2e (EXEC_ACTION multiset recall; None when ref has none),
  param_key_e2e (fixed denominator: 0 unless valid AND opseq),
  param_key_cond (conditional on valid AND opseq),
  tokens, latency, constraint stats (deadlock/violation/forced rates).

Usage:
  python scripts/eval_online_large.py --seed s43 \
      --ids data/phase7/confirmation_ids.json [--dev30] [--arms 0,1,2,3,4]
"""
import json, os, sys, time, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from collections import Counter

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.prompt_format import SYSTEM_PROMPT
from src.compiler.online_constrained_decoder import generate_constrained

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
ARM_NAMES = {0: "G0", 1: "G1", 2: "G2a", 3: "G2", 4: "G3"}
RUNS_PATH = "results/phase7/online_large_runs.jsonl"


def load_tasks():
    refs = {}
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        refs[r["id"]] = r.get("plan_text") or r.get("plan_target") or ""
    tasks = {}
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        gold = [c for c in t.get("candidates", []) if c.get("is_gold")]
        if not gold:
            continue
        tasks[t["id"]] = {
            "task": t["task"],
            "cap_blocks": "\n\n".join(c.get("block", "") for c in gold[:5]),
            "ref_text": refs.get(t["id"], ""),
        }
    return tasks


def filtered_nodes(mod):
    return [n for n in mod.program.nodes if n.op not in POLICY_OPS]


def opseq(mod):
    return [n.op for n in filtered_nodes(mod)]


def skill_f1(pred_ops, ref_ops):
    p, r = Counter(pred_ops), Counter(ref_ops)
    inter = sum((p & r).values())
    if not inter:
        return 0.0
    return round(2 * inter / (sum(p.values()) + sum(r.values())), 4)


def exec_recall(pred_ops, ref_ops):
    ref_e = sum(1 for o in ref_ops if o == "EXEC_ACTION")
    if not ref_e:
        return None
    pred_e = sum(1 for o in pred_ops if o == "EXEC_ACTION")
    # count-limited recall (cannot exceed ref count)
    return round(min(pred_e, ref_e) / ref_e, 4)


def param_key_jaccard(pred_mod, ref_mod):
    """Pairwise param key-set Jaccard over the FILTERED node sequence."""
    js = []
    for pn, rn in zip(filtered_nodes(pred_mod), filtered_nodes(ref_mod)):
        a, b = set(pn.params.keys()), set(rn.params.keys())
        js.append(1.0 if not a and not b else len(a & b) / len(a | b))
    return round(sum(js) / max(1, len(js)), 4)


def judge(text, ref_text):
    out = {"parse_ok": False, "valid": False, "opseq_match": None,
           "e2e": False, "skill_f1_e2e": 0.0, "exec_recall_e2e": 0.0,
           "param_key_e2e": 0.0, "param_key_cond": None, "error": None}
    try:
        mod = parse_text(text)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out
    out["parse_ok"] = True
    rep = validate(mod)
    out["valid"] = rep.valid
    if not rep.valid:
        out["error"] = "; ".join(str(e) for e in rep.errors[:3])[:200]
    if not ref_text:
        return out
    try:
        ref = parse_text(ref_text)
    except Exception:
        return out
    p_ops, r_ops = opseq(mod), opseq(ref)
    match = (p_ops == r_ops)
    out["opseq_match"] = match
    if rep.valid:
        out["skill_f1_e2e"] = skill_f1(p_ops, r_ops)
        er = exec_recall(p_ops, r_ops)
        out["exec_recall_e2e"] = er if er is not None else 0.0
        out["exec_recall_defined"] = er is not None
    else:
        out["exec_recall_defined"] = any(o == "EXEC_ACTION" for o in r_ops)
    if rep.valid and match:
        out["e2e"] = True
        jac = param_key_jaccard(mod, ref)
        out["param_key_e2e"] = jac
        out["param_key_cond"] = jac
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--ids", default="data/phase7/confirmation_ids.json")
    ap.add_argument("--dev30", action="store_true",
                    help="run the development-30 instead of confirmation")
    ap.add_argument("--arms", default="0,1,2,3,4")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: first N ids only (records tagged set unchanged)")
    args = ap.parse_args()

    arms = [int(a) for a in args.arms.split(",")]
    tasks = load_tasks()

    if args.dev30:
        id_spec = json.load(open(args.ids, encoding="utf-8"))
        ids = id_spec["dev30_ids"]
        tag = "dev30"
    else:
        id_spec = json.load(open(args.ids, encoding="utf-8"))
        ids = id_spec["confirmation_ids"]
        tag = "confirmation"
    if args.limit:
        ids = ids[:args.limit]
        tag = f"{tag}-smoke"

    # resume support
    done = set()
    if os.path.exists(RUNS_PATH):
        for line in open(RUNS_PATH, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["seed"], r["id"], r["arm"]))
    print(f"=== {tag}: {len(ids)} ids, seed={args.seed}, arms={arms}, "
          f"already-done={sum(1 for i in ids for a in arms if (args.seed, i, ARM_NAMES[a]) in done)} ===",
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(
        base, f"runs/phase6b/composer_C2_{args.seed}/final")
    model.eval()

    os.makedirs("results/phase7", exist_ok=True)
    fout = open(RUNS_PATH, "a", encoding="utf-8")

    t_start = time.time()
    n_done = 0
    n_total = len(ids) * len(arms)
    for tid in ids:
        s = tasks[tid]
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",
              "content": f"{s['task']}\n\nSelected capabilities:\n{s['cap_blocks']}"}],
            tokenize=False, add_generation_prompt=True)
        for level in arms:
            arm = ARM_NAMES[level]
            if (args.seed, tid, arm) in done:
                continue
            t0 = time.time()
            text, stats, _ = generate_constrained(
                model, tok, prompt, level,
                max_new_tokens=args.max_new_tokens)
            dt = round(time.time() - t0, 2)
            j = judge(text, s["ref_text"])
            rec = {"seed": args.seed, "id": tid, "arm": arm,
                   "set": tag, "latency_s": dt,
                   "tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                   "text": text, "judge": j}
            if stats is not None:
                rec["constraint_stats"] = {
                    "steps": stats.steps,
                    "forced_steps": stats.forced_steps,
                    "deadlocks": len(stats.deadlocks),
                    "dup_ext": stats.dup_forced_extensions,
                    "type_pref": stats.type_pref_applied,
                    "type_pref_fb": stats.type_pref_fallbacks,
                    "fsm_violation": stats.fsm_violation is not None,
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                eta = el / n_done * (n_total - sum(1 for i in ids for a in arms if (args.seed, i, ARM_NAMES[a]) in done) - n_done)
                print(f"  {n_done} records | {el/60:.0f}min elapsed | ETA {eta/60:.0f}min",
                      flush=True)
    fout.close()
    print(f"SEED-DONE {args.seed} ({n_done} new records)")


if __name__ == "__main__":
    main()
