"""Phase 7C-1 Task 4: online constraint-guided decoding, first round (s43).

Arms on the SAME 30 tasks and CapabilityIR blocks as 7C-0.1 v2:
  G0  frozen greedy baseline (no FSM)
  G1  grammar FSM (EOS gate, line grammar, inputs '@' requirement, ...)
  G2  G1 + dynamic SSA (defined-before-use, arity cap, dup-def refusal,
      type-preference operand trie)

All arms use batch=1 greedy generation through the same wrapper. Final
legality is judged by the official parse_text() + validate() on the completed
text — the FSM never lowers that bar.

Per-sample records include both transition directions (fixed AND broken),
constraint stats (banned/forced/deadlock/...), tokens, and latency.
"""
import json, os, sys, time, argparse

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.compiler.prompt_format import SYSTEM_PROMPT
from src.compiler.online_constrained_decoder import generate_constrained

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
ARMS = [("G0", 0), ("G1", 1), ("G2", 2)]


def load_test_data(n=30):
    refs = {}
    for line in open("data/compiler_corpus_v4/test.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        refs[r["id"]] = r.get("plan_text", "") or r.get("plan_target", "")

    data = []
    for line in open("data/phase6b_resolver/common_eval_tasks.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        gold = [c for c in t.get("candidates", []) if c.get("is_gold")]
        if not gold:
            continue
        data.append({
            "id": t["id"], "task": t["task"],
            "cap_blocks": "\n\n".join(c.get("block", "") for c in gold[:5]),
            "ref_text": refs.get(t["id"], ""),
        })
        if len(data) >= n:
            break
    return data


def opseq(mod):
    return [n.op for n in mod.program.nodes if n.op not in POLICY_OPS]


def param_key_accuracy(pred_mod, ref_mod):
    """Mean Jaccard of param key sets over positionally aligned nodes
    (computed only when the op sequences match)."""
    ps, rs = opseq(pred_mod), opseq(ref_mod)
    if ps != rs:
        return None
    js = []
    for pn, rn in zip(pred_mod.program.nodes, ref_mod.program.nodes):
        a, b = set(pn.params.keys()), set(rn.params.keys())
        if not a and not b:
            js.append(1.0)
        else:
            js.append(len(a & b) / len(a | b))
    return round(sum(js) / max(1, len(js)), 3)


def judge(text, ref_text):
    """Official parser + validator verdict for one completion."""
    out = {"parse_ok": True, "valid": False, "opseq_match": None,
           "param_key_acc": None, "error": None}
    try:
        mod = parse_text(text)
    except Exception as e:
        out.update(parse_ok=False, error=f"{type(e).__name__}: {e}"[:200])
        return out
    rep = validate(mod)
    out["valid"] = rep.valid
    if not rep.valid:
        out["error"] = "; ".join(str(e) for e in rep.errors[:3])[:200]
    if ref_text:
        try:
            ref = parse_text(ref_text)
            out["opseq_match"] = (opseq(mod) == opseq(ref))
            if out["opseq_match"]:
                out["param_key_acc"] = param_key_accuracy(mod, ref)
        except Exception:
            pass
    return out


def pct(vals):
    return round(100 * sum(vals) / max(1, len(vals)), 1)


def pXX(lat, q):
    if not lat:
        return 0.0
    s = sorted(lat)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def main(n=30, seed="s43", arms=("G0", "G1", "G2")):
    print("=" * 64)
    print(f"Phase 7C-1 Task 4: online constrained decoding  seed={seed} n={n}")
    print("=" * 64)

    data = load_test_data(n)
    print(f"test samples: {len(data)}")

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
        base, f"runs/phase6b/composer_C2_{seed}/final")
    model.eval()

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": f"{s['task']}\n\nSelected capabilities:\n{s['cap_blocks']}"}],
        tokenize=False, add_generation_prompt=True) for s in data]

    samples = {s["id"]: {} for s in data}
    arm_summary = {}

    for arm, level in ARMS:
        if arm not in arms:
            continue
        print(f"\n=== {arm} (level={level}) ===", flush=True)
        lat, ntok, stats_list = [], [], []
        n_parse = n_valid = n_opseq = 0
        param_accs = []

        for i, (s, prompt) in enumerate(zip(data, prompts)):
            t0 = time.time()
            text, stats, _plen = generate_constrained(
                model, tok, prompt, level, max_new_tokens=512)
            dt = time.time() - t0
            lat.append(round(dt, 2))
            ntok.append(len(tok(text, add_special_tokens=False)["input_ids"]))

            j = judge(text, s["ref_text"])
            n_parse += j["parse_ok"]
            n_valid += j["valid"]
            n_opseq += bool(j["opseq_match"])
            if j["param_key_acc"] is not None:
                param_accs.append(j["param_key_acc"])

            rec = {"id": s["id"], "arm": arm, "latency_s": round(dt, 2),
                   "text": text, "judge": j}
            if stats is not None:
                rec["constraint_stats"] = {
                    "steps": stats.steps,
                    "mean_banned": round(sum(stats.banned_counts) /
                                         max(1, len(stats.banned_counts)), 0),
                    "forced_steps": stats.forced_steps,
                    "deadlocks": stats.deadlocks,
                    "dup_forced_extensions": stats.dup_forced_extensions,
                    "type_pref_applied": stats.type_pref_applied,
                    "type_pref_fallbacks": stats.type_pref_fallbacks,
                    "fsm_violation": stats.fsm_violation,
                }
                stats_list.append(stats)
            samples[s["id"]][arm] = rec

            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(data)}  valid so far {n_valid}",
                      flush=True)

        arm_summary[arm] = {
            "n": len(data),
            "parse_ok": n_parse, "parse_pct": pct(
                [r[arm]["judge"]["parse_ok"] for r in samples.values()
                 if arm in r]),
            "valid": n_valid, "valid_pct": pct(
                [r[arm]["judge"]["valid"] for r in samples.values()
                 if arm in r]),
            "opseq": n_opseq, "opseq_pct": pct(
                [bool(r[arm]["judge"]["opseq_match"])
                 for r in samples.values() if arm in r]),
            "param_key_acc_mean": (round(sum(param_accs) / len(param_accs), 3)
                                   if param_accs else None),
            "tokens_mean": round(sum(ntok) / max(1, len(ntok)), 1),
            "latency_p50": pXX(lat, 0.50), "latency_p95": pXX(lat, 0.95),
            "latency_mean": round(sum(lat) / max(1, len(lat)), 2),
        }
        if stats_list:
            arm_summary[arm]["constraints"] = {
                "forced_steps_total": sum(s.forced_steps for s in stats_list),
                "deadlocks_total": sum(len(s.deadlocks) for s in stats_list),
                "dup_ext_total": sum(s.dup_forced_extensions
                                     for s in stats_list),
                "type_pref_total": sum(s.type_pref_applied
                                       for s in stats_list),
                "fsm_violations": sum(1 for s in stats_list
                                      if s.fsm_violation),
            }
        a = arm_summary[arm]
        print(f"  parse={a['parse_pct']}% valid={a['valid_pct']}% "
              f"opseq={a['opseq_pct']}% param_acc={a['param_key_acc_mean']} "
              f"p50/p95={a['latency_p50']}/{a['latency_p95']}s")

    # transitions table (both directions)
    transitions = []
    for s in data:
        row = {"id": s["id"]}
        for arm, _ in ARMS:
            if arm in samples[s["id"]]:
                j = samples[s["id"]][arm]["judge"]
                row[arm] = ("valid" if j["valid"] else
                            "parse_fail" if not j["parse_ok"] else
                            "invalid")
        if "G0" in row:
            for arm in ("G1", "G2"):
                if arm in row:
                    if row["G0"] != "valid" and row[arm] == "valid":
                        row[f"{arm}_effect"] = "FIXED"
                    elif row["G0"] == "valid" and row[arm] != "valid":
                        row[f"{arm}_effect"] = "BROKEN"
                    elif row["G0"] == "valid" and row[arm] == "valid":
                        row[f"{arm}_effect"] = "kept"
                    else:
                        row[f"{arm}_effect"] = "still_failing"
        transitions.append(row)

    n_fixed_g1 = sum(1 for r in transitions if r.get("G1_effect") == "FIXED")
    n_broken_g1 = sum(1 for r in transitions if r.get("G1_effect") == "BROKEN")
    n_fixed_g2 = sum(1 for r in transitions if r.get("G2_effect") == "FIXED")
    n_broken_g2 = sum(1 for r in transitions if r.get("G2_effect") == "BROKEN")

    print("\n" + "=" * 64)
    print("SUMMARY (s43, n=%d)" % len(data))
    print("=" * 64)
    for arm in arms:
        if arm in arm_summary:
            a = arm_summary[arm]
            print(f"{arm}: parse={a['parse_pct']}% valid={a['valid_pct']}% "
                  f"opseq={a['opseq_pct']}%")
    print(f"G1: FIXED={n_fixed_g1} BROKEN={n_broken_g1}")
    print(f"G2: FIXED={n_fixed_g2} BROKEN={n_broken_g2}")
    print("\nper-sample transitions:")
    for r in transitions:
        print(f"  {r['id'][:40]:<42s} G0={r.get('G0','-'):<10s} "
              f"G1={r.get('G1','-'):<10s} ({r.get('G1_effect','-'):<13s}) "
              f"G2={r.get('G2','-'):<10s} ({r.get('G2_effect','-')})")

    os.makedirs("results/phase7", exist_ok=True)
    out = {"seed": seed, "n": len(data), "arms": arm_summary,
           "transitions": transitions, "samples": samples}
    with open("results/phase7/online_constrained.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved results/phase7/online_constrained.json")
    print("ONLINE-EVAL-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=str, default="s43")
    ap.add_argument("--arms", type=str, default="G0,G1,G2")
    args = ap.parse_args()
    main(args.n, args.seed, tuple(args.arms.split(",")))
