"""Phase 6B Stage 5-6: Composer internal evaluation.

Arms evaluated: C0 (frozen E1-A + E5C-S, no training) / C1 / C2 / C3
(seeds aggregated). Metrics:
  Parse / Validator / OpSeq / Skill F1 / EXEC_ACTION recall / argument fidelity
  NEW — Condition Adherence: of the output TaskIR action nodes' skills,
  how many come from the PROVIDED selected capabilities:
    provided-skill recall / precision / hallucinated-skill rate
Depth buckets (1 / 2-3 / 4-6 / 7+): provided-skill recall, OpSeq, Valid.
Sequential fresh-base loading per adapter (6A lesson).
"""
import json, os, sys, gc
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.ir.parser import parse_text
from src.validator.validator import validate

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

test = [json.loads(l) for l in open("data/compiler_corpus_v4/test.jsonl",
                                    encoding="utf-8") if l.strip()]
print(f"test records: {len(test)}", flush=True)


def depth_bucket(n):
    return "1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7+"


def ref_skills(rec):
    canon = {c["capability_id"]: c for c in rec["canonical_capabilities"]}
    return {canon[cid]["canonical_skill"] for cid in rec["selected_capabilities"]
            if cid in canon}


def run_adapter(tok, base, adapter, arm):
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    if adapter:
        model.eval()
    prompts, provided, buckets = [], [], []
    for r in test:
        user = tpc_build_user(arm, r)
        if user is None:
            continue  # same filtering as training (fair comparison)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
        provided.append(ref_skills(r))
        n_act = len([n for n in (r.get("plan_json") or {}).get("program", {}).get("nodes", [])
                     if n.get("op") not in POLICY_OPS])
        buckets.append(depth_bucket(n_act))
    comps = []
    for bs in range(0, len(prompts), 16):
        if bs % 320 == 0:
            print(f"    {bs}/{len(prompts)}", flush=True)
        batch = prompts[bs:bs+16]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                 temperature=None, pad_token_id=tok.pad_token_id)
        comps.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True) for o in out)

    g = Counter()
    n = len(prompts)
    per_depth = defaultdict(lambda: Counter())
    for comp, prov, bk, r in zip(comps, provided, buckets, test):
        try:
            module = parse_text(extract_taskir_text(comp))
            g["parse"] += 1
            per_depth[bk]["n"] += 1
        except Exception:
            continue
        if validate(module).valid:
            g["valid"] += 1
            per_depth[bk]["valid"] += 1
        else:
            continue
        nodes = [x for x in module.program.nodes if x.op not in POLICY_OPS]
        pred_skills = [x.op for x in nodes]
        # OpSeq
        try:
            ref = parse_text(r.get("plan_target", ""))
            ref_ops = [x.op for x in ref.program.nodes if x.op not in POLICY_OPS]
            if pred_skills == ref_ops:
                g["opseq"] += 1
                per_depth[bk]["opseq"] += 1
        except Exception:
            pass
        # skill multiset F1 vs reference ops + EXEC_ACTION recall
        from collections import Counter as C
        pc, rc = C(pred_skills), C(ref_ops)
        inter = sum((pc & rc).values())
        prec = inter / max(1, sum(pc.values()))
        rec_ = inter / max(1, sum(rc.values()))
        f1 = 2 * prec * rec_ / max(1e-9, prec + rec_)
        g["skill_f1_sum"] += f1
        g["skill_micro_inter"] += inter
        g["skill_micro_pred"] += sum(pc.values())
        g["skill_micro_ref"] += sum(rc.values())
        n_ea_ref = rc.get("EXEC_ACTION", 0)
        if n_ea_ref:
            g["ea_ref_total_tasks"] += 1
            g["ea_hit_tasks"] += pc.get("EXEC_ACTION", 0) / n_ea_ref
        # argument fidelity: plan params non-empty share among tool-use recs
        if any((x.params or {}) for x in nodes):
            g["args_nonempty"] += 1
        # ── Condition Adherence ──
        if prov:
            prov_c = Counter(prov)
            pred_c = Counter(pred_skills)
            inter_p = sum((prov_c & pred_c).values())
            g["cond_tasks"] += 1
            g["provided_recall_sum"] += inter_p / max(1, sum(prov_c.values()))
            g["provided_precision_sum"] += inter_p / max(1, sum(pred_c.values()))
            halluc = sum((pred_c - prov_c).values())
            g["halluc_total"] += halluc
            g["pred_total"] += sum(pred_c.values())
            per_depth[bk]["cond_n"] += 1
            per_depth[bk]["provided_recall_sum"] += inter_p / max(1, sum(prov_c.values()))

    res = {
        "n": n,
        "parse_pct": round(100 * g["parse"] / n, 2),
        "valid_pct": round(100 * g["valid"] / n, 2),
        "opseq_pct": round(100 * g["opseq"] / n, 2),
        "skill_f1_macro_over_tasks": round(g["skill_f1_sum"] / max(1, n), 4),
        "skill_micro_f1": round(2 * g["skill_micro_inter"] /
                                max(1e-9, g["skill_micro_pred"] + g["skill_micro_ref"]), 4),
        "exec_action_recall_taskavg_pct": round(100 * g["ea_hit_tasks"] /
                                                max(1, g["ea_ref_total_tasks"]), 2),
        "argument_fidelity_nonempty_pct": round(100 * g["args_nonempty"] /
                                                max(1, g["valid"]), 2),
        "condition_adherence": {
            "tasks_with_condition": g["cond_tasks"],
            "provided_skill_recall_pct": round(100 * g["provided_recall_sum"] /
                                               max(1, g["cond_tasks"]), 2),
            "provided_skill_precision_pct": round(100 * g["provided_precision_sum"] /
                                                  max(1, g["cond_tasks"]), 2),
            "hallucinated_skill_rate_pct": round(100 * g["halluc_total"] /
                                                 max(1, g["pred_total"]), 2),
        },
        "by_depth": {},
    }
    for bk, c in sorted(per_depth.items()):
        nn = max(1, c["n"])
        res["by_depth"][bk] = {
            "n": c["n"],
            "valid_pct": round(100 * c["valid"] / nn, 2),
            "opseq_pct": round(100 * c["opseq"] / nn, 2),
            "provided_skill_recall_pct": round(100 * c["provided_recall_sum"] /
                                               max(1, c["cond_n"]), 2) if c["cond_n"] else None,
        }
    if adapter:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return res


def tpc_build_user(arm, r):
    """Same conditioning logic as training, parameterized by arm."""
    caps = {c["capability_id"]: c for c in r["available_capabilities"]}
    canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
    sel = [cid for cid in r["selected_capabilities"]
           if canon.get(cid, {}).get("label_tier") != "G3"]
    task = r["instruction"]
    if arm is None:  # C0 bare protocol: no conditioning block at all
        return task
    if not sel and r["available_capabilities"]:
        return None

    def blk(cap, cn):
        lines = [f"[{cap['capability_id']}] name: {cap['name']}"]
        if cap.get("description"):
            lines.append(f"description: {cap['description'][:200]}")
        if cap.get("parameters", {}).get("properties"):
            p = cap["parameters"]["properties"]
            lines.append("parameters: " + ", ".join(f"{k}:{v}" for k, v in list(p.items())[:6]))
        lines.append(f"canonical_skill: {cn['canonical_skill']}")
        return "\n".join(lines)

    if arm == "C1":
        skills = sorted({canon[cid]["canonical_skill"] for cid in sel if cid in canon})
        if not skills:
            return None
        return f"{task}\n\nSelected skills:\n" + "\n".join(f"- {s}" for s in skills)
    if arm == "C2":
        blocks = [blk(caps[cid], canon[cid]) for cid in sel if cid in caps and cid in canon]
        if not blocks:
            return None
        return f"{task}\n\nSelected capabilities:\n" + "\n\n".join(blocks)
    blocks = [blk(c, canon[cid]) for cid, c in caps.items() if cid in canon]
    if not blocks:
        return None
    return f"{task}\n\nAvailable capabilities:\n" + "\n\n".join(blocks[:15])


def main():
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = {}
    # C0 baseline: frozen E1-A on bare prompts (its matched protocol)
    SPECS = [("C0_e1a", "runs/phase5b1/e1_3b_qlora/final", None)]
    for seed in (42, 43, 44):
        for arm in ("C1", "C2"):
            p = f"runs/phase6b/composer_{arm}_s{seed}/final"
            if os.path.exists(p):
                SPECS.append((f"{arm}_s{seed}", p, arm))
    p3 = "runs/phase6b/composer_C3_s42/final"
    if os.path.exists(p3):
        SPECS.append(("C3_s42", p3, "C3"))

    for name, adapter, arm in SPECS:
        print(f"\n[{name}]", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        r = run_adapter(tok, base, adapter, arm)
        results[name] = r
        ca = r["condition_adherence"]
        print(f"  valid={r['valid_pct']} opseq={r['opseq_pct']} "
              f"skillF1={r['skill_f1_macro_over_tasks']} EA-R={r['exec_action_recall_taskavg_pct']} "
              f"| cond-recall={ca['provided_skill_recall_pct']}% "
              f"cond-prec={ca['provided_skill_precision_pct']}% "
              f"halluc={ca['hallucinated_skill_rate_pct']}%", flush=True)
        del base
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs("results/phase6b", exist_ok=True)
    with open("results/phase6b/composer_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved results/phase6b/composer_eval.json", flush=True)
    print("COMPOSER-EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()
