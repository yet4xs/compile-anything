"""Phase 6B-1.1 evaluator v3 — final statistical freeze.

A1 CondR-E2E: adherence credit ONLY when valid (parse-fail/invalid = 0)
A2 SkillF1 reported as |Valid AND E2E (invalid/parse-fail -> empty pred multiset,
   reference still enters denominator)
A3 EA-R reported as |Valid AND E2E (denominator = ALL retained cases' ref EA count)
A5 validator failure taxonomy per C2 seed (error codes + structural stats);
   predictions saved to results/phase6b/preds/ for reuse.
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


def build_user(arm, r):
    caps = {c["capability_id"]: c for c in r["available_capabilities"]}
    canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
    sel = [cid for cid in r["selected_capabilities"]
           if canon.get(cid, {}).get("label_tier") != "G3"]
    task = r["instruction"]
    if arm is None:
        return task
    if not sel:
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
    if arm == "C3":
        blocks = [blk(c, canon[cid]) for cid, c in caps.items() if cid in canon]
        if not blocks:
            return None
        return f"{task}\n\nAvailable capabilities:\n" + "\n\n".join(blocks[:15])
    raise ValueError(arm)


def depth_bucket(n):
    return "1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7+"


def run_arm(tok, adapter, arm, restrict_ids, save_name=None):
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    if adapter:
        model.eval()

    prompts, provided, buckets, kept = [], [], [], []
    for r in test:
        if restrict_ids is not None and r["id"] not in restrict_ids:
            continue
        user = build_user(arm, r)
        if user is None:
            continue
        kept.append(r)
        canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
        provided.append({canon[cid]["canonical_skill"] for cid in r["selected_capabilities"]
                         if cid in canon})
        n_act = len([n for n in (r.get("plan_json") or {}).get("program", {}).get("nodes", [])
                     if n.get("op") not in POLICY_OPS])
        buckets.append(depth_bucket(n_act))
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

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

    assert len(comps) == len(provided) == len(buckets) == len(kept)

    os.makedirs("results/phase6b/preds", exist_ok=True)
    if save_name:
        with open(f"results/phase6b/preds/{save_name}.jsonl", "w", encoding="utf-8") as f:
            for r, comp in zip(kept, comps):
                f.write(json.dumps({"id": r["id"], "output": comp[:3000]},
                                   ensure_ascii=False) + "\n")

    n = len(kept)
    g = Counter()
    tax = Counter()
    struct = Counter()
    per_depth = defaultdict(lambda: Counter())
    # E2E denominators
    skill_pred_e2e = skill_ref_e2e = skill_inter_e2e = 0
    ea_hit_e2e = ea_ref_e2e = 0
    for comp, prov, bk, r in zip(comps, provided, buckets, kept):
        per_depth[bk]["n"] += 1
        try:
            ref_ops = [x.op for x in parse_text(r["plan_target"]).program.nodes
                       if x.op not in POLICY_OPS]
        except Exception:
            ref_ops = None
        pred_ops = None
        valid = False
        module = None
        try:
            module = parse_text(extract_taskir_text(comp))
            g["parse"] += 1
            per_depth[bk]["parse"] += 1
            pred_ops = [x.op for x in module.program.nodes if x.op not in POLICY_OPS]
        except Exception:
            pass
        if pred_ops is not None:
            try:
                rep = validate(module)
                valid = rep.valid
                if not valid:
                    for e in rep.errors:
                        tax[e.code] += 1
                    nodes = module.program.nodes
                    struct["invalid_node_count"] += len(nodes)
                    struct["invalid_prog_len"] += len(comp)
                    ids = [x.id for x in nodes]
                    struct["dup_ssa"] += len(ids) - len(set(ids))
            except Exception:
                valid = False
        if valid:
            g["valid"] += 1
            per_depth[bk]["valid"] += 1

        # A3/A2 E2E denominators: reference always counts
        if ref_ops is not None:
            rc = Counter(ref_ops)
            pc = Counter(pred_ops) if (pred_ops is not None and valid) else Counter()
            inter = sum((pc & rc).values())
            skill_inter_e2e += inter
            skill_pred_e2e += sum(pc.values())
            skill_ref_e2e += sum(rc.values())
            if rc.get("EXEC_ACTION"):
                ea_ref_e2e += rc["EXEC_ACTION"]
                ea_hit_e2e += (pc & rc)["EXEC_ACTION"]
            if valid:
                g["skill_inter"] += inter
                g["skill_pred"] += sum(pc.values())
                g["skill_ref"] += sum(rc.values())
                if rc.get("EXEC_ACTION"):
                    g["ea_hit"] += (pc & rc)["EXEC_ACTION"]
                    g["ea_ref"] += rc["EXEC_ACTION"]
                if pred_ops == ref_ops:
                    g["opseq"] += 1
                    per_depth[bk]["opseq"] += 1
                g["unsupported"] += sum((pc - rc).values())
                g["pred_total_valid"] += sum(pc.values())
        # A1 CondR-E2E: valid required
        if prov:
            g["cond_tasks"] += 1
            inter_e = 0
            if pred_ops is not None and valid:
                g["cond_valid_tasks"] += 1
                inter_p = sum((Counter(prov) & Counter(pred_ops)).values())
                g["provided_recall_valid"] += inter_p / max(1, len(prov))
                inter_e = inter_p
                g["off_cond"] += sum((Counter(pred_ops) - Counter(prov)).values())
                g["pred_total_parsed"] += len(pred_ops)
            g["provided_recall_e2e"] += inter_e / max(1, len(prov))
            per_depth[bk]["cond_n"] += 1
            per_depth[bk]["provided_recall_e2e_sum"] += inter_e / max(1, len(prov))

    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    res = {
        "n": n,
        "parse_pct": round(100 * g["parse"] / n, 2),
        "valid_e2e_pct": round(100 * g["valid"] / n, 2),
        "opseq_e2e_pct": round(100 * g["opseq"] / n, 2),
        "skill_f1_given_valid": round(2 * g["skill_inter"] /
                                      max(1e-9, g["skill_pred"] + g["skill_ref"]), 4),
        "skill_f1_e2e": round(2 * skill_inter_e2e /
                              max(1e-9, skill_pred_e2e + skill_ref_e2e), 4),
        "ea_recall_given_valid_pct": round(100 * g["ea_hit"] / max(1, g["ea_ref"]), 2),
        "ea_recall_e2e_pct": round(100 * ea_hit_e2e / max(1, ea_ref_e2e), 2),
        "unsupported_vs_reference_skill_rate_pct": round(
            100 * g["unsupported"] / max(1, g["pred_total_valid"]), 2),
        "off_condition_skill_rate_pct": round(
            100 * g["off_cond"] / max(1, g["pred_total_parsed"]), 2),
        "provided_skill_recall_given_valid_pct": round(
            100 * g["provided_recall_valid"] / max(1, g["cond_valid_tasks"]), 2),
        "provided_skill_recall_e2e_pct": round(
            100 * g["provided_recall_e2e"] / max(1, g["cond_tasks"]), 2),
        "validator_error_taxonomy": dict(tax.most_common()),
        "structural_stats": {k: round(v / max(1, sum(tax.values())), 2)
                              if "count" in k or "len" in k else v
                              for k, v in struct.items()},
        "by_depth_e2e": {},
    }
    for bk, c in sorted(per_depth.items()):
        nn = max(1, c["n"])
        res["by_depth_e2e"][bk] = {
            "n": c["n"], "parse_pct": round(100 * c["parse"] / nn, 2),
            "valid_pct": round(100 * c["valid"] / nn, 2),
            "opseq_pct": round(100 * c["opseq"] / nn, 2),
            "provided_recall_e2e_pct": round(100 * c["provided_recall_e2e_sum"] /
                                             max(1, c["cond_n"]), 2) if c["cond_n"] else None,
        }
    return res


def main():
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    matched_ids = set()
    for r in test:
        canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
        if [cid for cid in r["selected_capabilities"]
                if canon.get(cid, {}).get("label_tier") != "G3"]:
            matched_ids.add(r["id"])
    print(f"matched subset: {len(matched_ids)}", flush=True)

    results = {"_matched_subset_n": len(matched_ids)}
    SPECS = [
        ("C0_e1a_matched", "runs/phase5b1/e1_3b_qlora/final", None, "c0_matched"),
        ("C0T_matched_control", "runs/phase6b/composer_C0T_s42/final", None, "c0t"),
    ]
    for seed in (42, 43, 44):
        p = f"runs/phase6b/composer_C1_s{seed}/final"
        if os.path.exists(p):
            SPECS.append((f"C1_s{seed}", p, "C1", f"c1_s{seed}"))
    for seed in (42, 43, 44):
        p = f"runs/phase6b/composer_C2_s{seed}/final"
        if os.path.exists(p):
            SPECS.append((f"C2_s{seed}", p, "C2", f"c2_s{seed}"))
    p = "runs/phase6b/composer_C3_s42/final"
    if os.path.exists(p):
        SPECS.append(("C3_s42", p, "C3", "c3_s42"))

    for name, adapter, arm, save in SPECS:
        print(f"\n[{name}]", flush=True)
        results[name] = run_arm(tok, adapter, arm, matched_ids, save_name=save)
        r = results[name]
        print(f"  valid={r['valid_e2e_pct']} opseq={r['opseq_e2e_pct']} "
              f"skF1|V={r['skill_f1_given_valid']} skF1e2e={r['skill_f1_e2e']} "
              f"EAR|V={r['ea_recall_given_valid_pct']} EARe2e={r['ea_recall_e2e_pct']} "
              f"condRe2e={r['provided_skill_recall_e2e_pct']} "
              f"unsup={r['unsupported_vs_reference_skill_rate_pct']} "
              f"tax={r['validator_error_taxonomy']}", flush=True)

    os.makedirs("results/phase6b", exist_ok=True)
    with open("results/phase6b/composer_eval_v3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nCOMPOSER-EVALV3-DONE", flush=True)


if __name__ == "__main__":
    main()
