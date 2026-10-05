"""Phase 6B-1 evaluator v2 — implements ALL reviewer fixes.

T1 explicit kept_records + zip alignment + assertions
T2 C0_e1a_matched on the identical retained subset (full-test C0 kept separately)
T4 multiset EXEC_ACTION recall: hit = (pred_multiset & ref_multiset)['EXEC_ACTION']
   reported micro + task-average
T5 Condition adherence in two versions: |Valid (conditional) and E2E
   (parse/invalid scored 0)
T6 'hallucination' renamed off-condition skill rate (vs selected caps);
   NEW unsupported-vs-reference skill rate = pred_multiset - ref_multiset share
T7 depth buckets counted BEFORE parse; report Parse/Valid-E2E/OpSeq-E2E/
   cond-recall-E2E per bucket
Arms: C0_e1a_full, C0_e1a_matched, C0T(post-hoc control, bare on same subset
training), C1x2, C2x3, C3x1. Sequential fresh-base loading.
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
        return None  # matches training retention exactly (tool-use only)

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


def run_arm(tok, adapter, arm, restrict_ids=None):
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
        kept.append(r)  # T1: paired records
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

    assert len(comps) == len(provided) == len(buckets) == len(kept), \
        f"alignment broken: {len(comps)}/{len(provided)}/{len(buckets)}/{len(kept)}"

    n = len(kept)
    g = Counter()
    per_depth = defaultdict(lambda: Counter())
    ea_micro_hit = ea_micro_ref = 0
    for comp, prov, bk, r in zip(comps, provided, buckets, kept):
        per_depth[bk]["n"] += 1  # T7: denominator BEFORE parse
        n_act_ref = None
        try:
            ref_ops = [x.op for x in parse_text(r["plan_target"]).program.nodes
                       if x.op not in POLICY_OPS]
        except Exception:
            ref_ops = None
        pred_ops = None
        try:
            module = parse_text(extract_taskir_text(comp))
            g["parse"] += 1
            per_depth[bk]["parse"] += 1
            pred_ops = [x.op for x in module.program.nodes if x.op not in POLICY_OPS]
        except Exception:
            pass
        valid = False
        if pred_ops is not None:
            try:
                valid = validate(module).valid
            except Exception:
                valid = False
        if valid:
            g["valid"] += 1
            per_depth[bk]["valid"] += 1
        # reference-dependent metrics require parse+ref (invalid kept 0 in E2E sense)
        if valid and ref_ops is not None:
            if pred_ops == ref_ops:
                g["opseq"] += 1
                per_depth[bk]["opseq"] += 1
            pc, rc = Counter(pred_ops), Counter(ref_ops)
            inter = sum((pc & rc).values())
            g["skill_inter"] += inter
            g["skill_pred"] += sum(pc.values())
            g["skill_ref"] += sum(rc.values())
            # T4: multiset EXEC_ACTION recall
            if rc.get("EXEC_ACTION"):
                ea_micro_hit += (pc & rc)["EXEC_ACTION"]
                ea_micro_ref += rc["EXEC_ACTION"]
                g["ea_tasks"] += 1
                g["ea_hit_sum"] += (pc & rc)["EXEC_ACTION"] / rc["EXEC_ACTION"]
            # T6: unsupported-vs-reference
            g["unsupported"] += sum((pc - rc).values())
            g["pred_total_valid"] += sum(pc.values())
        # T5/T6: adherence both versions
        if prov:
            g["cond_tasks"] += 1
            if pred_ops is not None:
                prov_c = Counter(prov)
                pred_c_all = Counter(pred_ops)
                inter_p = sum((prov_c & pred_c_all).values())
                g["off_cond"] += sum((pred_c_all - prov_c).values())
                g["pred_total_parsed"] += sum(pred_c_all.values())
                if valid:
                    g["cond_valid_tasks"] += 1
                    g["provided_recall_valid"] += inter_p / max(1, sum(prov_c.values()))
            # E2E: parse-fail counts as 0
            if pred_ops is not None and valid:
                pass
            inter_e = sum((Counter(prov) & Counter(pred_ops)).values()) if pred_ops is not None else 0
            g["provided_recall_e2e"] += inter_e / max(1, len(prov))
            per_depth[bk]["cond_n"] += 1
            per_depth[bk]["provided_recall_e2e_sum"] += inter_e / max(1, len(prov))

    del model
    if adapter:
        pass
    del base
    gc.collect()
    torch.cuda.empty_cache()

    res = {
        "n": n,
        "parse_pct": round(100 * g["parse"] / n, 2),
        "valid_e2e_pct": round(100 * g["valid"] / n, 2),
        "opseq_e2e_pct": round(100 * g["opseq"] / n, 2),
        "skill_micro_f1": round(2 * g["skill_inter"] /
                                max(1e-9, g["skill_pred"] + g["skill_ref"]), 4),
        "exec_action_recall_micro_pct": round(100 * ea_micro_hit / max(1, ea_micro_ref), 2),
        "exec_action_recall_taskavg_pct": round(100 * g["ea_hit_sum"] / max(1, g["ea_tasks"]), 2),
        "unsupported_vs_reference_skill_rate_pct": round(
            100 * g["unsupported"] / max(1, g["pred_total_valid"]), 2),
        "off_condition_skill_rate_pct": round(
            100 * g["off_cond"] / max(1, g["pred_total_parsed"]), 2),
        "provided_skill_recall_given_valid_pct": round(
            100 * g["provided_recall_valid"] / max(1, g["cond_valid_tasks"]), 2),
        "provided_skill_recall_e2e_pct": round(
            100 * g["provided_recall_e2e"] / max(1, g["cond_tasks"]), 2),
        "by_depth_e2e": {},
    }
    for bk, c in sorted(per_depth.items()):
        nn = max(1, c["n"])
        res["by_depth_e2e"][bk] = {
            "n": c["n"],
            "parse_pct": round(100 * c["parse"] / nn, 2),
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

    # T2: the C1/C2/C3 retained subset ids (identical rule across arms)
    matched_ids = []
    for r in test:
        canon = {c["capability_id"]: c for c in r["canonical_capabilities"]}
        sel = [cid for cid in r["selected_capabilities"]
               if canon.get(cid, {}).get("label_tier") != "G3"]
        if sel:
            matched_ids.append(r["id"])
    print(f"matched retained subset: {len(matched_ids)} / {len(test)}", flush=True)

    results = {"_matched_subset_n": len(matched_ids),
               "_note": "C0T is a POST-HOC matched-training control (bare retrain "
                        "on the C2-filtered training records), not preregistered"}

    SPECS = [
        ("C0_e1a_fulltest", "runs/phase5b1/e1_3b_qlora/final", None, None),
        ("C0_e1a_matched", "runs/phase5b1/e1_3b_qlora/final", None, set(matched_ids)),
        ("C0T_matched_control", "runs/phase6b/composer_C0T_s42/final", None, set(matched_ids)),
    ]
    for seed in (42, 43):
        p = f"runs/phase6b/composer_C1_s{seed}/final"
        if os.path.exists(p):
            SPECS.append((f"C1_s{seed}", p, "C1", set(matched_ids)))
    for seed in (42, 43, 44):
        p = f"runs/phase6b/composer_C2_s{seed}/final"
        if os.path.exists(p):
            SPECS.append((f"C2_s{seed}", p, "C2", set(matched_ids)))
    p = "runs/phase6b/composer_C3_s42/final"
    if os.path.exists(p):
        SPECS.append(("C3_s42", p, "C3", set(matched_ids)))

    for name, adapter, arm, ids in SPECS:
        if adapter and not os.path.exists(adapter):
            print(f"[skip missing {adapter}]", flush=True)
            continue
        print(f"\n[{name}]", flush=True)
        results[name] = run_arm(tok, adapter, arm, ids)
        r = results[name]
        print(f"  parse={r['parse_pct']} validE2E={r['valid_e2e_pct']} "
              f"opseqE2E={r['opseq_e2e_pct']} skillF1={r['skill_micro_f1']} "
              f"EA-R-micro={r['exec_action_recall_micro_pct']} "
              f"condR|valid={r['provided_skill_recall_given_valid_pct']} "
              f"condR-E2E={r['provided_skill_recall_e2e_pct']} "
              f"offCond={r['off_condition_skill_rate_pct']} "
              f"unsup-vs-ref={r['unsupported_vs_reference_skill_rate_pct']}", flush=True)

    os.makedirs("results/phase6b", exist_ok=True)
    with open("results/phase6b/composer_eval_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved results/phase6b/composer_eval_v2.json", flush=True)
    print("COMPOSER-EVALV2-DONE", flush=True)


if __name__ == "__main__":
    main()
