"""Phase 6A evaluation: baselines B0/B1/B2 + grounder, probes A/B/C,
boundary + exact metrics, counterfactual name-shortcut tests, NO_CALL.

Internal only (dev / test_id / test_ood / action_boundary_test).
External diagnostics run separately after model freeze (eval_phase6a_external.py).
"""
import json, os, sys, random, re, argparse
from collections import Counter

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.skill_grounder import (build_probe_prompt, parse_label,
                                         parse_probe_c, boundary_metrics,
                                         exact_metrics, skill_family,
                                         pick_hard_negatives, make_no_call)
from src.lifter.toolmap import map_tool

DATA = "data/phase6a_grounding"


def load(name):
    return [json.loads(l) for l in open(f"{DATA}/{name}", encoding="utf-8") if l.strip()]


def run_model(tok, model, samples, probe, transform="full", train_pool=None,
              batch=16, max_new=8):
    prompts, meta = [], []
    for i, s in enumerate(samples):
        cands = None
        if probe == "C":
            cands = pick_hard_negatives(s, train_pool, k=4, rng=random.Random(i))
            pos_idx = random.Random(i * 7).randrange(5)
            cands = cands[:pos_idx] + [s] + cands[pos_idx:]
        p = build_probe_prompt(s, probe, transform=transform, candidates=cands)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"]}],
            tokenize=False, add_generation_prompt=True))
        meta.append((s, cands))
    outs = []
    for bs in range(0, len(prompts), batch):
        batch_p = prompts[bs:bs + batch]
        inputs = tok(batch_p, return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        outs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True) for o in out)
    labels = []
    for (s, cands), o in zip(meta, outs):
        if probe == "C":
            _, lab = parse_probe_c(o)
        else:
            lab = parse_label(o)
        labels.append(lab)
    return labels


def eval_set(samples, preds):
    return {
        "boundary": boundary_metrics(samples, preds),
        "exact": exact_metrics(samples, preds),
    }


def toolmap_baseline(samples):
    preds = []
    for s in samples:
        try:
            preds.append(map_tool(s["capability"]["name"], {}).get("skill"))
        except Exception:
            preds.append(None)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounder-runs", nargs="*", default=[
        "runs/phase6/grounder_g1_s42/final",
        "runs/phase6/grounder_g1_s43/final",
        "runs/phase6/grounder_g1_s44/final",
        "runs/phase6/grounder_g1g2_s42/final",
    ])
    ap.add_argument("--skip-e5cs", action="store_true")
    args = ap.parse_args()

    dev = load("dev.jsonl")
    test_id = load("test_id.jsonl")
    test_ood = load("test_ood.jsonl")
    abt = load("action_boundary_test.jsonl")
    train_pool = load("train.jsonl")
    print(f"dev={len(dev)} test_id={len(test_id)} test_ood={len(test_ood)} abt={len(abt)}")

    results = {}

    # ── B0: deterministic toolmap ──
    print("\n[B0 toolmap]", flush=True)
    for name, ss in (("dev", dev), ("test_id", test_id), ("test_ood", test_ood)):
        results[f"B0_toolmap/{name}"] = eval_set(ss, toolmap_baseline(ss))
        b = results[f"B0_toolmap/{name}"]["boundary"]
        print(f"  {name}: boundary acc={b.get('binary_accuracy')} macroF1={b.get('macro_f1')}")

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    def model_suite(key, model):
        print(f"\n[{key}]", flush=True)
        for probe in ("A", "B"):
            for name, ss in (("dev", dev), ("test_id", test_id), ("test_ood", test_ood)):
                preds = run_model(tok, model, ss, probe)
                results[f"{key}/probe{probe}/{name}"] = eval_set(ss, preds)
                b = results[f"{key}/probe{probe}/{name}"]["boundary"]
                e = results[f"{key}/probe{probe}/{name}"]["exact"]
                print(f"  probe{probe} {name}: boundary acc={b.get('binary_accuracy')} "
                      f"macroF1={b.get('macro_f1')} | exact micro={e.get('micro_f1')} "
                      f"macro={e.get('macro_f1')}", flush=True)
        # action boundary slice (probe A)
        preds = run_model(tok, model, abt, "A")
        results[f"{key}/action_boundary"] = eval_set(abt, preds)
        b = results[f"{key}/action_boundary"]["boundary"]
        ea_rec = results[f"{key}/action_boundary"]["exact"]["per_skill"].get("EXEC_ACTION", {})
        print(f"  action_boundary: acc={b.get('binary_accuracy')} macroF1={b.get('macro_f1')} "
              f"EXEC_ACTION recall={ea_rec.get('recall')}", flush=True)

    # ── B1: zero-shot 3B ──
    model_suite("B1_zeroshot_3b", base)

    # ── B2: E5C-S as classifier ──
    if not args.skip_e5cs:
        m = PeftModel.from_pretrained(base, "runs/phase5c/e5c_s/final")
        m.eval()
        model_suite("B2_e5cs", m)
        del m
        torch.cuda.empty_cache()

    # ── G1 grounder (per run) ──
    per_run = {}
    for run_path in args.grounder_runs:
        if not os.path.exists(run_path):
            print(f"\n[skip missing {run_path}]", flush=True)
            continue
        name = os.path.basename(os.path.dirname(run_path))  # grounder_g1_s42
        m = PeftModel.from_pretrained(base, run_path)
        m.eval()
        print(f"\n[G1 {name}]", flush=True)
        r = {}
        for probe in ("A", "B"):
            for split, ss in (("dev", dev), ("test_id", test_id), ("test_ood", test_ood)):
                preds = run_model(tok, m, ss, probe)
                r[f"probe{probe}/{split}"] = eval_set(ss, preds)
                b = r[f"probe{probe}/{split}"]["boundary"]
                print(f"  probe{probe} {split}: acc={b.get('binary_accuracy')} "
                      f"macroF1={b.get('macro_f1')}", flush=True)
        preds = run_model(tok, m, abt, "A")
        r["action_boundary"] = eval_set(abt, preds)
        per_run[name] = r
        del m
        torch.cuda.empty_cache()

    # aggregate grounder mean±std across g1 seeds
    g1_runs = [v for k, v in per_run.items() if k.startswith("grounder_g1_")]
    if g1_runs:
        summary = {}
        for probe in ("A", "B"):
            for split in ("dev", "test_id", "test_ood"):
                f1s = [r[f"probe{probe}/{split}"]["boundary"]["macro_f1"] for r in g1_runs]
                accs = [r[f"probe{probe}/{split}"]["boundary"]["binary_accuracy"] for r in g1_runs]
                mu = lambda xs: round(sum(xs) / len(xs), 4)
                sd = lambda xs: round((sum((x - sum(xs)/len(xs))**2 for x in xs)/len(xs))**0.5, 4)
                summary[f"probe{probe}/{split}"] = {
                    "boundary_macro_f1_mean": mu(f1s), "boundary_macro_f1_std": sd(f1s),
                    "boundary_acc_mean": mu(accs), "boundary_acc_std": sd(accs)}
        ea = [r["action_boundary"]["exact"]["per_skill"].get("EXEC_ACTION", {}).get("recall", 0)
              for r in g1_runs]
        summary["action_boundary_EXEC_ACTION_recall"] = {
            "mean": round(sum(ea) / len(ea), 4),
            "std": round((sum((x - sum(ea)/len(ea))**2 for x in ea)/len(ea))**0.5, 4)}
        results["G1_grounder_agg"] = summary
        results["G1_grounder_per_run"] = per_run

    # ── Task 14: counterfactuals (grounder seed 42, probe A, test_ood) ──
    first_run = args.grounder_runs[0] if args.grounder_runs else None
    if first_run and os.path.exists(first_run):
        m = PeftModel.from_pretrained(base, first_run)
        m.eval()
        print("\n[Task 14 counterfactuals]", flush=True)
        cf = {}
        for transform in ("full", "name_masked", "desc_masked", "name_perturbed"):
            preds = run_model(tok, m, test_ood, "A", transform=transform)
            mres = eval_set(test_ood, preds)
            cf[transform] = {
                "boundary_macro_f1": mres["boundary"]["macro_f1"],
                "boundary_acc": mres["boundary"]["binary_accuracy"],
                "exact_micro": mres["exact"]["micro_f1"],
                "action_recall": mres["exact"]["per_skill"].get("EXEC_ACTION", {}).get("recall"),
            }
            print(f"  {transform}: F1={cf[transform]['boundary_macro_f1']} "
                  f"acc={cf[transform]['boundary_acc']} "
                  f"EXEC_ACTION recall={cf[transform]['action_recall']}", flush=True)
        results["counterfactuals"] = cf
        del m
        torch.cuda.empty_cache()

    # ── Probe C + NO_CALL (grounder seed 42, 500-sample subset) ──
    if first_run and os.path.exists(first_run):
        m = PeftModel.from_pretrained(base, first_run)
        m.eval()
        rng = random.Random(7)
        sub = rng.sample(test_ood, min(500, len(test_ood)))
        print(f"\n[Probe C on {len(sub)} samples]", flush=True)
        c_preds, golds = [], []
        prompts = []
        metas = []
        for i, s in enumerate(sub):
            cands = pick_hard_negatives(s, train_pool, k=4, rng=random.Random(i))
            pos_idx = random.Random(i * 7).randrange(5)
            cands = cands[:pos_idx] + [s] + cands[pos_idx:]
            p = build_probe_prompt(s, "C", candidates=cands)
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": p["system"]},
                 {"role": "user", "content": p["user"]}],
                tokenize=False, add_generation_prompt=True))
            metas.append((s, pos_idx))
        outs = []
        for bs in range(0, len(prompts), 32):
            inputs = tok(prompts[bs:bs+32], return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(m.device)
            with torch.no_grad():
                out = m.generate(**inputs, max_new_tokens=16, do_sample=False,
                                 temperature=None, pad_token_id=tok.pad_token_id)
            outs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True) for o in out)
        cap_ok = skill_ok = joint = 0
        for (s, pos), o in zip(metas, outs):
            cid, lab = parse_probe_c(o)
            if cid == pos + 1:
                cap_ok += 1
            if lab == s["target_skill"]:
                skill_ok += 1
            if cid == pos + 1 and lab == s["target_skill"]:
                joint += 1
        results["probe_C"] = {"n": len(sub),
                              "capability_selection_acc": round(cap_ok/len(sub), 4),
                              "skill_acc": round(skill_ok/len(sub), 4),
                              "joint_exact": round(joint/len(sub), 4)}
        print(f"  cap_sel={results['probe_C']['capability_selection_acc']} "
              f"skill={results['probe_C']['skill_acc']} joint={results['probe_C']['joint_exact']}",
              flush=True)

        # NO_CALL: remove correct capability, keep 4 distractors
        print("\n[NO_CALL probe]", flush=True)
        nc_prompts, nc_meta = [], []
        for i, s in enumerate(sub):
            cands = pick_hard_negatives(s, train_pool, k=4, rng=random.Random(i))
            p = build_probe_prompt(s, "C", candidates=cands)
            nc_prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": p["system"]},
                 {"role": "user", "content": p["user"]}],
                tokenize=False, add_generation_prompt=True))
            nc_meta.append(s)
        outs = []
        for bs in range(0, len(nc_prompts), 32):
            inputs = tok(nc_prompts[bs:bs+32], return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(m.device)
            with torch.no_grad():
                out = m.generate(**inputs, max_new_tokens=16, do_sample=False,
                                 temperature=None, pad_token_id=tok.pad_token_id)
            outs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True) for o in out)
        nc = sum(1 for o in outs if parse_label(o) == "NO_CALL")
        results["no_call"] = {"n": len(sub), "synthetic_negative": True,
                              "no_call_rate": round(nc/len(sub), 4)}
        print(f"  NO_CALL rate={results['no_call']['no_call_rate']}", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs("results/phase6", exist_ok=True)
    with open("results/phase6/phase6a_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved results/phase6/phase6a_metrics.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
