"""Phase 6A follow-up diagnostics (reviewer-directed):

1. E1-A as classifier (post-hoc attribution diagnostic, not preregistered):
   same Probe A/B prompts as B1/B2 -> is E5C-S's 0.782 boundary F1 from generic
   TaskIR training (E1-A similar) or 5C conditioning (E5C-S >> E1-A)?

2. Description-supported counterfactual slice: gold class independently
   supported by DESCRIPTION evidence (retrieval/action verb regexes), tool
   name excluded from subset selection. Run 4 transforms on this slice for
   grounder_g1g2_s42 + B2_e5cs + E1-A. Primary credibility readout:
   description_supported + name_masked.

Waits for the recovery chain (GPU) to finish.
"""
import json, os, sys, time, re
from collections import Counter

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

while True:
    log = ""
    try:
        with open("/tmp/phase6a_recovery.log", errors="ignore") as f:
            log = f.read()
    except FileNotFoundError:
        pass
    if "RECOVERY-CHAIN-DONE" in log:
        break
    print("waiting for recovery chain ...", flush=True)
    time.sleep(120)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.skill_grounder import (build_probe_prompt, parse_label,
                                         boundary_metrics, exact_metrics,
                                         format_capability, skill_family)

DATA = "data/phase6a_grounding"

RETRIEVAL_EVIDENCE = re.compile(
    r"\b(retriev|return|fetch|get|list|lookup|search|read|show|find|view|"
    r"information|details? of|query)\w*", re.I)
ACTION_EVIDENCE = re.compile(
    r"\b(create|modif|updat|delet|book|cancel|toggl|grant|reset|send|"
    r"purchas|order|reserv|submit|post|add|edit|remov|activate|deactivate|"
    r"transfer|subscrib|publish|upload|install)\w*", re.I)


def load(name):
    return [json.loads(l) for l in open(f"{DATA}/{name}", encoding="utf-8") if l.strip()]


def run_transform(tok, model, samples, transform, probe="A"):
    prompts = []
    for s in samples:
        p = build_probe_prompt(s, probe, transform=transform)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"]}],
            tokenize=False, add_generation_prompt=True))
    preds = []
    for bs in range(0, len(prompts), 16):
        inputs = tok(prompts[bs:bs+16], return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 temperature=None, pad_token_id=tok.pad_token_id)
        preds.extend(parse_label(tok.decode(o[inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True))
                     for o in out)
    return preds


def score(samples, preds):
    b = boundary_metrics(samples, preds)
    e = exact_metrics(samples, preds)
    per = e["per_skill"]
    return {
        "boundary_acc": b.get("binary_accuracy"), "boundary_macro_f1": b.get("macro_f1"),
        "exact_micro": e.get("micro_f1"), "exact_macro": e.get("macro_f1"),
        "ACTION_family_recall": b.get("per_class", {}).get("ACTION", {}).get("recall"),
        "EXEC_ACTION_recall": per.get("EXEC_ACTION", {}).get("recall"),
        "SEND_recall": per.get("SEND", {}).get("recall"),
        "SAVE_recall": per.get("SAVE", {}).get("recall"),
        "n": b.get("n"),
    }


def main():
    test_ood = load("test_ood.jsonl")
    abt = load("action_boundary_test.jsonl")

    # description-supported slice: subset decided by DESCRIPTION only
    ds = []
    for s in test_ood:
        if skill_family(s["target_skill"]) not in ("RETRIEVAL", "ACTION"):
            continue
        desc = (s["capability"].get("description") or "")
        ret, act = bool(RETRIEVAL_EVIDENCE.search(desc)), bool(ACTION_EVIDENCE.search(desc))
        if ret == act:  # ambiguous or empty -> exclude
            continue
        fam = skill_family(s["target_skill"])
        if (fam == "RETRIEVAL" and ret) or (fam == "ACTION" and act):
            ds.append(s)
    print(f"description_supported_test: {len(ds)} "
          f"({Counter(skill_family(s['target_skill']) for s in ds)})", flush=True)
    with open(f"{DATA}/description_supported_test.jsonl", "w", encoding="utf-8") as f:
        for s in ds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model specs: load ONE fresh base per adapter (no shared-base PEFT state
    # pollution, no lingering GPU references), evaluate, release, continue
    SPECS = [
        ("E1A_classifier", "runs/phase5b1/e1_3b_qlora/final"),
        ("B2_e5cs", "runs/phase5c/e5c_s/final"),
        ("grounder_g1g2_s42", "runs/phase6/grounder_g1g2_s42/final"),
        ("grounder_g1g2_s43", "runs/phase6/grounder_g1g2_s43/final"),
        ("grounder_g1g2_s44", "runs/phase6/grounder_g1g2_s44/final"),
    ]

    results = {"_note": "E1-A rows are post-hoc attribution diagnostics, not "
                        "preregistered primary results; the subset is "
                        "description-CORROBORATED (target skill still originates "
                        "from toolmap labels; description evidence only "
                        "corroborates it) — independent evidence comes from the "
                        "BFCL/tau3/AgentBoard oracles"}

    import gc
    for name, adapter in SPECS:
        if not os.path.exists(adapter):
            print(f"\n[skip missing {adapter}]", flush=True)
            continue
        print(f"\n[{name}] loading fresh base + adapter", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()

        r = {}
        for probe in ("A", "B"):
            preds = run_transform(tok, model, test_ood, "full", probe)
            r[f"probe{probe}_test_ood"] = score(test_ood, preds)
        preds = run_transform(tok, model, abt, "full", "A")
        r["action_boundary"] = score(abt, preds)
        print(f"  probeA ood: F1={r['probeA_test_ood']['boundary_macro_f1']} "
              f"EXR={r['probeA_test_ood']['EXEC_ACTION_recall']} "
              f"ACTION_R={r['probeA_test_ood']['ACTION_family_recall']} "
              f"SEND_R={r['probeA_test_ood']['SEND_recall']}", flush=True)
        cf = {}
        for transform in ("full", "name_masked", "desc_masked", "name_perturbed"):
            preds_full = run_transform(tok, model, test_ood, transform, "A")
            preds_ds = run_transform(tok, model, ds, transform, "A")
            cf[transform] = {"full_test_ood": score(test_ood, preds_full),
                             "description_corroborated": score(ds, preds_ds)}
            print(f"  {transform:15s} full F1={cf[transform]['full_test_ood']['boundary_macro_f1']} "
                  f"| dc F1={cf[transform]['description_corroborated']['boundary_macro_f1']} "
                  f"dc EXR={cf[transform]['description_corroborated']['EXEC_ACTION_recall']}",
                  flush=True)
        r["counterfactuals"] = cf
        results[name] = r

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs("results/phase6", exist_ok=True)
    with open("results/phase6/phase6a_followups.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved results/phase6/phase6a_followups.json", flush=True)
    print("FOLLOWUPS-DONE", flush=True)


if __name__ == "__main__":
    main()
