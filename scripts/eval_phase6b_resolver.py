"""Phase 6B-2 B10-B11: Resolver evaluation — set metrics, NO_CALL, slices.

R2: score every (task, capability) pair; select label=='relevant' (greedy);
    empty selection -> NONE prediction.
R1: generative id list; parse c-ids / NONE.
Slices: single/multi, ACTION/RETRIEVAL selected, unseen family, false-select
rate by negative_type.
"""
import json, os, sys, re, gc
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DATA = "data/phase6b_resolver"
SYSTEM_R = ("You are a capability resolver for a task compiler. Given a task and "
            "capability descriptions, decide which capabilities the task requires. "
            "Answer with the label ONLY.")

ACTION_FAMS = {"EXEC_ACTION", "SEND", "SAVE"}
RETR_FAMS = {"SEARCH", "FETCH", "QUERY_DB"}


def fam_class(skills):
    if any(s in ACTION_FAMS for s in skills):
        return "ACTION"
    if any(s in RETR_FAMS for s in skills):
        return "RETRIEVAL"
    return "OTHER"


def load_train_families():
    fams = set()
    for line in open("data/compiler_corpus_v4/train.jsonl", encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for c in r["canonical_capabilities"]:
            fams.add("_".join(c["name"].split("_")[:2]))
    return fams


def set_metrics(y_true, y_pred):
    tp = len(y_true & y_pred)
    prec = tp / max(1, len(y_pred))
    rec = tp / max(1, len(y_true))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1, y_true == y_pred


def main():
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tasks = [json.loads(l) for l in open(f"{DATA}/r2_tasks_test.jsonl",
                                        encoding="utf-8") if l.strip()]
    none_tasks = [json.loads(l) for l in open(f"{DATA}/none_tasks_test.jsonl",
                                              encoding="utf-8") if l.strip()]
    train_fams = load_train_families()
    print(f"tasks={len(tasks)} none_tasks={len(none_tasks)}", flush=True)

    SPECS = [("R2_s42", "runs/phase6b/resolver_R2_s42/final", "R2"),
             ("R2_s43", "runs/phase6b/resolver_R2_s43/final", "R2"),
             ("R2_s44", "runs/phase6b/resolver_R2_s44/final", "R2"),
             ("R1_s42", "runs/phase6b/resolver_R1_s42/final", "R1")]
    results = {}
    for name, adapter, mode in SPECS:
        if not os.path.exists(adapter):
            print(f"[skip {adapter}]", flush=True)
            continue
        base = AutoModelForCausalLM.from_pretrained(
            "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        print(f"\n[{name}]", flush=True)

        def gen(prompts, max_new=8, batch=32):
            outs = []
            for bs in range(0, len(prompts), batch):
                inputs = tok(prompts[bs:bs+batch], return_tensors="pt", padding=True,
                             truncation=True, max_length=2048).to(model.device)
                with torch.no_grad():
                    o = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                       temperature=None, pad_token_id=tok.pad_token_id)
                outs.extend(tok.decode(x[inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True) for x in o)
            return outs

        g = Counter()
        slices = defaultdict(lambda: Counter())
        false_sel_by_negtype = Counter()
        negtype_totals = Counter()
        if mode == "R2":
            for i, t in enumerate(tasks):
                if i % 100 == 0:
                    print(f"  {i}/{len(tasks)}", flush=True)
                prompts = [tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_R},
                     {"role": "user", "content":
                      f"Task:\n{t['task']}\n\nCapability:\n{blk}\n\n"
                      "Is this capability required by the task? "
                      "Answer 'relevant' or 'irrelevant'."}],
                    tokenize=False, add_generation_prompt=True)
                    for blk in t["blocks"].values()]
                outs = gen(prompts)
                pred_ids = {cid for cid, o in zip(t["blocks"].keys(), outs)
                            if "relevant" in o.lower() and "irrelevant" not in o.lower()}
                gold = set(t["gold_ids"])
                prec, rec, f1, exact = set_metrics(gold, pred_ids)
                g["n"] += 1
                g["prec_sum"] += prec
                g["rec_sum"] += rec
                g["f1_sum"] += f1
                g["exact"] += exact
                g["top1"] += int(bool(pred_ids) and
                                 (max(pred_ids, key=lambda c: 0) in gold if pred_ids else False))
                # false selects feed negative-type analysis (all non-gold selected)
                for cid in pred_ids - gold:
                    negtype_totals["selected_negatives"] += 1
                # slices
                n_gold = len(gold)
                sl = slices["multi" if n_gold > 1 else "single"]
                sl["n"] += 1
                sl["f1_sum"] += f1
                sl["exact"] += exact
                fc = fam_class(t["gold_skills"])
                slices[fc]["n"] += 1
                slices[fc]["f1_sum"] += f1
                # unseen family: any selected tool family unseen in train
                unseen = any("_".join(blk.split("\n")[0].split("name: ")[-1]
                                      .split("_")[:2]) not in train_fams
                             for blk in (t["blocks"][cid] for cid in gold
                                         if cid in t["blocks"]))
                slices["unseen_family" if unseen else "seen_family"]["n"] += 1
                slices["unseen_family" if unseen else "seen_family"]["f1_sum"] += f1
            # NONE tasks: predict NONE if no relevant
            none_ok = 0
            for i, t in enumerate(none_tasks):
                prompts = [tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_R},
                     {"role": "user", "content":
                      f"Task:\n{t['task']}\n\nCapability:\n{blk}\n\n"
                      "Is this capability required by the task? "
                      "Answer 'relevant' or 'irrelevant'."}],
                    tokenize=False, add_generation_prompt=True)
                    for blk in t["blocks"].values()]
                outs = gen(prompts)
                if not any("relevant" in o.lower() and "irrelevant" not in o.lower()
                           for o in outs):
                    none_ok += 1
            g["none_ok"] = none_ok
            g["none_n"] = len(none_tasks)
        else:  # R1
            prompts = []
            for t in tasks:
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_R},
                     {"role": "user", "content":
                      f"Task:\n{t['task']}\n\nAvailable capabilities:\n" +
                      "\n\n".join(t["blocks"].values()) +
                      "\n\nWhich capabilities does the task require? "
                      "Answer as a comma-separated id list (e.g. c3,c7) or NONE."}],
                    tokenize=False, add_generation_prompt=True))
            outs = gen(prompts, max_new=24)
            for t, o in list(zip(tasks, outs))[:3]:
                print(f"  R1 RAW: {o[:120]!r}", flush=True)
            for t, o in zip(tasks, outs):
                ou = o.upper()
                if "NONE" in ou:
                    pred_ids = set()
                else:
                    pred_ids = set(re.findall(r"\bc\d+\b", ou))
                gold = set(t["gold_ids"])
                prec, rec, f1, exact = set_metrics(gold, pred_ids)
                g["n"] += 1
                g["prec_sum"] += prec
                g["rec_sum"] += rec
                g["f1_sum"] += f1
                g["exact"] += exact
                n_gold = len(gold)
                sl = slices["multi" if n_gold > 1 else "single"]
                sl["n"] += 1
                sl["f1_sum"] += f1
                sl["exact"] += exact
                fc = fam_class(t["gold_skills"])
                slices[fc]["n"] += 1
                slices[fc]["f1_sum"] += f1
            # R1 NONE
            prompts = []
            for t in none_tasks:
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_R},
                     {"role": "user", "content":
                      f"Task:\n{t['task']}\n\nAvailable capabilities:\n" +
                      "\n\n".join(t["blocks"].values()) +
                      "\n\nWhich capabilities does the task require? "
                      "Answer as a comma-separated id list (e.g. c3,c7) or NONE."}],
                    tokenize=False, add_generation_prompt=True))
            outs = gen(prompts, max_new=24)
            g["none_ok"] = sum(1 for o in outs if "NONE" in o.upper())
            g["none_n"] = len(none_tasks)

        n = max(1, g["n"])
        results[name] = {
            "n": g["n"],
            "set_precision": round(g["prec_sum"] / n, 4),
            "set_recall": round(g["rec_sum"] / n, 4),
            "set_f1": round(g["f1_sum"] / n, 4),
            "exact_set_match": round(g["exact"] / n, 4),
            "none_recall": round(g["none_ok"] / max(1, g["none_n"]), 4),
            "none_n": g["none_n"],
            "slices": {k: {"n": v["n"], "f1": round(v["f1_sum"] / max(1, v["n"]), 4),
                            "exact": round(v["exact"] / max(1, v["n"]), 4)}
                        for k, v in slices.items()},
        }
        r = results[name]
        print(f"  setF1={r['set_f1']} exact={r['exact_set_match']} "
              f"P={r['set_precision']} R={r['set_recall']} "
              f"NONE-R={r['none_recall']} slices={r['slices']}", flush=True)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs("results/phase6b", exist_ok=True)
    with open("results/phase6b/resolver_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nRESOLVER-EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()
