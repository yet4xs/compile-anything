"""Phase 6B-2 evaluator v2 — implements reviewer Fixes 1/4/5/9.

Fix 1  R1 parser case-corrected + first-20 raw dump
Fix 4  R2 real scores: score = logP("relevant") - logP("irrelevant") over the
       full label sequence (teacher-forced); select score>0 (no dev tuning);
       real top-1/top-3; greedy-label sanity check secondary
Fix 5  NO_CALL as true P/R/F1: pairwise-empty refusal (primary mechanism) +
       explicit full-table NONE generation (diagnostic, training-format prompt)
Fix 9  unseen-family slice with n reported + caveat
Primary comparison: matched common candidate set (15, typed) for R1 AND R2.
Secondary: R2 full-table deployment diagnostic.
"""
import json, os, sys, re, gc
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

D = "data/phase6b_resolver"
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


def pair_prompt(tok, task, block):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_R},
         {"role": "user", "content":
          f"Task:\n{task}\n\nCapability:\n{block}\n\n"
          "Is this capability required by the task? "
          "Answer 'relevant' or 'irrelevant'."}],
        tokenize=False, add_generation_prompt=True)


def table_prompt(tok, task, blocks):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_R},
         {"role": "user", "content":
          f"Task:\n{task}\n\nAvailable capabilities:\n" + "\n\n".join(blocks) +
          "\n\nWhich capabilities does the task require? "
          "Answer as a comma-separated id list (e.g. c3,c7) or NONE."}],
        tokenize=False, add_generation_prompt=True)


def label_scores(tok, model, prompts, batch=8):
    """score = logP('relevant'|p) - logP('irrelevant'|p) as FULL sequence
    likelihoods (irrelevant = 'ir'+'relevant' for this tokenizer)."""
    r_ids = tok.encode("relevant", add_special_tokens=False)
    i_ids = tok.encode("irrelevant", add_special_tokens=False)
    scores = []
    for bs in range(0, len(prompts), batch):
        chunk = prompts[bs:bs+batch]
        # build two sequences per prompt: prompt+relevant, prompt+irrelevant
        seqs, kinds = [], []
        for p in chunk:
            seqs.append(tok(p + "relevant", return_tensors="pt"))
            seqs.append(tok(p + "irrelevant", return_tensors="pt"))
            kinds.extend(["r", "i"])
        maxlen = max(s["input_ids"].shape[1] for s in seqs)
        pad = tok.pad_token_id
        ii, am = [], []
        for s in seqs:
            n = maxlen - s["input_ids"].shape[1]
            ii.append([pad] * n + s["input_ids"][0].tolist())
            am.append([0] * n + s["attention_mask"][0].tolist())
        ii = torch.tensor(ii).to(model.device)
        am = torch.tensor(am).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ii, attention_mask=am).logits
        # log-softmax ONLY at the <=2 label positions per sequence (slicing
        # avoids materializing the full [B,L,V] float tensor — the earlier
        # version OOM'd on exactly that)
        seq_lp = []
        for j in range(len(seqs)):
            lab_len = len(r_ids) if kinds[j] == "r" else len(i_ids)
            total = 0.0
            for t_pos in range(maxlen - lab_len, maxlen):
                tok_id = ii[j, t_pos]
                row_lp = torch.log_softmax(logits[j, t_pos - 1].float(), -1)
                total += float(row_lp[tok_id])
            seq_lp.append(total)
        for j in range(0, len(seqs), 2):
            scores.append(seq_lp[j] - seq_lp[j + 1])  # relevant - irrelevant
        if bs % 1600 == 0:
            print(f"    seqscore {bs}/{len(prompts)}", flush=True)
    return scores

def gen(tok, model, prompts, max_new=24, batch=32):
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

    tasks = [json.loads(l) for l in open(f"{D}/common_eval_tasks.jsonl",
                                        encoding="utf-8") if l.strip()]
    none_tasks = [json.loads(l) for l in open(f"{D}/none_tasks_common.jsonl",
                                              encoding="utf-8") if l.strip()]
    clean_none = [t for t in none_tasks if not t["potential_collision"]]
    train_fams = set()
    for line in open("data/compiler_corpus_v4/train.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            for c in r["canonical_capabilities"]:
                train_fams.add("_".join(c["name"].split("_")[:2]))

    results = {"_protocol": "primary = matched common 15-candidate set, typed; "
                            "secondary = R2 full table; NONE = synthetic refusal diagnostic"}
    SPECS = [("R1_s42", "runs/phase6b/resolver_R1_s42/final", "R1"),
             ("R2_s42", "runs/phase6b/resolver_R2_s42/final", "R2"),
             ("R2_s43", "runs/phase6b/resolver_R2_s43/final", "R2"),
             ("R2_s44", "runs/phase6b/resolver_R2_s44/final", "R2")]

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

        g = Counter()
        slices = defaultdict(lambda: Counter())
        neg_stats = defaultdict(lambda: Counter())
        raw_dump = []

        if mode == "R2":
            # score every candidate of the common set
            all_prompts, owners = [], []
            for ti, t in enumerate(tasks):
                for c in t["candidates"]:
                    all_prompts.append(pair_prompt(tok, t["task"], c["block"]))
                    owners.append(ti)
            scores = label_scores(tok, model, all_prompts)
            task_scores = defaultdict(dict)
            idx = 0
            for ti, t in enumerate(tasks):
                for c in t["candidates"]:
                    task_scores[ti][c["capability_id"]] = scores[idx]
                    idx += 1

            for ti, t in enumerate(tasks):
                sc = task_scores[ti]
                gold = set(t["gold_ids"])
                pred = {cid for cid, s in sc.items() if s > 0}
                if not pred:  # pairwise-empty refusal -> NONE
                    pred = set()
                    g["pred_none"] += 1
                    if gold:
                        g["false_none"] += 1
                prec, rec, f1, exact = set_metrics(gold, pred)
                g["n"] += 1
                g["prec_sum"] += prec
                g["rec_sum"] += rec
                g["f1_sum"] += f1
                g["exact"] += exact
                ranked = sorted(sc.items(), key=lambda x: -x[1])
                g["top1"] += bool(ranked) and ranked[0][0] in gold
                top3 = {cid for cid, _ in ranked[:3]}
                g["top3"] += bool(gold & top3)
                # slices
                for key in (("multi" if len(gold) > 1 else "single"), fam_class(t["gold_skills"])):
                    slices[key]["n"] += 1
                    slices[key]["f1_sum"] += f1
                    slices[key]["exact"] += exact
                fams = {re.search(r"name: (\S+)", c["block"].split("\n")[0]).group(1)
                        for c in t["candidates"] if c["is_gold"]}
                fams = {"_".join(f.split("_")[:2]) for f in fams}
                uk = "unseen_family" if fams and not (fams & train_fams) else "seen_family"
                slices[uk]["n"] += 1
                slices[uk]["f1_sum"] += f1
                # negative-type stats
                for c in t["candidates"]:
                    if not c["is_gold"]:
                        nt = c["negative_type"]
                        neg_stats[nt]["n"] += 1
                        neg_stats[nt]["score_sum"] += sc.get(c["capability_id"], 0)
                        if sc.get(c["capability_id"], 0) > 0:
                            neg_stats[nt]["fp"] += 1
            # NONE tasks: pairwise-empty
            for t in clean_none:
                prompts_n = [pair_prompt(tok, t["task"], c["block"])
                             for c in t["candidates"]]
                sc_n = label_scores(tok, model, prompts_n)
                if not any(s > 0 for s in sc_n):
                    g["none_ok"] += 1
            g["none_n"] = len(clean_none)
            # explicit NONE generation diagnostic (training-format prompt)
            prompts_e = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_R},
                 {"role": "user", "content":
                  f"Task:\n{t['task']}\n\nAvailable capabilities:\n" +
                  "\n\n".join(c["block"] for c in t["candidates"][:8]) +
                  "\n\nWhich capabilities does the task require? "
                  "Answer as a comma-separated id list (e.g. c3,c7) or NONE."}],
                tokenize=False, add_generation_prompt=True) for t in clean_none]
            outs_e = gen(tok, model, prompts_e)
            g["explicit_none_ok"] = sum(1 for o in outs_e if "NONE" in o.upper())
        else:
            prompts = [table_prompt(tok, t["task"],
                                    [c["block"] for c in t["candidates"]])
                       for t in tasks]
            outs = gen(tok, model, prompts)
            for t, o in list(zip(tasks, outs))[:20]:
                raw_dump.append({"gold": t["gold_ids"], "raw": o[:100]})
            for t, o in zip(tasks, outs):
                ol = o.lower()
                pred = set() if "none" in ol else set(re.findall(r"\bc\d+\b", ol))
                gold = set(t["gold_ids"])
                if not pred and gold:
                    g["false_none"] += 1
                prec, rec, f1, exact = set_metrics(gold, pred)
                g["n"] += 1
                g["prec_sum"] += prec
                g["rec_sum"] += rec
                g["f1_sum"] += f1
                g["exact"] += exact
                ordered = re.findall(r"[c][0-9]+", ol)
                ranked = list(dict.fromkeys(ordered))
                g["top1"] += bool(ranked) and ranked[0] in gold
                top3 = set(ranked[:3])
                g["top3"] += bool(gold & top3)
                for key in (("multi" if len(gold) > 1 else "single"), fam_class(t["gold_skills"])):
                    slices[key]["n"] += 1
                    slices[key]["f1_sum"] += f1
                    slices[key]["exact"] += exact
                fams = {re.search(r"name: (\S+)", c["block"].split("\n")[0]).group(1)
                        for c in t["candidates"] if c["is_gold"]}
                fams = {"_".join(f.split("_")[:2]) for f in fams}
                uk = "unseen_family" if fams and not (fams & train_fams) else "seen_family"
                slices[uk]["n"] += 1
                slices[uk]["f1_sum"] += f1
            prompts_n = [table_prompt(tok, t["task"],
                                      [c["block"] for c in t["candidates"]])
                         for t in clean_none]
            outs_n = gen(tok, model, prompts_n)
            g["none_ok"] = sum(1 for o in outs_n if "none" in o.lower()
                               and not re.findall(r"\bc\d+\b", o.lower()))
            g["none_n"] = len(clean_none)
            g["explicit_none_ok"] = g["none_ok"]

        n = max(1, g["n"])
        none_p = g["none_ok"] / max(1, g["none_ok"] + g["false_none"])
        none_r = g["none_ok"] / max(1, g["none_n"])
        results[name] = {
            "n": g["n"],
            "set_precision": round(g["prec_sum"] / n, 4),
            "set_recall": round(g["rec_sum"] / n, 4),
            "set_f1": round(g["f1_sum"] / n, 4),
            "exact_set": round(g["exact"] / n, 4),
            "top1": round(g["top1"] / n, 4),
            "top3": round(g["top3"] / n, 4),
            "none_precision": round(none_p, 4),
            "none_recall": round(none_r, 4),
            "none_f1": round(2 * none_p * none_r / max(1e-9, none_p + none_r), 4),
            "false_refusal_rate": round(g["false_none"] / n, 4),
            "explicit_none_accuracy": round(g["explicit_none_ok"] / max(1, g["none_n"]), 4),
            "clean_none_n": g["none_n"],
            "negative_type_stats": {
                nt: {"n": v["n"],
                     "FPR": round(v["fp"] / max(1, v["n"]), 4),
                     "mean_score": round(v["score_sum"] / max(1, v["n"]), 3)}
                for nt, v in neg_stats.items()},
            "slices": {k: {"n": v["n"], "f1": round(v["f1_sum"] / max(1, v["n"]), 4),
                            "exact": round(v["exact"] / max(1, v["n"]), 4)}
                        for k, v in slices.items()},
        }
        r = results[name]
        print(f"  setF1={r['set_f1']} exact={r['exact_set']} P={r['set_precision']} "
              f"R={r['set_recall']} top1={r['top1']} top3={r['top3']} "
              f"noneF1={r['none_f1']} explicitNONE={r['explicit_none_accuracy']}", flush=True)
        print(f"  negtypes: {json.dumps(r['negative_type_stats'])}", flush=True)
        if raw_dump:
            print(f"  R1 raw samples: {json.dumps(raw_dump[:5], ensure_ascii=False)}",
                  flush=True)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    # R2 mean±std
    r2s = [v for k, v in results.items() if k.startswith("R2_")]
    if r2s:
        import statistics as st
        agg = {}
        for f in ("set_f1", "exact_set", "none_f1", "top1"):
            xs = [v[f] for v in r2s]
            agg[f] = f"{st.mean(xs):.4f}±{st.pstdev(xs):.4f}"
        results["R2_mean_std"] = agg
        print(f"\nR2 mean±std: {agg}", flush=True)

    os.makedirs("results/phase6b", exist_ok=True)
    with open("results/phase6b/resolver_eval_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nRESOLVER-EVALV2-DONE", flush=True)


if __name__ == "__main__":
    main()
