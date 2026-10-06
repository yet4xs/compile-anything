"""Phase 6B-2.1: refusal gate freeze — BOTH sides, no training, no prompt change.

Task 1: explicit full-table NONE-gate prompt (training format) on
        994 positive matched tasks (accept expected) + 101 clean synthetic
        NONE tasks (refuse expected), for frozen R2 s42/s43/s44.
        NONE precision/recall/F1, accept rates, false refusal/call.
Task 2: tokenizer assertion — 'relevant'/'irrelevant' single-token check;
        records whether the v2 next-token score is the complete-label score.
"""
import json, os, sys, re, gc
from collections import Counter

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


def table_prompt(tok, task, blocks):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_R},
         {"role": "user", "content":
          f"Task:\n{task}\n\nAvailable capabilities:\n" + "\n\n".join(blocks) +
          "\n\nWhich capabilities does the task require? "
          "Answer as a comma-separated id list (e.g. c3,c7) or NONE."}],
        tokenize=False, add_generation_prompt=True)


def gen(tok, model, prompts, batch=32):
    outs = []
    for bs in range(0, len(prompts), batch):
        inputs = tok(prompts[bs:bs+batch], return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=24, do_sample=False,
                               temperature=None, pad_token_id=tok.pad_token_id)
        outs.extend(tok.decode(x[inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True) for x in o)
    return outs


def main():
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Task 2: tokenizer assertion
    r_ids = tok.encode("relevant", add_special_tokens=False)
    i_ids = tok.encode("irrelevant", add_special_tokens=False)
    single = len(r_ids) == 1 and len(i_ids) == 1
    print(f"tokenizer: relevant={r_ids} irrelevant={i_ids} single-token={single}")
    assertion = {
        "relevant_ids": r_ids, "irrelevant_ids": i_ids, "single_token": single,
        "note": ("current next-token score IS the complete-label score for this "
                 "tokenizer" if single else
                 "label is multi-token — v2 scores were first-token only; "
                 "sequence-likelihood rescoring required"),
    }

    pos = [json.loads(l) for l in open(f"{D}/common_eval_tasks.jsonl",
                                       encoding="utf-8") if l.strip()]
    nones = [json.loads(l) for l in open(f"{D}/none_tasks_common.jsonl",
                                         encoding="utf-8") if l.strip()]
    clean_none = [t for t in nones if not t["potential_collision"]]
    print(f"positive={len(pos)} clean_NONE={len(clean_none)}")

    results = {"_tokenizer_assertion": assertion}
    for seed in (42, 43, 44):
        adapter = f"runs/phase6b/resolver_R2_s{seed}/final"
        if not os.path.exists(adapter):
            continue
        base = AutoModelForCausalLM.from_pretrained(
            "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()

        def blocks_of(t):
            return [c["block"] for c in t["candidates"][:15]]

        prompts_p = [table_prompt(tok, t["task"], blocks_of(t)) for t in pos]
        outs_p = gen(tok, model, prompts_p)
        # positive side: NONE output = false refusal
        false_refusal = 0
        for o in outs_p:
            ol = o.lower()
            if "none" in ol and not re.findall(r"\bc\d+\b", ol):
                false_refusal += 1
        prompts_n = [table_prompt(tok, t["task"], blocks_of(t)) for t in clean_none]
        outs_n = gen(tok, model, prompts_n)
        correct_refusal = sum(1 for o in outs_n
                              if "none" in o.lower()
                              and not re.findall(r"\bc\d+\b", o.lower()))
        tp = correct_refusal                      # refused when should
        fp = len(pos) - false_refusal - (len(pos) - false_refusal)  # placeholder
        # confusion: NONE-predicted on positives = false refusal (FP for NONE)
        fp = false_refusal
        fn = len(clean_none) - correct_refusal
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        results[f"R2_s{seed}"] = {
            "positive_n": len(pos),
            "clean_none_n": len(clean_none),
            "NONE_precision": round(prec, 4),
            "NONE_recall": round(rec, 4),
            "NONE_f1": round(f1, 4),
            "positive_accept_rate": round(1 - false_refusal / len(pos), 4),
            "false_refusal_rate": round(false_refusal / len(pos), 4),
            "negative_refusal_rate": round(correct_refusal / len(clean_none), 4),
            "false_call_rate": round(fn / len(clean_none), 4),
        }
        print(f"s{seed}: {json.dumps(results[f'R2_s{seed}'])}")
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs("results/phase6b", exist_ok=True)
    with open("results/phase6b/refusal_gate_freeze.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("GATE-FREEZE-DONE")


if __name__ == "__main__":
    main()
