"""Phase 5C follow-up: internal test WITH capability schema in prompt.

The main eval used bare SYSTEM_PROMPT for all models — a train/eval mismatch
for schema-conditioned models (E5C-S, E5C-SD). This script reruns the internal
test using build_capability_prompt(...) exactly as in training, to answer:
  does the internal OpSeq collapse recover when schema is present at inference?
"""
import json, sys, os
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.capability_format import build_capability_prompt
from src.ir.parser import parse_text
from src.validator.validator import validate
from collections import Counter

MODELS = {
    "E1-A":   "runs/phase5b1/e1_3b_qlora/final",
    "E5C-S":  "runs/phase5c/e5c_s/final",
    "E5C-SD": "runs/phase5c/e5c_sd/final",
}
BATCH = 16

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

internal = []
with open("data/compiler_corpus_v3_1/test.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            internal.append({
                "instruction": r["instruction"],
                "capabilities": r.get("capabilities") or [],
                "plan_target": r.get("plan_target", ""),
            })
n_schema = sum(1 for s in internal if s["capabilities"])
print(f"Internal test: {len(internal)} | with schema: {n_schema} | without: {len(internal)-n_schema}", flush=True)


def run_inference(model, prompts):
    inputs = tok(prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
    return [tok.decode(outputs[j][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True) for j in range(len(prompts))]


def evaluate(model, label):
    texts = []
    for s in internal:
        user = build_capability_prompt(s["instruction"], s["capabilities"]) if s["capabilities"] else s["instruction"]
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        texts.append(prompt)

    completions = []
    for bs in range(0, len(texts), BATCH):
        if bs % 320 == 0:
            print(f"    {bs}/{len(texts)}", flush=True)
        completions.extend(run_inference(model, texts[bs:bs+BATCH]))

    stats = {k: Counter() for k in ("all", "schema", "noschema")}
    per_sample = []
    for comp, s in zip(completions, internal):
        bucket = "schema" if s["capabilities"] else "noschema"
        entry = {"bucket": bucket, "parse": 0, "valid": 0, "opseq": 0}
        try:
            module = parse_text(extract_taskir_text(comp))
            entry["parse"] = 1
            if validate(module).valid:
                entry["valid"] = 1
                pred_ops = [n.op for n in module.program.nodes if n.op not in POLICY_OPS]
                try:
                    ref_mod = parse_text(s["plan_target"])
                    ref_ops = [n.op for n in ref_mod.program.nodes if n.op not in POLICY_OPS]
                    if pred_ops == ref_ops:
                        entry["opseq"] = 1
                except Exception:
                    pass
        except Exception:
            pass
        for k in ("all", bucket):
            for m in ("parse", "valid", "opseq"):
                stats[k][m] += entry[m]
        stats[k]["n"] += 1
        per_sample.append(entry)

    print(f"\n  [{label}] internal (schema-prompted):", flush=True)
    out = {}
    for k in ("all", "schema", "noschema"):
        n = stats[k]["n"] or 1
        out[k] = {"n": stats[k]["n"],
                  "parse_pct": round(100*stats[k]["parse"]/n, 2),
                  "valid_pct": round(100*stats[k]["valid"]/n, 2),
                  "opseq_pct": round(100*stats[k]["opseq"]/n, 2)}
        print(f"    {k:9s} n={stats[k]['n']:5d} parse={out[k]['parse_pct']:6.2f}% "
              f"valid={out[k]['valid_pct']:6.2f}% opseq={out[k]['opseq_pct']:6.2f}%", flush=True)
    return out


base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")

results = {}
for name, adapter in MODELS.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    results[name] = evaluate(model, name)
    del model
    torch.cuda.empty_cache()

os.makedirs("results/phase5c", exist_ok=True)
with open("results/phase5c/internal_schema_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/phase5c/internal_schema_eval.json", flush=True)
print("DONE", flush=True)
