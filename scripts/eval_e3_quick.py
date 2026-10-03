"""Quick E3 (7B QLoRA) sanity eval: parse/valid/opseq on corpus val split (1,561).

Mirrors the Phase 5C internal protocol so E3 can be reported in the same table
as E0/E1/E2 (supplementary, Phase 6 era — does NOT modify paper_snapshot_v1).
"""
import json, os, sys
ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.ir.parser import parse_text
from src.validator.validator import validate

ADAPTER = "runs/phase6/e3_7b_qlora/final"
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}

rows = [json.loads(l) for l in open("data/compiler_corpus_v3_1/val.jsonl", encoding="utf-8") if l.strip()]
print(f"val samples: {len(rows)}", flush=True)

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=__import__("transformers").BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

texts = [tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": r["instruction"]}],
    tokenize=False, add_generation_prompt=True) for r in rows]

comps = []
for bs in range(0, len(texts), 16):
    if bs % 320 == 0:
        print(f"  {bs}/{len(texts)}", flush=True)
    batch = texts[bs:bs+16]
    inputs = tok(batch, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             temperature=None, pad_token_id=tok.pad_token_id)
    comps.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True) for o in out)

parse_ok = valid = opseq = 0
for comp, r in zip(comps, rows):
    try:
        module = parse_text(extract_taskir_text(comp))
        parse_ok += 1
        if validate(module).valid:
            valid += 1
            pred_ops = [n.op for n in module.program.nodes if n.op not in POLICY_OPS]
            try:
                ref = parse_text(r.get("plan_target", ""))
                ref_ops = [n.op for n in ref.program.nodes if n.op not in POLICY_OPS]
                if pred_ops == ref_ops:
                    opseq += 1
            except Exception:
                pass
    except Exception:
        pass

n = len(rows)
res = {"model": "E3_7B_QLoRA", "split": "val(n=1561)",
       "parse_pct": round(100*parse_ok/n, 2),
       "valid_pct": round(100*valid/n, 2),
       "opseq_pct": round(100*opseq/n, 2),
       "note": "Phase 6-era supplementary; NOT part of paper_snapshot_v1"}
os.makedirs("results/phase6", exist_ok=True)
with open("results/phase6/e3_quick_eval.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res), flush=True)
print("E3-EVAL-DONE", flush=True)
