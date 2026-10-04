"""Probe C + NO_CALL on the G1+G2 grounder (completes preregistered Task 9;
the internal run had bound Probe C to the G1-only model by first_run choice)."""
import json, os, sys, random
ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.skill_grounder import build_probe_prompt, parse_probe_c, parse_label, pick_hard_negatives

DATA = "data/phase6a_grounding"
test_ood = [json.loads(l) for l in open(f"{DATA}/test_ood.jsonl", encoding="utf-8") if l.strip()]
train_pool = [json.loads(l) for l in open(f"{DATA}/train.jsonl", encoding="utf-8") if l.strip()]

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

out = {}
import gc
for seed in (42, 43, 44):
    adapter = f"runs/phase6/grounder_g1g2_s{seed}/final"
    if not os.path.exists(adapter):
        continue
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    rng = random.Random(7)
    sub = rng.sample(test_ood, min(500, len(test_ood)))
    prompts, metas = [], []
    for i, s in enumerate(sub):
        cands = pick_hard_negatives(s, train_pool, k=4, rng=random.Random(i))
        pos = random.Random(i * 7).randrange(5)
        cands = cands[:pos] + [s] + cands[pos:]
        p = build_probe_prompt(s, "C", candidates=cands)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"]}],
            tokenize=False, add_generation_prompt=True))
        metas.append((s, pos))
    outs = []
    for bs in range(0, len(prompts), 32):
        inputs = tok(prompts[bs:bs+32], return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                               temperature=None, pad_token_id=tok.pad_token_id)
        outs.extend(tok.decode(x[inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True) for x in o)
    cap_ok = skill_ok = joint = 0
    for (s, pos), o in zip(metas, outs):
        cid, lab = parse_probe_c(o)
        cap_ok += cid == pos + 1
        skill_ok += lab == s["target_skill"]
        joint += (cid == pos + 1 and lab == s["target_skill"])
    # NO_CALL: same samples, correct capability removed
    nc_prompts = []
    for i, s in enumerate(sub):
        cands = pick_hard_negatives(s, train_pool, k=4, rng=random.Random(i))
        p = build_probe_prompt(s, "C", candidates=cands)
        nc_prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"]}],
            tokenize=False, add_generation_prompt=True))
    ncouts = []
    for bs in range(0, len(nc_prompts), 32):
        inputs = tok(nc_prompts[bs:bs+32], return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                               temperature=None, pad_token_id=tok.pad_token_id)
        ncouts.extend(tok.decode(x[inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True) for x in o)
    nc = sum(1 for o in ncouts if parse_label(o) == "NO_CALL")
    out[f"s{seed}"] = {"n": len(sub),
                       "capability_selection_acc": round(cap_ok/len(sub), 4),
                       "skill_acc": round(skill_ok/len(sub), 4),
                       "joint_exact": round(joint/len(sub), 4),
                       "no_call_rate": round(nc/len(sub), 4)}
    print(f"s{seed}: {out[f's{seed}']}", flush=True)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

with open("results/phase6/phase6a_probe_c_g1g2.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE")
