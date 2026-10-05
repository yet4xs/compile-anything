"""6B-1 diagnostic: eyeball C2 predictions + fair C0-on-filtered-subset metrics."""
import json, os, sys, gc
ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.ir.parser import parse_text
from src.validator.validator import validate
from collections import Counter

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
    if not sel and r["available_capabilities"]:
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
    return task


tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# the C2-filtered subset
sub = [(r, build_user("C2", r)) for r in test]
sub = [(r, u) for r, u in sub if u is not None]
print(f"C2-filtered subset: {len(sub)} / {len(test)}", flush=True)

# ── fair C0 metrics on the SAME subset ──
base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")
c0 = PeftModel.from_pretrained(base, "runs/phase5b1/e1_3b_qlora/final")
c0.eval()
prompts = [tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["instruction"]}],
    tokenize=False, add_generation_prompt=True) for r, _ in sub]
g = Counter()
comps0 = []
for bs in range(0, len(prompts), 16):
    batch = prompts[bs:bs+16]
    inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                 max_length=2048).to(c0.device)
    with torch.no_grad():
        out = c0.generate(**inputs, max_new_tokens=512, do_sample=False,
                          temperature=None, pad_token_id=tok.pad_token_id)
    comps0.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True) for o in out)
n = len(sub)
for comp, (r, _) in zip(comps0, sub):
    try:
        m = parse_text(extract_taskir_text(comp))
        g["parse"] += 1
        if validate(m).valid:
            g["valid"] += 1
            po = [x.op for x in m.program.nodes if x.op not in POLICY_OPS]
            ro = [x.op for x in parse_text(r["plan_target"]).program.nodes
                  if x.op not in POLICY_OPS]
            if po == ro:
                g["opseq"] += 1
            if "EXEC_ACTION" in ro:
                g["ea_tasks"] += 1
                g["ea_hit"] += po.count("EXEC_ACTION") / ro.count("EXEC_ACTION")
    except Exception:
        pass
print(f"\nC0 on C2-subset (n={n}): parse={100*g['parse']/n:.2f}% "
      f"valid={100*g['valid']/n:.2f}% opseq={100*g['opseq']/n:.2f}% "
      f"EA-R={100*g['ea_hit']/max(1,g['ea_tasks']):.2f}%", flush=True)
del c0
gc.collect(); torch.cuda.empty_cache()

# ── eyeball 12 C2_s42 predictions ──
c2 = PeftModel.from_pretrained(base, "runs/phase6b/composer_C2_s42/final")
c2.eval()
sample = sub[:6] + [x for x in sub if "EXEC_ACTION" in
                    {c["canonical_skill"] for c in x[0]["canonical_capabilities"]
                     if c["capability_id"] in x[0]["selected_capabilities"]}][:6]
prompts2 = [tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u}],
    tokenize=False, add_generation_prompt=True) for _, u in sample]
outs = []
for bs in range(0, len(prompts2), 6):
    inputs = tok(prompts2[bs:bs+6], return_tensors="pt", padding=True,
                 truncation=True, max_length=2048).to(c2.device)
    with torch.no_grad():
        out = c2.generate(**inputs, max_new_tokens=400, do_sample=False,
                          temperature=None, pad_token_id=tok.pad_token_id)
    outs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True) for o in out)
for (r, u), o in zip(sample, outs):
    prov = sorted({c["canonical_skill"] for c in r["canonical_capabilities"]
                   if c["capability_id"] in r["selected_capabilities"]})
    ro = [x.op for x in parse_text(r["plan_target"]).program.nodes
          if x.op not in POLICY_OPS]
    try:
        po = [x.op for x in parse_text(extract_taskir_text(o)).program.nodes
              if x.op not in POLICY_OPS]
    except Exception:
        po = ["PARSE_FAIL"]
    print(f"\n--- provided={prov} | ref_ops={ro} | pred_ops={po}", flush=True)
    print("PRED:", o[:350].replace("\n", " | "), flush=True)
print("\nDIAG-DONE", flush=True)
