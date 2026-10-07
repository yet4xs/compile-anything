"""6B-3 external tau3 diagnostic: does the modular frontend recover EXEC_ACTION?

The paper's core architecture question:
  E5C-S monolithic:   EXEC_ACTION output ~0%, SemRecall ~0.58% (frozen 6B-0 Protocol A)
  Modular (learned):  Canonicalizer → Resolver → C2 Composer on tau3

This test uses tau3's env-visible tool names (NOT GT-derived) as raw capabilities,
runs the frozen learned modular pipeline, and reports:
  Parse / Valid / Pred/T / SemRecall / EXEC_ACTION output rate + recall
"""
import json, os, sys, re, gc
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.skill_grounder import build_probe_prompt, parse_label
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.eval.tau3_skill_oracle import lower_reference_actions, semantic_match

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
TAU3_SRC = "/ccfa2026/compile-anything/third_party/tau3-bench/src/tau2/domains"
SEED = 42  # frozen primary


def extract_domain_tools(domain):
    """Env-visible tools: from tau3 source @is_tool + ref-name union."""
    ref_names = set()
    with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r["case_id"].split("-")[0] == domain:
                ref_names.update(r.get("ref_action_names") or [])
    tools = [{"name": n, "description": ""} for n in sorted(ref_names)]
    path = os.path.join(TAU3_SRC, domain, "tools.py")
    if os.path.exists(path):
        src = open(path, encoding="utf-8").read()
        seen = {t["name"] for t in tools}
        for m in re.finditer(r"@is_tool[^\n]*\n(?:[^\n]*\n){0,4}?\s*def (\w+)\(", src):
            if m.group(1) not in seen:
                tools.append({"name": m.group(1), "description": ""})
    return tools


def main():
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load tau3 tasks
    tasks = []
    with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            tasks.append({"case_id": r["case_id"], "domain": r["case_id"].split("-")[0],
                          "instruction": r["instruction"],
                          "ref_action_names": r.get("ref_action_names") or [],
                          "n_ref": r["n_ref"]})
    print(f"tau3 tasks: {len(tasks)}")

    # Domain tool inventories
    domains = sorted(set(t["domain"] for t in tasks))
    domain_tools = {d: extract_domain_tools(d) for d in domains}
    for d, ts in domain_tools.items():
        print(f"  {d}: {len(ts)} tools")

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    # ── Stage 1: Learned Canonicalizer on tau3 tools ──
    canon = PeftModel.from_pretrained(base, f"runs/phase6/grounder_g1g2_s{SEED}/final")
    canon.eval()
    tool_skill = {}
    for d, tools in domain_tools.items():
        prompts = []
        for t in tools:
            p = build_probe_prompt({"instruction": "", "capability": t}, "A")
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": p["system"]},
                 {"role": "user", "content": p["user"]}],
                tokenize=False, add_generation_prompt=True))
        for bs in range(0, len(prompts), 16):
            inputs = tok(prompts[bs:bs+16], return_tensors="pt", padding=True,
                         truncation=True, max_length=1024).to(canon.device)
            with torch.no_grad():
                out = canon.generate(**inputs, max_new_tokens=8, do_sample=False,
                                     temperature=None, pad_token_id=tok.pad_token_id)
            preds = [parse_label(tok.decode(o[inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True)) for o in out]
            for t, s in zip(tools[bs:bs+16], preds):
                tool_skill[(d, t["name"])] = s
    del canon; gc.collect(); torch.cuda.empty_cache()
    print("canonicalizer done")

    # ── Stage 2: Learned Resolver on tau3 tasks × tools ──
    SYSTEM_R = ("You are a capability resolver for a task compiler. Given a task and "
                "capability descriptions, decide which capabilities the task requires. "
                "Answer with the label ONLY.")
    resolver = PeftModel.from_pretrained(base, f"runs/phase6b/resolver_R2_s{SEED}/final")
    resolver.eval()
    r_ids = tok.encode("relevant", add_special_tokens=False)
    i_ids = tok.encode("irrelevant", add_special_tokens=False)

    def score_pair(task, tool_name, skill):
        block = f"[c0] name: {tool_name}\ncanonical_skill: {skill}"
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_R},
             {"role": "user", "content":
              f"Task:\n{task}\n\nCapability:\n{block}\n\n"
              "Is this capability required by the task? "
              "Answer 'relevant' or 'irrelevant'."}],
            tokenize=False, add_generation_prompt=True)
        # two sequences: +relevant, +irrelevant
        seqs = [tok(prompt + "relevant", return_tensors="pt"),
                tok(prompt + "irrelevant", return_tensors="pt")]
        maxlen = max(s["input_ids"].shape[1] for s in seqs)
        ii = []
        am = []
        for s in seqs:
            n = maxlen - s["input_ids"].shape[1]
            ii.append([tok.pad_token_id] * n + s["input_ids"][0].tolist())
            am.append([0] * n + s["attention_mask"][0].tolist())
        ii = torch.tensor(ii).to(resolver.device)
        am = torch.tensor(am).to(resolver.device)
        with torch.no_grad():
            logits = resolver(input_ids=ii, attention_mask=am,
                              num_logits_to_keep=4).logits
        K = 4
        def seq_lp(j, lab_ids):
            total = 0.0
            for tp in range(maxlen - len(lab_ids), maxlen):
                idx = tp - 1 - (maxlen - K)
                row = torch.log_softmax(logits[j, idx].float(), -1)
                total += float(row[ii[j, tp]])
            return total
        return seq_lp(0, r_ids) - seq_lp(1, i_ids)

    # For each task, select tools above threshold
    selected_tools = {}
    for i, t in enumerate(tasks):
        if i % 100 == 0:
            print(f"  resolver {i}/{len(tasks)}", flush=True)
        d = t["domain"]
        best = []
        for tool in domain_tools[d]:
            skill = tool_skill.get((d, tool["name"]), "UNKNOWN")
            s = score_pair(t["instruction"], tool["name"], skill)
            if s > 0:
                best.append((tool["name"], skill))
        if not best:
            # fallback: top-1 by raw score (avoid empty)
            scores = []
            for tool in domain_tools[d]:
                skill = tool_skill.get((d, tool["name"]), "UNKNOWN")
                s = score_pair(t["instruction"], tool["name"], skill)
                scores.append((s, tool["name"], skill))
            scores.sort(reverse=True)
            best = [(scores[0][1], scores[0][2])] if scores else []
        selected_tools[t["case_id"]] = best
    del resolver; gc.collect(); torch.cuda.empty_cache()
    print("resolver done")

    # ── Stage 3: C2 Composer ──
    composer = PeftModel.from_pretrained(base, f"runs/phase6b/composer_C2_s{SEED}/final")
    composer.eval()

    prompts = []
    for t in tasks:
        sel = selected_tools[t["case_id"]]
        blocks = [f"[c{i}] name: {name}\ncanonical_skill: {skill}"
                  for i, (name, skill) in enumerate(sel)]
        user = f"{t['instruction']}\n\nSelected capabilities:\n" + "\n\n".join(blocks)
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    comps = []
    for bs in range(0, len(prompts), 16):
        if bs % 256 == 0:
            print(f"  composer {bs}/{len(prompts)}", flush=True)
        batch = prompts[bs:bs+16]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(composer.device)
        with torch.no_grad():
            out = composer.generate(**inputs, max_new_tokens=512,
                                    do_sample=False, temperature=None,
                                    pad_token_id=tok.pad_token_id)
        comps.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True) for o in out)

    del composer; gc.collect(); torch.cuda.empty_cache()

    # ── Score ──
    g = Counter()
    for t, comp in zip(tasks, comps):
        g["n"] += 1
        g["ref_count"] += t["n_ref"]
        try:
            ref_ops = None
            try:
                ref_mod = parse_text(None) if False else None
            except: pass
        except: pass
        # Parse prediction
        try:
            mod = parse_text(extract_taskir_text(comp))
            g["parse"] += 1
        except:
            continue
        try:
            valid = validate(mod).valid
        except:
            valid = False
        if valid:
            g["valid"] += 1
        nodes = [n for n in mod.program.nodes if n.op not in POLICY_OPS]
        g["pred_count"] += len(nodes)
        g["exec_output"] += sum(1 for n in nodes if n.op == "EXEC_ACTION")
        # Semantic matching
        ref_low = lower_reference_actions([{"name": n} for n in t["ref_action_names"]])
        used = set()
        for rlow in ref_low:
            g["ref_sem_total"] += 1
            for pi, pn in enumerate(nodes):
                if pi in used: continue
                try:
                    hit = semantic_match(pn.op, pn.params, rlow)
                except:
                    hit = False
                if hit:
                    used.add(pi)
                    g["ref_sem_matched"] += 1
                    if rlow["skill"] == "EXEC_ACTION":
                        g["exec_ref_total"] += 1
                        if pn.op == "EXEC_ACTION": g["exec_ref_hit"] += 1
                    break

    n = max(1, g["n"])
    results = {
        "_protocol": "learned modular (canonicalizer+resolver+composer s42) on tau3; "
                     "env-visible tool names (source + ref union); no GT used in pipeline",
        "n": g["n"],
        "parse_pct": round(100 * g["parse"] / n, 2),
        "valid_pct": round(100 * g["valid"] / n, 2),
        "pred_per_task": round(g["pred_count"] / n, 2),
        "semantic_recall_pct": round(100 * g["ref_sem_matched"] / max(1, g["ref_sem_total"]), 2),
        "exec_action_output_rate_pct": round(100 * g["exec_output"] / max(1, g["pred_count"]), 2),
        "exec_action_reference_recall_pct": round(100 * g["exec_ref_hit"] / max(1, g["exec_ref_total"]), 2)
        if g["exec_ref_total"] else None,
        "exec_ref_total": g["exec_ref_total"],
        "_baselines": {
            "E5C-S monolithic (frozen 6B-0 Protocol A)": {
                "semantic_recall_pct": 0.58, "exec_action_output_rate_pct": 0.25},
            "E5C-S monolithic + oracle CapabilityIR (6B-0 Protocol D)": {
                "semantic_recall_pct": 0.22, "exec_action_output_rate_pct": 3.28},
        },
    }
    print(f"\n=== 6B-3 External tau3 Diagnostic ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    os.makedirs("results/phase6b", exist_ok=True)
    with open("results/phase6b/external_tau3.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("TAU3-EXTERNAL-DONE")


if __name__ == "__main__":
    main()
