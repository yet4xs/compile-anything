"""Phase 6B-0: matched-information tau3 diagnostic — information routing vs composition.

Four protocols on the SAME frozen model (E5C-S), tau3 external-development
benchmark (mechanism diagnostic only, no training):

  A  bare instruction (reuse frozen Phase 5C preds)        — capability info: none
  B  task + available concrete tool NAMES                  — names
  C  task + learned CapabilityIR (grounder-predicted skills) — learned annotations
  D  task + oracle CapabilityIR (hardened tau3 oracle)     — ORACLE UPPER-BOUND DIAGNOSTIC

Metrics per protocol: Parse, Valid, Pred/T, semantic recall, EXEC_ACTION
output rate, EXEC_ACTION reference recall.

B vs A: did capability names simply never reach the frontend?
C vs B: do explicit learned canonical annotations help the compiler?
D vs C/B: ORACLE grounded table — if EXEC_ACTION still ~0, composition is the
bottleneck; if it recovers, information routing is.
"""
import json, os, re, sys
from collections import Counter

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.capability_format import format_capabilities
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.eval.tau3_skill_oracle import (lower_reference_actions,
                                        semantic_match, tool_to_semantic_skill)

POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
TAU3_SRC = "/ccfa2026/compile-anything/third_party/tau3-bench/src/tau2/domains"
DOMAINS = {"airline": "airline", "retail": "retail",
           "telecom": "telecom", "banking_knowledge": "banking_knowledge"}


def extract_domain_tools(domain, ref_names):
    """Env-visible tool inventory: the tau3 environment exposes the domain's
    full tool list to the agent at runtime. We take (a) every tool name that
    appears in the frozen reference set for this domain (union over tasks —
    the provably-complete env tool list, not per-task GT; WHICH task uses
    WHICH tool stays out of the prompts), plus (b) @is_tool methods parsed
    from the tau3 source as a supplement."""
    path = os.path.join(TAU3_SRC, domain, "tools.py")
    tools, seen = [], set()
    for n in sorted(ref_names):
        if n and n not in seen:
            seen.add(n)
            tools.append({"name": n, "description": ""})
    if os.path.exists(path):
        src = open(path, encoding="utf-8").read()
        for m in re.finditer(r"@is_tool[^\n]*\n(?:[^\n]*\n){0,4}?\s*def (\w+)\(", src):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                tools.append({"name": m.group(1), "description": ""})
    return tools


def main():
    # ── tool inventories + coverage check ──
    # per-domain union of reference tool names = the env-visible tool list
    dom_ref_names = {d: set() for d in DOMAINS}
    with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d = r["case_id"].split("-")[0]
            if d in dom_ref_names:
                dom_ref_names[d].update(r.get("ref_action_names") or [])
    domain_tools = {d: extract_domain_tools(d, dom_ref_names[d]) for d in DOMAINS}
    for d, ts in domain_tools.items():
        print(f"{d}: {len(ts)} tools", flush=True)

    # ── load tau3 tasks (same set as Phase 5C: from frozen preds file) ──
    tasks = []
    with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                dom = r["case_id"].split("-")[0]
                tasks.append({"case_id": r["case_id"], "domain": dom,
                              "instruction": r["instruction"],
                              "ref_action_names": r.get("ref_action_names") or [],
                              "n_ref": r["n_ref"]})
    print(f"tau3 tasks: {len(tasks)}", flush=True)

    # coverage: ref tools present in inventory?
    have = {d: {t["name"] for t in ts} for d, ts in domain_tools.items()}
    cov = total = 0
    for t in tasks:
        if not t["ref_action_names"]:
            continue
        total += 1
        if t["ref_action_names"][0] in have.get(t["domain"], set()):
            cov += 1
    print(f"first-ref-tool coverage: {cov}/{total}", flush=True)

    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, "runs/phase5c/e5c_s/final")
    model.eval()

    def compile_and_score(prompts_user, label):
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for u in prompts_user]
        comps = []
        for bs in range(0, len(prompts), 16):
            if bs % 512 == 0:
                print(f"    {label} {bs}/{len(prompts)}", flush=True)
            batch = prompts[bs:bs+16]
            inputs = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512,
                                     do_sample=False, temperature=None,
                                     pad_token_id=tok.pad_token_id)
            comps.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True) for o in out)
        g = Counter()
        g["n"] = len(tasks)
        for comp, t in zip(comps, tasks):
            g["ref_count"] += t["n_ref"]
            try:
                module = parse_text(extract_taskir_text(comp))
                g["parse"] += 1
            except Exception:
                continue
            try:
                ok = validate(module).valid
            except Exception:
                ok = False
            if not ok:
                continue
            g["valid"] += 1
            nodes = [n for n in module.program.nodes if n.op not in POLICY_OPS]
            g["pred_count"] += len(nodes)
            g["exec_output"] += sum(1 for n in nodes if n.op == "EXEC_ACTION")
            ref_low = lower_reference_actions([{"name": n} for n in t["ref_action_names"]])
            used = set()
            for rlow in ref_low:
                g["ref_sem_total"] += 1
                for pi, pn in enumerate(nodes):
                    if pi in used:
                        continue
                    try:
                        hit = semantic_match(pn.op, pn.params, rlow)
                    except Exception:
                        hit = False
                    if hit:
                        used.add(pi)
                        g["ref_sem_matched"] += 1
                        if rlow["skill"] == "EXEC_ACTION":
                            g["exec_ref_total"] += 1
                            if pn.op == "EXEC_ACTION":
                                g["exec_ref_hit"] += 1
                        break
        n = g["n"]
        res = {
            "n": n,
            "parse_pct": round(100 * g["parse"] / n, 2),
            "valid_pct": round(100 * g["valid"] / n, 2),
            "pred_per_task": round(g["pred_count"] / n, 2),
            "semantic_recall_pct": round(100 * g["ref_sem_matched"] / max(1, g["ref_sem_total"]), 2),
            "exec_action_output_rate_pct": round(100 * g["exec_output"] / max(1, g["pred_count"]), 2),
            "exec_action_reference_recall_pct": round(100 * g["exec_ref_hit"] / max(1, g["exec_ref_total"]), 2)
            if g["exec_ref_total"] else None,
        }
        print(f"  [{label}] {json.dumps(res)}", flush=True)
        return res

    results = {"_protocol": "matched-information tau3 diagnostic; frozen E5C-S; "
                "D is an ORACLE CAPABILITYIR UPPER-BOUND DIAGNOSTIC, not model performance"}

    # ── Protocol A: reuse frozen Phase 5C preds ──
    print("\n=== Protocol A: bare compiler (frozen preds) ===", flush=True)
    g = Counter()
    g["n"] = len(tasks)
    with open("runs/phase5c/e5c_s_tau3_preds.jsonl") as f:
        preds = [json.loads(l) for l in f if l.strip()]
    for p in preds:
        g["ref_count"] += p["n_ref"]
        try:
            module = parse_text(p["taskir_text"])
            g["parse"] += 1
        except Exception:
            continue
        try:
            ok = validate(module).valid
        except Exception:
            ok = False
        if not ok:
            continue
        g["valid"] += 1
        nodes = [n for n in module.program.nodes if n.op not in POLICY_OPS]
        g["pred_count"] += len(nodes)
        g["exec_output"] += sum(1 for n in nodes if n.op == "EXEC_ACTION")
        ref_low = lower_reference_actions([{"name": n} for n in p.get("ref_action_names", [])])
        used = set()
        for rlow in ref_low:
            g["ref_sem_total"] += 1
            for pi, pn in enumerate(nodes):
                if pi in used:
                    continue
                try:
                    hit = semantic_match(pn.op, pn.params, rlow)
                except Exception:
                    hit = False
                if hit:
                    used.add(pi)
                    g["ref_sem_matched"] += 1
                    if rlow["skill"] == "EXEC_ACTION":
                        g["exec_ref_total"] += 1
                        if pn.op == "EXEC_ACTION":
                            g["exec_ref_hit"] += 1
                    break
    n = g["n"]
    results["A_bare"] = {
        "n": n,
        "parse_pct": round(100 * g["parse"] / n, 2),
        "valid_pct": round(100 * g["valid"] / n, 2),
        "pred_per_task": round(g["pred_count"] / n, 2),
        "semantic_recall_pct": round(100 * g["ref_sem_matched"] / max(1, g["ref_sem_total"]), 2),
        "exec_action_output_rate_pct": round(100 * g["exec_output"] / max(1, g["pred_count"]), 2),
        "exec_action_reference_recall_pct": round(100 * g["exec_ref_hit"] / max(1, g["exec_ref_total"]), 2)
        if g["exec_ref_total"] else None,
    }
    print(f"  [A] {json.dumps(results['A_bare'])}", flush=True)

    # ── Protocol B: name-visible ──
    print("\n=== Protocol B: names ===", flush=True)
    users_b = []
    for t in tasks:
        caps = [c["name"] for c in domain_tools.get(t["domain"], [])][:15]
        users_b.append(f"{t['instruction']}\n\nAvailable capabilities:\n" +
                       "\n".join(f"[{i}] {c}" for i, c in enumerate(caps, 1)))
    results["B_names"] = compile_and_score(users_b, "B")

    # ── learned + oracle CapabilityIR tables (grounder needs GPU; do here) ──
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    grounder_base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    grounder = PeftModel.from_pretrained(grounder_base, "runs/phase6/grounder_g1g2_s42/final")
    grounder.eval()
    from src.compiler.skill_grounder import build_probe_prompt, parse_label
    learned_skill = {}
    for d, ts in domain_tools.items():
        prompts = []
        for c in ts:
            p = build_probe_prompt({"instruction": "", "capability": c}, "A")
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": p["system"]},
                 {"role": "user", "content": p["user"]}],
                tokenize=False, add_generation_prompt=True))
        preds = []
        for bs in range(0, len(prompts), 16):
            inputs = tok(prompts[bs:bs+16], return_tensors="pt", padding=True,
                         truncation=True, max_length=1024).to(grounder.device)
            with torch.no_grad():
                out = grounder.generate(**inputs, max_new_tokens=8, do_sample=False,
                                        temperature=None, pad_token_id=tok.pad_token_id)
            preds.extend(parse_label(tok.decode(o[inputs["input_ids"].shape[1]:],
                                                skip_special_tokens=True)) for o in out)
        for c, sk in zip(ts, preds):
            learned_skill[(d, c["name"])] = sk or "UNKNOWN"
        print(f"  learned CapabilityIR {d}: {Counter(preds).most_common(5)}", flush=True)
    del grounder, grounder_base
    gc.collect()
    torch.cuda.empty_cache()

    base2 = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base2, "runs/phase5c/e5c_s/final")
    model.eval()

    print("\n=== Protocol C: learned CapabilityIR ===", flush=True)
    users_c = []
    for t in tasks:
        entries = [f"[{i}] {c['name']} :: {learned_skill.get((t['domain'], c['name']), 'UNKNOWN')}"
                   for i, c in enumerate(domain_tools.get(t["domain"], [])[:15], 1)]
        users_c.append(f"{t['instruction']}\n\nGrounded capabilities:\n" + "\n".join(entries))
    results["C_learned_ir"] = compile_and_score(users_c, "C")

    print("\n=== Protocol D: oracle CapabilityIR (UPPER-BOUND DIAGNOSTIC) ===", flush=True)
    users_d = []
    for t in tasks:
        entries = [f"[{i}] {c['name']} :: {tool_to_semantic_skill(c['name'], c['description'])['skill']}"
                   for i, c in enumerate(domain_tools.get(t["domain"], [])[:15], 1)]
        users_d.append(f"{t['instruction']}\n\nGrounded capabilities:\n" + "\n".join(entries))
    results["D_oracle_ir"] = compile_and_score(users_d, "D")

    os.makedirs("results/phase6", exist_ok=True)
    with open("results/phase6/phase6b0_routing.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved results/phase6/phase6b0_routing.json", flush=True)
    print("PHASE6B0-DONE", flush=True)


if __name__ == "__main__":
    main()
