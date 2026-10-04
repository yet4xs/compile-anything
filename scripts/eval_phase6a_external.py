"""Phase 6A Tasks 15-18: external grounding diagnostics (after internal freeze).

BFCL  : task + first function schema -> skill; gold = frozen oracle first op
        (full/partial cases). Noise floor disclosed: first-listed function may
        differ from first-called; also report the single-op subset.
tau3  : reference action tool name (name-only capability; tau3 ships no tool
        descriptions in our data) -> skill; gold = hardened tau3 oracle.
AgentBoard: env-visible schemas (Raw tool_set_message) -> predicted skill
        distribution only (auxiliary failure analysis, no gold).

Models: B0 toolmap (deterministic) | B1 3B zero-shot | B2 E5C-S | G1 grounder s42/43/44.
External only in the diagnostic sense — all three benchmarks are already
external-development or seen; nothing here is an untouched test.
"""
import json, os, sys, re
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# wait for internal chain to finish (frozen order: internal before external)
import time
while True:
    log = ""
    try:
        with open("/tmp/phase6a_chain.log", errors="ignore") as f:
            log = f.read()
    except FileNotFoundError:
        pass
    if "PHASE6A-CHAIN-DONE" in log:
        break
    print("waiting for internal chain ...", flush=True)
    time.sleep(120)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.skill_grounder import (build_probe_prompt, parse_label,
                                         boundary_metrics, skill_family)
from src.lifter.toolmap import map_tool

AB_RAW = "/ccfa2026/AgentBoard/agentboard/prompts/Raw"


def classify(tok, model, caps_texts, tasks=None, batch=16):
    prompts = []
    for cap_txt, task in zip(caps_texts, tasks or [None] * len(caps_texts)):
        sample = {"instruction": task or "", "capability": _cap_from_text(cap_txt)}
        p = build_probe_prompt(sample, "B" if task else "A")
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"]}],
            tokenize=False, add_generation_prompt=True))
    outs = []
    for bs in range(0, len(prompts), batch):
        inputs = tok(prompts[bs:bs + batch], return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 temperature=None, pad_token_id=tok.pad_token_id)
        outs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True) for o in out)
    return [parse_label(o) for o in outs]


def _cap_from_text(txt):
    """Rebuild a capability dict from a formatted block (name/description lines)."""
    name = desc = ""
    for line in txt.splitlines():
        if line.startswith("name:"):
            name = line[5:].strip()
        elif line.startswith("description:"):
            desc = line[12:].strip()
    return {"name": name, "description": desc, "parameters": {}}


def cap_text(cap):
    name = cap.get("name", "")
    desc = (cap.get("description") or "").strip()[:400]
    return f"name: {name}\ndescription: {desc}" if desc else f"name: {name}"


def evaluate(name, golds, preds, meta_notes=""):
    samples = [{"target_skill": g} for g in golds]
    b = boundary_metrics(samples, preds)
    ea_rec = None
    inb = [(g, p) for g, p in zip(golds, preds)
           if skill_family(g) in ("RETRIEVAL", "ACTION")]
    if inb:
        act_gold = [1 for g, p in inb if skill_family(g) == "ACTION"]
        act_hit = [1 for g, p in inb
                   if skill_family(g) == "ACTION" and p == "EXEC_ACTION"]
        ea_rec = round(sum(act_hit) / max(1, sum(act_gold)), 4) if sum(act_gold) else None
    return {"model": name, "n": len(golds),
            "boundary_n": b.get("n", 0),
            "boundary_acc": b.get("binary_accuracy"),
            "boundary_macro_f1": b.get("macro_f1"),
            "boundary_confusion": b.get("confusion"),
            "exec_action_recall_in_boundary": ea_rec,
            "note": meta_notes}


def main():
    results = {"_protocol": "external DIAGNOSTIC only (all three benchmarks already "
                            "external-development or seen); golds: BFCL frozen oracle, "
                            "tau3 hardened oracle; AgentBoard distribution-only"}

    # ── Build BFCL grounding set ──
    from src.eval.adapters import bfcl as bfcl_adapter
    from src.eval.oracle_bfcl import oracle as bfcl_oracle
    samples = bfcl_adapter.load()
    bfcl_rows = []
    for s in samples:
        o = bfcl_oracle(s)
        mod = o.get("module")
        if not mod or o.get("status") not in ("full", "partial"):
            continue
        ops = [n.op for n in mod.program.nodes
               if n.op not in ("EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE")]
        if not ops or not s.capabilities:
            continue
        bfcl_rows.append({"task": s.instruction,
                          "cap": s.capabilities[0],
                          "gold": ops[0],
                          "single_op": len(ops) == 1})
    print(f"BFCL grounding rows: {len(bfcl_rows)} "
          f"(single-op subset {sum(r['single_op'] for r in bfcl_rows)})", flush=True)

    # ── Build tau3 grounding set ──
    from src.eval.tau3_skill_oracle import lower_reference_actions
    tau3_rows = []
    with open("runs/phase5b1/tau3_preds_2048.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("ref_action_names"):
                continue
            low = lower_reference_actions([{"name": n} for n in r["ref_action_names"]])
            if not low:
                continue
            tau3_rows.append({"task": r["instruction"],
                              "tool": r["ref_action_names"][0],
                              "gold": low[0]["skill"]})
    print(f"tau3 grounding rows: {len(tau3_rows)}", flush=True)

    # ── AgentBoard env-visible schemas ──
    ab_schemas = {}
    for tool in ("todo", "sheet", "academia", "movie", "weather"):
        try:
            with open(f"{AB_RAW}/{tool}_raw.json") as f:
                ab_schemas[tool] = json.load(f).get("tool_set_message") or []
        except FileNotFoundError:
            continue
    print(f"AgentBoard tools: {list(ab_schemas)}", flush=True)

    # ── B0 toolmap (deterministic, no GPU) ──
    bfcl_b0 = [map_tool(r["cap"].get("name", ""), {}).get("skill") for r in bfcl_rows]
    results["bfcl/B0_toolmap"] = evaluate("B0_toolmap", [r["gold"] for r in bfcl_rows], bfcl_b0,
                                          "regex toolmap on unseen BFCL function names")
    tau3_b0 = [map_tool(r["tool"], {}).get("skill") for r in tau3_rows]
    results["tau3/B0_toolmap"] = evaluate("B0_toolmap", [r["gold"] for r in tau3_rows], tau3_b0,
                                          "regex toolmap on tau3 tool names (the Phase 5C failure axis)")
    for k in ("bfcl/B0_toolmap", "tau3/B0_toolmap"):
        r = results[k]
        print(f"{k}: acc={r['boundary_acc']} F1={r['boundary_macro_f1']} "
              f"EXEC_ACTION_R={r['exec_action_recall_in_boundary']}", flush=True)

    # ── Neural models ──
    tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")

    # SEQUENTIAL loading: one fresh base per adapter. The first run of this
    # script shared one base across PeftModel wrappers and every neural row
    # came out bit-identical (adapters never affected generation) — those rows
    # were invalid and have been discarded. Sanity guard below verifies each
    # adapter actually changes outputs before the expensive sweep.
    SPECS = [("B1_zeroshot_3b", None),
             ("B2_e5cs", "runs/phase5c/e5c_s/final"),
             ("G1_grounder_G1_s42", "runs/phase6/grounder_g1_s42/final"),
             ("G1_grounder_G1_s43", "runs/phase6/grounder_g1_s43/final"),
             ("G1_grounder_G1_s44", "runs/phase6/grounder_g1_s44/final"),
             ("G1_grounder_G1G2_s42", "runs/phase6/grounder_g1g2_s42/final")]

    import gc
    for name, adapter in SPECS:
        del base
        gc.collect()
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:
            if not os.path.exists(adapter):
                print(f"[skip missing {adapter}]", flush=True)
                continue
            model = PeftModel.from_pretrained(base, adapter)
            model.eval()
            # sanity guard: adapter must change outputs vs the plain base
            guard_caps = ["name: book_reservation", "name: get_weather_data"]
            g_base = classify(tok, base, guard_caps)
            g_ad = classify(tok, model, guard_caps)
            if g_base == g_ad:
                print(f"  WARNING: {name} outputs identical to base on guard set", flush=True)
        else:
            model = base
        # BFCL
        preds = classify(tok, model,
                         [cap_text(r["cap"]) for r in bfcl_rows],
                         [r["task"] for r in bfcl_rows])
        results[f"bfcl/{name}"] = evaluate(name, [r["gold"] for r in bfcl_rows], preds)
        r = results[f"bfcl/{name}"]
        print(f"bfcl/{name}: acc={r['boundary_acc']} F1={r['boundary_macro_f1']} "
              f"EXEC_ACTION_R={r['exec_action_recall_in_boundary']}", flush=True)
        # single-op subset
        idx = [i for i, rr in enumerate(bfcl_rows) if rr["single_op"]]
        results[f"bfcl/{name}__single_op"] = evaluate(
            name, [bfcl_rows[i]["gold"] for i in idx], [preds[i] for i in idx])
        # tau3 (name-only capability, no task -> probe A on name)
        preds_t = classify(tok, model, [f"name: {r['tool']}" for r in tau3_rows])
        results[f"tau3/{name}"] = evaluate(name, [r["gold"] for r in tau3_rows], preds_t,
                                           "name-only capability (tau3 ships no descriptions)")
        r = results[f"tau3/{name}"]
        print(f"tau3/{name}: acc={r['boundary_acc']} F1={r['boundary_macro_f1']} "
              f"EXEC_ACTION_R={r['exec_action_recall_in_boundary']}", flush=True)
        # AgentBoard distribution (no gold)
        dist = {}
        for tool, caps in ab_schemas.items():
            preds_ab = classify(tok, model, [cap_text(c) for c in caps[:15]])
            dist[tool] = dict(Counter(p or "MISS" for p in preds_ab).most_common(5))
        results[f"agentboard/{name}"] = {"skill_distribution": dist}
        print(f"agentboard/{name}: EXEC_ACTION rates:",
              {t: d.get("EXEC_ACTION", 0) for t, d in dist.items()}, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs("results/phase6", exist_ok=True)
    with open("results/phase6/phase6a_external.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved results/phase6/phase6a_external.json", flush=True)
    print("PHASE6A-EXTERNAL-DONE", flush=True)


if __name__ == "__main__":
    main()
