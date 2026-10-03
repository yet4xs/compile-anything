"""AgentBoard untouched first test — E5C-S, pre-registered protocol.

docs/agentboard-preregistration.md (registered BEFORE this run):
- 351 samples: tool-query 60 / tool-operation 40 / webshop 251
- capability schemas ONLY from env-visible sources:
  tool tasks: agentboard/prompts/Raw/{tool}_raw.json tool_set_message
  tool-operation: + init_config (runtime state)
  webshop: standard WebShop action space (public env interface)
- GT (subgoals/answer/product_id) never enters prompt
- protocol: SYSTEM_PROMPT + build_capability_prompt, input 2048, new 512, greedy
- metrics: parse/valid, pred/T, EXEC_ACTION rate, op distribution
- one-shot; no result-driven changes
"""
import json, sys, os, time, re
sys.path.insert(0, "/ccfa2026/compile-anything")
os.chdir("/ccfa2026/compile-anything")

# wait for the E1-A normalized rerun to release the GPU
while True:
    if not os.popen("pgrep -f rerun_e1a_tau3[.]py").read().strip():
        break
    print("waiting for rerun_e1a_tau3 ...", flush=True)
    time.sleep(60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.compiler.capability_format import build_capability_prompt
from src.ir.parser import parse_text
from src.validator.validator import validate
from collections import Counter, defaultdict

AB = "/ccfa2026/compile-anything/data/external_benchmarks/agentboard/data"
RAW = "/ccfa2026/AgentBoard/agentboard/prompts/Raw"

TOOL_MAP = {"academia": "academia", "movie": "movie", "weather": "weather",
            "todo": "todo", "sheet": "sheet"}

WEBSHOP_ACTIONS = [
    "search[keywords] - search products by keywords",
    "click[item] - click a product item from results",
    "click[button] - click a page button (e.g. Next, Prev, Buy Now)",
    "click[tab] - navigate to an attribute tab",
    "buy[now] - purchase the currently viewed item",
]

# ── Load env-visible schemas ──
schemas = {}
for tool in TOOL_MAP.values():
    with open(f"{RAW}/{tool}_raw.json") as f:
        d = json.load(f)
    schemas[tool] = d.get("tool_set_message") or []

print("env-visible schemas:", {k: len(v) for k, v in schemas.items()}, flush=True)

# ── Build samples ──
samples = []
for task, path in (("tool-query", f"{AB}/tool-query/test.jsonl"),
                   ("tool-operation", f"{AB}/tool-operation/test.jsonl"),
                   ("webshop", f"{AB}/webshop/test.jsonl")):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            info = r.get("additional_info") or {}
            goal = r.get("goal", "")
            if task == "webshop":
                caps = WEBSHOP_ACTIONS
                user_base = goal
            else:
                tool = TOOL_MAP.get(info.get("tool"))
                caps = schemas.get(tool, [])
                user_base = goal
                if task == "tool-operation" and info.get("init_config"):
                    user_base = (f"{goal}\n\nCurrent environment state:\n"
                                 f"{json.dumps(info['init_config'], ensure_ascii=False)}")
            samples.append({
                "task": task, "case_id": f"{task}-{r.get('id','')}",
                "user": user_base, "caps": caps,
            })
print(f"samples: {len(samples)}", flush=True)
assert len(samples) == 351, len(samples)

# ── Inference (E5C-S, frozen protocol) ──
tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "runs/phase5c/e5c_s/final")
model.eval()

prompts = [tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": build_capability_prompt(s["user"], s["caps"])}],
    tokenize=False, add_generation_prompt=True) for s in samples]

completions = []
for bs in range(0, len(prompts), 16):
    if bs % 64 == 0:
        print(f"    {bs}/{len(prompts)}", flush=True)
    batch = prompts[bs:bs+16]
    inputs = tok(batch, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
    completions.extend(tok.decode(outputs[j][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                       for j in range(len(batch)))

del model
torch.cuda.empty_cache()

# ── Score ──
POLICY_OPS = {"EXTRACT", "GENERATE", "VERIFY", "SELECT", "MERGE"}
by_task = defaultdict(lambda: Counter())
op_dist = defaultdict(Counter)
os.makedirs("results/agentboard", exist_ok=True)
with open("results/agentboard/first_test_preds.jsonl", "w") as fout:
    for s, comp in zip(samples, completions):
        t = s["task"]
        by_task[t]["n"] += 1
        text = extract_taskir_text(comp)
        fout.write(json.dumps({"case_id": s["case_id"], "task": t,
                               "taskir_text": text}, ensure_ascii=False) + "\n")
        try:
            module = parse_text(text)
            by_task[t]["parse"] += 1
        except Exception:
            continue
        try:
            ok = validate(module).valid
        except Exception:
            ok = False
        if ok:
            by_task[t]["valid"] += 1
            nodes = [n for n in module.program.nodes if n.op not in POLICY_OPS]
            by_task[t]["pred_actions"] += len(nodes)
            for n in nodes:
                op_dist[t][n.op] += 1
                if n.op == "EXEC_ACTION":
                    by_task[t]["exec_action"] += 1

results = {}
for t, c in by_task.items():
    n = c["n"]
    results[t] = {
        "n": n,
        "parse_pct": round(100 * c["parse"] / n, 2),
        "valid_pct": round(100 * c["valid"] / n, 2),
        "pred_per_task": round(c["pred_actions"] / n, 2),
        "exec_action_use_pct": round(100 * c["exec_action"] / max(1, c["pred_actions"]), 2),
        "top_ops": op_dist[t].most_common(6),
    }

# official baselines (read-only reference)
ref = {}
for m in ("gpt-4", "gpt-35-turbo", "claude2"):
    for t, fname in (("tool-query", "tool-query"), ("tool-operation", "tool-operation"),
                     ("webshop", "webshop")):
        p = f"{AB}/baseline_results/{m}/{fname}.txt"
        if not os.path.exists(p):
            continue
        succ, prog = [], []
        for line in open(p):
            ms = re.search(r"\[success_rate\]: (\d+), \[progress_rate\]: ([\d.]+)", line)
            if ms:
                succ.append(int(ms.group(1)))
                prog.append(float(ms.group(2)))
        if succ:
            ref[f"{m}/{t}"] = {"n": len(succ),
                               "success_rate": round(sum(succ) / len(succ), 4),
                               "progress_rate": round(sum(prog) / len(prog), 4)}
results["_official_baselines_reference"] = ref
results["_protocol"] = ("pre-registered docs/agentboard-preregistration.md; "
                        "env-visible schemas only; one-shot; E5C-S")

with open("results/agentboard/first_test.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== AgentBoard untouched first test (E5C-S) ===", flush=True)
for t in ("tool-query", "tool-operation", "webshop"):
    r = results[t]
    print(f"{t:16s} n={r['n']:3d} parse={r['parse_pct']:6.2f}% valid={r['valid_pct']:6.2f}% "
          f"pred/T={r['pred_per_task']:5.2f} EXEC_ACTION={r['exec_action_use_pct']:5.1f}%", flush=True)
    print(f"                 top ops: {r['top_ops']}", flush=True)
print("\nbaseline reference (official, read-only):", flush=True)
for k, v in ref.items():
    print(f"  {k:28s} success={v['success_rate']:6.2f} progress={v['progress_rate']:6.2f} (n={v['n']})", flush=True)
print("\nSaved to results/agentboard/first_test.json", flush=True)
print("DONE", flush=True)
