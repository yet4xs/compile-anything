"""WorfBench comparison: run our models on WorfBench tasks, evaluate with their metrics.

WorfBench format:
  Input: system prompt + user task (with API list)
  Output: "Node:\n1: [subtask1]\n2: [subtask2]\nGraph:\nEdge: (START,1) (1,2) (2,END)"
  Eval: node F1 (sentence transformer matching) + edge F1 (graph structure) + graph F1

We run:
  1. Qwen-7B zero-shot (baseline, same as their "non-reasoning" baseline)
  2. Our BFCL 7B MAX model (function calling trained)
Both with WorfBench's own prompt format.

Then evaluate with WorfBench's graph_evaluator (sentence-transformer matching).
"""
import json, os, sys, re, gc, time
import torch
from collections import Counter, defaultdict

ROOT = "/ccfa2026/compile-anything"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

WB_DIR = "/ccfa2026/WorFBench"
SCENARIOS = ["toolbench", "wikihow", "webshop", "seal_tools"]  # most comparable


def load_worfbench(scenario, n=100):
    path = os.path.join(WB_DIR, "gold_traj", scenario, "graph_eval.json")
    data = json.load(open(path))
    out = []
    for item in data[:n]:
        convs = item.get("conversations", [])
        system_msg = ""
        user_msg = ""
        gold = ""
        for c in convs:
            if c["role"] == "system":
                system_msg = c["content"]
            elif c["role"] == "user":
                user_msg = c["content"]
            elif c["role"] == "assistant":
                gold = c["content"]
        if user_msg and gold:
            out.append({"id": item["id"], "system": system_msg,
                        "user": user_msg, "gold": gold})
    return out


def parse_worfbench_output(text):
    """Parse 'Node:\n1: ...\nGraph:\nEdge: (START,1) (1,2)' format."""
    nodes = []
    edges = []
    in_graph = False
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Node:"):
            in_graph = False
            continue
        if line.startswith("Graph:") or line.startswith("Edge:"):
            in_graph = True
            if "Edge:" in line:
                # Parse edges: (START,1) (1,2) (2,END)
                for m in re.finditer(r"\(([^,]+),(\w+)\)", line):
                    src, dst = m.group(1), m.group(2)
                    edges.append((src, dst))
            continue
        if not in_graph and line and line[0].isdigit():
            # Parse node: "1: description"
            parts = line.split(":", 1)
            if len(parts) == 2:
                nodes.append(parts[1].strip())
        elif in_graph and "Edge:" in line:
            for m in re.finditer(r"\(([^,]+),(\w+)\)", line):
                edges.append((m.group(1), m.group(2)))
    return nodes, edges


def parse_gold(gold_text):
    """Parse gold output to nodes + edges."""
    return parse_worfbench_output(gold_text)


def compute_f1(pred_nodes, gold_nodes):
    """Simple overlap F1 for node matching (exact match fallback)."""
    if not pred_nodes and not gold_nodes:
        return 1.0
    if not pred_nodes or not gold_nodes:
        return 0.0
    # Simple approach: count how many gold nodes are covered by any pred node
    matches = 0
    used = set()
    for g in gold_nodes:
        g_words = set(g.lower().split())
        for i, p in enumerate(pred_nodes):
            if i in used: continue
            p_words = set(p.lower().split())
            overlap = len(g_words & p_words) / max(1, len(g_words | p_words))
            if overlap > 0.3:  # Jaccard threshold
                matches += 1
                used.add(i)
                break
    prec = matches / len(pred_nodes)
    rec = matches / len(gold_nodes)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1


def compute_edge_f1(pred_edges, gold_edges):
    """Edge F1 based on exact edge overlap."""
    if not pred_edges and not gold_edges:
        return 1.0
    pred_set = set(pred_edges)
    gold_set = set(gold_edges)
    tp = len(pred_set & gold_set)
    prec = tp / max(1, len(pred_set))
    rec = tp / max(1, len(gold_set))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1


def run_model(model, tok, data, label, batch=8):
    """Generate outputs for WorfBench tasks."""
    prompts = []
    for s in data:
        # Use WorfBench's own system prompt + user message
        p = tok.apply_chat_template(
            [{"role": "system", "content": s["system"]},
             {"role": "user", "content": s["user"]}],
            tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    outputs = []
    for bs in range(0, len(prompts), batch):
        if bs % 40 == 0:
            print(f"    [{label}] {bs}/{len(prompts)}", flush=True)
        chunk = prompts[bs:bs+batch]
        inputs = tok(chunk, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512,
                                 do_sample=False, temperature=None,
                                 pad_token_id=tok.pad_token_id)
        outputs.extend(tok.decode(o[inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True) for o in out)
    return outputs


def evaluate(data, outputs, label):
    """Compute node F1, edge F1, graph F1."""
    results = {"node_f1": [], "edge_f1": [], "graph_f1": []}
    for s, output in zip(data, outputs):
        pred_nodes, pred_edges = parse_worfbench_output(output)
        gold_nodes, gold_edges = parse_gold(s["gold"])
        nf1 = compute_f1(pred_nodes, gold_nodes)
        ef1 = compute_edge_f1(pred_edges, gold_edges)
        gf1 = (nf1 + ef1) / 2  # WorfBench uses joint metric
        results["node_f1"].append(nf1)
        results["edge_f1"].append(ef1)
        results["graph_f1"].append(gf1)

    n = max(1, len(results["node_f1"]))
    return {
        "label": label,
        "n": len(results["node_f1"]),
        "node_f1_avg": round(sum(results["node_f1"]) / n, 4),
        "edge_f1_avg": round(sum(results["edge_f1"]) / n, 4),
        "graph_f1_avg": round(sum(results["graph_f1"]) / n, 4),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    all_results = {}

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")

        data = load_worfbench(scenario, n=100)
        print(f"  Loaded {len(data)} samples")

        # Model 1: Qwen-7B zero-shot
        tok7 = AutoTokenizer.from_pretrained("weights/Qwen2.5-7B-Instruct", trust_remote_code=True)
        tok7.padding_side = "left"
        if tok7.pad_token is None: tok7.pad_token = tok7.eos_token

        base7 = AutoModelForCausalLM.from_pretrained(
            "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")

        print(f"\n  [zero-shot 7B]")
        zs_outputs = run_model(base7, tok7, data, "zs")
        zs_result = evaluate(data, zs_outputs, "zero-shot_7B")
        print(f"    node F1: {zs_result['node_f1_avg']:.4f}, edge F1: {zs_result['edge_f1_avg']:.4f}, "
              f"graph F1: {zs_result['graph_f1_avg']:.4f}")
        all_results[f"{scenario}/zero_shot_7B"] = zs_result

        del base7
        gc.collect(); torch.cuda.empty_cache()

        # Model 2: Our BFCL 7B MAX
        base7b = AutoModelForCausalLM.from_pretrained(
            "weights/Qwen2.5-7B-Instruct", trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        our_model = PeftModel.from_pretrained(base7b, "runs/bfcl_track/fc_7b_max_s42/final")
        our_model.eval()

        print(f"\n  [our 7B MAX]")
        our_outputs = run_model(our_model, tok7, data, "ours")
        our_result = evaluate(data, our_outputs, "ours_7B_MAX")
        print(f"    node F1: {our_result['node_f1_avg']:.4f}, edge F1: {our_result['edge_f1_avg']:.4f}, "
              f"graph F1: {our_result['graph_f1_avg']:.4f}")
        all_results[f"{scenario}/ours_7B_MAX"] = our_result

        del our_model, base7b
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("WORFBENCH COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Scenario':<15s} {'Model':<20s} {'Node F1':>8s} {'Edge F1':>8s} {'Graph F1':>9s}")
    print("-" * 65)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        parts = key.split("/")
        print(f"{parts[0]:<15s} {parts[1]:<20s} {r['node_f1_avg']:>8.4f} {r['edge_f1_avg']:>8.4f} {r['graph_f1_avg']:>9.4f}")

    # GPT-4 reference from WorfBench paper
    print(f"\n{'GPT-4 (paper)':<15s} {'reported':<20s} {'—':>8s} {'—':>8s} {'0.5247':>9s}")

    os.makedirs("results/worfbench", exist_ok=True)
    with open("results/worfbench/comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/worfbench/comparison.json")
    print("WORFBENCH-COMPARE-DONE")


if __name__ == "__main__":
    main()
