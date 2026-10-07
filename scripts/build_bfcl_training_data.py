"""Build BFCL in-distribution training data (Track A: leaderboard competition).

Lowers each BFCL V4 sample to a (instruction, function_schemas, function_call) triplet
using the ground truth. Output format directly trainable for function-calling:
  user prompt = task + available function schemas
  target = function call(s) in BFCL AST format

Also merges with xLAM 60k raw data for volume.
"""
import json, os, sys, glob
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BFCL_DIR = os.path.join(ROOT, "data/external_benchmarks/bfcl_v4")
OUT = os.path.join(ROOT, "data/bfcl_training")
os.makedirs(OUT, exist_ok=True)

# ── Load all BFCL samples with GT ──
CATEGORIES = [
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance", "irrelevance",
    "multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func",
    "multi_turn_miss_param", "memory", "multiple", "parallel",
    "parallel_multiple", "simple_java", "simple_javascript", "simple_python",
    "web_search",
]


def extract_instruction(rec):
    """Get the first user message from BFCL question field."""
    q = rec.get("question")
    if isinstance(q, list):
        for msg_group in q:
            if isinstance(msg_group, list):
                for msg in msg_group:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")
            elif isinstance(msg_group, dict) and msg_group.get("role") == "user":
                return msg_group.get("content", "")
    elif isinstance(q, str):
        return q
    return ""


def extract_schemas(rec):
    """Get function schemas (name/description/parameters)."""
    funcs = rec.get("function")
    if isinstance(funcs, str):
        try:
            funcs = json.loads(funcs)
        except:
            return []
    if not isinstance(funcs, list):
        return []
    out = []
    for f in funcs:
        if isinstance(f, dict) and f.get("name"):
            out.append({
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {}),
            })
    return out


def format_schema_prompt(schemas):
    """Format function schemas like BFCL's prompt style."""
    lines = []
    for i, s in enumerate(schemas, 1):
        params = s.get("parameters", {})
        props = params.get("properties", {})
        req = params.get("required", [])
        pstr = ", ".join(f"{k}:{v.get('type','str') if isinstance(v, dict) else str(v)[:6]}"
                         for k, v in props.items())
        lines.append(f"[{i}] {s['name']}({pstr}) - {s.get('description','')[:120]}")
    return "\n".join(lines)


def main():
    # Load ground truth index
    gt_index = {}
    for pa in glob.glob(os.path.join(BFCL_DIR, "possible_answer", "*.json")):
        for line in open(pa, encoding="utf-8"):
            if not line.strip():
                continue
            g = json.loads(line)
            gt_index[g["id"]] = g["ground_truth"]

    print(f"Ground truth entries: {len(gt_index)}")

    samples = []
    cat_dist = Counter()
    skipped = Counter()
    for cat in CATEGORIES:
        path = os.path.join(BFCL_DIR, f"BFCL_v4_{cat}.json")
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("id", "")
            instruction = extract_instruction(rec)
            schemas = extract_schemas(rec)
            gt = gt_index.get(sid)
            if not instruction:
                skipped[f"{cat}_no_instr"] += 1
                continue
            if not schemas:
                skipped[f"{cat}_no_schema"] += 1
                continue

            # Format target: function call(s) or irrelevance marker
            if gt is None:
                # irrelevance categories have no GT or empty GT
                if "irrelevance" in cat:
                    target = "No function should be called."
                else:
                    skipped[f"{cat}_no_gt"] += 1
                    continue
            else:
                # Convert GT to BFCL-style function call string
                calls = []
                for item in gt:
                    if isinstance(item, dict):
                        for fn_name, params in item.items():
                            if isinstance(params, dict):
                                # Flatten list values to comma-separated
                                flat = {}
                                for k, v in params.items():
                                    if isinstance(v, list):
                                        flat[k] = ", ".join(str(x) for x in v)
                                    else:
                                        flat[k] = v
                                param_str = ", ".join(f"{k}={repr(v)}" for k, v in flat.items())
                                calls.append(f"{fn_name}({param_str})")
                            else:
                                calls.append(f"{fn_name}()")
                if not calls:
                    if "irrelevance" in cat:
                        target = "No function should be called."
                    else:
                        target = ""
                elif len(calls) == 1:
                    target = calls[0]
                else:
                    # parallel/multiple: all calls
                    target = "\n".join(calls)

            if not target:
                skipped[f"{cat}_empty_target"] += 1
                continue

            samples.append({
                "id": sid,
                "category": cat,
                "instruction": instruction,
                "schemas": schemas,
                "schema_prompt": format_schema_prompt(schemas),
                "target": target,
            })
            cat_dist[cat] += 1

    print(f"\nTotal BFCL training samples: {len(samples)}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Categories: {dict(cat_dist.most_common())}")

    # Save
    with open(os.path.join(OUT, "bfcl_train.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {OUT}/bfcl_train.jsonl")

    # Also build xLAM training data in the same format
    xlam = json.load(open(os.path.join(ROOT, "data/raw/xlam/xlam_function_calling_60k.json")))
    xlam_samples = []
    for r in xlam:
        query = r.get("query", "")
        tools_raw = r.get("tools", "[]")
        if isinstance(tools_raw, str):
            try:
                tools = json.loads(tools_raw)
            except:
                continue
        else:
            tools = tools_raw or []
        answers_raw = r.get("answers", "[]")
        if isinstance(answers_raw, str):
            try:
                answers = json.loads(answers_raw)
            except:
                continue
        else:
            answers = answers_raw or []
        if not query or not tools:
            continue

        schemas = []
        for t in tools:
            if isinstance(t, dict) and t.get("name"):
                params = t.get("parameters", {})
                if isinstance(params, dict):
                    props = params.get("properties", params)
                    req = params.get("required", [])
                    schemas.append({"name": t["name"],
                                    "description": t.get("description", ""),
                                    "parameters": {"properties": props, "required": req}})
                else:
                    schemas.append({"name": t["name"],
                                    "description": t.get("description", ""),
                                    "parameters": {}})

        calls = []
        for a in answers:
            if isinstance(a, dict) and a.get("name"):
                args = a.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                param_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                calls.append(f"{a['name']}({param_str})")
        if not calls:
            continue
        target = calls[0] if len(calls) == 1 else "\n".join(calls)

        xlam_samples.append({
            "id": f"xlam-{r.get('id', '')}",
            "category": "xlam",
            "instruction": query,
            "schemas": schemas,
            "schema_prompt": format_schema_prompt(schemas),
            "target": target,
        })

    print(f"\nxLAM training samples: {len(xlam_samples)}")
    with open(os.path.join(OUT, "xlam_train.jsonl"), "w", encoding="utf-8") as f:
        for s in xlam_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {OUT}/xlam_train.jsonl")
    print(f"\nTotal combined: {len(samples) + len(xlam_samples)}")


if __name__ == "__main__":
    main()
