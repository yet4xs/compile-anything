"""Strong-model semantic judge interface (Task 10) — interface ONLY.

    python scripts/run_semantic_judge.py \
        --sample data/reports/semantic_audit_sample.jsonl \
        --judge openai://api_key [--model gpt-4o] [--limit 100]

No judge configured -> skip (exit 2). Results are NEVER used to rewrite
labels (Task 11 analyzes agreement only). The judge prompt ALWAYS shows
instruction + raw ground truth + TaskIR — never instruction+TaskIR alone.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JUDGE_PROMPT = """You are auditing a neural compiler's training label.

Given:
1. INSTRUCTION: the natural-language task
2. GROUND TRUTH: the benchmark's own reference (tool-call trajectory /
   SQL / Python AST summary / RTL prompt)
3. TASKIR: the compiled intermediate representation the model is trained
   to emit

Question: does the TaskIR faithfully represent the semantics of the task
AS DEFINED BY THE GROUND TRUTH? The ground truth is the primary evidence;
the instruction alone is not.

Answer with EXACTLY this JSON (no markdown fence):
{"semantic_match": true|false,
 "missing_steps": [str], "extra_steps": [str],
 "wrong_dependencies": [str], "wrong_arguments": [str],
 "confidence": 0.0-1.0, "reason": "one short sentence"}

INSTRUCTION:
{instruction}

GROUND TRUTH:
{ground_truth}

TASKIR:
{taskir}"""


def call_judge_openai(api_key: str, model: str, prompt: str) -> dict:
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0,
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": prompt}]}
                      ).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    return json.loads(content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(
        ROOT / "data" / "reports" / "semantic_audit_sample.jsonl"))
    ap.add_argument("--judge", default=None,
                    help="e.g. openai://sk-... (omit -> skip)")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(
        ROOT / "data" / "reports" / "semantic_judge_results.jsonl"))
    args = ap.parse_args()

    if not args.judge:
        print("SKIP: no judge configured (--judge openai://<key>). "
              "No results are fabricated.")
        return 2
    kind, _, key = args.judge.partition("://")
    if kind != "openai" or not key:
        print("SKIP: unsupported judge (only openai://<key> implemented)")
        return 2

    rows = [json.loads(l) for l in
            open(args.sample, encoding="utf-8").read().splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            prompt = JUDGE_PROMPT.format(
                instruction=r["instruction"],
                ground_truth=str(r.get("raw_ground_truth_summary"))[:3000],
                taskir=r["taskir"])
            try:
                verdict = call_judge_openai(key, args.model, prompt)
            except Exception as e:                       # noqa: BLE001
                verdict = {"semantic_match": None, "error": str(e)}
            f.write(json.dumps({"sample_id": r["sample_id"],
                                "deterministic_status":
                                    r["deterministic_audit"]["status"],
                                **verdict}, ensure_ascii=False) + "\n")
            n += 1
    print(f"judge verdicts: {n} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
