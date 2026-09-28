"""Neural Compiler inference: corpus test set (or raw instructions) ->
predictions jsonl for benchmark/neural_compiler_eval.py.

    python -m src.compiler.train.infer --model runs/qwen3b-lora/final \
        --test data/compiler_corpus_v3/test.jsonl --out preds/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True, help="corpus jsonl with "
                    "instruction + reference targets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--capability-context", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch                                       # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
    from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
    from src.compiler.train.dataset import build_user_text

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")

    rows = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            caps = r.get("capabilities") if args.capability_context else None
            user = build_user_text(r.get("instruction", ""), caps)
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=None)
            completion = tok.decode(
                out_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)
            f.write(json.dumps({
                "id": r.get("id", ""), "source": r.get("source", ""),
                "input_text": user,
                "output_text": extract_taskir_text(completion),
                "reference_plan": r.get("plan_target", ""),
            }, ensure_ascii=False) + "\n")
    print(f"predictions -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
