"""One-command Neural Compiler run: inference over the corpus test set ->
evaluation report.

    python scripts/run_neural_compiler.py --model runs/qwen3b-lora/final \
        [--capability-context] [--limit 500]

(Delegates to src.compiler.train.infer + benchmark/neural_compiler_eval.py.)
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default=str(ROOT / "data" / "compiler_corpus_v3"
                                          / "test.jsonl"))
    ap.add_argument("--preds", default="preds/neural_compiler_test.jsonl")
    ap.add_argument("--report", default=str(ROOT / "data" / "reports"
                                            / "neural_compiler_eval.json"))
    ap.add_argument("--capability-context", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    cmd = [sys.executable, "-m", "src.compiler.train.infer",
           "--model", args.model, "--test", args.test,
           "--out", args.preds, "--max-new-tokens",
           str(args.max_new_tokens)]
    if args.capability_context:
        cmd.append("--capability-context")
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print(">", " ".join(cmd))
    if subprocess.call(cmd, cwd=str(ROOT)) != 0:
        return 1

    eval_cmd = [sys.executable, str(ROOT / "benchmark"
                                    / "neural_compiler_eval.py"),
                "--predictions", args.preds, "--train",
                str(ROOT / "data" / "compiler_corpus_v3" / "train.jsonl"),
                "--out", args.report]
    print(">", " ".join(eval_cmd))
    return subprocess.call(eval_cmd, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
