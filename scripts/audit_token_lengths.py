"""Token length audit (Phase 5B-1 Task 8).

    python scripts/audit_token_lengths.py                       # approximate
    python scripts/audit_token_lengths.py --tokenizer weights/Qwen2.5-3B-Instruct

Reports p50/p90/p95/p99/max and >1024/1536/2048 fractions for prompt /
target / total tokens, per split. Without a tokenizer (dev machine), uses
a conservative chars/3.5 approximation and marks the report approximate —
server rerun with the real tokenizer overwrites it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.prompt_format import SYSTEM_PROMPT        # noqa: E402
from src.compiler.train.dataset import build_user_text      # noqa: E402


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(round(q / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def stats_for(vals):
    s = sorted(vals)
    return {
        "n": len(s),
        "p50": percentile(s, 50), "p90": percentile(s, 90),
        "p95": percentile(s, 95), "p99": percentile(s, 99),
        "max": max(s) if s else 0,
        "gt_1024_pct": round(100 * sum(1 for v in s if v > 1024) / len(s), 2)
        if s else 0.0,
        "gt_1536_pct": round(100 * sum(1 for v in s if v > 1536) / len(s), 2)
        if s else 0.0,
        "gt_2048_pct": round(100 * sum(1 for v in s if v > 2048) / len(s), 2)
        if s else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "compiler_corpus_v3"))
    ap.add_argument("--tokenizer", default=None,
                    help="model dir; omit for char-approximate mode")
    ap.add_argument("--out", default=str(ROOT / "data" / "reports"
                                         / "token_length_audit.json"))
    args = ap.parse_args()

    tok = None
    approximate = True
    if args.tokenizer and pathlib.Path(args.tokenizer).exists():
        try:
            from transformers import AutoTokenizer          # noqa: PLC0415
            tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                                trust_remote_code=True)
            approximate = False
        except Exception as e:                             # noqa: BLE001
            print(f"[WARN] tokenizer unavailable ({e}); approximate mode")

    def n_tokens(text: str) -> int:
        if tok is not None:
            return len(tok.encode(text))
        return int(len(text) / 3.5) + 1

    report = {"approximate": approximate,
              "tokenizer": args.tokenizer if tok else None,
              "splits": {}}
    for split in ("train", "val", "test"):
        p = pathlib.Path(args.corpus) / f"{split}.jsonl"
        if not p.exists():
            continue
        prompts, targets, totals = [], [], []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                user = build_user_text(r.get("instruction", ""),
                                       r.get("capabilities"))
                pt = n_tokens(SYSTEM_PROMPT) + n_tokens(user)
                tt = n_tokens(r.get("plan_target", ""))
                prompts.append(pt)
                targets.append(tt)
                totals.append(pt + tt)
        report["splits"][split] = {
            "prompt": stats_for(prompts),
            "target": stats_for(targets),
            "total": stats_for(totals),
        }
        t = report["splits"][split]["total"]
        print(f"{split:5s} total: p50={t['p50']} p90={t['p90']} "
              f"p95={t['p95']} p99={t['p99']} max={t['max']} "
              f">1024:{t['gt_1024_pct']}% >1536:{t['gt_1536_pct']}% "
              f">2048:{t['gt_2048_pct']}%")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"{'APPROXIMATE' if approximate else 'EXACT'} report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
