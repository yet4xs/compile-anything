"""Judge agreement analysis (Task 11) — deterministic checker vs
strong-model judge. Ground-truth deterministic evidence outranks LLM
opinion; the judge can only CONFIRM suspicion, never rewrite labels.

    python scripts/analyze_semantic_judge.py \
        --judge data/reports/semantic_judge_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=str(
        ROOT / "data" / "reports" / "semantic_judge_results.jsonl"))
    args = ap.parse_args()
    p = pathlib.Path(args.judge)
    if not p.exists():
        print("SKIP: no judge results (run scripts/run_semantic_judge.py "
              "first). Nothing fabricated.")
        return 2

    rows = [json.loads(l) for l in p.read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    n = agree = both_suspect = judge_only = det_only = 0
    conflicts = []
    for r in rows:
        det = r.get("deterministic_status")
        judge_ok = r.get("semantic_match")
        if judge_ok is None:
            continue
        n += 1
        det_susp = det == "suspect"
        judge_susp = judge_ok is False
        if det_susp == judge_susp:
            agree += 1
        if det_susp and judge_susp:
            both_suspect += 1
        elif judge_susp:
            judge_only += 1
            if (r.get("confidence") or 0) >= 0.8:
                conflicts.append(r)
        elif det_susp:
            det_only += 1
    out = {"judged": n,
           "agreement_pct": round(100 * agree / n, 2) if n else None,
           "both_suspect": both_suspect,
           "judge_only_suspect": judge_only,
           "deterministic_only_suspect": det_only,
           "high_confidence_judge_only": [
               c["sample_id"] for c in conflicts][:50]}
    outp = ROOT / "data" / "reports" / "semantic_judge_agreement.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"-> {outp}")
    print("policy: deterministic evidence > LLM opinion; labels are "
          "never rewritten by the judge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
