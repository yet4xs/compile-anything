"""BIRD mini-dev adapter (evaluation only)."""
from __future__ import annotations

import json
import pathlib
from typing import List

from ..schema import EvalSample

ROOT = pathlib.Path(__file__).resolve().parents[3]
BIRD_DIR = ROOT / "data" / "external_benchmarks" / "bird" / "mini_dev"


def load(dialect: str = "sqlite", bird_dir: pathlib.Path = None
         ) -> List[EvalSample]:
    d = pathlib.Path(bird_dir) if bird_dir else BIRD_DIR
    f = d / f"mini_dev_{dialect}-00000-of-00001.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("data", [])
    out: List[EvalSample] = []
    for r in rows:
        out.append(EvalSample(
            benchmark="bird_mini_dev",
            case_id=f"{dialect}-{r.get('question_id', '')}",
            instruction=(r.get("question", "")
                         + (f"\n\nEvidence: {r['evidence']}"
                            if r.get("evidence") else "")),
            reference_answer=r.get("SQL"),
            metadata={"db_id": r.get("db_id"),
                      "difficulty": r.get("difficulty"),
                      "dialect": dialect,
                      "official_metrics": ["execution_accuracy (EX)"],
                      "requires_db_environment": True}))
    return out
