"""AgentBoard adapter — tool-query / tool-operation / webshop (+ webarena).

Uniform record shape: {task, id, goal, subgoals, difficulty,
additional_info}. Success/progress semantics stay in metadata (AgentBoard
official metrics: progress rate + success rate)."""
from __future__ import annotations

import json
import pathlib
from typing import List

from ..schema import EvalSample

ROOT = pathlib.Path(__file__).resolve().parents[3]
AB_DIR = ROOT / "data" / "external_benchmarks" / "agentboard" / "data"

SUPPORTED = ("tool-query", "tool-operation", "webshop", "webarena")


def load(tasks: List[str] = None, data_dir: pathlib.Path = None
         ) -> List[EvalSample]:
    d = pathlib.Path(data_dir) if data_dir else AB_DIR
    out: List[EvalSample] = []
    for name in (tasks or list(SUPPORTED)):
        f = d / name / "test.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.append(EvalSample(
                benchmark=f"agentboard_{r.get('task', name).replace('-', '_')}",
                case_id=f"{name}-{r.get('id', '')}",
                instruction=r.get("goal", ""),
                reference_answer=None,
                metadata={
                    "subgoals": r.get("subgoals"),
                    "difficulty": r.get("difficulty"),
                    "additional_info": r.get("additional_info"),
                    "official_metrics": ["progress_rate", "success_rate"],
                    "interaction": True,       # needs the AgentBoard env
                }))
    return out
