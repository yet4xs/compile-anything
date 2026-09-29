"""RTL-Repo adapter — line-level RTL completion with repository context.

Audited schema (parquet): {repo_name, file_path, next_line, context:
[{path, snippet}], created_at, all_code, cropped_code, level}.

IMPORTANT honesty note (docs/benchmark-schema-audit.md): RTL-Repo is a
NEXT-LINE completion benchmark with cross-file repo context, NOT
module-level generation with testbenches. Rows carry no executable
testbench — official metrics are exact-match / syntax-match (pass@1),
NOT functional correctness. We record that explicitly and do not invent
"functional correctness" numbers. Future TaskIR orchestration evaluation
(repo-level multi-file reasoning) is a different axis, tracked in
metadata.eval_axes.
"""
from __future__ import annotations

import pathlib
from typing import List

import pyarrow.parquet as pq

from ..schema import EvalSample

ROOT = pathlib.Path(__file__).resolve().parents[3]
RTL_DIR = ROOT / "data" / "external_benchmarks" / "rtl_repo"


def load(split: str = "test", rtl_dir: pathlib.Path = None
         ) -> List[EvalSample]:
    d = pathlib.Path(rtl_dir) if rtl_dir else RTL_DIR
    f = d / f"{split}-00000-of-00001.parquet"
    table = pq.read_table(f)
    out: List[EvalSample] = []
    for i, row in enumerate(table.to_pylist()):
        ctx = row.get("context") or []
        out.append(EvalSample(
            benchmark="rtl_repo",
            case_id=f"{split}-{row.get('repo_name', '')}-"
                    f"{row.get('file_path', '')}-{i}",
            instruction=(
                f"Complete the next line of {row.get('file_path', '')} in "
                f"repository {row.get('repo_name', '')}, given the "
                f"cross-file context."),
            metadata={
                "split": split,
                "repo_name": row.get("repo_name"),
                "file_path": row.get("file_path"),
                "context_files": [c.get("path") for c in ctx],
                "context_snippets": ctx[:6],
                "cropped_code": row.get("cropped_code"),
                "level": row.get("level"),
                "created_at": row.get("created_at"),
                "official_metrics": ["exact_match_pass@1",
                                     "syntax_match_pass@1"],
                "executable_testbench": False,
                "eval_axes": {"generation_correctness": True,
                              "taskir_orchestration": False},
            },
            reference_answer=row.get("next_line"),
        ))
    return out
