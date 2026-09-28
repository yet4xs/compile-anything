"""Code dataset adapters: HumanEval (official jsonl.gz), MBPP (CSV mirror)."""
from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Dict, List

from ..schema import make_sample
from ..downloader import gunzip_file


class HumanEvalAdapter:
    sources = ["humaneval"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        gz = raw_dir / "humaneval" / "HumanEval.jsonl.gz"
        if not gz.exists():
            return []
        plain = gunzip_file(gz)
        out = []
        for line in plain.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(self.normalize(json.loads(line)))
        return out

    def normalize(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        # canonical full function = prompt (signature + docstring) + solution
        code = (rec.get("prompt", "") + rec.get("canonical_solution", ""))
        return make_sample(
            id=str(rec.get("task_id", "")).replace("/", "_"),
            source="humaneval",
            input_text=rec.get("prompt", ""),
            code=code,
            metadata={"entry_point": rec.get("entry_point", "")},
            raw_payload={"prompt": rec.get("prompt", ""),
                         "canonical_solution": rec.get(
                             "canonical_solution", "")})


class MBPPAdapter:
    sources = ["mbpp"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        f = raw_dir / "mbpp" / "cleaned_mbpp.csv"
        if not f.exists():
            return []
        out = []
        with open(f, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                s = self.normalize(dict(row))
                if s:
                    out.append(s)
        return out

    def normalize(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        prompt = rec.get("prompt") or rec.get("text") or ""
        code = rec.get("code_solution") or rec.get("code") or ""
        tid = rec.get("task_id") or rec.get("id") or ""
        if not prompt or not code:
            return {}
        return make_sample(id=f"mbpp-{tid}", source="mbpp",
                           input_text=prompt.strip(),
                           code=self._wrap(code, prompt),
                           metadata={"tests": rec.get("test_list", "")},
                           raw_payload={"prompt": prompt, "code": code})

    @staticmethod
    def _wrap(code: str, prompt: str) -> str:
        """MBPP code bodies are statements/expressions; ensure a function
        wrapper exists so the ast-based lifter can parse a return."""
        if "def " in code:
            return code
        body = "\n".join("    " + ln for ln in code.splitlines() if ln.strip())
        return f"def f(x):\n    \"\"\"{prompt.strip()[:120]}\"\"\"\n    return (\n{body}\n    )"
