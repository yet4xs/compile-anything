"""RTL dataset adapters: VerilogEval (official repo, HDLBits-derived)."""
from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

from ..schema import make_sample


def _signals_from_verilog(text: str) -> List[str]:
    sigs = []
    for m in re.finditer(r"\b(?:input|output|inout)\s+(?:wire|reg|logic)?"
                         r"\s*(?:\[[^\]]*\]\s*)?([a-zA-Z_]\w*)", text):
        if m.group(1) not in sigs:
            sigs.append(m.group(1))
        if len(sigs) >= 6:
            break
    return sigs or ["clk", "rst_n", "data"]


class VerilogEvalAdapter:
    sources = ["verilogeval"]

    def load(self, raw_dir: pathlib.Path) -> List[Dict[str, Any]]:
        root = raw_dir / "verilogeval"
        out: List[Dict[str, Any]] = []
        for sub in ("code-complete", "spec-to-rtl"):
            d = root / sub
            if not d.is_dir():
                continue
            prompts = sorted(d.glob("*_prompt.txt"))
            for p in prompts:
                stem = p.name[: -len("_prompt.txt")]
                test = d / f"{stem}_test.sv"
                if not test.exists():
                    continue
                out.append(self.normalize(
                    p.read_text(encoding="utf-8", errors="replace"),
                    test.read_text(encoding="utf-8", errors="replace"),
                    f"{sub}/{stem}"))
        return out

    def normalize(self, prompt: str, test_sv: str, sid: str) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            return {}
        return make_sample(
            id=f"verilogeval-{sid.replace('/', '_')}", source="verilogeval",
            input_text=prompt, rtl=test_sv,
            metadata={"signals": _signals_from_verilog(test_sv),
                      "track": sid.split("/")[0]},
            raw_payload={"prompt": prompt})
