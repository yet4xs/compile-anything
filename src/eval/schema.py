"""External evaluation schema — STRICTLY separate from training data.

EvalSample is the normalized representation of a benchmark's ground
truth. It deliberately has NO taskir_target: evaluation benchmarks are
never supervision. The deterministic oracle TaskIR (Task 12/13) lives in
src/eval/oracle_bfcl.py and data/external_benchmarks/derived_oracle/ and
is blocked from training by src/dataset/external_guard.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

FORBIDDEN_FIELDS = ("taskir_target", "taskir_text", "taskir_json")


@dataclass
class EvalSample:
    benchmark: str
    case_id: str
    instruction: str = ""                 # model input (NL goal)
    capabilities: Optional[List[Any]] = None   # tool/function schemas
    conversation: Optional[List[Any]] = None   # multi-turn message turns
    initial_state: Optional[Dict[str, Any]] = None  # env/DB state
    reference_actions: Optional[List[Any]] = None   # GT calls (never input)
    reference_answer: Optional[str] = None          # GT final answer
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for f in FORBIDDEN_FIELDS:
            if f in self.metadata:
                raise ValueError(
                    f"EvalSample must never carry {f} — evaluation data is "
                    f"not supervision")

    def to_dict(self) -> Dict[str, Any]:
        return {"benchmark": self.benchmark, "case_id": self.case_id,
                "instruction": self.instruction,
                "capabilities": self.capabilities,
                "conversation": self.conversation,
                "initial_state": self.initial_state,
                "reference_actions": self.reference_actions,
                "reference_answer": self.reference_answer,
                "metadata": self.metadata}
