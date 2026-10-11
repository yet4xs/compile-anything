"""SSA Error Slice Extractor — extracts minimal repairable regions from failed TaskIR.

Given a failed TaskIR module + validator report + error node IDs, produces:
- The faulty node(s) and their SSA definitions
- Upstream definitions (inputs) feeding into faulty nodes
- Downstream consumers affected by the faulty nodes
- Type constraints at each operand position
- The bounded repair scope (max nodes allowed to change)
- Program output dependency chain

All nodes NOT in the repair scope are frozen — patches touching them are rejected.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
from src.ir.taskir import Module, Node
from src.validator.validator import Report


@dataclass
class SSASlice:
    """A bounded repairable region of a TaskIR program."""
    module: Module
    error_nodes: Set[str]                          # nodes with validator errors
    upstream: Dict[str, List[str]]                 # node_id -> upstream provider IDs
    downstream: Set[str]                            # nodes consuming error node outputs
    frozen: Set[str]                                # nodes NOT in repair scope (immutable)
    repairable: Set[str]                            # nodes allowed to be modified
    program_output: Optional[str]                    # current program output node
    output_in_scope: bool                           # whether program output is repairable
    slice_size: int                                 # len(repairable)
    max_repair_nodes: int                           # budget
    overflow: bool                                  # True if slice exceeds budget

    def can_modify(self, node_id: str) -> bool:
        return node_id in self.repairable

    def summary(self) -> str:
        return (f"slice: {self.slice_size}/{self.max_repair_nodes} nodes, "
                f"errors={sorted(self.error_nodes)}, "
                f"upstream={ {k: v for k, v in self.upstream.items()} }, "
                f"downstream={sorted(self.downstream)}")


def _providers(node: Node, all_ids: Set[str]) -> List[str]:
    """Return SSA node IDs that this node reads from."""
    return [inp for inp in node.inputs if inp in all_ids]


def _consumers(node_id: str, nodes: List[Node]) -> List[str]:
    """Return IDs of nodes that consume this node's output."""
    return [n.id for n in nodes if node_id in n.inputs]


def extract_slice(module: Module,
                  report: Report,
                  error_node_ids: Optional[Set[str]] = None,
                  max_repair_nodes: int = 3) -> SSASlice:
    """Extract a bounded SSA slice around validator error nodes.

    The slice includes:
    - The error nodes themselves
    - Their direct upstream providers (SSA definitions they read)
    - Their direct downstream consumers (nodes that read their output)
    - The program output node if it's in the affected chain

    Nodes beyond this boundary are frozen.
    """
    nodes = module.program.nodes
    all_ids = {n.id for n in nodes}
    inputs_set = {f"@{inp['name'].lstrip('@')}" for inp in module.program.inputs}

    # Identify error nodes: from explicit list or infer from validator report
    if error_node_ids is None:
        error_node_ids = set()
        for issue in report.errors:
            # issue.node can be a Node object, a string ID, or None
            if issue.node is not None:
                if hasattr(issue.node, "id"):
                    if issue.node.id in all_ids:
                        error_node_ids.add(issue.node.id)
                elif isinstance(issue.node, str) and issue.node in all_ids:
                    error_node_ids.add(issue.node)

            # Also try to infer from message
            import re
            for m in re.finditer(r"'(%\w+)'", issue.msg or ""):
                if m.group(1) in all_ids:
                    error_node_ids.add(m.group(1))

    if not error_node_ids:
        error_node_ids = {nodes[-1].id} if nodes else set()

    # Compute upstream (providers) for each error node
    upstream: Dict[str, List[str]] = {}
    node_map = {n.id: n for n in nodes}

    for eid in error_node_ids:
        if eid in node_map:
            upstream[eid] = _providers(node_map[eid], all_ids | inputs_set)

    # Compute downstream (consumers of error node outputs)
    downstream: Set[str] = set()
    for eid in error_node_ids:
        downstream.update(_consumers(eid, nodes))

    # Repairable set: error nodes + their upstream (to fix references) + downstream (to fix types)
    repairable = set(error_node_ids)
    repairable.update(d for d in downstream if d in all_ids)
    # Don't include upstream in repairable — upstream is context, not to be modified
    # unless the error is IN the upstream node

    # Include program output if it depends on any error node
    prog_output = module.program.output
    output_in_scope = False
    if prog_output and prog_output in repairable:
        output_in_scope = True

    # Frozen: everything not in repairable
    frozen = all_ids - repairable

    overflow = len(repairable) > max_repair_nodes

    return SSASlice(
        module=module,
        error_nodes=error_node_ids,
        upstream=upstream,
        downstream=downstream - error_node_ids,
        frozen=frozen,
        repairable=repairable,
        program_output=prog_output,
        output_in_scope=output_in_scope,
        slice_size=len(repairable),
        max_repair_nodes=max_repair_nodes,
        overflow=overflow,
    )
