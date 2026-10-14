"""Phase 8D-2A Task 5: HEFT scheduler + critical path analysis for dataflow DAGs.

CPU-only simulation. All costs are declared, not measured on hardware.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DAGNode:
    id: str
    tool: str
    effect_class: str = "read_only"
    compute_cost: float = 1.0    # abstract compute units
    is_extract: bool = False     # EXTRACT_FIELD nodes are free


@dataclass
class DAGEdge:
    src: str
    dst: str
    kind: str  # DATA_VALUE | DATA_OPERAND | ORDER | EFFECT
    transfer_cost: float = 0.1  # abstract data transfer units


@dataclass
class ScheduleEntry:
    node_id: str
    processor: int
    start_time: float
    finish_time: float


@dataclass
class ScheduleResult:
    entries: List[ScheduleEntry] = field(default_factory=list)
    makespan: float = 0.0
    critical_path_length: float = 0.0
    max_parallelism: int = 0


class DataflowDAG:
    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: List[DAGEdge] = []

    def add_node(self, node: DAGNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: DAGEdge):
        self.edges.append(edge)

    def successors(self, nid: str) -> List[str]:
        return [e.dst for e in self.edges if e.src == nid]

    def predecessors(self, nid: str) -> List[str]:
        return [e.src for e in self.edges if e.dst == nid]

    def topological_order(self) -> List[str]:
        in_deg = {n: 0 for n in self.nodes}
        for e in self.edges:
            in_deg[e.dst] += 1
        queue = sorted([n for n, d in in_deg.items() if d == 0])
        order = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for succ in self.successors(n):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)
            queue.sort()
        return order

    def critical_path_length(self) -> float:
        """Longest weighted path through the DAG."""
        topo = self.topological_order()
        dist = {n: 0.0 for n in self.nodes}
        for n in topo:
            node_cost = self.nodes[n].compute_cost
            for succ in self.successors(n):
                edge_cost = next(
                    (e.transfer_cost for e in self.edges
                     if e.src == n and e.dst == succ), 0.0)
                new_dist = dist[n] + node_cost + edge_cost
                dist[succ] = max(dist[succ], new_dist)
        return max(dist.values()) if dist else 0.0

    def max_parallelism(self) -> int:
        """Maximum number of nodes that could execute simultaneously."""
        topo = self.topological_order()
        # Simplified: count nodes at each "level" (longest path from root)
        level = {n: 0 for n in self.nodes}
        for n in topo:
            for succ in self.successors(n):
                level[succ] = max(level[succ], level[n] + 1)
        from collections import Counter
        level_counts = Counter(level.values())
        return max(level_counts.values()) if level_counts else 0


def heft_schedule(dag: DataflowDAG, num_processors: int = 2) -> ScheduleResult:
    """HEFT (Heterogeneous Earliest Finish Time) scheduling.

    Simplified for homogeneous processors with uniform compute costs.
    """
    topo = dag.topological_order()
    if not topo:
        return ScheduleResult()

    # upward rank (critical path from each node to exit)
    rank = {}
    for n in reversed(topo):
        node_cost = dag.nodes[n].compute_cost
        succ_costs = []
        for succ in dag.successors(n):
            edge_cost = next(
                (e.transfer_cost for e in dag.edges
                 if e.src == n and e.dst == succ), 0.0)
            succ_costs.append(edge_cost + rank.get(succ, 0))
        rank[n] = node_cost + (max(succ_costs) if succ_costs else 0)

    # Sort by descending upward rank
    sorted_nodes = sorted(topo, key=lambda n: -rank[n])

    # Track processor availability and node finish times
    proc_available = [0.0] * num_processors
    node_finish: Dict[str, float] = {}
    entries = []

    for nid in sorted_nodes:
        node = dag.nodes[nid]
        if node.is_extract:
            # EXTRACT_FIELD: negligible cost, same processor as producer
            preds = dag.predecessors(nid)
            est = max(node_finish.get(p, 0) for p in preds) if preds else 0
            proc = 0  # assign to proc 0 for simplicity
            entry = ScheduleEntry(nid, proc, est, est + 0.01)
            entries.append(entry)
            node_finish[nid] = est + 0.01
            continue

        # Earliest start considering data dependencies
        preds = dag.predecessors(nid)
        data_ready = 0.0
        for p in preds:
            edge_cost = next(
                (e.transfer_cost for e in dag.edges
                 if e.src == p and e.dst == nid), 0.0)
            data_ready = max(data_ready,
                              node_finish.get(p, 0) + edge_cost)

        # Find processor with earliest finish time
        best_proc = 0
        best_eft = float("inf")
        for p in range(num_processors):
            est = max(proc_available[p], data_ready)
            eft = est + node.compute_cost
            if eft < best_eft:
                best_eft = eft
                best_proc = p

        est = max(proc_available[best_proc], data_ready)
        eft = est + node.compute_cost
        entries.append(ScheduleEntry(nid, best_proc, est, eft))
        node_finish[nid] = eft
        proc_available[best_proc] = eft

    makespan = max(node_finish.values()) if node_finish else 0
    return ScheduleResult(
        entries=sorted(entries, key=lambda e: e.start_time),
        makespan=makespan,
        critical_path_length=dag.critical_path_length(),
        max_parallelism=dag.max_parallelism())


def dag_from_taskir(taskir_text: str,
                    default_compute: float = 1.0) -> Optional[DataflowDAG]:
    """Build a DataflowDAG from TaskIR text."""
    from src.ir.parser import parse_text
    try:
        mod = parse_text(taskir_text)
    except Exception:
        return None

    dag = DataflowDAG()
    ext_chain = {}
    for n in mod.program.nodes:
        tool = n.params.get("action") or (n.hints or {}).get("tool", "")
        eff = (n.hints or {}).get("effect", "read_only")
        cost = 0.01 if n.op == "EXTRACT_FIELD" else default_compute
        dag.add_node(DAGNode(n.id, tool, eff, cost,
                              n.op == "EXTRACT_FIELD"))
        if n.op == "EXTRACT_FIELD" and n.inputs:
            ext_chain[n.id] = n.inputs[0]

    for n in mod.program.nodes:
        for inp in n.inputs:
            if inp in dag.nodes:
                dag.add_edge(DAGEdge(inp, n.id, "DATA_VALUE"))
        for a in (n.after or []):
            if a in dag.nodes:
                dag.add_edge(DAGEdge(a, n.id, "ORDER"))

    return dag
