"""ReAct baseline for the neural-compiler evaluation (corpus v3.1).

A standard agentic-execution baseline to contrast with our up-front
compiler.  Both run on the same test set and are scored by the same
metrics (benchmark/neural_compiler_eval.py):

  ours (compiler) : NL task -> full TaskIR program in one shot
  ReAct (this)    : NL task -> Thought/Action/Action Input, mock-tool
                    Observation, repeat (<= max turns), Final Answer;
                    the executed tool calls are then replayed as TaskIR

Honesty rules (what keeps the comparison apples-to-apples):
  - available functions = the sample's `capabilities` (the same semantic
    Skill-ISA contract the compiler sees); empty capabilities -> full ISA
  - mock tool execution is the SAME simulated world as the compiler eval
    (Simulator + src/runtime/executors.py), seeded per sample id
  - only calls the mock world actually executed enter the converted TaskIR;
    unknown ops, malformed JSON and arity violations become Observation
    errors the agent can self-correct on (ReAct's whole point) — nothing
    is silently repaired afterwards

    python -m src.eval.react_baseline --model weights/Qwen2.5-3B-Instruct \
        --test data/compiler_corpus_v3_1/test.jsonl --out runs/react_baseline/

Outputs under --out:
    predictions.jsonl    eval-ready records (+ trajectory/reference_actions)
    eval_report.json/.md same metrics as benchmark/neural_compiler_eval.py
    unseen_dump.json     per-case unseen-composition records
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ..compiler.prompt_format import extract_taskir_text     # noqa: E402
from ..ir.taskir import Module, Node, Program, to_text        # noqa: E402
from ..isa.registry import ALL_OPS                            # noqa: E402
from ..runtime.executors import _digest                       # noqa: E402
from ..runtime.simulator import Simulator                     # noqa: E402

GLOBAL_NAME = "@task"          # the single program input ReAct may reference


# --------------------------------------------------------------- prompting

def available_ops(capabilities: Optional[List[str]]) -> List[str]:
    """Function set exposed to the agent: the sample's declared semantic
    capabilities; the full Skill ISA when the sample declares none (the
    compiler's SYSTEM_PROMPT grants the ISA unconditionally too)."""
    caps = [c for c in (capabilities or []) if c in ALL_OPS]
    return list(dict.fromkeys(caps)) if caps else list(ALL_OPS)


def _output_type_str(rule: str) -> str:
    if rule.startswith("FIXED:"):
        return rule[len("FIXED:"):].strip()
    kind, idx = rule.split(":", 1)
    return {"SAME_AS": f"same as input #{idx}",
            "ITEM_OF": f"item of input #{idx}"}.get(kind, "Any")


def _signature(op: str) -> str:
    spec = ALL_OPS[op]
    args = list(spec.input_types)
    if spec.max_inputs is None:                     # variadic skill
        args.append("...")
    return f"{op}({', '.join(args)}) -> {_output_type_str(spec.output_rule)}"


def build_system_prompt(capabilities: Optional[List[str]],
                        max_turns: int) -> str:
    """Standard ReAct system prompt; the function list is the sample's
    capability contract rendered from the Skill ISA registry."""
    funcs = "\n".join(f"- {_signature(op)}  # {ALL_OPS[op].desc}"
                      for op in available_ops(capabilities))
    return (
        "You are a helpful assistant. Use Thought/Action/Action Input/"
        "Observation format.\n"
        f"Available functions:\n{funcs}\n\n"
        "At every step reply with exactly:\n"
        "Thought: your reasoning about what to do next\n"
        "Action: one function name from the list above\n"
        'Action Input: a JSON object, e.g. {"inputs": ["@task"], '
        '"domain": "papers"}\n'
        '  - "inputs" (optional): positional input references — "@task" is '
        'the user\'s task text; "%t0", "%t1", ... are your earlier action '
        "results in order\n"
        "  - every other key is a function parameter (JSON value)\n\n"
        "You will then receive:\n"
        "Observation: the function result\n\n"
        f"You have at most {max_turns} action steps. Never invent an "
        "Observation. When you can answer, reply with exactly:\n"
        "Thought: I now know the final answer.\n"
        "Final Answer: the answer for the user\n"
    )


# ----------------------------------------------------------------- parsing

_LABEL = r"[*>`\t ]*"          # tolerate markdown emphasis around labels
_FINAL_RE = re.compile(
    rf"^\s*{_LABEL}(?:final[_ ]?answer|finish){_LABEL}:\s*(.*)$",
    re.I | re.M | re.S)
_ACTION_RE = re.compile(
    rf"^\s*{_LABEL}action{_LABEL}:\s*(.+?)\s*$", re.I | re.M)
_INPUT_MARK_RE = re.compile(
    rf"^\s*{_LABEL}action[_ ]?input{_LABEL}:\s*", re.I | re.M)
_THOUGHT_RE = re.compile(
    rf"thought{_LABEL}:\s*(.*?)(?=\n\s*{_LABEL}(?:action|final|observation)"
    r"\b|\Z)",
    re.I | re.S)
_OBS_RE = re.compile(rf"^\s*{_LABEL}observation{_LABEL}:\s*", re.I | re.M)


def cut_at_observation(text: str) -> str:
    """Drop everything from a (hallucinated) Observation marker onwards."""
    m = _OBS_RE.search(text)
    return text[: m.start()] if m else text


def parse_action_input(text: str) -> Optional[Any]:
    """Action Input -> python value; {} when absent; None when unparseable."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, re.S)              # embedded/fenced object
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def parse_step(text: str) -> Tuple[str, Any]:
    """Classify one model turn.

    Returns ("final", answer_text) | ("action", (op, action_input_text))
    | ("none", None).  A Final Answer marker wins over an Action one."""
    m = _FINAL_RE.search(text)
    if m:
        return "final", cut_at_observation(m.group(1)).strip()
    m = _ACTION_RE.search(text)
    if not m:
        return "none", None
    raw = m.group(1).strip().strip("*").strip()
    ident = re.match(r"[A-Za-z_][A-Za-z0-9_]*", raw)
    op = (ident.group(0) if ident else raw).upper()
    im = _INPUT_MARK_RE.search(text, m.end()) or _INPUT_MARK_RE.search(text)
    input_text = cut_at_observation(text[im.end():]).strip() if im else ""
    return "action", (op, input_text)


def extract_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()[:500]


# ---------------------------------------------------------- mock execution

def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:limit].rstrip("_") or "react_program"


def run_mock_world(nodes: List[Node], new_node: Node, instruction: str,
                   seed: str) -> Tuple[bool, str]:
    """Execute `new_node` on top of the trajectory's earlier nodes via the
    corpus Simulator (the same mock world the compiler eval runs in)."""
    name = _slug(instruction)
    prog = Program(name=name, description=(instruction or "").strip()[:400],
                   inputs=[{"name": GLOBAL_NAME, "type": "Str"}],
                   nodes=list(nodes) + [new_node], output=new_node.id)
    try:
        res = Simulator(Module(program=prog, meta={"name": name}),
                        seed=seed, jitter=0).run()
    except Exception as e:                          # noqa: BLE001
        return False, f"Error: simulator crashed: {e}"
    if res.status != "completed":
        return False, f"Error: execution failed: {res.output_digest}"
    return True, _digest(res.values.get(new_node.id), 200)


def _split_action_input(data: Any) -> Tuple[List[Any], Dict[str, Any],
                                            Optional[str]]:
    """-> (positional refs, params, error message)."""
    if isinstance(data, list):
        return list(data), {}, None
    if not isinstance(data, dict):
        return [], {}, ("Action Input must be a JSON object with an "
                        'optional "inputs" list plus function parameters.')
    refs = data.pop("inputs", data.pop("args", []))
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        return [], {}, '"inputs" must be a list of references.'
    return refs, dict(data), None


def execute_action(nodes: List[Node], op: str, input_text: str,
                   instruction: str, seed: str
                   ) -> Tuple[Optional[Node], str, Dict[str, Any]]:
    """Run one ReAct action in the mock world.

    Returns (node, observation, sanitized_action_input); node is None when
    the action did not execute (malformed input / arity / simulator
    failure) — the observation then carries the error the agent reacts to."""
    data = parse_action_input(input_text)
    if data is None:
        return None, ('Error: Action Input is not valid JSON — reply with a '
                      'JSON object, e.g. {"inputs": ["@task"], "domain": '
                      '"papers"}.'), {}
    refs, params, err = _split_action_input(data)
    if err:
        return None, f"Error: {err}", {}

    spec = ALL_OPS.get(op)                          # caller checked membership
    if spec is None:
        return None, f"Error: unknown function {op!r}.", {}
    if len(refs) < spec.min_inputs:
        return None, (f'Error: {op} needs >= {spec.min_inputs} input '
                      f'reference(s) in "inputs" (e.g. ["@task"]); got '
                      f"{len(refs)}."), {}
    if spec.max_inputs is not None and len(refs) > spec.max_inputs:
        return None, (f"Error: {op} takes <= {spec.max_inputs} input "
                      f"reference(s); got {len(refs)}."), {}

    defined = {GLOBAL_NAME} | {n.id for n in nodes}
    kept, dropped = [], []
    for r in refs:
        r = str(r).strip()
        (kept if r in defined else dropped).append(r)
    if len(kept) < spec.min_inputs:
        return None, (f'Error: {op} input references must be defined here '
                      f'("@task" or earlier "%tN" results); got '
                      f"{refs!r}."), {}

    node = Node(id=f"%t{len(nodes)}", op=op, inputs=kept, params=params)
    ok, obs = run_mock_world(nodes, node, instruction, seed)
    if not ok:
        return None, obs, {}
    if dropped:
        obs += "  (ignored invalid references: " + ", ".join(dropped) + ")"
    return node, obs, {"inputs": kept, **params}


# --------------------------------------------------------------- the agent

@dataclass
class ReactStep:
    step: int
    thought: str
    action: str
    action_input: Any
    observation: str
    executed: bool


@dataclass
class Trajectory:
    steps: List[ReactStep]
    final_answer: str
    finished: bool
    nodes: List[Node]

    @property
    def turns_used(self) -> int:
        return len(self.steps)


GenerateFn = Callable[[List[Dict[str, str]]], str]


class ReactAgent:
    """Multi-turn ReAct loop over a chat model.

    `generate_fn` wraps tokenizer+model (same decoding contract as
    src/compiler/train/infer.py); injecting a scripted callable makes the
    loop unit-testable without torch."""

    def __init__(self, generate_fn: GenerateFn, max_turns: int = 5,
                 max_consecutive_errors: int = 3):
        self.generate_fn = generate_fn
        self.max_turns = max_turns
        self.max_consecutive_errors = max_consecutive_errors

    def run(self, instruction: str, capabilities: Optional[List[str]],
            sample_id: str) -> Trajectory:
        functions = set(available_ops(capabilities))
        messages: List[Dict[str, str]] = [
            {"role": "system",
             "content": build_system_prompt(capabilities, self.max_turns)},
            {"role": "user", "content": (instruction or "").strip()}]
        steps: List[ReactStep] = []
        nodes: List[Node] = []
        errors = 0
        final_answer, finished = "", False
        seed = f"react:{sample_id}"

        while len(steps) < self.max_turns:
            completion = cut_at_observation(self.generate_fn(messages))
            kind, payload = parse_step(completion)
            thought = extract_thought(completion)

            if kind == "final":
                final_answer = payload.strip().strip("*").strip()
                finished = True
                break

            sanitized: Any = None
            if kind == "action":
                op, input_text = payload
                sanitized = {"raw": input_text}
                if op not in functions:
                    errors += 1
                    obs = (f"Error: unknown function {op!r}. Choose one "
                           f"of: {', '.join(sorted(functions))}.")
                else:
                    node, obs, sanitized = execute_action(
                        nodes, op, input_text, instruction, seed)
                    if node is None:
                        sanitized = {"raw": input_text}
                    else:
                        errors = 0
                        nodes.append(node)
                        steps.append(ReactStep(len(steps) + 1, thought, op,
                                               sanitized, obs, True))
                        messages.append({"role": "assistant",
                                         "content": completion})
                        messages.append({"role": "user",
                                         "content": f"Observation: {obs}"})
                        continue
                    errors += 1
            else:
                op = ""
                errors += 1
                obs = ("Error: no action found. Reply with 'Action: "
                       "<function>' and 'Action Input: {json}', or "
                       "'Final Answer: ...'.")

            steps.append(ReactStep(len(steps) + 1, thought, op, sanitized,
                                   obs, False))
            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            if errors >= self.max_consecutive_errors:
                break

        return Trajectory(steps, final_answer, finished, nodes)


# --------------------------------------------------------------- conversion

def trajectory_to_taskir(traj: Trajectory, instruction: str) -> str:
    """Replay the executed tool calls as a TaskIR program — the artifact
    our compiler emits up front.  Empty string when nothing executed (the
    eval then counts a parse failure, honestly)."""
    if not traj.nodes:
        return ""
    name = _slug(instruction)
    nodes = [Node(id=n.id, op=n.op, inputs=list(n.inputs),
                  params=dict(n.params)) for n in traj.nodes]
    prog = Program(name=name, description=(instruction or "").strip()[:400],
                   inputs=[{"name": GLOBAL_NAME, "type": "Str"}],
                   nodes=nodes, output=nodes[-1].id)
    mod = Module(program=prog, meta={
        "name": name,
        "provenance": {"source": "react_baseline", "view": "plan"}})
    # same cosmetic post-processing gate the compiler path applies
    return extract_taskir_text(to_text(mod))


def to_prediction(row: dict, traj: Trajectory) -> dict:
    """Corpus record -> prediction record for neural_compiler_eval."""
    instruction = row.get("instruction", "")
    return {
        "id": row.get("id", ""),
        "source": row.get("source", ""),
        "input_text": instruction.strip(),
        "output_text": trajectory_to_taskir(traj, instruction),
        "reference_plan": row.get("plan_target", ""),
        # ReAct-specific extras (ignored by neural_compiler_eval.evaluate):
        "reference_actions": [asdict(s) for s in traj.steps],
        "final_answer": traj.final_answer,
        "finished": traj.finished,
        "turns_used": traj.turns_used,
        "n_actions": len(traj.nodes),
    }


# --------------------------------------------------------------- evaluation

def _load_eval_module():
    """Import benchmark/neural_compiler_eval.py (a script, not a package)
    so the baseline is scored by the exact same code path."""
    path = ROOT / "benchmark" / "neural_compiler_eval.py"
    spec = importlib.util.spec_from_file_location("neural_compiler_eval",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("neural_compiler_eval", mod)
    spec.loader.exec_module(mod)
    return mod


def build_train_opseqs(nce, train_path) -> set:
    """Seen/unseen-composition support set, mirroring nce.main."""
    opseqs = set()
    p = pathlib.Path(train_path) if train_path else None
    if not p or not p.exists():
        return opseqs
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                opseqs.add("|".join(nce.ops_of(
                    nce.parse_text(r.get("plan_target", "")))))
            except (nce.TaskIRSyntaxError, ValueError):
                continue
    return opseqs


def react_stats(preds: List[dict]) -> dict:
    """Trajectory-level behavior of the baseline (beyond compiler metrics)."""
    n = len(preds)
    if not n:
        return {"trajectories": 0}
    turns = [p.get("turns_used", 0) for p in preds]
    acts = [p.get("n_actions", 0) for p in preds]
    err_steps = sum(1 for p in preds
                    for s in p.get("reference_actions", [])
                    if not s.get("executed"))
    return {
        "trajectories": n,
        "final_answer_pct": round(
            100 * sum(bool(p.get("finished")) for p in preds) / n, 2),
        "mean_turns": round(sum(turns) / n, 3),
        "mean_actions": round(sum(acts) / n, 3),
        "zero_action_pct": round(100 * sum(1 for a in acts if not a) / n, 2),
        "error_steps": err_steps,
        "error_step_pct": round(100 * err_steps / max(1, sum(turns)), 2),
    }


def write_report(report: dict, out_dir: pathlib.Path) -> None:
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rs = report["react"]
    L = ["# ReAct Baseline Eval",
         "",
         f"model: `{report['model']}` — predictions: {report['predictions']}"
         f" — train_opseqs: {report['train_opseqs']}",
         f"react: final-answer {rs.get('final_answer_pct', 0)}% | mean "
         f"turns {rs.get('mean_turns', 0)} | mean actions "
         f"{rs.get('mean_actions', 0)} | error steps "
         f"{rs.get('error_step_pct', 0)}%",
         "",
         "| source | n | parse% | valid% | exec% | opseq% | GES | P | R | "
         "F1 | generic-action% | unseen n | unseen opseq% |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for src, s in report["per_source"].items():
        u = s["unseen_composition"]
        L.append(
            f"| {src} | {s['n']} | {s['parse_rate_pct']} "
            f"| {s['validator_pass_pct']} | {s['execution_pct']} "
            f"| {s['opseq_exact_pct']} | {s['graph_edit_similarity_mean']} "
            f"| {s['skill_precision_mean']} | {s['skill_recall_mean']} "
            f"| {s['skill_f1_micro']} | {s['generic_action_rate_pct']} "
            f"| {u['n']} | {u['opseq_exact_pct']} |")
    (out_dir / "eval_report.md").write_text("\n".join(L) + "\n",
                                            encoding="utf-8")


# ---------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ReAct baseline on the compiler corpus test set")
    ap.add_argument("--model", required=True,
                    help="HF model dir (base or LoRA-merged checkpoint)")
    ap.add_argument("--test", required=True,
                    help="corpus jsonl (instruction + capabilities + "
                         "plan_target)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--train", default=None,
                    help="train.jsonl for seen/unseen composition "
                         "(default: train.jsonl next to --test)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-consecutive-errors", type=int, default=3)
    ap.add_argument("--dump-unseen", default=None,
                    help="default: <out>/unseen_dump.json")
    args = ap.parse_args()

    import torch                                       # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

    rows = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")

    def generate(messages: List[Dict[str, str]]) -> str:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, temperature=None)
        return tok.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    agent = ReactAgent(generate, max_turns=args.max_turns,
                       max_consecutive_errors=args.max_consecutive_errors)

    preds: List[dict] = []
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            traj = agent.run(row.get("instruction", ""),
                             row.get("capabilities"), row.get("id", ""))
            rec = to_prediction(row, traj)
            preds.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"[{i + 1}/{len(rows)}] id={rec['id']} "
                      f"actions={rec['n_actions']} "
                      f"turns={rec['turns_used']} "
                      f"finished={rec['finished']}", flush=True)

    nce = _load_eval_module()
    train_path = args.train or (pathlib.Path(args.test).parent
                                / "train.jsonl")
    train_opseqs = build_train_opseqs(nce, train_path)
    ev = nce.evaluate(preds, train_opseqs,
                      dump_unseen=args.dump_unseen
                      or str(out_dir / "unseen_dump.json"))
    report = {
        "model": args.model,
        "test": str(args.test),
        "max_turns": args.max_turns,
        "predictions": len(preds),
        "train_opseqs": len(train_opseqs),
        "react": react_stats(preds),
        "per_skill": ev.pop("per_skill"),
        "per_source": ev,
    }
    write_report(report, out_dir)

    print(json.dumps({k: v for k, v in report.items() if k != "per_skill"},
                     indent=2, ensure_ascii=False)[:2000], flush=True)
    print(f"predictions -> {pred_path}")
    print(f"report -> {out_dir / 'eval_report.json'} "
          f"({out_dir / 'eval_report.md'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
