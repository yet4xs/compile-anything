"""Validator-Guided Neural Repair — zero-training repair prompting with frozen Qwen Composer.

Input to the LLM:
  - Original NL task
  - Relevant capability metadata
  - Validator error code + message
  - Faulty node description (op, params, inputs)
  - SSA dependency slice (upstream providers, downstream consumers)
  - Allowed patch operations (Patch ISA)

Output from the LLM:
  - A single JSON patch (NOT a full TaskIR program)

Processing:
  Failed TaskIR → Validator → SSA Slice → Neural Patch → Boundary Check → Apply → Full Validator → Accept/Reject
"""
from __future__ import annotations
import json, os, re, sys, copy
from typing import List, Dict, Optional, Any
from src.ir.taskir import Module
from src.validator.validator import validate
from src.repair.ssa_slicer import extract_slice, SSASlice
from src.repair.patch_ir import apply_patch, PatchResult

REPAIR_SYSTEM = (
    "You are a TaskIR repair engine. Given a validator error, the faulty node, "
    "and its SSA dependency context, output a single JSON patch to fix the error.\n"
    "You must ONLY output a JSON object with 'op' and required fields. "
    "Do NOT output the entire program.\n\n"
    "Allowed operations:\n"
    '{"op": "REPLACE_INPUT", "node_id": "...", "position": 0, "new_reference": "..."}\n'
    '{"op": "REPLACE_NODE", "node_id": "...", "new_op": "...", "new_params": {...}}\n'
    '{"op": "INSERT_BEFORE", "anchor_id": "...", "new_id": "%new1", "new_op": "...", "new_params": {...}, "new_inputs": [...]}\n'
    '{"op": "UPDATE_ANNOTATION", "node_id": "...", "field": "params", "value": {...}}\n'
    '{"op": "RENAME_SSA", "old_id": "...", "new_id": "..."}\n'
    '{"op": "UPDATE_OUTPUT", "new_output": "..."}'
)


def build_repair_prompt(task: str, module: Module, report, slice_info: SSASlice,
                        capabilities: str = "") -> str:
    """Build the repair prompt with SSA context."""
    # Describe the faulty nodes
    node_desc = []
    for nid in sorted(slice_info.error_nodes):
        node = next((n for n in module.program.nodes if n.id == nid), None)
        if node:
            node_desc.append(
                f"  {node.id} = {node.op}(inputs={node.inputs}, params={dict(node.params) if node.params else {}})")

    # Describe upstream providers (context, not to be modified)
    upstream_desc = []
    for eid, ups in slice_info.upstream.items():
        for up_id in ups:
            node = next((n for n in module.program.nodes if n.id == up_id), None)
            if node:
                upstream_desc.append(
                    f"  {node.id} = {node.op}(output_type={node.output_type}) [FROZEN - do not modify]")

    # Describe downstream consumers
    downstream_desc = []
    for did in sorted(slice_info.downstream):
        node = next((n for n in module.program.nodes if n.id == did), None)
        if node:
            downstream_desc.append(
                f"  {node.id} = {node.op}(inputs={node.inputs}) [FROZEN]")

    # Describe errors
    error_desc = [f"  {e.code}: {e.msg}" for e in report.errors[:5]]

    # List available SSA references
    all_ids = [n.id for n in module.program.nodes if n.id not in slice_info.error_nodes]
    inputs = [f"@{inp['name'].lstrip('@')}" for inp in module.program.inputs]
    valid_refs = all_ids + inputs

    prompt = f"""Task: {task}

The following TaskIR program has validation errors:

Nodes:
{chr(10).join(node_desc)}

Upstream providers (do NOT modify):
{chr(10).join(upstream_desc) if upstream_desc else "  (none)"}

Downstream consumers (do NOT modify):
{chr(10).join(downstream_desc) if downstream_desc else "  (none)"}

Validator errors:
{chr(10).join(error_desc)}

Valid SSA references you can use: {valid_refs}
Program output: {slice_info.program_output}
Repair scope: {sorted(slice_info.repairable)} (max {slice_info.max_repair_nodes} nodes)
{"WARNING: slice overflow, repair may need to be split" if slice_info.overflow else ""}
{f"Capabilities: {capabilities}" if capabilities else ""}

Output ONE JSON patch to fix the error. Only output the JSON, nothing else."""

    return prompt


def parse_patch(text: str) -> Optional[Dict]:
    """Extract JSON patch from model output."""
    # Try direct JSON parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```", r"(\{.*?\})"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

    return None


def neural_repair(task: str, module: Module, report, capabilities: str,
                  model, tok, max_attempts: int = 2,
                  max_repair_nodes: int = 3) -> Dict:
    """Attempt to repair a failed TaskIR module using validator-guided patches.

    Returns dict with:
      - success (bool): final module passes validator
      - module: the repaired module (deep copy, original untouched)
      - attempts: list of patch attempts with results
      - total_tokens: estimated tokens used
      - changed_nodes: list of node IDs that were modified
    """
    import torch

    original = copy.deepcopy(module)
    current = copy.deepcopy(module)
    attempts = []
    total_tokens = 0
    changed_nodes = []

    for attempt in range(max_attempts):
        try:
            report = validate(current)
            if report.valid:
                return {
                    "success": True,
                    "module": current,
                    "attempts": attempts,
                    "total_tokens": total_tokens,
                    "changed_nodes": changed_nodes,
                }
        except Exception:
            pass

        # Extract SSA slice
        slice_info = extract_slice(current, report, max_repair_nodes=max_repair_nodes)

        # Build prompt
        prompt_text = build_repair_prompt(task, current, report, slice_info, capabilities)
        full_prompt = tok.apply_chat_template(
            [{"role": "system", "content": REPAIR_SYSTEM},
             {"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True)

        # Generate
        inputs = tok(full_prompt, return_tensors="pt", truncation=True,
                     max_length=2048).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256,
                                    do_sample=False, temperature=None,
                                    pad_token_id=tok.pad_token_id)
        output_text = tok.decode(output[0][input_len:], skip_special_tokens=True)
        output_tokens = output.shape[1] - input_len
        total_tokens += input_len + output_tokens

        # Parse patch
        patch = parse_patch(output_text)
        if patch is None:
            attempts.append({
                "attempt": attempt + 1,
                "success": False,
                "error": "failed to parse JSON patch",
                "raw_output": output_text[:200],
            })
            continue

        # Apply patch
        result = apply_patch(current, patch, slice_info)
        attempts.append({
            "attempt": attempt + 1,
            "success": result.success,
            "patch": patch,
            "errors": result.errors,
            "changed_nodes": result.changed_nodes,
            "output_tokens": output_tokens,
        })

        if result.success and result.module:
            current = result.module
            changed_nodes.extend(result.changed_nodes)

    # Final validation
    try:
        final_valid = validate(current).valid
    except Exception:
        final_valid = False

    return {
        "success": final_valid,
        "module": current if final_valid else None,
        "attempts": attempts,
        "total_tokens": total_tokens,
        "changed_nodes": changed_nodes,
    }
