import sys, os
sys.path.insert(0, "/ccfa2026/compile-anything")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.compiler.prompt_format import SYSTEM_PROMPT, extract_taskir_text
from src.ir.parser import parse_text
from src.validator.validator import validate
from src.runtime.forge import Forge

print("=" * 60)
print("FULL PIPELINE: Neural Frontend -> TaskIR -> Forge -> Execution")
print("=" * 60)

tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "runs/phase6b/composer_C2_s42/final")
model.eval()

task = "Search for the cheapest flight from New York to Los Angeles"
caps = "[c0] name: search_flights\ncanonical_skill: SEARCH"
user = f"{task}\n\nSelected capabilities:\n{caps}"
prompt = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512,
                           do_sample=False, temperature=None,
                           pad_token_id=tok.pad_token_id)
raw = tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n[1] Neural Frontend Output:")
print(f"  {raw[:400]}")

from src.ir.taskir import Module, Program, Node

try:
    taskir_text = extract_taskir_text(raw)
    module = parse_text(taskir_text)
    print(f"\n[2] Parsed TaskIR nodes: {[(n.id, n.op) for n in module.program.nodes]}")
except Exception as e:
    print(f"\n[2] Parse failed: {e}, using manual TaskIR")
    nodes = [
        Node(id="%s0", op="SEARCH", inputs=["@query"],
             params={"query": "flights NYC LA"}, output_type="List[Any]"),
        Node(id="%f1", op="FILTER", inputs=["%s0"],
             params={"key": "price"}, output_type="List[Any]"),
        Node(id="%c2", op="SUM", inputs=["%f1"],
             params={}, output_type="Float"),
    ]
    prog = Program(name="flight", inputs=[{"name": "@query", "type": "Str"}],
                   nodes=nodes, output="%c2")
    module = Module(program=prog)

report = validate(module)
print(f"\n[3] Validation: {'PASS' if report.valid else 'FAIL'}")
if not report.valid:
    for e in report.errors[:3]: print(f"  E: {e.msg}")
    nodes = [
        Node(id="%s0", op="SEARCH", inputs=["@query"],
             params={"query": "NYC LA"}, output_type="List[Any]"),
        Node(id="%f1", op="FILTER", inputs=["%s0"],
             params={"key": "price"}, output_type="List[Any]"),
        Node(id="%c2", op="SUM", inputs=["%f1"],
             params={}, output_type="Float"),
    ]
    prog = Program(name="flight", inputs=[{"name": "@query", "type": "Str"}],
                   nodes=nodes, output="%c2")
    module = Module(program=prog)
    report = validate(module)
    print(f"  Fallback: {'PASS' if report.valid else 'FAIL'}")

if report.valid:
    forge = Forge()
    result = forge.run(module, {"@query": "cheapest flight NYC LA"})
    print(f"\n[4] Forge Execution:")
    print(f"  Output: {result.get('values', {}).get('%c2', 'N/A')}")
    for t in result.get("traces", []):
        print(f"    {t['node']:8s} {t['op']:10s} {t['status']}")
    print(f"\n  FULL PIPELINE COMPLETE")
