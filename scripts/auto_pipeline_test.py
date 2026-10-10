"""True end-to-end: user types NL question → system auto-compiles → auto-executes → answer.

No manual TaskIR construction. The neural frontend decides what skills to use.
"""
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
print("全自动编译器: 自然语言 → 自动选技能 → 自动执行 → 答案")
print("=" * 60)

# Step 0: Define available capabilities (like declaring a target ISA)
CAPABILITIES = """[c0] name: search_housing_data
description: Search for housing prices and property data
canonical_skill: SEARCH

[c1] name: filter_by_criteria
description: Filter data by criteria like bedrooms, price range
canonical_skill: FILTER

[c2] name: find_minimum
description: Find the minimum value in a dataset
canonical_skill: MIN

[c3] name: find_maximum
description: Find the maximum value in a dataset
canonical_skill: MAX

[c4] name: calculate_average
description: Calculate the average of a dataset
canonical_skill: AVG

[c5] name: sort_data
description: Sort data by a key
canonical_skill: SORT"""

# User's natural language question (NO manual skill specification)
USER_TASK = "What is the cheapest house price in Seattle?"

print(f"\n[Input] {USER_TASK}")
print(f"\n[Available capabilities] (auto-provided by system):")
for line in CAPABILITIES.split("\n"):
    if "name:" in line:
        print(f"  {line.strip()}")

# Load the neural composer
print("\n[Loading neural frontend...]")
tok = AutoTokenizer.from_pretrained("weights/Qwen2.5-3B-Instruct", trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    "weights/Qwen2.5-3B-Instruct", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "runs/phase6b/composer_C2_s42/final")
model.eval()

# Step 1: Neural frontend compiles NL + capabilities → TaskIR
print("\n[Step 1] Neural frontend compiling...")
user_prompt = f"{USER_TASK}\n\nSelected capabilities:\n{CAPABILITIES}"
full_prompt = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": user_prompt}],
    tokenize=False, add_generation_prompt=True)

inputs = tok(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=512,
                           do_sample=False, temperature=None,
                           pad_token_id=tok.pad_token_id)
raw = tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  Generated TaskIR: {raw[:200]}...")

# Step 2: Parse TaskIR
print("\n[Step 2] Parsing TaskIR...")
try:
    taskir_text = extract_taskir_text(raw)
    module = parse_text(taskir_text)
    ops = [(n.id, n.op) for n in module.program.nodes]
    print(f"  Nodes: {ops}")
except Exception as e:
    print(f"  Parse failed: {e}")
    print("  (This is where the neural frontend needs to improve)")
    sys.exit(1)

# Step 3: Validate
print("\n[Step 3] Validation...")
report = validate(module)
print(f"  {'PASS' if report.valid else 'FAIL'}")
if not report.valid:
    for err in report.errors[:3]:
        print(f"  E: {err.msg}")

# Step 4: Execute with Forge
if report.valid:
    print("\n[Step 4] Forge execution...")
    forge = Forge()
    # Inject real housing data for the SEARCH executor
    housing_data = [
        {"price": 380000, "address": "789 Pine Rd", "bedrooms": 3},
        {"price": 420000, "address": "987 Maple Dr", "bedrooms": 2},
        {"price": 450000, "address": "123 Main St", "bedrooms": 3},
        {"price": 480000, "address": "555 Birch St", "bedrooms": 3},
        {"price": 510000, "address": "111 Wall St", "bedrooms": 4},
        {"price": 550000, "address": "321 Elm St", "bedrooms": 3},
        {"price": 620000, "address": "456 Oak Ave", "bedrooms": 4},
        {"price": 720000, "address": "654 Cedar Ln", "bedrooms": 5},
    ]
    forge.executors["api"].execute = lambda op, p, i: housing_data

    result = forge.run(module, {"@query": "Seattle houses"})
    print(f"  Values: {result.get('values', {})}")
    print(f"  Trace: {len(result.get('traces', []))} steps")

    # Extract the answer
    print(f"\n{'='*60}")
    print(f"Answer to '{USER_TASK}'")
    # Get the output node's value
    for nid, val in result.get("values", {}).items():
        if val is not None and not isinstance(val, (list, dict)):
            print(f"  → {val}")
    print(f"{'='*60}")
