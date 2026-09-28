"""Dataset registry — real benchmark sources for the compiler corpus.

Each DatasetSpec records provenance (url, license, task type, expected
size) and a download plan consumed by src/dataset/downloader.py.

Availability note (recorded honestly in metadata.json at download time):
- reachable from this environment: GitHub raw hosts
- unreachable: huggingface.co (ToolBench full corpus, BIRD mirrors,
  AgentBench LMUData); BIRD itself is form-gated upstream.
Re-running scripts/download_datasets.py in an HF-capable environment
picks those up with no code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Plan = Tuple[str, dict]     # ("file", {...}) | ("github-dir", {...})


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_url: str
    license: str
    task_type: str                 # tooluse | code | sql | rtl
    adapter: str
    expected_size: int
    plan: List[Plan] = field(default_factory=list)
    notes: str = ""


def _f(url: str, dest: str) -> Plan:
    return ("file", {"url": url, "dest": dest})


def _gh(repo: str, ref: str, prefix: str, suffixes: List[str],
        dest_dir: Optional[str] = None) -> Plan:
    return ("github-dir", {"repo": repo, "ref": ref, "prefix": prefix,
                           "suffixes": tuple(suffixes),
                           "dest_dir": dest_dir or prefix.rstrip("/").split("/")[-1]})


REGISTRY: dict = {s.name: s for s in [
    DatasetSpec(
        name="humaneval",
        source_url="https://github.com/openai/human-eval",
        license="MIT",
        task_type="code", adapter="humaneval", expected_size=164,
        plan=[_f("https://raw.githubusercontent.com/openai/human-eval/master/"
                 "data/HumanEval.jsonl.gz", "HumanEval.jsonl.gz")],
        notes="164 real prompts + canonical solutions"),
    DatasetSpec(
        name="mbpp",
        source_url="https://github.com/nahid-hasan-raju/mbpp-dataset "
                   "(mirror of google-research MBPP)",
        license="CC-BY-4.0 (upstream MBPP)",
        task_type="code", adapter="mbpp", expected_size=427,
        plan=[_f("https://raw.githubusercontent.com/nahid-hasan-raju/"
                 "mbpp-dataset/main/data/cleaned_mbpp.csv",
                 "cleaned_mbpp.csv")],
        notes="cleaned split of the real MBPP set via GitHub mirror"),
    DatasetSpec(
        name="spider",
        source_url="https://github.com/taoyds/spider",
        license="CC-BY-SA-4.0",
        task_type="sql", adapter="spider", expected_size=8034,
        plan=[
            _f("https://raw.githubusercontent.com/taoyds/spider/master/"
               "evaluation_examples/examples/train_spider.json",
               "train_spider.json"),
            _f("https://raw.githubusercontent.com/taoyds/spider/master/"
               "evaluation_examples/examples/dev.json", "dev.json"),
        ],
        notes="official Spider train (7000) + dev (1034)"),
    DatasetSpec(
        name="toolbench",
        source_url="https://github.com/OpenBMB/ToolBench",
        license="Apache-2.0 (code); see repo for data terms",
        task_type="tooluse", adapter="toolbench", expected_size=18,
        plan=[_gh("OpenBMB/ToolBench", "master",
                  "data_example/answer/", [".json"], dest_dir="answer")],
        notes="official repo's real G1/G2/G3 answer examples; the FULL "
              "corpus is HF-hosted (blocked here) — re-run downloader "
              "in an HF-capable env to scale tool-use volume"),
    DatasetSpec(
        name="verilogeval",
        source_url="https://github.com/NVlabs/verilog-eval",
        license="see upstream repository (NVIDIA)",
        task_type="rtl", adapter="verilogeval", expected_size=312,
        plan=[_gh("NVlabs/verilog-eval", "main",
                  "dataset_code-complete-iccad2023/",
                  ["_prompt.txt", "_test.sv"], dest_dir="code-complete"),
              _gh("NVlabs/verilog-eval", "main",
                  "dataset_spec-to-rtl/", ["_prompt.txt", "_test.sv"],
                  dest_dir="spec-to-rtl")],
        notes="312 real RTL problems (HDLBits-derived prompts)"),
    # ---- registered but gated / unreachable from this environment ----
    DatasetSpec(
        name="bird",
        source_url="https://bird-bench.github.io/",
        license="request-gated upstream (Google form)",
        task_type="sql", adapter="bird", expected_size=0, plan=[],
        notes="UNAVAILABLE: upstream requires a request form"),
    DatasetSpec(
        name="apibank",
        source_url="https://github.com/ShishirPatil/apibank",
        license="Apache-2.0",
        task_type="tooluse", adapter="apibank", expected_size=0, plan=[],
        notes="UNAVAILABLE: upstream repo not found at probed paths; "
              "adapter kept for when a mirror is located"),
    DatasetSpec(
        name="agentbench",
        source_url="https://github.com/THUDM/AgentBench",
        license="see upstream",
        task_type="tooluse", adapter="agentbench", expected_size=0, plan=[],
        notes="UNAVAILABLE: data ships via HF LMUData (blocked here)"),
    DatasetSpec(
        name="hdlbits",
        source_url="https://hdlbits.01xz.net/",
        license="no redistributable dump",
        task_type="rtl", adapter="hdlbits", expected_size=0, plan=[],
        notes="UNAVAILABLE: covered indirectly via verilogeval "
              "(HDLBits-derived)"),
]}


def available() -> List[str]:
    return [n for n, s in REGISTRY.items() if s.plan]


def unavailable() -> List[str]:
    return [n for n, s in REGISTRY.items() if not s.plan]
