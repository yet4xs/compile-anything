"""Dataset adapters: raw benchmark files -> unified samples.

load(dir) reads whatever the downloader fetched; normalize(record) maps one
raw record onto the unified schema (src/dataset/schema.py). Adapters never
duplicate lifting logic — that stays in src/lifter/benchmark/."""
from .tooluse import ToolBenchAdapter, ApiBankAdapter          # noqa: F401
from .code import HumanEvalAdapter, MBPPAdapter                # noqa: F401
from .sql import SpiderAdapter, BirdAdapter                    # noqa: F401
from .rtl import VerilogEvalAdapter                            # noqa: F401

ADAPTERS = [ToolBenchAdapter(), ApiBankAdapter(), HumanEvalAdapter(),
            MBPPAdapter(), SpiderAdapter(), BirdAdapter(),
            VerilogEvalAdapter()]


def adapters_for(sources=None):
    return [a for a in ADAPTERS
            if sources is None or set(a.sources) & set(sources)]
