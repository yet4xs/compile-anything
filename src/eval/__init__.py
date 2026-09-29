"""External evaluation package (data/external_benchmarks → EvalSample).

Completely independent of src/dataset/ (the training pipeline): different
schema, different adapters, and outputs that can never feed SFT (see
src/dataset/external_guard.py)."""
from .schema import EvalSample            # noqa: F401
