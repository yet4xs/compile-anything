"""Benchmark lifter registry — importing this package registers all
lifters with src/lifter/benchmark/base.lift_sample."""
from .base import BenchmarkLifter, LIFTERS, register, lift_sample  # noqa: F401
from . import toolbench, humaneval, spider, rtl                    # noqa: F401
