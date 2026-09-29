"""Firewall tests: external benchmark paths must be blocked everywhere,
training paths must pass, and every corpus-builder entry point must be
wired through the guard."""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dataset.external_guard import (is_external_path,      # noqa: E402
                                        assert_not_external)

EXT = ["data/external_benchmarks/bfcl_v4/BFCL_v4_simple.json",
       "data/external_benchmarks/agentboard/data/tool-query/test.jsonl",
       "D:/x/compile-anything/data/external_benchmarks/rtl_repo/train.parquet",
       "external_benchmarks/toolbench_full/toolbench.jsonl"]
OK = ["data/compiler_corpus_v3/train.jsonl", "data/raw/spider/train_spider.json",
      "weights/Qwen2.5-3B-Instruct/config.json"]


class TestExternalGuard(unittest.TestCase):
    def test_external_paths_detected(self):
        for p in EXT:
            self.assertTrue(is_external_path(p), p)

    def test_training_paths_pass(self):
        for p in OK:
            self.assertFalse(is_external_path(p), p)

    def test_assert_raises(self):
        for p in EXT:
            with self.assertRaises(RuntimeError):
                assert_not_external(p)

    def test_assert_passes_normal(self):
        assert_not_external(OK[0])
        assert_not_external(OK)

    def test_corpus_builders_are_wired(self):
        """The four builder entry points must import/call the guard."""
        import src.compiler.train.dataset as ds
        import src.dataset.pipeline as pl
        self.assertTrue(hasattr(ds, "assert_not_external"))
        self.assertTrue(hasattr(pl, "assert_not_external"))
        for script in ("scripts/build_real_corpus.py",
                       "scripts/prepare_sft.py",
                       "scripts/build_compiler_corpus.py"):
            src = pathlib.Path(__file__).resolve().parents[1] / script
            if src.exists():
                self.assertIn("assert_not_external",
                              src.read_text(encoding="utf-8"),
                              f"{script} not wired to the external guard")

    def test_end_to_end_block_on_dataset_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "external_benchmarks" / "x" / "train.jsonl"
            p.parent.mkdir(parents=True)
            p.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                from src.compiler.train.dataset import iter_corpus_records
                list(iter_corpus_records([str(p)]))


if __name__ == "__main__":
    unittest.main()
