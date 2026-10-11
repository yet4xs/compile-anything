"""Phase 7C-2 Task 1: TokenMask vs the REAL Qwen tokenizer (server-only).

Skipped automatically when weights/Qwen2.5-3B-Instruct is absent (e.g. on a
laptop). Verifies the byte-level decode, special-token handling and the
roundtrip property: tokenizing any valid corpus program yields tokens whose
texts ALL pass the FSM when fed in sequence.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.online_constrained_decoder import (
    TokenMask, TaskIRGenFSM, GenStats)
from src.ir.parser import parse_text
from src.ir.taskir import to_text
from src.validator.validator import validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_PATH = os.path.join(ROOT, "weights", "Qwen2.5-3B-Instruct")


@unittest.skipIf(not os.path.isdir(TOK_PATH), "real tokenizer not present")
class TestRealTokenizerMask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
        cls.mask = TokenMask(cls.tok)

    def _corpus(self, n=25):
        out = []
        path = os.path.join(ROOT, "data", "compiler_corpus_v4", "test.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                text = r.get("plan_text") or r.get("plan_target") or ""
                if not text:
                    continue
                try:
                    if validate(parse_text(text)).valid:
                        out.append(text)
                except Exception:
                    continue
                if len(out) >= n:
                    break
        return out

    def test_byte_decode_matches_tokenizer(self):
        """token_texts[i] == tok.decode([i]) for regular tokens."""
        import random
        random.seed(7)
        vocab = len(self.tok)
        special = set(self.tok.all_special_ids)
        for tid in random.sample(range(vocab), 3000):
            if tid in special or not self.mask.token_texts[tid]:
                continue
            self.assertEqual(
                self.mask.token_texts[tid], self.tok.decode([tid]),
                f"mismatch at id {tid}")

    def test_special_tokens_banned_everywhere(self):
        """Added/special tokens must never be allowed by the mask (EOS is
        handled separately by the processor via fsm.eos_ok())."""
        fsm = TaskIRGenFSM(level=1)
        for sid in self.tok.all_special_ids:
            self.assertNotIn(sid, self.mask.allowed_ids(fsm))
        # and deep in a program
        fsm2 = TaskIRGenFSM(level=3)
        for c in "inputs: @task: Str\n%c0 = SEARCH(@task":
            fsm2.feed(c)
        for sid in self.tok.all_special_ids:
            self.assertNotIn(sid, self.mask.allowed_ids(fsm2))

    def test_non_ascii_tokens_banned(self):
        """Tokens whose text contains non-ASCII (Chinese etc.) can never be
        legal TaskIR at any reachable state."""
        fsm = TaskIRGenFSM(level=1)
        allowed = set(self.mask.allowed_ids(fsm))
        bad = [t for t in allowed
               if any(ord(ch) > 127 for ch in self.mask.token_texts[t])]
        self.assertEqual(bad, [], f"{len(bad)} non-ASCII tokens allowed")

    def test_tokenization_roundtrip_all_pass_fsm(self):
        """THE consistency check: tokenize(to_text(valid program)) and feed
        each token's text through the FSM in sequence — every token must be
        acceptable (this exercises multi-char/multi-state tokens such as
        '"])\n' and UTF-8 edge cases end to end)."""
        for text in self._corpus(25):
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            for lvl in (1, 2, 3, 4):
                fsm = TaskIRGenFSM(level=lvl)
                for tid in ids:
                    txt = self.mask.token_texts[tid]
                    self.assertTrue(
                        self.mask._token_ok(fsm, txt),
                        f"L{lvl}: token {txt!r} (id {tid}) rejected in "
                        f"{text[:60]!r}")
                    for ch in txt:          # advance the live FSM
                        self.assertTrue(fsm.feed(ch), f"feed {ch!r} failed")

    def test_eos_allowed_only_via_processor(self):
        """allowed_ids() returns no EOS id even when the program is complete
        — the processor adds real eos ids gated by fsm.eos_ok()."""
        fsm = TaskIRGenFSM(level=1)
        for c in "; TaskIR v0.1\ninputs: @task: Str\n%c0 = FETCH(@task)\nreturn %c0":
            fsm.feed(c)
        self.assertTrue(fsm.eos_ok())
        ids = self.mask.allowed_ids(fsm)
        self.assertNotIn(self.tok.eos_token_id, ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
