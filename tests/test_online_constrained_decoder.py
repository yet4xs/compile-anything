"""Phase 7C-1 Task 2 tests: online constraint decoder (CPU only).

Covers: FSM acceptance over the whole valid corpus (G1+G2), rejection of the
observed failure modes (prose openers, bad ops, undefined refs, duplicate
defs, arity overflow, truncation), token masking with a mock tokenizer, and
the LogitsProcessor mask/force/deadlock bookkeeping.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.ir.parser import parse_text
from src.ir.taskir import to_text, module_from_dict
from src.validator.validator import validate
from src.compiler.online_constrained_decoder import (
    TaskIRGenFSM, TokenMask, ConstrainedLogitsProcessor, GenStats,
    _OP_TRIE, VALID_OPS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOOD = """; TaskIR v0.1  module=demo
inputs: @task: Str

%c0 = SEARCH(@task, domain=web)  -> List
%c1 = EXTRACT(@task)  -> Str
%c2 = GENERATE(prompt=%c1, context=%c0, n=3)  -> Str  retry(on=error, max=2)
%c3 = VERIFY(%c2, check="factual_consistency")  -> Bool
%c4 = SELECT(%c3, %c2)  -> Dict

return %c4
"""


def _walk(fsm: TaskIRGenFSM, text: str):
    for i, c in enumerate(text):
        if not fsm.feed(c):
            return i, c
    return None, None


def _valid_corpus(n=60):
    path = os.path.join(ROOT, "data", "compiler_corpus_v4", "test.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("plan_text") or r.get("plan_target") or ""
            if not text:
                continue
            try:
                mod = parse_text(text)
                if validate(mod).valid:
                    out.append(text)
            except Exception:
                continue
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------- G1 grammar

def test_fsm_accepts_valid_corpus_g1():
    corpus = _valid_corpus(60)
    assert len(corpus) >= 20
    for text in corpus:
        fsm = TaskIRGenFSM(level=1)
        i, c = _walk(fsm, text)
        assert i is None, f"rejected at {i} ({c!r}): {text[:120]!r}"
        assert fsm.node_count >= 1
        assert fsm.return_line_done


def test_fsm_accepts_valid_corpus_g2():
    """Valid corpus programs satisfy the dynamic SSA constraints at EVERY
    frozen arm level (G2a=2, G2=3, G3=4 — corpus input types are canonical)."""
    corpus = _valid_corpus(30)
    for text in corpus:
        for lvl in (2, 3, 4):
            fsm = TaskIRGenFSM(level=lvl)
            i, c = _walk(fsm, text)
            assert i is None, f"L{lvl} rejected at {i} ({c!r}): {text[:120]!r}"


def test_fsm_accepts_handwritten_examples():
    import glob
    for p in glob.glob(os.path.join(ROOT, "data", "taskir", "examples",
                                    "*.json")):
        mod = module_from_dict(json.load(open(p, encoding="utf-8")))
        assert validate(mod).valid
        for lvl in (1, 2, 3, 4):
            fsm = TaskIRGenFSM(level=lvl)
            i, c = _walk(fsm, to_text(mod))
            assert i is None, f"{os.path.basename(p)} L{lvl} at {i} ({c!r})"


def test_prose_opener_rejected():
    """The dominant parse failure ('no nodes found') is prose instead of
    TaskIR — G1 must reject it at the first character."""
    fsm = TaskIRGenFSM(level=1)
    ok = fsm.feed("H")
    assert ok is False


def test_bad_op_rejected():
    fsm = TaskIRGenFSM(level=1)
    for c in "%c0 = SEARCC(":
        fsm.feed(c)
    # 'C' (second C) is not a valid op continuation
    # walk manually to find the rejection point
    fsm2 = TaskIRGenFSM(level=1)
    rejected_at = None
    for i, c in enumerate("%c0 = SEARCC(@task)"):
        if not fsm2.feed(c):
            rejected_at = i
            break
    assert rejected_at is not None


def test_truncated_program_cannot_emit_eos():
    """Unclosed paren / missing return: EOS must stay banned."""
    fsm = TaskIRGenFSM(level=1)
    _walk(fsm, GOOD.replace("return %c4\n", ""))
    assert not fsm.eos_ok()
    fsm2 = TaskIRGenFSM(level=1)
    _walk(fsm2, GOOD[:GOOD.index(")")])   # cut mid-node
    assert not fsm2.eos_ok()


def test_eos_legal_after_complete_program():
    fsm = TaskIRGenFSM(level=1)
    _walk(fsm, GOOD)
    assert fsm.eos_ok()


# ---------------------------------------------------------------- G2 SSA

def test_g2_rejects_undefined_ref():
    bad = GOOD.replace("SELECT(%c3, %c2)", "SELECT(%c3, %ghost)")
    fsm = TaskIRGenFSM(level=2)
    i, c = _walk(fsm, bad)
    assert i is not None, "G2 accepted an undefined reference"
    # G1 (grammar only) accepts the same text
    fsm1 = TaskIRGenFSM(level=1)
    i1, _ = _walk(fsm1, bad)
    assert i1 is None


def test_g2_duplicate_def_forces_extension():
    # insert a second '%c0' node BEFORE the return line
    text = GOOD.replace(
        "%c4 = SELECT(%c3, %c2)  -> Dict",
        "%c4 = SELECT(%c3, %c2)  -> Dict\n%c0 = SEARCH(@task, domain=web)")
    fsm = TaskIRGenFSM(level=2)
    i, c = _walk(fsm, text)
    assert i is not None            # terminating '%c0 ' was refused
    assert fsm.stats.dup_forced_extensions >= 1
    # extending the name is legal
    text2 = GOOD.replace(
        "%c4 = SELECT(%c3, %c2)  -> Dict",
        "%c4 = SELECT(%c3, %c2)  -> Dict\n%c05 = SEARCH(@task, domain=web)")
    fsm2 = TaskIRGenFSM(level=2)
    i2, _ = _walk(fsm2, text2)
    assert i2 is None


def test_g2_return_ref_must_be_defined():
    bad = GOOD.replace("return %c4", "return %zz")
    fsm = TaskIRGenFSM(level=2)
    i, _ = _walk(fsm, bad)
    assert i is not None


def test_g2_arity_cap():
    from src.isa.registry import get as get_skill
    spec = get_skill("GENERATE")
    if not (spec and spec.max_inputs):
        return
    over = GOOD.replace("GENERATE(prompt=%c1, context=%c0, n=3)",
                        "GENERATE(%c1, %c0, %c2)")
    fsm = TaskIRGenFSM(level=2)
    i, _ = _walk(fsm, over)
    assert i is not None, "arity cap not enforced"
    fsm1 = TaskIRGenFSM(level=1)
    i1, _ = _walk(fsm1, over)
    assert i1 is None   # grammar alone accepts it


# ---------------------------------------------------------------- TokenMask

class MockTokenizer:
    """Tiny byte-level-BPE-shaped vocab for mask logic tests."""

    def __init__(self, pieces):
        self.pieces = pieces

    def __len__(self):
        return len(self.pieces)

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[i] for i in ids]


def _mock_mask():
    pieces = ["%", "1", "Ġ=", "ĠSEARCH", "(", ")", "Ġreturn", "Ċ", "@task",
              "%c0", "%c1", "garbage", " inputs:", "%c0", "H", "Ġdomain",
              "=", '"', "web", "Ġ-", "> ", "List", "ĠEXTRACT"]
    return TokenMask(MockTokenizer(pieces))


def test_token_mask_line_start():
    m = _mock_mask()
    fsm = TaskIRGenFSM(level=1)
    ids = m.allowed_ids(fsm)
    texts = {m.token_texts[i] for i in ids}
    assert " inputs:" in texts and "%" in texts and "\n" in texts
    assert "garbage" not in texts and "H" not in texts


def test_comment_newline_is_legal_exit():
    """Regression: '\\n' was banned inside comments, forcing the model into
    garbage at every line break. The terminator must be emittable."""
    m = _mock_mask()
    fsm = TaskIRGenFSM(level=1)
    for c in "; TaskIR v0.1  module=demo":
        fsm.feed(c)
    ids = set(m.allowed_ids(fsm))
    nl_id = m.token_texts.index("\n")
    assert nl_id in ids                     # '\n' ends the comment legally
    assert fsm.feed("\n")                   # and feeding it closes the line
    assert fsm.state == TaskIRGenFSM.S_LINE_START


def test_comment_cap_forces_exit():
    """Runaway comments burn the token budget; past COMMENT_CAP only the
    newline is legal. feed() itself stays permissive (FSM losslessness)."""
    from src.compiler.online_constrained_decoder import COMMENT_CAP
    fsm = TaskIRGenFSM(level=1)
    fsm.feed(";")
    for _ in range(COMMENT_CAP + 10):
        assert fsm.feed("x") is True        # feed never rejects comment text
    assert fsm.comment_len >= COMMENT_CAP
    spec = fsm.allowed()
    assert spec == {"kind": "chars", "set": {"\n"}}
    assert fsm.feed("\n")
    assert fsm.state == TaskIRGenFSM.S_LINE_START
    assert fsm.allowed() != {"kind": "chars", "set": {"\n"}}  # reset


def test_token_mask_op_state():
    m = _mock_mask()
    fsm = TaskIRGenFSM(level=1)
    for c in "%1 = ":
        fsm.feed(c)
    ids = m.allowed_ids(fsm)
    texts = {m.token_texts[i] for i in ids}
    assert " SEARCH" in texts or "SEARCH" in texts
    assert ")" not in texts and "%" not in texts


def test_token_mask_g2_operand_trie():
    m = _mock_mask()
    fsm = TaskIRGenFSM(level=2)
    for c in "inputs: @task: Str\n%c0 = SEARCH(":
        fsm.feed(c)
    ids = m.allowed_ids(fsm)
    texts = {m.token_texts[i] for i in ids}
    # only defined ids are legal operands here: '@task' yes, any '%' no
    assert "@task" in texts
    assert not any(t.startswith("%") for t in texts)


def test_token_ok_full_token_crosses_states():
    """A single token may cross a state boundary ('%c0' = id + terminator is
    NOT legal mid-id, but '(' after a terminal op in one token would be)."""
    m = _mock_mask()
    fsm = TaskIRGenFSM(level=1)
    for c in "%c0 = SEARCH":
        fsm.feed(c)
    assert m._token_ok(fsm, "(") is True
    assert m._token_ok(fsm, ")") is False
    assert m._token_ok(fsm, "H") is False


# ---------------------------------------------------------------- processor

def _mock_processor():
    pieces = ["%", "1", "Ġ=", "ĠSEARCH", "(", ")", "Ċ", "@task", "garbage",
              "H", " inputs:", "Ġreturn"]
    # eos id 99 collides with no vocab piece
    proc = ConstrainedLogitsProcessor(MockTokenizer(pieces), level=1,
                                      eos_token_ids=[99])
    return proc


def test_processor_masks_top1_and_records_force():
    proc = _mock_processor()
    vocab = proc.mask.vocab_size
    prompt = torch.tensor([[5, 5, 5]])
    scores = torch.full((1, vocab), -10.0)
    scores[0, 9] = 5.0          # 'H' — illegal at line start
    scores[0, 10] = 4.0         # ' inputs:' — legal
    out = proc(prompt, scores)
    assert torch.isinf(out[0, 9]) and out[0, 10] == 4.0
    assert proc.stats.forced_steps == 1
    # legal tokens at line start in this mock: '%', '@task', '\n',
    # ' inputs:', ' return'  (no eos — program not complete)
    assert proc.stats.banned_counts[-1] == vocab - 5


def test_processor_deadlock_fallback():
    proc = _mock_processor()
    vocab = proc.mask.vocab_size
    # drive into a dead branch: operand trie empty at G2 impossible here,
    # so craft: return ref pointing nowhere via mock state
    proc.fsm.state = TaskIRGenFSM.S_RETURN_REF
    proc.fsm.buf = "%zz"
    proc.fsm.level = 2
    proc.fsm.defined = ["@task"]
    prompt = torch.tensor([[5]])
    scores = torch.full((1, vocab), -1.0)
    out = proc(prompt, scores)
    assert torch.equal(out, scores)          # untouched fallback
    assert proc.stats.deadlocks


def test_processor_feeds_generated_tokens():
    proc = _mock_processor()
    vocab = proc.mask.vocab_size
    prompt = torch.tensor([[5, 5]])
    scores = torch.full((1, vocab), -10.0)
    scores[0, 0] = 3.0                       # '%' legal at line start
    proc(prompt, scores)
    # simulate generate() appending '%'
    ids2 = torch.tensor([[5, 5, 0]])
    scores2 = torch.full((1, vocab), -10.0)
    scores2[0, 1] = 3.0                      # '1'
    proc(ids2, scores2)
    assert proc.fsm.state == TaskIRGenFSM.S_NODE_ID
    assert proc.fsm.buf == "%"


# =================================================================
# Phase 7C-2 regressions: arm isolation, G3 type trie, EOS sentinel
# =================================================================

def test_g2a_has_no_type_preference():
    """G2a (level 2) restricts refs to defined ids but applies NO type
    preference; G2 (level 3) does. SELECT operand #0 wants Bool."""
    fsm2a = TaskIRGenFSM(level=2)
    for c in "inputs: @task: Str, @flag: Bool\n%c0 = SELECT(":
        fsm2a.feed(c)
    fsm2a._begin_operand("%", is_arg=True)
    assert fsm2a._cur_operand_choices == ["@task", "@flag"]
    assert fsm2a.stats.type_pref_applied == 0

    fsm3 = TaskIRGenFSM(level=3)
    for c in "inputs: @task: Str, @flag: Bool\n%c0 = SELECT(":
        fsm3.feed(c)
    fsm3._begin_operand("%", is_arg=True)
    assert fsm3._cur_operand_choices == ["@flag"]
    assert fsm3.stats.type_pref_applied == 1


def test_g3_type_trie_normalizes_names():
    """G3 allows only canonical type literals in the inputs line."""
    m = _mock_mask()
    fsm = TaskIRGenFSM(level=4)
    for c in "inputs: @task: ":
        fsm.feed(c)
    # 'S' is a canonical-prefix char; 's' (lowercase str) is not
    spec = fsm.allowed()
    assert "S" in spec["set"] and "s" not in spec["set"]
    assert "L" in spec["set"]                      # List[...]
    # walk 'Str' to completion -> terminator legal
    for c in "Str":
        fsm.feed(c)
    spec = fsm.allowed()
    assert "\n" in spec["set"] and "," in spec["set"]
    # feed stays permissive (FSM losslessness): lowercase type text feeds
    # fine — the G3 restriction lives only in allowed()/the mask layer
    fsm2 = TaskIRGenFSM(level=4)
    for c in "inputs: @task: str":
        assert fsm2.feed(c) is True
    # ...but the mask bans continuing a non-canonical literal: 'str' is
    # off-trie, so nothing is emittable until a terminator, and only
    # canonical prefixes were allowed in the first place ('s' was banned)
    assert "s" not in fsm2.allowed()["set"]
    assert fsm2.feed("\n")                       # permissive feed accepts
    # int -> 'I' forced (name normalization); honest: does not claim
    # semantic compatibility for a genuinely wrong type
    fsm3 = TaskIRGenFSM(level=4)
    for c in "inputs: @page: ":
        fsm3.feed(c)
    assert "I" in fsm3.allowed()["set"] and "i" not in fsm3.allowed()["set"]


def test_g3_list_any_walks_trie():
    fsm = TaskIRGenFSM(level=4)
    for c in "inputs: @data: List[Any]":
        fsm.feed(c)
    spec = fsm.allowed()
    assert "\n" in spec["set"]                     # complete literal
    fsm4 = TaskIRGenFSM(level=4)
    for c in "inputs: @data: List[An":
        fsm4.feed(c)
    assert "y" in fsm4.allowed()["set"]            # trie continues


def test_allowed_ids_never_contains_negative():
    """Regression: the old EOS -1 sentinel leaked into the allowed set as a
    negative tensor index, un-banning an arbitrary vocab slot."""
    m = _mock_mask()
    fsm = TaskIRGenFSM(level=1)
    ids = m.allowed_ids(fsm)
    assert all(i >= 0 for i in ids)
    # and when EOS is legal the caller adds it — not this method
    fsm2 = TaskIRGenFSM(level=1)
    _walk(fsm2, GOOD)
    ids2 = m.allowed_ids(fsm2)
    assert all(i >= 0 for i in ids2)


if __name__ == "__main__":
    # NOTE: this runner must stay at the END of the file — tests defined
    # after a mid-file __main__ block would silently never run (found in
    # review; use pytest to double-check the count).
    import traceback
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests)} tests: "
          f"{'ALL PASSED' if not failed else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
