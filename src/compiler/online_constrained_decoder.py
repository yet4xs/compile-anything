"""Online (generation-time) constraint-guided TaskIR decoding.

Phase 7C-1: G0/G1/G2 mechanism validation on the s43 development set.
Phase 7C-2: arms re-frozen + isolation ablations + G3, evaluated on a
held-out confirmation set across seeds.

Unlike constrained_assembly.py (post-hoc: parse a finished completion, then
repair), this module constrains the model DURING decoding: a character-level
incremental FSM over the TaskIR text grammar tracks the generation prefix,
and a HuggingFace LogitsProcessor masks, at every step, the tokens whose
text would drive the prefix off the grammar BEFORE they are sampled.

Arm levels (frozen for 7C-2):
  G0  (0): no constraints (frozen greedy baseline, identical prompts)
  G1  (1): grammar FSM — header/comments, inputs line, node lines
           (%id = OP(args) annotations), return line, literals, EOS gate,
           comment cap. Target: eliminate preventable parse failures.
  G2a (2): G1 + dynamic SSA ENFORCED AT DECODE TIME (never by substituting
           references afterwards):
             - operand / return references restricted to ids DEFINED so far
             - duplicate node-id definition prevented by banning the
               terminator (forced extension, max 8, then logged fallback)
             - operand count capped at spec.max_inputs
  G2  (3): G2a + Skill-ISA type PREFERENCE (trie restricted to
           type-compatible ∪ unknown-typed ids; empty preferred set falls
           back to all defined ids, logged). G2 == the 7C-1 arm of the
           same name (pure refactor of the level constant).
  G3  (4): G2 + canonical input-type literal trie (name normalization only:
           forces str->Str spelling; does NOT claim semantic compatibility
           — a genuinely incompatible declaration still fails validation).

Conservative policies (recorded):
  - unknown/uninferable types are compatible with everything
  - annotation bodies (after/guard/retry/hints) are grammar-checked only;
    reference legality inside them is left to the final validator
  - deadlock fallback returns UNMASKED scores (best-effort constrained
    decoding, not a soundness guarantee) — every such event is logged and
    reported as a rate
  - the FSM never lowers the final bar: legality is always re-judged by
    src/ir/parser.py::parse_text() + validator on the completed text

Stats per generation: banned-token count per step, forced-choice steps
(model top-1 was banned, with state + the banned top-1 text), deadlocks +
fallback, forced duplicate extensions, type-preference applications and
fallbacks, FSM violations observed while feeding emitted tokens.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.isa.registry import get as get_skill, REGISTRY

VALID_OPS = set(REGISTRY.keys()) | {"VERIFY", "SELECT", "MERGE"}
ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
KEY_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789")
TYPE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_[]")
NUM_CHARS = set("-+0123456789.eE")

# Generation-time comment cap: >99% of corpus comment lines are <500 chars;
# runaway comments (model reproducing task-embedded JSON inside '; task:')
# burn the whole token budget. Cap forces '\n' via the MASK layer only —
# feed() stays permissive, so the FSM remains lossless on valid corpus.
COMMENT_CAP = 800

# G3 canonical input-type literals — FROZEN before the Phase 7C-2
# confirmation run. Sources: ISA signatures (src/isa/registry.py FIXED/input
# types) ∪ input-type literals observed across data/compiler_corpus_v4
# (Str, List[Any]) ∪ spec scalars. Scope: NAME NORMALIZATION only — the trie
# forces canonical spelling (str->Str, int->Int); it does NOT claim semantic
# compatibility (declaring @page: Int when FETCH wants Str still fails the
# validator, honestly).
CANONICAL_SCALARS = ["Any", "Bool", "Date", "Dict", "Entity", "Float",
                     "Int", "Json", "Str", "Table"]
CANONICAL_INPUT_TYPES = (CANONICAL_SCALARS
                         + [f"List[{s}]" for s in CANONICAL_SCALARS])


# ---------------- trie ----------------

class Trie:
    __slots__ = ("children", "terminal")

    def __init__(self):
        self.children: Dict[str, "Trie"] = {}
        self.terminal = False

    @classmethod
    def from_words(cls, words) -> "Trie":
        t = cls()
        for w in words:
            if w:
                n = t
                for ch in w:
                    n = n.children.setdefault(ch, Trie())
                n.terminal = True
        return t

    def node_for(self, prefix: str) -> Optional["Trie"]:
        n = self
        for c in prefix:
            n = n.children.get(c)
            if n is None:
                return None
        return n

    def next_chars(self) -> Set[str]:
        return set(self.children.keys())


_OP_TRIE = Trie.from_words(sorted(VALID_OPS))
_TYPE_TRIE = Trie.from_words(sorted(CANONICAL_INPUT_TYPES))


# ---------------- stats ----------------

@dataclass
class GenStats:
    steps: int = 0
    banned_counts: List[int] = field(default_factory=list)
    forced_steps: int = 0
    forced_events: List[Dict] = field(default_factory=list)
    deadlocks: List[str] = field(default_factory=list)
    dup_forced_extensions: int = 0
    type_pref_applied: int = 0
    type_pref_fallbacks: int = 0
    fsm_violation: Optional[str] = None


# ---------------- FSM ----------------

class TaskIRGenFSM:
    """Incremental char-level lexer for canonical TaskIR text.

    feed(ch) -> False on grammar violation.
    allowed() -> {"kind": "chars", "set": {...}} or
                 {"kind": "free", "banned": {...}} for the next character.
    eos_ok()  -> True only after >=1 node and a completed return line.
    """

    # top-level
    S_LINE_START = "line_start"
    S_COMMENT = "comment"
    S_KW = "kw"
    # inputs line
    S_INPUT_AT = "input_at"
    S_INPUT_NAME = "input_name"
    S_INPUT_COLON = "input_colon"
    S_INPUT_TYPE = "input_type"
    S_INPUT_SEP = "input_sep"
    # node line
    S_NODE_ID = "node_id"
    S_NODE_EQ = "node_eq"
    S_OP = "op"
    S_ARGS = "args"              # position where an arg or ')' may start
    S_OPERAND = "operand"
    S_KEY = "key"
    S_VALUE = "value"
    S_NUM = "num"
    S_BARE = "bare"
    # literals with a resume context
    S_STR = "str"
    S_STR_ESC = "str_esc"
    S_LIST = "list"
    # annotations / return
    S_ANN_WS = "ann_ws"
    S_ANN_TYPE = "ann_type"
    S_ANN_BODY = "ann_body"
    S_RETURN_WS = "return_ws"
    S_RETURN_REF = "return_ref"
    S_TRAIL_NL = "trail_nl"

    def __init__(self, level: int = 1, stats: Optional[GenStats] = None):
        assert level in (1, 2, 3, 4), \
            "levels: 1=G1 grammar, 2=G2a (+SSA/arity), 3=G2/G2b (+type " \
            "preference), 4=G3 (+canonical input type trie); G0 uses no FSM"
        self.level = level
        self.stats = stats or GenStats()
        self.state = self.S_LINE_START
        self.buf = ""
        self.kw_candidates: List[str] = []
        self.resume: List[str] = []     # context stack for literals
        self.line_kind: Optional[str] = None
        # node context
        self.cur_node_id: Optional[str] = None
        self.cur_op: Optional[str] = None
        self.operand_count = 0
        self.cur_op_node = _OP_TRIE
        self.dup_ext_attempts = 0
        # G2 bookkeeping
        self.defined: List[str] = []                # ordered, '@in' + '%nodes'
        self.type_map: Dict[str, str] = {}
        self.node_count = 0
        self.return_line_done = False
        self.comment_len = 0
        self._last_node_inputs: List[str] = []
        self._cur_operand_choices: Optional[List[str]] = None
        self._operand_is_arg: bool = True

    # ---------- snapshot (for cheap what-if simulation) ----------
    # NOTE: must copy the mutable containers — feed() appends to them during
    # token simulation, and a shallow copy would pollute the live state.

    def snapshot(self) -> Dict:
        return {
            "state": self.state, "buf": self.buf,
            "kw_candidates": list(self.kw_candidates),
            "resume": list(self.resume),
            "line_kind": self.line_kind,
            "cur_node_id": self.cur_node_id,
            "cur_op": self.cur_op,
            "operand_count": self.operand_count,
            "cur_op_node": self.cur_op_node,
            "dup_ext_attempts": self.dup_ext_attempts,
            "defined": list(self.defined),
            "type_map": dict(self.type_map),
            "node_count": self.node_count,
            "return_line_done": self.return_line_done,
            "_last_node_inputs": list(self._last_node_inputs),
            "_cur_operand_choices": (list(self._cur_operand_choices)
                                     if self._cur_operand_choices is not None
                                     else None),
            "_operand_is_arg": self._operand_is_arg,
            "stats": self.stats,
        }

    def restore(self, snap: Dict):
        self.state = snap["state"]
        self.buf = snap["buf"]
        self.kw_candidates = snap["kw_candidates"]
        self.resume = snap["resume"]
        self.line_kind = snap["line_kind"]
        self.cur_node_id = snap["cur_node_id"]
        self.cur_op = snap["cur_op"]
        self.operand_count = snap["operand_count"]
        self.cur_op_node = snap["cur_op_node"]
        self.dup_ext_attempts = snap["dup_ext_attempts"]
        self.defined = snap["defined"]
        self.type_map = snap["type_map"]
        self.node_count = snap["node_count"]
        self.return_line_done = snap["return_line_done"]
        self._last_node_inputs = snap["_last_node_inputs"]
        self._cur_operand_choices = snap["_cur_operand_choices"]
        self._operand_is_arg = snap["_operand_is_arg"]
        self.stats = snap["stats"]

    # ---------- G2 helpers ----------

    def _operand_cap_reached(self) -> bool:
        if self.level < 2:
            return False
        spec = get_skill(self.cur_op) if self.cur_op else None
        return bool(spec and spec.max_inputs
                    and self.operand_count >= spec.max_inputs)

    def _ref_choices(self, want_type: Optional[str]) -> List[str]:
        """Legal reference ids for the current position (def-use from G2a;
        type PREFERENCE only from G2b — G2a returns the plain defined set)."""
        if self.level < 2:
            return None  # unrestricted (G1)
        ids = list(self.defined)
        if want_type is not None and self.level >= 3:
            from src.ir.types import is_compatible
            preferred = [d for d in ids
                         if self.type_map.get(d) is None
                         or is_compatible(want_type, self.type_map[d])]
            if preferred:
                if len(preferred) < len(ids):
                    self.stats.type_pref_applied += 1
                return preferred
            self.stats.type_pref_fallbacks += 1
        return ids

    def _want_type(self) -> Optional[str]:
        if self.level < 3:
            return None
        spec = get_skill(self.cur_op) if self.cur_op else None
        if spec and spec.input_types:
            return spec.input_types[
                min(self.operand_count, len(spec.input_types) - 1)]
        return None

    def _begin_operand(self, prefix: str, is_arg: bool = True):
        self.state, self.buf = self.S_OPERAND, prefix
        self._operand_is_arg = is_arg
        self._cur_operand_choices = self._ref_choices(
            self._want_type() if is_arg else None)

    def _operand_choices_trie(self) -> Optional[Trie]:
        if self._cur_operand_choices is None:
            return None
        return Trie.from_words(sorted(self._cur_operand_choices))

    # ---------- lexer ----------

    def feed(self, ch: str) -> bool:
        st = self.state

        # ---- literals with resume context ----
        if st == self.S_STR:
            if ch == "\\":
                self.state = self.S_STR_ESC
                return True
            if ch == '"':
                self.state = self.resume.pop()
                return True
            return True                                # free content
        if st == self.S_STR_ESC:
            self.state = self.S_STR
            return True
        if st == self.S_LIST:
            if ch == "]":
                self.state = self.resume.pop()
                return True
            if ch == '"':
                self.resume.append(self.S_LIST)
                self.state = self.S_STR
                return True
            return True                                # permissive content

        # ---- top level ----
        if st == self.S_LINE_START:
            if ch in "\n ":
                return True
            if ch == ";":
                self.state, self.line_kind = self.S_COMMENT, "comment"
                return True
            if ch == "i":
                self.state, self.buf, self.kw_candidates = \
                    self.S_KW, "i", ["inputs:"]
                self.line_kind = "inputs"
                return True
            if ch == "r":
                self.state, self.buf, self.kw_candidates = \
                    self.S_KW, "r", ["return"]
                self.line_kind = "return"
                return True
            if ch in "%@":
                self.line_kind = "node"
                self.state, self.buf = self.S_NODE_ID, ch
                self.operand_count = 0
                self.dup_ext_attempts = 0
                self._last_node_inputs = []
                return True
            return False

        if st == self.S_COMMENT:
            self.comment_len += 1
            if ch == "\n":
                self.comment_len = 0
                self._end_line()
            return True                                # free content

        if st == self.S_KW:
            self.buf += ch
            self.kw_candidates = [k for k in self.kw_candidates
                                  if k.startswith(self.buf)]
            if not self.kw_candidates:
                return False
            if self.buf in self.kw_candidates:
                self._kw_complete()
            return True

        # ---- inputs line ----
        if st == self.S_INPUT_AT:
            if ch == " ":
                return True                          # stay, skip spaces
            if ch == "@":
                self.state, self.buf = self.S_INPUT_NAME, "@"
                return True
            return False

        if st == self.S_INPUT_NAME:
            if ch in ID_CHARS:
                self.buf += ch
                return True
            if ch == ":":
                self.defined.append(self.buf)
                self.state, self.buf = self.S_INPUT_COLON, ""
                return True
            return False

        if st == self.S_INPUT_COLON:
            if ch == " ":
                return True
            if ch in TYPE_CHARS:
                self.state, self.buf = self.S_INPUT_TYPE, ch
                return True
            return False

        if st == self.S_INPUT_TYPE:
            if ch in TYPE_CHARS or ch == " ":
                self.buf += ch
                return True
            if ch == ",":
                self.type_map[self.defined[-1]] = self.buf.strip() or "Any"
                self.state, self.buf = self.S_INPUT_SEP, ""
                return True
            if ch == "\n":
                self.type_map[self.defined[-1]] = self.buf.strip() or "Any"
                self._end_line()
                return True
            return False

        if st == self.S_INPUT_SEP:
            if ch == " ":
                return True
            if ch == "@":
                self.state, self.buf = self.S_INPUT_NAME, "@"
                return True
            return False

        # ---- node line ----
        if st == self.S_NODE_ID:
            if ch in ID_CHARS:
                self.buf += ch
                return True
            if ch in " =":
                if (self.level >= 2 and self.buf in self.defined
                        and self.dup_ext_attempts < 8):
                    self.stats.dup_forced_extensions += 1
                    self.dup_ext_attempts += 1
                    return False                        # force name extension
                self.cur_node_id = self.buf
                self.state = self.S_NODE_EQ
                return True
            return False

        if st == self.S_NODE_EQ:
            if ch == " ":
                return True
            if ch == "=":
                self.state, self.buf = self.S_OP, ""
                self.cur_op_node = _OP_TRIE
                return True
            return False

        if st == self.S_OP:
            if ch == " " and not self.buf:
                return True                     # space between '=' and op
            nxt = self.cur_op_node.children.get(ch)
            if nxt is not None:
                self.buf += ch
                self.cur_op_node = nxt
                return True
            if self.cur_op_node.terminal and ch in "( ":
                if ch == "(":
                    self.cur_op = self.buf
                    self.buf = ""
                    self.operand_count = 0
                    self.state = self.S_ARGS
                return True
            return False

        if st == self.S_ARGS:
            if ch == ")":
                self.state = self.S_ANN_WS
                return True
            if ch in "%@":
                if self._operand_cap_reached():
                    return False
                self._begin_operand(ch)
                return True
            if ch in KEY_CHARS:
                self.state, self.buf = self.S_KEY, ch
                return True
            return False

        if st == self.S_OPERAND:
            if ch in ID_CHARS:
                self.buf += ch
                return True
            if ch in ", )":
                if (self.level >= 2 and self.buf not in self.defined):
                    return False                       # undefined ref
                if self._operand_is_arg:
                    self.operand_count += 1
                    self._last_node_inputs.append(self.buf)
                self.buf = ""
                self._cur_operand_choices = None
                if ch == ",":
                    self.state = self.S_ARGS_SEP_WS
                elif ch == ")":
                    self.state = self.S_ANN_WS
                else:                                   # space then ',' or key
                    self.state = self.S_ARGS
                return True
            return False

        if st == self.S_KEY:
            if ch in KEY_CHARS:
                self.buf += ch
                return True
            if ch == "=":
                self.state, self.buf = self.S_VALUE, ""
                return True
            return False

        if st == self.S_VALUE:
            if ch == '"':
                self.resume.append(self.S_ARGS_SEP_WS)
                self.state = self.S_STR
                return True
            if ch == "[":
                self.resume.append(self.S_ARGS_SEP_WS)
                self.state = self.S_LIST
                return True
            if ch in NUM_CHARS:
                self.state, self.buf = self.S_NUM, ch
                return True
            if ch in ("t", "f"):
                self.state, self.buf = self.S_KW, ch
                self.kw_candidates = ["true", "false"]
                return True
            if ch in "%@":
                self._begin_operand(ch, is_arg=False)
                return True
            if ch in KEY_CHARS:
                self.state, self.buf = self.S_BARE, ch
                return True
            return False

        if st == self.S_NUM:
            if ch in NUM_CHARS:
                return True
            return self._value_delim(ch)

        if st == self.S_BARE:
            if ch in KEY_CHARS or ch in ".-/:":
                return True
            return self._value_delim(ch)

        # ---- annotations ----
        if st == self.S_ANN_WS:
            if ch == "\n":
                self._end_node_line()
                return True
            if ch == " ":
                return True
            if ch == "-":
                self.state, self.buf, self.kw_candidates = \
                    self.S_KW, "-", ["->"]
                return True
            if ch in "agrh":
                words = {"a": ["after("], "g": ["guard("],
                         "r": ["retry("], "h": ["hints("]}[ch]
                self.state, self.buf, self.kw_candidates = \
                    self.S_KW, ch, words
                return True
            return False

        if st == self.S_ANN_TYPE:
            # annotation keyword may follow the type after a space — check
            # BEFORE the type-char branch ('r' is also a valid type letter)
            if ch in "agrh" and self.buf.endswith(" "):
                words = {"a": ["after("], "g": ["guard("],
                         "r": ["retry("], "h": ["hints("]}[ch]
                self.state, self.buf, self.kw_candidates = \
                    self.S_KW, ch, words
                return True
            if ch in TYPE_CHARS or ch == " ":
                self.buf += ch
                return True
            if ch == "\n":
                self._end_node_line()
                return True
            return False

        if st == self.S_ANN_BODY:
            # conservative: grammar-level only; ref legality left to validator
            if ch == ")":
                self.state = self.S_ANN_WS
                return True
            if ch == '"':
                self.resume.append(self.S_ANN_BODY)
                self.state = self.S_STR
                return True
            if ch == "\n":
                return False                           # annot must close
            return True

        # ---- return line ----
        if st == self.S_RETURN_WS:
            if ch == " ":
                return True
            if ch in "%@":
                self.state, self.buf = self.S_RETURN_REF, ch
                self._cur_operand_choices = self._ref_choices(None)
                return True
            return False

        if st == self.S_RETURN_REF:
            if ch in ID_CHARS:
                self.buf += ch
                return True
            if ch in " \n":
                if self.level >= 2 and self.buf not in self.defined:
                    return False
                self.return_line_done = True
                self.state = self.S_TRAIL_NL
                return True
            return False

        if st == self.S_TRAIL_NL:
            return ch == "\n"

        # helper state: after ',' + optional space -> next arg
        if st == "args_sep_ws":
            if ch in " ,":
                return True                  # ', ' or bare ','
            return self.feed_args(ch)

        return False

    S_ARGS_SEP_WS = "args_sep_ws"

    def feed_args(self, ch: str) -> bool:
        """S_ARGS logic reused by args_sep_ws (without re-entering ws)."""
        if ch == ")":
            self.state = self.S_ANN_WS
            return True
        if ch in "%@":
            if self._operand_cap_reached():
                return False
            self._begin_operand(ch)
            return True
        if ch in KEY_CHARS:
            self.state, self.buf = self.S_KEY, ch
            return True
        return False

    # ---------- helpers ----------

    def _value_delim(self, ch: str) -> bool:
        if ch in ", ":
            self.state = self.S_ARGS_SEP_WS
            return True
        if ch == ")":
            self.state = self.S_ANN_WS
            return True
        if ch == "]":
            self.state = self.S_ARGS_SEP_WS
            return True
        return False

    def _kw_complete(self):
        kw = self.buf
        if kw == "inputs:":
            self.state = self.S_INPUT_AT
        elif kw == "return":
            self.state = self.S_RETURN_WS
        elif kw in ("true", "false"):
            self.state = self.S_ARGS_SEP_WS
        elif kw == "->":
            self.state, self.buf = self.S_ANN_TYPE, ""
        else:  # after( / guard( / retry( / hints(
            self.state = self.S_ANN_BODY

    def _end_node_line(self):
        if self.line_kind == "node" and self.cur_node_id:
            nid = self.cur_node_id
            if nid not in self.defined:
                self.defined.append(nid)
            spec = get_skill(self.cur_op) if self.cur_op else None
            if spec is not None:
                in_types = [self.type_map.get(r, "Any")
                            for r in self._last_node_inputs]
                try:
                    from src.ir.types import normalize
                    self.type_map[nid] = normalize(
                        str(spec.output_type(in_types)))
                except Exception:
                    self.type_map[nid] = "Any"
            else:
                self.type_map[nid] = "Any"
            self.node_count += 1
        self._end_line()

    def _end_line(self):
        self.state = self.S_LINE_START
        self.line_kind = None
        self.buf = ""
        self.cur_node_id = None
        self.cur_op = None
        self.cur_op_node = _OP_TRIE
        self.operand_count = 0
        self.resume = []

    def eos_ok(self) -> bool:
        if not (self.node_count >= 1):
            return False
        if self.state in (self.S_TRAIL_NL, self.S_LINE_START):
            return self.return_line_done
        # EOS right after a complete return ref, before its newline:
        # 'return %c4' then EOS is a finished program (the trailing
        # newline is cosmetic; parse_text does not require it)
        if self.state == self.S_RETURN_REF and self.buf:
            return self.level < 2 or self.buf in self.defined
        return False

    # ---------- allowed continuation ----------

    def allowed(self) -> Dict:
        st = self.state

        if st == self.S_LINE_START:
            return {"kind": "chars", "set": set(";ir%@\n ")}
        if st == self.S_COMMENT:
            if self.comment_len >= COMMENT_CAP:
                return {"kind": "chars", "set": {"\n"}}   # force line exit
            return {"kind": "free", "banned": {"\n"}}
        if st == self.S_KW:
            cand = {k[len(self.buf)] for k in self.kw_candidates
                    if len(k) > len(self.buf)}
            return {"kind": "chars", "set": cand}
        if st == self.S_INPUT_AT:
            return {"kind": "chars", "set": {" ", "@"}}
        if st == self.S_INPUT_NAME:
            return {"kind": "chars", "set": ID_CHARS | {":"}}
        if st == self.S_INPUT_COLON:
            if self.level >= 4:
                # G3 applies from the FIRST type character (this state
                # holds it before S_INPUT_TYPE takes over)
                node = _TYPE_TRIE.node_for("")
                return {"kind": "chars", "set": set(node.next_chars())}
            return {"kind": "chars", "set": TYPE_CHARS | {" "}}
        if st == self.S_INPUT_TYPE:
            if self.level >= 4:
                # G3: canonical type literals only (name normalization).
                # Terminators legal only at a complete literal.
                node = _TYPE_TRIE.node_for(self.buf.strip())
                if node is None:
                    return {"kind": "chars", "set": set()}
                chars = set(node.next_chars())
                if node.terminal:
                    chars |= {",", "\n"}
                return {"kind": "chars", "set": chars}
            return {"kind": "chars", "set": TYPE_CHARS | {" ", ",", "\n"}}
        if st == self.S_INPUT_SEP:
            return {"kind": "chars", "set": {" ", "@"}}
        if st == self.S_NODE_ID:
            base = set(ID_CHARS)
            if (self.level >= 2 and self.buf in self.defined
                    and self.dup_ext_attempts < 8):
                return {"kind": "chars", "set": base}   # extend, no terminator
            return {"kind": "chars", "set": base | {" ", "="}}
        if st == self.S_NODE_EQ:
            return {"kind": "chars", "set": {" ", "="}}
        if st == self.S_OP:
            chars = set(self.cur_op_node.next_chars())
            if self.cur_op_node.terminal:
                chars |= {"(", " "}
            if not self.buf:
                chars |= {" "}               # space after '='
            return {"kind": "chars", "set": chars}
        if st == self.S_ARGS:
            return self.allowed_args()
        if st == self.S_ARGS_SEP_WS:
            return self.allowed_args()
        if st == self.S_OPERAND:
            if self.level >= 2:
                trie = self._operand_choices_trie()
                if trie is None:
                    return {"kind": "chars",
                            "set": ID_CHARS | {",", ")", " "}}
                node = trie.node_for(self.buf)
                if node is None:
                    return {"kind": "chars", "set": set()}  # off-trie
                chars = set(node.next_chars())
                if self.buf in self.defined:
                    chars |= {",", ")", " "}
                return {"kind": "chars", "set": chars}
            return {"kind": "chars", "set": ID_CHARS | {",", ")", " "}}
        if st == self.S_KEY:
            return {"kind": "chars", "set": KEY_CHARS | {"="}}
        if st == self.S_VALUE:
            return {"kind": "chars",
                    "set": {'"', "[", "t", "f", "%", "@"} | NUM_CHARS
                    | KEY_CHARS}
        if st == self.S_STR:
            return {"kind": "free_str", "banned": {'"', "\n"}}
        if st == self.S_STR_ESC:
            return {"kind": "free", "banned": set()}
        if st == self.S_LIST:
            return {"kind": "free_list",
                    "banned": {"]", '"', "\n"}}
        if st == self.S_NUM:
            return {"kind": "chars",
                    "set": NUM_CHARS | {",", ")", "]", " "}}
        if st == self.S_BARE:
            return {"kind": "chars",
                    "set": KEY_CHARS | {".", "-", "/", ":",
                                        ",", ")", "]", " "}}
        if st == self.S_ANN_WS:
            return {"kind": "chars", "set": {" ", "\n", "-"} | set("agrh")}
        if st == self.S_ANN_TYPE:
            chars = TYPE_CHARS | {" ", "\n"}
            if self.buf.endswith(" ") or not self.buf:
                chars |= set("agrh")
            return {"kind": "chars", "set": chars}
        if st == self.S_ANN_BODY:
            return {"kind": "free_ann", "banned": {")", "\n", '"'}}
        if st == self.S_RETURN_WS:
            return {"kind": "chars", "set": {" ", "%", "@"}}
        if st == self.S_RETURN_REF:
            if self.level >= 2:
                choices = (self._cur_operand_choices
                           if self._cur_operand_choices is not None
                           else self._ref_choices(None))
                trie = Trie.from_words(sorted(choices))
                node = trie.node_for(self.buf)
                if node is None and self.buf not in self.defined:
                    return {"kind": "chars", "set": set()}
                chars = set(node.next_chars()) if node else set()
                if self.buf in self.defined:
                    chars |= {"\n", " "}
                return {"kind": "chars", "set": chars}
            return {"kind": "chars", "set": ID_CHARS | {"\n", " "}}
        if st == self.S_TRAIL_NL:
            return {"kind": "chars", "set": {"\n"}}
        return {"kind": "chars", "set": set()}

    def allowed_args(self) -> Dict:
        chars = {")", " "} | KEY_CHARS
        if self.level < 2:
            chars |= {"%", "@"}
        elif not self._operand_cap_reached():
            # only ref prefixes that have at least one defined id
            chars |= {d[0] for d in self.defined if d[0] in "%@"}
        return {"kind": "chars", "set": chars}


# ---------------- token mask ----------------

class TokenMask:
    """Precomputes token-id <-> text; checks full-token FSM compatibility.

    A token may be emitted iff feeding EVERY character of its text keeps the
    FSM error-free (single tokens can cross state boundaries, e.g. `")` or
    ` = `), and — for `free_str` states — it contains no unescaped '"'.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.token_texts: List[str] = self._build_token_texts(tokenizer)
        self._first_char_buckets: Dict[str, List[int]] = {}
        for tid, txt in enumerate(self.token_texts):
            if not txt:
                continue
            self._first_char_buckets.setdefault(txt[0], []).append(tid)
        self._cache: Dict[tuple, List[int]] = {}

    @staticmethod
    def _build_token_texts(tokenizer) -> List[str]:
        n = len(tokenizer)
        texts = [""] * n
        # byte-level BPE (Qwen2/GPT-2 style): map merge pieces to chars
        try:
            from transformers.models.gpt2.tokenization_gpt2 import \
                bytes_to_unicode
            b2u = bytes_to_unicode()
            u2b = {v: k for k, v in b2u.items()}
            toks = tokenizer.convert_ids_to_tokens(list(range(n)))
            for tid, piece in enumerate(toks):
                try:
                    texts[tid] = bytes(u2b.get(c, ord(c)) for c in piece
                                       ).decode("utf-8", errors="replace")
                except Exception:
                    texts[tid] = ""
            return texts
        except Exception:
            pass
        # slow fallback
        for tid in range(n):
            try:
                texts[tid] = tokenizer.decode([tid])
            except Exception:
                texts[tid] = ""
        return texts

    def allowed_ids(self, fsm: TaskIRGenFSM) -> List[int]:
        """Legal non-special token ids for the current FSM state.

        EOS legality is the caller's concern (processor adds real eos ids
        when fsm.eos_ok()); a sentinel like -1 must never enter this list —
        it would become a negative tensor index and un-ban an arbitrary
        vocab slot.
        """
        spec = fsm.allowed()
        # cache key must cover EVERYTHING _token_ok depends on: the resume
        # stack (string/list exit context), keyword candidates, and — at
        # G2+ — the dynamic SSA sets. A stale entry here leaks bans across
        # contexts (observed: '"])\n' banned inside lists because a param-
        # value string computed the key first).
        key = (fsm.state, fsm.buf, tuple(fsm.resume),
               tuple(fsm.kw_candidates), spec["kind"],
               tuple(sorted(spec.get("set", []))),
               tuple(sorted(spec.get("banned", []))))
        if fsm.level >= 2:
            key += (tuple(fsm.defined), fsm.cur_op or "",
                    fsm.operand_count, fsm._operand_is_arg,
                    tuple(fsm._cur_operand_choices or []),
                    fsm.dup_ext_attempts)
        if key in self._cache:
            return list(self._cache[key])
        allowed = self._compute(fsm, spec)
        self._cache[key] = allowed
        return list(allowed)

    def _compute(self, fsm: TaskIRGenFSM, spec: Dict) -> List[int]:
        kind = spec["kind"]
        out: List[int] = []
        if kind == "chars":
            allowed_chars = spec["set"]
            for ch, ids in self._first_char_buckets.items():
                if ch not in allowed_chars:
                    continue
                for tid in ids:
                    if self._token_ok(fsm, self.token_texts[tid]):
                        out.append(tid)
        else:
            # free regions (comment body, string body, list, annotation):
            # tokens WITHOUT a terminator char are always legal; tokens WITH
            # one may still be legal when the full-token FSM simulation
            # accepts it — the terminator is the region's legal exit
            # (e.g. '\n' ends a comment, '"' closes a string).
            banned = spec["banned"]
            for tid, txt in enumerate(self.token_texts):
                if not txt:
                    continue
                if not any(b in txt for b in banned):
                    out.append(tid)
                elif self._token_ok(fsm, txt):
                    out.append(tid)
        return out

    def _token_ok(self, fsm: TaskIRGenFSM, text: str) -> bool:
        """Full-token simulation: every char must keep the FSM error-free."""
        snap = fsm.snapshot()
        try:
            for c in text:
                if not fsm.feed(c):
                    return False
            return True
        finally:
            fsm.restore(snap)

    def warm(self, fsm: TaskIRGenFSM, sample_text: str):
        """Warm the mask cache by walking a representative text."""
        for c in sample_text:
            fsm.feed(c)


# ---------------- HF LogitsProcessor ----------------

import torch  # noqa: E402  (kept local so CPU unit tests can skip GPU stack)


class ConstrainedLogitsProcessor:
    """Masks off-grammar tokens at every decode step (batch size must be 1).

    Usage with model.generate(logits_processor=[proc], ...). Feeds each newly
    generated token into the FSM, then replaces scores so only FSM-legal
    tokens (plus EOS when the program is complete) survive.

    Deadlock policy: if no legal token remains (grammar/mask inconsistency),
    the ORIGINAL scores are returned untouched and the event is logged; the
    final parser/validator still judges the text — the FSM never lowers that
    bar and never silently repairs anything.
    """

    def __init__(self, tokenizer, level: int, eos_token_ids):
        self.stats = GenStats()
        self.fsm = TaskIRGenFSM(level=level, stats=self.stats)
        self.mask = TokenMask(tokenizer)
        self.eos_ids = set(int(e) for e in
                           (eos_token_ids if isinstance(eos_token_ids,
                                                        (list, tuple))
                            else [eos_token_ids]))
        self.prompt_len: Optional[int] = None
        self.gen_fed = 0

    def __call__(self, input_ids, scores):
        assert scores.shape[0] == 1, "online constraints require batch=1"
        ids = input_ids[0].tolist()
        if self.prompt_len is None:
            self.prompt_len = len(ids)
        # 1) advance FSM with tokens generated since last call
        while self.prompt_len + self.gen_fed < len(ids):
            tid = ids[self.prompt_len + self.gen_fed]
            self.gen_fed += 1
            if tid in self.eos_ids:
                continue
            txt = self.mask.token_texts[tid] if 0 <= tid < len(
                self.mask.token_texts) else ""
            for c in txt:
                if not self.fsm.feed(c) and self.stats.fsm_violation is None:
                    self.stats.fsm_violation = (
                        f"state={self.fsm.state} char={c!r} after "
                        f"{self.gen_fed} tokens")
        # 2) compute legal token set (EOS added separately — never a sentinel)
        allowed = set(self.mask.allowed_ids(self.fsm))
        # ban every special token except EOS-when-legal
        for e in self.eos_ids:
            if self.fsm.eos_ok():
                allowed.add(e)
        # 3) bookkeeping + mask
        self.stats.steps += 1
        top1 = int(torch.argmax(scores[0]).item())
        n_vocab = scores.shape[-1]
        if allowed:
            keep = torch.full((n_vocab,), float("-inf"),
                              dtype=scores.dtype, device=scores.device)
            idx = torch.tensor(sorted(allowed), device=scores.device,
                               dtype=torch.long)
            keep[idx] = scores[0][idx]
            self.stats.banned_counts.append(n_vocab - len(allowed))
            if top1 not in allowed:
                self.stats.forced_steps += 1
                if len(self.stats.forced_events) < 40:
                    self.stats.forced_events.append({
                        "step": self.stats.steps, "state": self.fsm.state,
                        "buf": self.fsm.buf[-30:],
                        "top1": (self.mask.token_texts[top1]
                                 if 0 <= top1 < len(self.mask.token_texts)
                                 else f"id{top1}")})
            out = scores.clone()
            out[0] = keep
            return out
        # deadlock fallback: leave scores untouched, log
        self.stats.deadlocks.append(
            f"step={self.stats.steps} state={self.fsm.state} buf={self.fsm.buf!r}")
        self.stats.banned_counts.append(0)
        return scores


def generate_constrained(model, tokenizer, prompt_text: str, level: int,
                         max_new_tokens: int = 512):
    """Greedily generate one completion under G0(0)/G1(1)/G2a(2)/G2(3)/G3(4).

    Returns (text, stats_or_None, n_prompt_tokens).
    """
    assert 0 <= level <= 4, f"unknown arm level {level}"
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  temperature=None, top_p=None, top_k=None,
                  pad_token_id=tokenizer.pad_token_id)
    if level == 0:
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        return text, None, int(inputs["input_ids"].shape[1])
    proc = ConstrainedLogitsProcessor(tokenizer, level, eos_ids)
    with torch.no_grad():
        out = model.generate(**inputs, logits_processor=[proc], **gen_kw)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text, proc.stats, int(inputs["input_ids"].shape[1])
