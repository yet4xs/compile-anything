import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.ir.taskir import Guard, Module, Node, Program, Retry
from src.validator.validator import validate

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "taskir" / "examples"


def prog(nodes, output="%2", inputs=("@task", "Str")):
    return Module(program=Program(
        name="t", description="t",
        inputs=[{"name": inputs[0], "type": inputs[1]}],
        nodes=nodes, output=output))


def codes(rep):
    return {i.code for i in rep.issues if i.severity == "error"}


class TestValidator(unittest.TestCase):
    def test_examples_valid(self):
        for f in sorted(EXAMPLES.glob("*.json")):
            from src.ir.taskir import load_module
            rep = validate(load_module(f))
            self.assertTrue(rep.valid, f"{f}: {rep.summary()}")

    def test_undefined_ref(self):
        # %2 = FILTER(%3); %3 未定义
        m = prog([Node(id="%2", op="FILTER", inputs=["%3"])])
        self.assertIn("UNDEFINED_REF", codes(validate(m)))

    def test_use_before_def(self):
        # 逆序排列：FILTER 在 SEARCH 之前
        m = prog([Node(id="%2", op="FILTER", inputs=["%1"]),
                  Node(id="%1", op="SEARCH", inputs=["@task"])])
        self.assertIn("USE_BEFORE_DEF", codes(validate(m)))

    def test_unknown_skill(self):
        m = prog([Node(id="%1", op="NOT_A_SKILL", inputs=["@task"])], output="%1")
        self.assertIn("SKILL_UNKNOWN", codes(validate(m)))

    def test_type_mismatch(self):
        # FILTER 期待 List，收到 Str
        m = prog([Node(id="%1", op="FILTER", inputs=["@task"])], output="%1")
        self.assertIn("TYPE_MISMATCH", codes(validate(m)))

    def test_explicit_output_type_conflict(self):
        m = prog([Node(id="%1", op="SEARCH", inputs=["@task"],
                       params={"domain": "flight"}, output_type="Str")],
                 output="%1")
        self.assertIn("TYPE_MISMATCH", codes(validate(m)))

    def test_guard_must_be_bool(self):
        m = prog([Node(id="%1", op="SEARCH", inputs=["@task"]),
                  Node(id="%2", op="GENERATE", inputs=["%1"],
                       guard=Guard(cond="%1", expect=True))], output="%2")
        self.assertIn("TYPE_MISMATCH", codes(validate(m)))

    def test_retry_on_non_verify(self):
        m = prog([Node(id="%1", op="SEARCH", inputs=["@task"]),
                  Node(id="%2", op="GENERATE", inputs=["%1"],
                       retry=Retry(max_attempts=2, on="%1"))], output="%2")
        self.assertIn("CONTROL", codes(validate(m)))

    def test_retry_verify_must_depend(self):
        # retry 指向的 VERIFY 不依赖被重试节点
        m = prog([Node(id="%1", op="SEARCH", inputs=["@task"]),
                  Node(id="%2", op="SEARCH", inputs=["@task"]),
                  Node(id="%3", op="VERIFY", inputs=["%2"]),
                  Node(id="%4", op="GENERATE", inputs=["%1"],
                       retry=Retry(max_attempts=2, on="%3"))], output="%4")
        self.assertIn("CONTROL", codes(validate(m)))

    def test_select_branch_mismatch(self):
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"],
                 output_type="List[Flight]"),
            Node(id="%2", op="SEARCH", inputs=["@task"],
                 output_type="List[Hotel]"),
            Node(id="%3", op="SELECT", inputs=["%1", "%1", "%2"]),
        ], output="%3")
        self.assertIn("TYPE_MISMATCH", codes(validate(m)))

    def test_duplicate_id(self):
        m = prog([Node(id="%1", op="SEARCH", inputs=["@task"]),
                  Node(id="%1", op="SEARCH", inputs=["@task"])], output="%1")
        self.assertIn("STRUCT", codes(validate(m)))

    def test_dead_node_warning(self):
        m = prog([Node(id="%1", op="SEARCH", inputs=["@task"]),
                  Node(id="%2", op="SEARCH", inputs=["@task"])], output="%2")
        rep = validate(m)
        self.assertTrue(rep.valid)
        self.assertTrue(any(i.code == "DEAD_NODE" for i in rep.warnings))

    def test_bad_output(self):
        m = prog([Node(id="%1", op="SEARCH", inputs=["@task"])], output="%9")
        self.assertIn("OUTPUT", codes(validate(m)))

    def test_valid_verify_retry_chain(self):
        m = prog([
            Node(id="%1", op="SEARCH", inputs=["@task"]),
            Node(id="%2", op="GENERATE", inputs=["%1"]),
            Node(id="%3", op="VERIFY", inputs=["%2"]),
        ], output="%2")
        m.program.nodes[1].retry = Retry(max_attempts=3, on="%3")
        self.assertTrue(validate(m).valid)


if __name__ == "__main__":
    unittest.main()
