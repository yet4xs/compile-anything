"""Phase 8D-0 Task 5: runtime dataflow correctness tests."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ir.parser import parse_text
from src.validator.validator import validate
from src.ir.taskir import to_text
from src.runtime.mock_capability_registry import (
    MockCapabilityRegistry, DataflowRuntime)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_registry():
    reg = MockCapabilityRegistry()
    reg.register_simple("lookup_user", {"name": "Str"},
                        {"user_id": "Str", "email": "Str"},
                        executor=lambda p: {"user_id": f"U_{p['name']}",
                                            "email": f"{p['name']}@test.com"})
    reg.register_simple("get_account", {"user_id": "Str"},
                        {"account_id": "Str", "balance": "Float"},
                        executor=lambda p: {"account_id": f"A_{p['user_id']}",
                                            "balance": 100.0})
    reg.register_simple("get_latest_transaction", {"account_id": "Str"},
                        {"tx_id": "Str", "amount": "Float"},
                        executor=lambda p: {"tx_id": f"T_{p['account_id']}",
                                            "amount": 42.0})
    reg.register_simple("get_transaction_status", {"tx_id": "Str"},
                        {"status": "Str"},
                        executor=lambda p: {"status": "confirmed"})
    reg.register_simple("get_credit_score", {"user_id": "Str"},
                        {"score": "Int"},
                        executor=lambda p: {"score": 750})
    reg.register_simple("approve_loan",
                        {"user_id": "Str", "score": "Int", "amount": "Float"},
                        {"approved": "Bool", "loan_id": "Str"},
                        effect_class="state_creation",
                        executor=lambda p: {"approved": p["score"] >= 700,
                                            "loan_id": f"L_{p['user_id']}"})
    reg.register_simple("update_balance",
                        {"account_id": "Str", "new_balance": "Float"},
                        {"success": "Bool"},
                        effect_class="state_update_reversible",
                        executor=lambda p: {"success": True})
    reg.register_simple("send_notification",
                        {"user_id": "Str", "message": "Str"},
                        {"sent": "Bool"}, effect_class="communication",
                        executor=lambda p: {"sent": True})
    reg.register_simple("step_a", {"x": "Str"}, {"a_id": "Str"},
                        executor=lambda p: {"a_id": f"a_{p['x']}"})
    reg.register_simple("step_b", {"a_id": "Str"}, {"b_id": "Str"},
                        executor=lambda p: {"b_id": f"b_{p['a_id']}"})
    reg.register_simple("step_c", {"b_id": "Str"}, {"c_id": "Str"},
                        executor=lambda p: {"c_id": f"c_{p['b_id']}"})
    reg.register_simple("step_d", {"c_id": "Str"}, {"d_id": "Str"},
                        executor=lambda p: {"d_id": f"d_{p['c_id']}"})
    return reg


RT = DataflowRuntime(make_registry())


# ── Linear chains ──

def test_linear_3step():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Alice")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%acc = EXEC_ACTION(action="get_account")  bind(user_id=%uid)\n\n'
         "return %acc\n")
    mod = parse_text(t)
    assert validate(mod).valid
    tr = RT.execute(mod)
    assert tr.success, tr.error
    assert tr.steps[2].input_params["user_id"] == "U_Alice"
    assert tr.final_result["account_id"] == "A_U_Alice"


def test_linear_4step():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Bob")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%acc = EXEC_ACTION(action="get_account")  bind(user_id=%uid)\n'
         '%aid = EXTRACT_FIELD(%acc, field="account_id") -> Str\n'
         '%tx = EXEC_ACTION(action="get_latest_transaction")  '
         'bind(account_id=%aid)\n'
         '%tid = EXTRACT_FIELD(%tx, field="tx_id") -> Str\n'
         '%st = EXEC_ACTION(action="get_transaction_status")  '
         'bind(tx_id=%tid)\n\n'
         "return %st\n")
    mod = parse_text(t)
    assert validate(mod).valid
    tr = RT.execute(mod)
    assert tr.success, tr.error
    assert tr.final_result["status"] == "confirmed"
    assert tr.steps[6].input_params["tx_id"] == "T_A_U_Bob"


def test_fork_multi_consumer():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Cara")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%acc = EXEC_ACTION(action="get_account")  bind(user_id=%uid)\n'
         '%cr = EXEC_ACTION(action="get_credit_score")  bind(user_id=%uid)\n\n'
         "return %acc\n")
    mod = parse_text(t)
    assert validate(mod).valid
    tr = RT.execute(mod)
    assert tr.success
    s = {x.node_id: x for x in tr.steps}
    assert s["%acc"].input_params["user_id"] == "U_Cara"
    assert s["%cr"].input_params["user_id"] == "U_Cara"


def test_multi_source_params():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Dave")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%cr = EXEC_ACTION(action="get_credit_score")  bind(user_id=%uid)\n'
         '%sc = EXTRACT_FIELD(%cr, field="score") -> Int\n'
         '%ln = EXEC_ACTION(action="approve_loan", amount=50000.0)  '
         'bind(user_id=%uid, score=%sc)\n\n'
         "return %ln\n")
    mod = parse_text(t)
    assert validate(mod).valid
    tr = RT.execute(mod)
    assert tr.success, tr.error
    assert tr.final_result["approved"] == True
    s = {x.node_id: x for x in tr.steps}
    assert s["%ln"].input_params["user_id"] == "U_Dave"
    assert s["%ln"].input_params["score"] == 750


def test_read_then_write():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Eve")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%acc = EXEC_ACTION(action="get_account")  bind(user_id=%uid)\n'
         '%aid = EXTRACT_FIELD(%acc, field="account_id") -> Str\n'
         '%upd = EXEC_ACTION(action="update_balance", new_balance=200.0)  '
         'bind(account_id=%aid)\n\n'
         "return %upd\n")
    mod = parse_text(t)
    assert validate(mod).valid
    tr = RT.execute(mod)
    assert tr.success, tr.error
    assert tr.final_result["success"] == True


# ── Error cases ──

def test_missing_field():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Frank")\n'
         '%bad = EXTRACT_FIELD(%u, field="nonexistent")\n\n'
         "return %bad\n")
    mod = parse_text(t)
    tr = RT.execute(mod)
    assert not tr.success
    assert "nonexistent" in tr.error


def test_forward_ref_rejected():
    t = ("inputs: @task: Str\n\n"
         '%a = EXEC_ACTION(action="get_account")  bind(user_id=%later)\n'
         '%later = EXTRACT_FIELD(%u, field="user_id")\n'
         '%u = EXEC_ACTION(action="lookup_user", name="Gina")\n\n'
         "return %a\n")
    mod = parse_text(t)
    rep = validate(mod)
    assert not rep.valid
    assert any("USE_BEFORE_DEF" in str(e) for e in rep.errors)


def test_param_conflict():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Hank")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id")\n'
         '%acc = EXEC_ACTION(action="get_account", user_id="literal")  '
         'bind(user_id=%uid)\n\n'
         "return %acc\n")
    mod = parse_text(t)
    rep = validate(mod)
    assert not rep.valid
    assert any("PARAM_CONFLICT" in str(e) for e in rep.errors)


# ── Compatibility & roundtrip ──

def test_v01_compat():
    t = ("inputs: @task: Str\n\n"
         '%c0 = EXEC_ACTION(action="lookup_user", name="Jack")\n\n'
         "return %c0\n")
    mod = parse_text(t)
    assert validate(mod).valid
    tr = RT.execute(mod)
    assert tr.success


def test_roundtrip():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Ivy")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%acc = EXEC_ACTION(action="get_account")  bind(user_id=%uid)\n\n'
         "return %acc\n")
    mod1 = parse_text(t)
    t2 = to_text(mod1)
    mod2 = parse_text(t2)
    assert mod2.program.nodes[2].param_refs == {"user_id": "%uid"}
    assert mod2.program.nodes[1].params["field"] == "user_id"
    t3 = to_text(mod2)
    mod3 = parse_text(t3)
    assert mod3.program.nodes[2].param_refs == {"user_id": "%uid"}


# ── Deep chains ──

def test_8step():
    reg = make_registry()
    rt = DataflowRuntime(reg)
    lines = ["inputs: @task: Str", ""]
    lines.append('%s0 = EXEC_ACTION(action="lookup_user", name="Deep")')
    lines.append('%f0 = EXTRACT_FIELD(%s0, field="user_id") -> Str')
    # chain: each tool's input param name matches the upstream output field
    # step_a(x=...) → a_id, step_b(a_id=...) → b_id, etc.
    prev_ssa = "%f0"
    for i, (tool, input_param, outf) in enumerate(
            [("step_a", "x", "a_id"), ("step_b", "a_id", "b_id"),
             ("step_c", "b_id", "c_id"), ("step_d", "c_id", "d_id")]):
        ssa = f"%s{i+1}"
        lines.append(f'{ssa} = EXEC_ACTION(action="{tool}")  '
                     f'bind({input_param}={prev_ssa})')
        if i < 3:
            fsa = f"%f{i+1}"
            lines.append(f'{fsa} = EXTRACT_FIELD({ssa}, field="{outf}") -> Str')
            prev_ssa = fsa
    lines += ["", "return %s4"]
    mod = parse_text("\n".join(lines) + "\n")
    assert validate(mod).valid
    tr = rt.execute(mod)
    assert tr.success, tr.error
    assert tr.final_result["d_id"] == "d_c_b_a_U_Deep"


def test_side_effect_once():
    reg = make_registry()
    rt = DataflowRuntime(reg)
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Liam")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%upd = EXEC_ACTION(action="update_balance", new_balance=300.0)  '
         'bind(account_id=%uid)\n'
         '%ntf = EXEC_ACTION(action="send_notification", message="done")  '
         'bind(user_id=%uid)\n\n'
         "return %ntf\n")
    mod = parse_text(t)
    assert validate(mod).valid
    tr = rt.execute(mod)
    assert tr.success
    assert reg.get("update_balance").call_count == 1


def test_trace_proves_dataflow():
    t = ("inputs: @task: Str\n\n"
         '%u = EXEC_ACTION(action="lookup_user", name="Kate")\n'
         '%uid = EXTRACT_FIELD(%u, field="user_id") -> Str\n'
         '%acc = EXEC_ACTION(action="get_account")  bind(user_id=%uid)\n\n'
         "return %acc\n")
    mod = parse_text(t)
    tr = RT.execute(mod)
    assert tr.success
    ext = next(s for s in tr.steps if s.op == "EXTRACT_FIELD")
    assert ext.output == "U_Kate"
    exe = next(s for s in tr.steps if s.node_id == "%acc")
    assert exe.input_params["user_id"] == "U_Kate"


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{'ALL TESTS PASSED' if not failed else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
