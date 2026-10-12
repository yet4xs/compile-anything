"""tau3 semantic skill oracle — Phase 8A Task 2.

Hand-verified per-tool ground truth for the AGENT-side runtime pool
(non-discoverable @is_tool). Built ONLY from tool definitions and
implementations (third_party/tau3-bench); model predictions and task
reference answers were NOT consulted. Reference actions are used solely
by the frozen scorer, never here.

Fields per tool:
  semantic_skill: Skill ISA op (EXEC_ACTION keeps tool identity via the
                  action/params; FETCH/SEARCH/CALCULATE are semantic)
  effect_class:   read_only | state_update_reversible | state_creation |
                  state_transition_irreversible | communication |
                  agent_internal
  policy_guarded: explicit validation/policy checks in the implementation
  evidence:       what was read to justify the label
  confidence:     verified (implementation read) | heuristic (name+doc rule)

Review coverage (advisor requirement): ALL state-mutating pool tools were
implementation-reviewed; read tools use the documented name/description
rule and are marked heuristic unless their body was read.
"""
from __future__ import annotations

# (domain, tool_name) -> entry
TAU3_SEMANTIC_ORACLE = {
    # ---------------- airline ----------------
    ("airline", "book_reservation"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_creation", policy_guarded=True,
        evidence="impl read: creates Reservation, deepcopies payments, "
                 "_get_new_reservation_id; consumed payment history",
        confidence="verified"),
    ("airline", "cancel_reservation"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_transition_irreversible", policy_guarded=False,
        evidence="impl read: status='cancelled' + negative refunds; no "
                 "un-cancel path; seats release explicitly not implemented",
        confidence="verified"),
    ("airline", "update_reservation_baggages"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=True,
        evidence="impl read: re-settable baggage counts with payment_id",
        confidence="verified"),
    ("airline", "update_reservation_flights"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=True,
        evidence="impl read: replaces ENTIRE flight list; requires full "
                 "segment array + payment_id",
        confidence="verified"),
    ("airline", "update_reservation_passengers"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=False,
        evidence="impl read: passenger list replacement",
        confidence="verified"),
    ("airline", "send_certificate"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_creation", policy_guarded=True,
        evidence="impl read: adds certificate to user; docstring 'Be "
                 "careful!'; capped at 3 per task",
        confidence="verified"),
    ("airline", "get_user_details"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("airline", "get_reservation_details"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("airline", "get_flight_status"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("airline", "list_all_airports"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("airline", "search_direct_flight"): dict(
        semantic_skill="SEARCH", effect_class="read_only",
        policy_guarded=False,
        evidence="impl read: filters db.flights by origin/dest/date/status",
        confidence="verified"),
    ("airline", "search_onestop_flight"): dict(
        semantic_skill="SEARCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type (search rule)",
        confidence="heuristic"),
    ("airline", "calculate"): dict(
        semantic_skill="CALCULATE", effect_class="read_only",
        policy_guarded=False, evidence="registry generic; pure math",
        confidence="heuristic"),
    ("airline", "transfer_to_human_agents"): dict(
        semantic_skill="EXEC_ACTION", effect_class="communication",
        policy_guarded=False,
        evidence="registry generic, mutates_state=False (framework "
                 "override per toolkit docs)",
        confidence="verified"),

    # ---------------- banking_knowledge (agent pool only) ----------------
    ("banking_knowledge", "submit_transaction"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_creation", policy_guarded=True,
        evidence="impl read: COMPLETED transaction record with rewards; "
                 "validates CREDIT_CARD_REWARDS card types",
        confidence="verified"),
    ("banking_knowledge", "change_user_email"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=False,
        evidence="impl read: update_record_in_db users.email",
        confidence="verified"),
    ("banking_knowledge", "apply_for_credit_card"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_creation", policy_guarded=True,
        evidence="impl read: VALID_CREDIT_CARD_TYPES check; deterministic "
                 "application id",
        confidence="verified"),
    ("banking_knowledge", "submit_referral"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_creation", policy_guarded=False,
        evidence="impl read (4382): referral submission record",
        confidence="verified"),
    ("banking_knowledge", "log_verification"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_creation", policy_guarded=True,
        evidence="impl read: audit record; requires 2-of-4 identity "
                 "fields verified (documented protocol)",
        confidence="verified"),
    ("banking_knowledge", "get_user_information_by_id"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("banking_knowledge", "get_user_information_by_email"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("banking_knowledge", "get_user_information_by_name"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("banking_knowledge", "get_credit_card_accounts_by_user"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("banking_knowledge", "get_credit_card_transactions_by_user"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("banking_knowledge", "get_referrals_by_user"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("banking_knowledge", "get_current_time"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("banking_knowledge", "KB_search"): dict(
        semantic_skill="SEARCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type (search rule)",
        confidence="heuristic"),
    ("banking_knowledge", "grep"): dict(
        semantic_skill="SEARCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type (search rule)",
        confidence="heuristic"),
    ("banking_knowledge", "shell"): dict(
        semantic_skill="EXEC_ACTION", effect_class="read_only",
        policy_guarded=False,
        evidence="registry: read/GENERIC, mutates_state=False; generic "
                 "command execution keeps identity -> EXEC_ACTION",
        confidence="heuristic"),
    ("banking_knowledge", "rewrite_context"): dict(
        semantic_skill="EXEC_ACTION", effect_class="agent_internal",
        policy_guarded=False, evidence="docstring: replaces working context",
        confidence="verified"),
    ("banking_knowledge", "transfer_to_human_agents"): dict(
        semantic_skill="EXEC_ACTION", effect_class="communication",
        policy_guarded=False, evidence="registry generic, mutates False",
        confidence="verified"),
    ("banking_knowledge", "request_human_agent_transfer"): dict(
        semantic_skill="EXEC_ACTION", effect_class="communication",
        policy_guarded=False, evidence="docstring: request transfer",
        confidence="verified"),
    ("banking_knowledge", "call_discoverable_agent_tool"): dict(
        semantic_skill="EXEC_ACTION", effect_class="agent_internal",
        policy_guarded=True,
        evidence="docstring: gated on prior unlock of the target tool",
        confidence="verified"),
    ("banking_knowledge", "call_discoverable_user_tool"): dict(
        semantic_skill="EXEC_ACTION", effect_class="agent_internal",
        policy_guarded=True,
        evidence="docstring: only for tools previously given to the user",
        confidence="verified"),
    ("banking_knowledge", "give_discoverable_user_tool"): dict(
        semantic_skill="EXEC_ACTION", effect_class="agent_internal",
        policy_guarded=False, evidence="docstring: pass tool to user",
        confidence="verified"),
    ("banking_knowledge", "unlock_discoverable_agent_tool"): dict(
        semantic_skill="EXEC_ACTION", effect_class="agent_internal",
        policy_guarded=True,
        evidence="docstring: requires KB-found credentials",
        confidence="verified"),
    ("banking_knowledge", "list_discoverable_agent_tools"): dict(
        semantic_skill="FETCH", effect_class="agent_internal",
        policy_guarded=False, evidence="registry read",
        confidence="heuristic"),
    ("banking_knowledge", "list_discoverable_user_tools"): dict(
        semantic_skill="FETCH", effect_class="agent_internal",
        policy_guarded=False, evidence="registry read",
        confidence="heuristic"),

    # ---------------- retail ----------------
    ("retail", "cancel_pending_order"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_transition_irreversible", policy_guarded=True,
        evidence="impl read: pending-only guard + reason enum {'no longer "
                 "needed','ordered by mistake'}; issues refund",
        confidence="verified"),
    ("retail", "exchange_delivered_order_items"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_transition_irreversible", policy_guarded=True,
        evidence="impl read: delivered-only; positional item pairing; "
                 "payment delta",
        confidence="verified"),
    ("retail", "return_delivered_order_items"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_transition_irreversible", policy_guarded=True,
        evidence="impl read: status -> 'return requested', refund via "
                 "payment method",
        confidence="verified"),
    ("retail", "modify_pending_order_items"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=True,
        evidence="impl read: pending-only; same-product positional "
                 "constraint; payment delta",
        confidence="verified"),
    ("retail", "modify_pending_order_address"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=True,
        evidence="impl read (401): pending-only guard",
        confidence="verified"),
    ("retail", "modify_pending_order_payment"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=True,
        evidence="impl read (535): pending-only + payment overwrite",
        confidence="verified"),
    ("retail", "modify_user_address"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=False,
        evidence="impl read (615): default address overwrite",
        confidence="verified"),
    ("retail", "find_user_id_by_email"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("retail", "find_user_id_by_name_zip"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("retail", "get_user_details"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("retail", "get_order_details"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("retail", "get_product_details"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("retail", "get_item_details"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("retail", "list_all_product_types"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("retail", "calculate"): dict(
        semantic_skill="CALCULATE", effect_class="read_only",
        policy_guarded=False, evidence="registry generic; pure math",
        confidence="heuristic"),
    ("retail", "transfer_to_human_agents"): dict(
        semantic_skill="EXEC_ACTION", effect_class="communication",
        policy_guarded=False, evidence="registry generic, mutates False",
        confidence="verified"),

    # ---------------- telecom ----------------
    ("telecom", "suspend_line"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=True,
        evidence="impl read: requires status Active; records "
                 "suspension_start_date; max 6 months policy",
        confidence="verified"),
    ("telecom", "resume_line"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=True,
        evidence="impl read: requires status Suspended",
        confidence="verified"),
    ("telecom", "enable_roaming"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=False,
        evidence="impl read: sets roaming_enabled=True; idempotent-ish "
                 "('already enabled' message)",
        confidence="verified"),
    ("telecom", "disable_roaming"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_update_reversible", policy_guarded=False,
        evidence="impl read (566): sets roaming_enabled=False",
        confidence="verified"),
    ("telecom", "refuel_data"): dict(
        semantic_skill="EXEC_ACTION",
        effect_class="state_creation", policy_guarded=True,
        evidence="impl read: adds data + charge record at plan rate; "
                 "positive-amount guard",
        confidence="verified"),
    ("telecom", "send_payment_request"): dict(
        semantic_skill="EXEC_ACTION", effect_class="communication",
        policy_guarded=True,
        evidence="impl read (361): bill-scoped request with checks",
        confidence="verified"),
    ("telecom", "get_customer_by_id"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("telecom", "get_customer_by_name"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("telecom", "get_customer_by_phone"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("telecom", "get_details_by_id"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=True,
        evidence="docstring: ID must be a valid existing ID (guard)",
        confidence="heuristic"),
    ("telecom", "get_bills_for_customer"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("telecom", "get_data_usage"): dict(
        semantic_skill="FETCH", effect_class="read_only",
        policy_guarded=False, evidence="registry read type",
        confidence="heuristic"),
    ("telecom", "transfer_to_human_agents"): dict(
        semantic_skill="EXEC_ACTION", effect_class="communication",
        policy_guarded=False, evidence="registry generic, mutates False",
        confidence="verified"),
}

# Coarse ontology for L2 dual reporting: exact skill vs coarse class.
COARSE = {
    "FETCH": "read", "SEARCH": "read", "CALCULATE": "read",
    "EXEC_ACTION": "act", "GENERATE": "reason", "EXTRACT": "reason",
    "TRANSLATE": "reason", "CLASSIFY": "reason", "MERGE": "reason",
    "VERIFY": "reason", "SELECT": "control",
}
