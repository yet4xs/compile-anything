# τ³-bench Effect Semantics Audit

- tasks: **2546** across 4 domains; **2443** expect multi-action sequences (transactions), **162** carry NL assertions (verification criteria)
- classified ground-truth actions: **14834**

## Action effect classification

| class | count | share |
|---|---:|---:|
| reversible_state | 6920 | 46.6% |
| irreversible_world | 5606 | 37.8% |
| other | 1840 | 12.4% |
| read_only | 468 | 3.2% |

Top actions per class:

- **read_only**: `get_order_details`(168), `get_user_details`(71), `find_user_id_by_name_zip`(61), `get_reservation_details`(57), `get_product_details`(54)
- **irreversible_world**: `grant_app_permission`(2048), `refuel_data`(1120), `reboot_device`(1040), `reset_apn_settings`(1032), `unlock_discoverable_agent_tool`(275)
- **other**: `reseat_sim_card`(1000), `call_discoverable_agent_tool`(428), `disconnect_vpn`(128), `log_verification`(81), `call_discoverable_user_tool`(62)
- **reversible_state**: `set_network_mode_preference`(1152), `toggle_airplane_mode`(1127), `toggle_roaming`(1120), `enable_roaming`(1120), `toggle_data`(1120)

## Per domain

| domain | tasks | with actions | multi-action | NL-assert | read-only | reversible | irreversible |
|---|---:|---:|---:|---:|---:|---:|---:|
| airline | 50 | 43 | 25 | 50 | 92 | 0 | 49 |
| retail | 114 | 112 | 92 | 112 | 370 | 109 | 26 |
| telecom | 2285 | 2285 | 2240 | 0 | 0 | 6791 | 5256 |
| banking_knowledge | 97 | 97 | 86 | 0 | 6 | 20 | 275 |

## Expressibility vs the Phase 3 effect proposal

- read-only actions: pure skills — expressible today (468/14834 = 3.2%)
- reversible state changes: `state` effect class + rollback semantics — expressible per proposal (6920/14834 = 46.6%)
- irreversible world actions: `world` effect class, and the proposal's rule 'no verify-retry across a world action' is exactly the safety property τ³ rewards (5606/14834 = 37.8%)
- multi-action transactions (2443 tasks): the proposal's per-class linear effect chain serializes same-class actions — sufficient for ordering, but τ³ transactions may need all-or-nothing rollback ACROSS classes (world+state in one transaction), which the current single-class chains do NOT express -> **documented gap: cross-class transactional effect regions**
- verification: NL assertions map to VERIFY nodes; tasks with assertions need verify-after-effect ordering, expressible via after-edges + effect chain

**Verdict**: the proposal covers read/reversible/irreversible classes and retry safety; the real gap is cross-class transactional semantics — recorded for effect-system v0.2 design (not implemented this phase).
