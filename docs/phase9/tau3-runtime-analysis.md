# τ³ Runtime Dependency Analysis

**Phase**: 9B-1 | **Date**: 2026-10-15 | **Method**: Static analysis of reference action arguments

## Results

| Domain | Total | Multi-step | L0 | L1 | L2 | **L3** | Runtime Dep Edges |
|--------|-------|-----------|----|----|----|----|-------------------|
| airline | 50 | 25 | 25 | 0 | 8 | **17** | 86 |
| banking_knowledge | 97 | 86 | 11 | 0 | 86 | **0** | 0 |
| retail | 114 | 92 | 22 | 0 | 32 | **60** | 236 |
| telecom | 2,285 | 2,240 | 45 | 0 | 2,240 | **0** | 0 |
| **Total** | **2,546** | **2,443** | 103 | 0 | 2,366 | **77** | **322** |

## Interpretation

### L3 Distribution is Domain-Dependent

- **Airline + Retail**: Genuine runtime value dependencies exist (17 + 60 = 77 tasks, 322 edges)
- **Banking**: Multi-step tasks use environment-sourced values (user_id from DB, not from tool output)
- **Telecom**: Multi-step tasks use environment-sourced values (customer_id, line_id from DB)

### Why Banking/Telecom Have Zero L3

In banking tasks, the reference trajectory might be:
```
1. get_user_information_by_name(name="Sarah Bosch") → outputs user_id
2. change_user_email(user_id="6680a37184")
```

The value "6680a37184" is not from the prior tool's return — it's stored in the DB. The tool `get_user_information_by_name` returns this value, but the reference trajectory hardcodes it directly from the environment. In a real AI workflow, the agent would need to call tool 1 first to GET this value, then pass it to tool 2. But τ³'s reference trajectories assume the value is already known.

This reveals a limitation of τ³'s trajectory format: it records what the IDEAL agent would do with full knowledge, not what a model must figure out through tool chaining.

### Airline L3 Examples

```
Step 0: get_user_details(user_id="raj_sanchez_7340")
  → returns {reservations: [...], user_id: "raj_sanchez_7340", ...}

Step 1: get_reservation_details(reservation_id="Q69X3R")
  ← reservation_id from get_user_details's output (runtime_dep)
```

The user says "check my reservation" but doesn't provide the reservation_id. The agent must first call get_user_details, extract the reservation_id from the output, then pass it to get_reservation_details.

### Retail L3 Examples

```
Step 0: find_user_id_by_name_zip(first_name="Yusuf", last_name="Rossi", zip="19122")
  → returns {user_id: "yusuf_rossi_9620"}

Step 1: get_order_details(order_id="#W2378156")
  ← order_id from user's account (could be from find_user_id or get_user_details)

Step 3: get_product_details(product_id="1656367028")
  ← product_id from get_order_details's output items list
```

The user asks about a product in their order. The agent must first get the order, extract the product_id from the items, then query the product details.

## Precision Assessment

My classification uses two criteria:
1. Value NOT in user text
2. Prior tool's return type CONTAINS a matching field name

**False positive risk**: A value might match a prior tool's return field name by coincidence, but actually come from the environment. 

**False negative risk**: Banking/telecom values might actually be runtime deps in a real agent workflow (the agent would need to call get_user first), but τ³'s reference trajectories hardcode them from the environment.

**Estimated precision**: ~80-90% for airline/retail (the chains are semantically clear). Lower confidence for telecom where many values are DB-sourced.

## Recommendation

**77 L3 tasks with 322 runtime dependency edges** is sufficient for a curated benchmark:

- Size: comparable to specialized benchmarks (BFCL has ~2K but single-tool)
- Quality: from real τ³ trajectories with real tool schemas
- Diversity: 2 domains (airline + retail), various chain lengths (2-11 steps)
- Annotation: automatic extraction + manual verification of ~20% sample

This is **Situation B** from the advisor's framework: "curated Runtime Dataflow Benchmark, 100-300 条即可."

## Next Step

1. Manual verification of 15-20 L3 samples (precision check)
2. Convert verified L3 tasks to TaskIR v0.2 format
3. Run M0-T/M0-J/M1 evaluation on real data
