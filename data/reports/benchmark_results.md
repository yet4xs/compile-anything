# TaskIR Runtime Benchmark

- programs: 5 (valid 5, completed 5)
- executor pools: default (1 per resource class)
- simulator: nominal costs, jitter=0, deterministic
- makespan schedules BOTH guarded branches optimistically;
  critical-path and sequential reflect the actual execution

| program | status | seq ms | makespan ms | crit-path ms | energy J | peak MB | lm | api/db | retries |
|---|---|---|---|---|---|---|---|---|---|
| financial_analysis | completed | 784 | 784.0 | 734.0 | 30.04 | 4600.0 | 2 | 1 | 0 |
| find_cheapest_flight | completed | 126 | 126.0 | 126.0 | 0.015 | 30.0 | 0 | 1 | 0 |
| find_cheapest_flight_verified | completed | 776 | 776.0 | 726.0 | 30.035 | 4600.0 | 2 | 1 | 0 |
| branch_select_demo | completed | 1378.1 | 1978.1 | 1378.1 | 52.035 | 4600.0 | 3 | 1 | 0 |
| rtl_debug | completed | 1098 | 1070.0 | 1020.0 | 40.035 | 4600.0 | 2 | 1 | 0 |

## Aggregate

```json
{
  "programs": 5,
  "valid": 5,
  "completed": 5,
  "mean_latency_sequential_ms": 832.42,
  "mean_makespan_scheduled_ms": 946.82,
  "mean_critical_path_ms": 796.82,
  "mean_energy_j": 30.432,
  "mean_memory_peak_mb": 3686.0,
  "total_lm_calls": 9,
  "total_api_calls": 5,
  "mean_energy_vs_70b_pct": 3.381,
  "mean_latency_vs_70b_pct": 37.873
}
```

70B single-shot reference: ~2500 ms, ~900 J, ~140000 MB.
