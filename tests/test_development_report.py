"""Development reporting must preserve complete pairing and unfavorable outcomes."""
from copy import deepcopy

import pytest

from reports.audit_control_development_complete import aggregate


def rows():
    result = []
    for actuator in ("direct", "lagged"):
        for model in ("mlp_dt", "gru_dt", "node_generic", "power_correct", "node_structured"):
            for step in (40_000_000, 5_000_000):
                for episode in range(10):
                    result.append(dict(
                        actuator=actuator, model_id=model, model_partition_ns=[step]*(40_000_000//step),
                        episode_id=episode, seed=9000, phase="exploratory", condition="clean",
                        actuator_interval_ns=40_000_000, physics_step_ns=250_000,
                        initial_state=[episode / 10, 0, 0, 0] + ([0, 0] if actuator == "lagged" else []),
                        target=[0, 1], complete=True, realized_cost=.5 if model == "power_correct" else 1.,
                        success=False, fallback_count=0, power_mode=None,
                    ))
    return result


def test_complete_paired_workload_is_descriptive():
    result = aggregate(rows())
    assert len(result["means"]) == 20
    assert len(result["contrasts"]) == 16
    assert all(c["relative_reduction"] == .5 for c in result["contrasts"])
    assert result["statistical_status"] == "descriptive_only"
    assert result["n_training_seeds"] == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "confirmation", "different_task", "nonfinite"])
def test_incomplete_or_incompatible_evidence_is_rejected(change):
    data = rows()
    if change == "missing":
        data.pop()
    elif change == "duplicate":
        data[-1] = deepcopy(data[0])
    elif change == "confirmation":
        data[0]["phase"] = "confirmatory"
    elif change == "different_task":
        data[0]["target"] = [1, 1]
    else:
        data[0]["realized_cost"] = float("nan")
    with pytest.raises(ValueError):
        aggregate(data)


def test_losses_and_fallback_episodes_remain_in_the_summary():
    data = rows()
    for row in data:
        if row["model_id"] == "power_correct":
            row["realized_cost"] = 2.
            row["fallback_count"] = 1
    result = aggregate(data)
    assert all(c["relative_reduction"] == -1 for c in result["contrasts"])
    assert all(c["losses"] == 10 and c["wins"] == 0 for c in result["contrasts"])
    assert sum(m["fallbacks"] for m in result["means"]) == 40


def test_zero_baseline_cost_does_not_fabricate_a_percentage():
    data = rows()
    for row in data:
        if row["model_id"] == "mlp_dt":
            row["realized_cost"] = 0.
    result = aggregate(data)
    assert all(c["relative_reduction"] is None for c in result["contrasts"] if c["baseline_model"] == "mlp_dt")


@pytest.mark.parametrize("mutation", ["partition", "success", "fallback", "power"])
def test_malformed_partition_or_failure_status_is_rejected(mutation):
    data = rows()
    row = next(r for r in data if r["model_partition_ns"][0] == 5_000_000)
    if mutation == "partition":
        row["model_partition_ns"] = [5_000_000]
    elif mutation == "success":
        row["success"] = 1
    elif mutation == "fallback":
        row["fallback_count"] = -1
    else:
        row["power_mode"] = "low"
    with pytest.raises(ValueError):
        aggregate(data)
