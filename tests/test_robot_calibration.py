"""Robot qualification cannot come from partial or mismatched reference runs."""
import copy
import numpy as np
import pytest

from poc.robot_calibration import summarize, ranking_check, cost_envelope


def rows():
    result = []
    for actuator in ('direct', 'lagged'):
        for episode in range(16):
            for model in ('zero_command', 'physics_fit', 'sim_oracle'):
                result.append(dict(actuator=actuator, episode_id=episode, model_id=model,
                    phase='calibration', seed=9000, condition='clean', complete=True,
                    actuator_interval_ns=40_000_000, model_partition_ns=[40_000_000],
                    physics_step_ns=250_000, fallback_count=0,
                    initial_state=[0., 0., 0., 0.] + ([0., 0.] if actuator == 'lagged' else []),
                    target=[.2, .3], realized_cost=1. if model == 'zero_command' else .5))
    return result


def test_complete_same_task_reference_suite_qualifies_both_actuators():
    answer = summarize(rows())
    assert answer['qualified']
    assert answer['episode_count'] == 96
    assert len(answer['comparisons']) == 4
    assert all(c['mean_reduction'] == .5 for c in answer['comparisons'])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'clock', 'task', 'fallback', 'nan'])
def test_partial_confounded_or_invalid_runs_cannot_qualify(mutation):
    data = rows()
    if mutation == 'missing': data.pop()
    if mutation == 'duplicate': data.append(copy.deepcopy(data[0]))
    if mutation == 'clock': data[1]['actuator_interval_ns'] = 20_000_000
    if mutation == 'task': data[1]['target'] = [.3, .4]
    if mutation == 'fallback': data[1]['fallback_count'] = 1
    if mutation == 'nan': data[1]['realized_cost'] = float('nan')
    with pytest.raises(ValueError): summarize(data)


def test_no_headroom_on_one_actuator_remains_unqualified():
    data = rows()
    for row in data:
        if row['actuator'] == 'lagged' and row['model_id'] == 'physics_fit':
            row['realized_cost'] = .9
    assert not summarize(data)['qualified']


def test_resolved_ranking_inversion_is_detected_but_uncertain_tie_is_reported():
    assert ranking_check([1., 2., 3.], [1.01, 2.01, 3.01], .02, .02)['qualified']
    bad = ranking_check([1., 2., 3.], [2., 1., 3.], .01, .01)
    assert not bad['qualified'] and bad['resolved_inversions'] == 1
    near = ranking_check([1., 1.001], [1.001, 1.], .002, .002)
    assert near['qualified'] and near['uncertain_inversions'] == 1
    with pytest.raises(ValueError): ranking_check([np.inf], [0.], .01, .01)


def test_cost_envelope_preserves_largest_successive_refinement_discrepancy():
    result = cost_envelope([[1., 2.], [1.1, 2.01], [1.101, 2.009]])
    np.testing.assert_allclose(result, [.1, .01], atol=1e-12, rtol=0)
    with pytest.raises(ValueError): cost_envelope([[1., 2.], [1.1, 2.01]])


def test_common_bank_audit_reconciles_saved_scores_and_rejects_changed_recording():
    import torch
    from poc.reference_bank import score_bank
    from poc.mpc_calibration import ExactDoubleIntegrator
    from poc.mpc import rollout_candidates
    model = ExactDoubleIntegrator()
    bank = np.zeros((2, 6, 2)); bank[1] = .5
    state = torch.zeros(4, dtype=torch.float64)
    costs = rollout_candidates(model, state, [.2, .3], None, bank, (40_000_000,), 40_000_000)['costs']
    arrays, summary = score_bank([model] * 3, [model] * 3, state, [.2, .3], bank, costs)
    assert summary['ranking']['qualified']
    assert arrays['oracle_costs'].shape == (3, 2)
    with pytest.raises(ValueError, match='recorded'):
        score_bank([model] * 3, [model] * 3, state, [.2, .3], bank, costs + 1)
