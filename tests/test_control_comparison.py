import copy
import pytest
from poc.control_comparison import paired_cost_summary


def row(seed, episode, cost, fallback=0):
    return dict(seed=seed, episode_id=episode, realized_cost=cost, phase='exploratory',
        complete=True, actuator='direct', condition='clean', actuator_interval_ns=40_000_000,
        model_partition_ns=[5_000_000] * 8, physics_step_ns=250_000, target=[.2, .3],
        initial_state=[0., 0., 0., 0.], fallback_count=fallback)


def test_pair_by_seed_and_episode_before_averaging_instead_of_file_order():
    baseline = [row(0, 0, 2.), row(0, 1, 4.), row(1, 0, 3.), row(1, 1, 3.)]
    method = [row(0, 0, 1.), row(0, 1, 3.), row(1, 0, 2.), row(1, 1, 1.)]
    answer = paired_cost_summary(method[::-1], baseline, [(0, 0), (0, 1), (1, 0), (1, 1)])
    assert answer['baseline_mean_cost'] == 3.
    assert answer['method_mean_cost'] == 1.75
    assert answer['relative_reduction'] == pytest.approx(1 - 1.75 / 3.)
    assert answer['seed_contrasts'] == [1., 1.5]
    assert answer['n_training_seeds'] == 2
    assert answer['statistical_status'] == 'descriptive_only'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nonfinite', 'clock', 'task', 'partial'])
def test_invalid_pairing_or_missingness_is_not_an_improvement(mutation):
    method = [row(9000, 0, .5), row(9000, 1, .6)]
    baseline = [row(9000, 0, 1.), row(9000, 1, 1.)]
    if mutation == 'missing': method.pop()
    if mutation == 'duplicate': method.append(copy.deepcopy(method[0]))
    if mutation == 'nonfinite': method[0]['realized_cost'] = float('nan')
    if mutation == 'clock': method[0]['actuator_interval_ns'] = 20_000_000
    if mutation == 'task': method[0]['target'] = [.3, .4]
    if mutation == 'partial': method[0]['complete'] = False
    with pytest.raises(ValueError): paired_cost_summary(method, baseline, [(9000, 0), (9000, 1)])


def test_fallback_episode_is_retained_and_one_development_seed_has_no_seed_interval():
    answer = paired_cost_summary([row(9000, 0, 2., fallback=1)], [row(9000, 0, 1.)], [(9000, 0)])
    assert answer['relative_reduction'] == -1.
    assert answer['method_fallback_count'] == 1
    assert answer['n_training_seeds'] == 1
    assert answer['confidence_bounds'] is None
