import pytest
from poc.control_attribution import attribution_values


def test_power_only_and_full_structure_degradations_remain_separate():
    result = attribution_values([1., 2.], [1., 2.], [1.2, 1.9],
                                [4., 5.], [2., 3.], [2.5, 2.4])
    assert result['A_by_seed'] == [2., 2.]
    assert result['B_by_seed'] == pytest.approx([1.7, 2.5])
    assert result['A'] == 2.
    assert result['B'] == pytest.approx(2.1)
    assert result['fraction_explained'] is None
    assert result['statistical_status'] == 'descriptive_only'


def test_explained_fraction_requires_a_positive_external_bound_for_full_improvement():
    values = ([1., 2.], [1., 2.], [1.2, 1.9], [4., 5.], [2., 3.], [2.5, 2.4])
    result = attribution_values(*values, structured_lower_bound=.5)
    assert result['fraction_explained'] == pytest.approx(2. / 2.1)
    assert attribution_values(*values, structured_lower_bound=-.1)['fraction_explained'] is None
    with pytest.raises(ValueError): attribution_values(*values, structured_lower_bound=3.)


def test_no_positive_full_improvement_has_no_manufactured_explained_fraction():
    result = attribution_values([1.], [1.], [1.], [3.], [2.], [4.])
    assert result['A'] == 1.
    assert result['B'] == -1.
    assert result['fraction_explained'] is None


@pytest.mark.parametrize('mutation', ['nominal', 'missing', 'nan'])
def test_changed_nominal_map_or_missing_seed_is_not_a_scaling_attribution(mutation):
    arrays = [[1., 1.], [1., 1.], [1., 1.], [3., 3.], [2., 2.], [2., 2.]]
    if mutation == 'nominal': arrays[1][0] = 1.01
    if mutation == 'missing': arrays[3].pop()
    if mutation == 'nan': arrays[0][0] = float('nan')
    with pytest.raises(ValueError): attribution_values(*arrays)


@pytest.mark.parametrize('mutation', [None, 'knots', 'weights', 'time', 'missing_file'])
def test_nominal_identity_checks_raw_commands_and_candidates_not_only_equal_cost(tmp_path, monkeypatch, mutation):
    import numpy as np
    from test_control_episode import run
    from poc.control_artifacts import save_episode, _npz
    from poc.control_attribution import verify_nominal_power_outputs
    from poc.io import read_json, atomic_json
    _, result, _ = run(monkeypatch)
    result['model_partition_ns'] = (40_000_000,)
    low, correct = tmp_path / 'low', tmp_path / 'correct'
    result['power_mode'] = 'low'
    save_episode(low, result, model_id='power_correct', checkpoint_hash='a' * 64)
    result['power_mode'] = 'direct'
    save_episode(correct, result, model_id='power_correct', checkpoint_hash='a' * 64)
    if mutation == 'knots':
        with np.load(correct / 'candidates.npz') as original:
            arrays = {name: original[name].copy() for name in original.files}
        arrays['knots'][0, 0, 0, 0] = .5
        _npz(correct / 'candidates.npz', **arrays)
    if mutation in {'weights', 'time'}:
        value = read_json(correct / 'episode.json')
        value['checkpoint_hash' if mutation == 'weights' else 'actuator_interval_ns'] = 'b' * 64 if mutation == 'weights' else 20_000_000
        atomic_json(correct / 'episode.json', value)
    if mutation == 'missing_file': (correct / 'action_segments.jsonl').unlink()
    if mutation:
        with pytest.raises((ValueError, OSError)): verify_nominal_power_outputs(low, correct)
    else:
        assert verify_nominal_power_outputs(low, correct)['matched']


def test_nominal_identity_can_check_the_third_prespecified_power_mode(tmp_path, monkeypatch):
    from test_control_episode import run
    from poc.control_artifacts import save_episode
    from poc.control_attribution import verify_nominal_power_outputs
    _, result, _ = run(monkeypatch)
    result['model_partition_ns'] = (40_000_000,)
    result['power_mode'] = 'low'
    low, other = tmp_path / 'low', tmp_path / 'other'
    save_episode(low, result, model_id='power_correct', checkpoint_hash='a' * 64)
    result['power_mode'] = 'lagged'
    save_episode(other, result, model_id='power_correct', checkpoint_hash='a' * 64)
    with pytest.raises(ValueError):
        verify_nominal_power_outputs(low, other)
    assert verify_nominal_power_outputs(low, other, comparison_mode='lagged')['matched']
    with pytest.raises(ValueError):
        verify_nominal_power_outputs(low, other, comparison_mode='low')
