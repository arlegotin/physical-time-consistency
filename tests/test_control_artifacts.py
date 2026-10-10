"""Complete control costs retain every candidate and physical observation."""
import numpy as np
import pytest

from poc.io import read_json
from test_control_episode import run


def test_control_files_reconcile_cost_and_preserve_failed_candidates(tmp_path, monkeypatch):
    from poc.control_artifacts import save_episode
    _, result, _ = run(monkeypatch, failure=7)
    summary = save_episode(tmp_path, result)
    assert summary['realized_cost'] == result['realized_cost']
    assert summary['fallback_count'] == 1
    assert summary['complete'] is True
    assert len((tmp_path / 'decisions.jsonl').read_text().splitlines()) == 40
    with np.load(tmp_path / 'candidates.npz', allow_pickle=False) as arrays:
        assert arrays['knots'].shape == (40, 8, 6, 2)
        assert np.isinf(arrays['costs'][7]).all()
        assert (arrays['statuses'][7] == 'model_failure').all()
        assert (arrays['reasons'][7] == 'fixture failure').all()
    with np.load(tmp_path / 'true_trajectories.npz', allow_pickle=False) as arrays:
        assert arrays['time_ns'][-1] == 1_600_000_000
    assert read_json(tmp_path / 'episode.json')['success'] == result['success']


def test_actuator_rate_episode_keeps_its_distinct_experiment_identity(tmp_path, monkeypatch):
    from poc.control_artifacts import save_episode
    _, result, _ = run(monkeypatch, interval=20_000_000)
    assert save_episode(tmp_path, result)['experiment_id'] == 'E04'


@pytest.mark.parametrize('mutation', ['partial', 'missing_candidate', 'unknown_infinity', 'changed_cost'])
def test_invalid_control_data_cannot_be_written_as_complete(tmp_path, monkeypatch, mutation):
    from poc.control_artifacts import save_episode
    _, result, _ = run(monkeypatch)
    if mutation == 'partial':
        result['true_times_ns'] = result['true_times_ns'][:-1]
        result['true_states'] = result['true_states'][:-1]
    elif mutation == 'missing_candidate':
        result['candidates'][3]['knots'] = result['candidates'][3]['knots'][:-1]
    elif mutation == 'unknown_infinity':
        result['candidates'][3]['costs'][0] = np.inf
    else:
        result['realized_cost'] += 1
    with pytest.raises(ValueError):
        save_episode(tmp_path, result)
    assert not (tmp_path / 'episode.json').exists()
