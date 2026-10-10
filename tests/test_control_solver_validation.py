"""Nominal solver checks retain the selected field and all validation targets."""
import copy

import numpy as np
import pytest
import torch


def damped_fixture():
    from test_confirmation_solver import unit_mass_node
    from poc.models.common import Scales
    from poc.train import validate_nominal
    model = unit_mass_node().float()
    with torch.no_grad():
        model.network.weight[2, 4] = -.8
        model.network.weight[3, 5] = -.8
    rng = np.random.default_rng(4918)
    commands = rng.uniform(-1, 1, (4, 32, 2))
    states = np.zeros((4, 33, 4))
    states[:, 0] = rng.uniform(-1, 1, (4, 4))
    # Decay 20/s keeps the two/four-substep errors above float32 roundoff.
    decay = np.exp(-20 * .04)
    for i in range(32):
        q, v, u = states[:, i, :2], states[:, i, 2:4], commands[:, i]
        states[:, i + 1, :2] = q + v * (1 - decay) / 20 + u * (.04 / 20 - (1 - decay) / 400)
        states[:, i + 1, 2:4] = v * decay + u * (1 - decay) / 20
    arrays = dict(state=states, command=commands,
        duration_ns=np.full((4, 32), 40_000_000, dtype=np.int64),
        time_ns=np.tile(np.arange(33, dtype=np.int64) * 40_000_000, (4, 1)),
        trajectory_id=np.arange(4, dtype=np.int64))
    scales = Scales(torch.ones(2), torch.ones(2), torch.ones(2))
    selected = validate_nominal(model, arrays, scales, 'shared')
    return model, arrays, scales, selected


def test_nominal_solver_audit_keeps_tensors_targets_and_changes_only_integration():
    from poc.control_solver_validation import audit_validation
    model, arrays, scales, selected = damped_fixture()
    before = copy.deepcopy(model.state_dict())
    result = audit_validation(model, arrays, scales, selected)
    assert result['state'] == 'complete'
    assert result['training'] is False and result['reselection'] is False
    assert [row['substeps'] for row in result['resolutions']] == [1, 2, 4]
    assert all(row['observed']['one_count'] == 128 and row['observed']['four_count'] == 116
        for row in result['resolutions'])
    scores = [row['observed']['objective'] for row in result['resolutions']]
    assert scores[0] > scores[1] > scores[2]
    assert result['resolutions'][0]['observed'] == selected
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert result['checkpoint_changed'] is False
    assert result['material_nominal_fit_change'] is True
    assert result['continuous_field_claim_requires_interpretation'] is True


@pytest.mark.parametrize('mutation', ['score', 'targets'])
def test_nominal_audit_rejects_a_selected_score_or_coverage_mismatch(mutation):
    from poc.control_solver_validation import audit_validation
    model, arrays, scales, selected = damped_fixture()
    if mutation == 'score':
        selected['objective'] += .1
    else:
        selected['four_count'] -= 1
    with pytest.raises(ValueError, match='selected'):
        audit_validation(model, arrays, scales, selected)


def test_nominal_solver_audit_requires_a_node_and_original_training_precision():
    from poc.control_solver_validation import audit_validation
    model, arrays, scales, selected = damped_fixture()
    with pytest.raises(ValueError, match='float32'):
        audit_validation(model.double(), arrays, scales, selected)
    with pytest.raises(ValueError, match='NODE'):
        audit_validation(torch.nn.Linear(4, 4), arrays, scales, selected)


def test_validation_registry_uses_exact_ten_frozen_node_fits():
    from poc.control_solver_validation import validation_jobs
    from test_control_nominal_selection_audit import frozen
    jobs = validation_jobs(frozen())
    assert len(jobs) == 10
    assert {(row['actuator'], row['seed']) for row in jobs} == {
        (actuator, seed) for actuator in ('direct', 'lagged') for seed in range(5)}
    assert all(row['model_id'] == 'node_generic' for row in jobs)
    assert all(row['source_context'] == 'study/source' for row in jobs)
    value = frozen()
    value['selected'].pop()
    with pytest.raises(ValueError):
        validation_jobs(value)


def test_validation_prepare_rejects_incomplete_freeze_before_writing_jobs(tmp_path):
    from poc.control_solver_validation import prepare
    from poc.io import atomic_json
    atomic_json(tmp_path / 'config/control_confirmation_cells.json', dict(state='running'))
    with pytest.raises(ValueError):
        prepare(tmp_path)
    assert not (tmp_path / 'config/control_solver_validation_jobs.json').exists()
