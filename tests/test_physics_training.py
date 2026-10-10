"""Endpoint optimization learns physical parameters without acceleration labels."""
import numpy as np
import pytest
import torch

from poc.models.common import Scales
from poc.physics_fit import PhysicsFit
from poc.physics_training import endpoint_loss, optimize_endpoints


def models(actuator):
    scales = Scales(torch.full((2,), .01), torch.full((2,), .1), torch.ones(2))
    learned = PhysicsFit(torch.tensor([.10, .02, .02, .05, .05]), actuator, scales)
    teacher = PhysicsFit(torch.tensor([.13, .026, .012, .04, .07]), actuator, scales)
    return learned, teacher


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_small_endpoint_fit_reduces_heldout_loss_and_records_nominal_selection(actuator):
    model, teacher = models(actuator)
    rng = np.random.default_rng(73)
    dimension = 4 if actuator == 'direct' else 6
    x = torch.tensor(rng.uniform(-1, 1, (48, dimension)), dtype=torch.float32)
    u = torch.tensor(rng.uniform(-1, 1, (48, 2)), dtype=torch.float32)
    h = torch.full((48, 1), .04)
    with torch.no_grad():
        y = teacher.predict(x, u, h).next_state
    data = {'state': x, 'command': u, 'h_s': h, 'target': y}
    training = {key: value[:32] for key, value in data.items()}
    validation = {key: value[32:] for key, value in data.items()}
    initial = float(endpoint_loss(model, validation, .005).detach())
    result = optimize_endpoints(model, training, validation, epochs=10, batch_size=16, seed=8201)
    assert result['best_validation_loss'] < initial
    assert result['best_epoch'] in (5, 10)
    assert result['epochs_completed'] == 10
    assert result['optimizer_updates'] == 20
    assert len(result['training_history']) == 20
    assert [r['epoch'] for r in result['validation_history']] == [5, 10]
    assert result['integration_step_s'] <= .005
    assert result['evaluation_step_s'] <= .00125
    assert all(np.isfinite(r['loss']) for r in result['training_history'])
    assert result['optimizer_state']
    assert result['rng_state'].dtype == torch.uint8 and result['rng_state'].numel() > 0
    assert all(torch.equal(model.state_dict()[key], value) for key, value in result['best_state'].items())


def test_known_activation_is_excluded_from_loss():
    model, _ = models('lagged')
    x, u, h = torch.zeros((2, 6)), torch.ones((2, 2)), torch.full((2, 1), .04)
    with torch.no_grad():
        y = model.predict(x, u, h).next_state
    y[:, 4:] = 1234
    assert endpoint_loss(model, {'state': x, 'command': u, 'h_s': h, 'target': y}, .005) < 1e-7


def test_fine_time_or_privileged_acceleration_labels_are_rejected():
    model, _ = models('direct')
    data = {'state': torch.zeros((2, 4)), 'command': torch.zeros((2, 2)),
            'h_s': torch.full((2, 1), .04), 'target': torch.zeros((2, 4))}
    with pytest.raises(ValueError, match='Fine-time'):
        endpoint_loss(model, {**data, 'h_s': torch.full((2, 1), .005)}, .005)
    with pytest.raises(ValueError, match='endpoint labels only'):
        endpoint_loss(model, {**data, 'acceleration': torch.zeros((2, 2))}, .005)


def test_interrupted_fit_resumes_optimizer_shuffle_and_selection_exactly():
    model, teacher = models('direct')
    x = torch.tensor(np.random.default_rng(8).uniform(-1, 1, (24, 4)), dtype=torch.float32)
    u, h = torch.ones((24, 2)), torch.full((24, 1), .04)
    with torch.no_grad():
        y = teacher.predict(x, u, h).next_state
    data = {'state': x, 'command': u, 'h_s': h, 'target': y}
    uninterrupted = optimize_endpoints(model, data, data, epochs=10, batch_size=8)
    resumed, _ = models('direct')
    saved = []
    class Interrupted(Exception):
        pass
    def checkpoint(state):
        saved.append(state)
        if state['epoch'] == 5:
            raise Interrupted()
    with pytest.raises(Interrupted):
        optimize_endpoints(resumed, data, data, epochs=10, batch_size=8, on_epoch=checkpoint)
    restored = optimize_endpoints(resumed, data, data, epochs=10, batch_size=8, resume=saved[-1])
    for key in uninterrupted['best_state']:
        assert torch.equal(uninterrupted['best_state'][key], restored['best_state'][key])
        assert torch.equal(uninterrupted['last_state'][key], restored['last_state'][key])
    assert uninterrupted['training_history'] == restored['training_history']
    assert uninterrupted['validation_history'] == restored['validation_history']
    assert torch.equal(uninterrupted['rng_state'], restored['rng_state'])
    assert uninterrupted['optimizer_updates'] == restored['optimizer_updates']
