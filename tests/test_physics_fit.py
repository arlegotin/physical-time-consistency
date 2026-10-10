"""References learn endpoints; unit oracles never supply training parameters."""
import numpy as np
import pytest
import torch

from poc.models.common import Scales
from poc.physics_fit import AccelRidge, PhysicsFit, fit_ridge


def scales():
    return Scales(torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64),
                  torch.ones(2, dtype=torch.float64))


def test_physics_initialization_is_exact_and_mass_stays_positive_definite():
    model = PhysicsFit(torch.tensor([.10, .02, .02, .05, .05], dtype=torch.float64), 'direct', scales())
    np.testing.assert_allclose(model.physical_parameters().detach().numpy(), [.10, .02, .02, .05, .05], atol=1e-14)
    q = torch.zeros((1001, 2), dtype=torch.float64)
    q[:, 1] = torch.linspace(-np.pi, np.pi, 1001)
    assert torch.linalg.eigvalsh(model.mass_matrix(q)).min() > 0
    with torch.no_grad():
        model.raw[:] = torch.tensor([-5., -3., 2., -4., -6.])
    assert torch.linalg.eigvalsh(model.mass_matrix(q)).min() > 0


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_torch_reference_matches_independent_equations_and_converged_integration(actuator):
    from poc.oracle_rigidbody import integrate, rhs
    parameters = torch.tensor([.10, .02, .02, .05, .05], dtype=torch.float64)
    model = PhysicsFit(parameters, actuator, scales())
    state = np.array([.2, -.7, .4, -.3] + ([.6, -.5] if actuator == 'lagged' else []))
    command = np.array([-.4, .8])
    oracle = {'theta1': .10, 'theta2': .02, 'theta3': .02, 'armature': [.02, .02],
              'damping': [.05, .05], 'tau_s': .08}
    effort = state[4:] if actuator == 'lagged' else command
    acceleration = model.acceleration(torch.tensor(state[:4][None]), torch.tensor(effort[None]))
    expected_field = rhs(0., state[:4], command, state[4:] if actuator == 'lagged' else None, oracle)
    np.testing.assert_allclose(acceleration.detach().numpy()[0], expected_field[2:], atol=1e-12)
    predicted = model.predict(torch.tensor(state[None]), torch.tensor(command[None]), .04).next_state
    expected = integrate(state, command, .04, oracle, actuator)['state']
    np.testing.assert_allclose(predicted.detach().numpy()[0], expected, atol=2e-8, rtol=2e-8)
    split = torch.tensor(state[None])
    for _ in range(8):
        split = model.predict(split, torch.tensor(command[None]), .005).next_state
    np.testing.assert_allclose(split.detach().numpy(), predicted.detach().numpy(), atol=2e-8, rtol=2e-8)


def test_lagged_new_command_is_excluded_from_instantaneous_acceleration_and_w_is_exact():
    model = PhysicsFit(torch.tensor([.10, .02, .02, .05, .05], dtype=torch.float64), 'lagged', scales())
    state = torch.tensor([[.2, .3, 0., 0., .5, -.4]], dtype=torch.float64)
    command = torch.tensor([[.8, -.2]], dtype=torch.float64, requires_grad=True)
    before = state.clone()
    packet = model.predict(state, command, .005)
    expected_w = state[:, 4:] + (command - state[:, 4:]) * (-torch.expm1(torch.tensor(-.005 / .08)))
    torch.testing.assert_close(packet.next_state[:, 4:], expected_w, atol=1e-9, rtol=1e-7)
    assert torch.equal(state, before)
    packet.next_state[:, :4].sum().backward()
    assert command.grad is not None and model.raw.grad is not None
    assert torch.isfinite(model.raw.grad).all()


def test_ridge_uses_only_train_endpoints_and_does_not_regularize_the_constant(tmp_path, monkeypatch):
    import json
    import poc.dataset
    rng = np.random.default_rng(61)
    state = rng.uniform(-1, 1, (16, 33, 4))
    commands = rng.uniform(-2, 2, (16, 32, 2))
    h_ns = np.full((16, 32), 40_000_000, dtype=np.int64)
    # Pure constant acceleration provides an independent intercept answer.
    for t in range(32):
        state[:, t + 1, 2:] = state[:, t, 2:] + .04 * np.array([.7, -.4])
    arrays = {'state': state, 'command': commands, 'duration_ns': h_ns}
    opened = []
    def load(path, phase, role, purpose, *, audit_path=None):
        opened.append((phase, role, purpose))
        return arrays
    monkeypatch.setattr(poc.dataset, 'load_corpus', load)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'role': 'train', 'actuator': 'direct'}))
    result = fit_ridge(path)
    assert opened == [('exploratory', 'train', 'training')]
    np.testing.assert_allclose(result['coefficients'][0], [.7, -.4], atol=1e-10)
    np.testing.assert_allclose(np.asarray(result['coefficients'])[1:], 0, atol=1e-10)
    path.write_text(json.dumps({'role': 'confirmation', 'actuator': 'direct'}))
    with pytest.raises(ValueError):
        fit_ridge(path)
    assert len(opened) == 1


def test_lagged_ridge_training_uses_exact_average_activation(tmp_path, monkeypatch):
    import json
    import poc.dataset
    from poc.analytic import coefficients
    rng = np.random.default_rng(62)
    initial = rng.uniform(-1, 1, (16, 6))
    state = np.repeat(initial[:, None, :], 33, axis=1)
    commands = rng.uniform(-2, 2, (16, 32, 2))
    durations = rng.choice([20_000_000, 40_000_000, 60_000_000], size=(16, 32))
    for t in range(32):
        h = durations[:, t] / 1e9
        kv = np.array([coefficients(float(value))[1] for value in h])
        w = state[:, t, 4:]
        mean_effort = w + (commands[:, t] - w) * (kv / h)[:, None]
        state[:, t + 1, 2:4] = state[:, t, 2:4] + h[:, None] * mean_effort
    monkeypatch.setattr(poc.dataset, 'load_corpus', lambda *a, **k:
                        {'state': state, 'command': commands, 'duration_ns': durations})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'role': 'train', 'actuator': 'lagged'}))
    result = np.asarray(fit_ridge(path)['coefficients'])
    np.testing.assert_allclose(result[7:9], np.eye(2), atol=1e-6)


def test_mixed_zero_queries_and_bad_physical_parameter_values():
    model = PhysicsFit(torch.tensor([.10, .02, .02, .05, .05], dtype=torch.float64), 'direct', scales())
    state = torch.ones((2, 4), dtype=torch.float64)
    command = torch.zeros((2, 2), dtype=torch.float64)
    packet = model.predict(state, command, torch.tensor([[0.], [.005]], dtype=torch.float64))
    assert torch.equal(packet.next_state[0], state[0])
    for values in ([.001, .02, .02, .05, .05], [.1, -.02, .02, .05, .05]):
        with pytest.raises(ValueError):
            PhysicsFit(torch.tensor(values), 'direct', scales())


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_ridge_instantaneous_effort_matches_exact_unit_mass_maps(actuator):
    from poc.analytic import transition
    coefficients = np.zeros((12, 2))
    coefficients[7:9] = np.eye(2)
    model = AccelRidge(coefficients, actuator, scales())
    state = np.array([.2, -.4, .3, -.1] + ([.7, -.5] if actuator == 'lagged' else []))
    command = np.array([-.4, .8])
    actual = model.predict(torch.tensor(state[None]), torch.tensor(command[None]), .04).next_state
    np.testing.assert_allclose(actual.numpy()[0], transition(state, command, .04, actuator), atol=1e-9)
    # The shared API must also work with float32 scales and states.
    float_model = AccelRidge(coefficients, actuator, Scales(torch.ones(2), torch.ones(2), torch.ones(2)))
    result = float_model.predict(torch.tensor(state[None], dtype=torch.float32),
                                 torch.tensor(command[None], dtype=torch.float32), .04)
    assert result.next_state.dtype == torch.float32
