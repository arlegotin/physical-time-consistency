"""Same weights isolate powers, including nominal training and command derivatives."""
import importlib
import numpy as np
import pytest
import torch

from poc.analytic import action_jacobian


def fixture(actuator, dtype=torch.float64):
    assert importlib.util.find_spec('poc.models'), 'Shared power-head contract is not implemented'
    from poc.models.common import Scales
    from poc.models.power import PowerHead
    torch.manual_seed(8021)
    network = torch.nn.Linear(9 if actuator == 'direct' else 11, 4).to(dtype)
    scales = Scales(torch.tensor([.1, .2], dtype=dtype), torch.tensor([.3, .4], dtype=dtype),
                    torch.ones(2, dtype=dtype))
    model = PowerHead(network, actuator, scales)
    state = torch.tensor([[.5, -.7, .3, -.1] + ([.8, -.5] if actuator == 'lagged' else [])], dtype=dtype)
    command = torch.tensor([[.9, -.6]], dtype=dtype)
    return model, state, command


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_nominal_outputs_state_command_gradients_weights_and_four_step_loss(actuator, dtype):
    model, state, command = fixture(actuator, dtype)
    for update in range(2):
        records = {}
        for mode in ('low', 'direct', 'lagged'):
            predict = lambda x, u: model.predict(x, u, .04, power_mode=mode).next_state
            output = predict(state, command)
            sj = torch.autograd.functional.jacobian(lambda x: predict(x, command), state)
            uj = torch.autograd.functional.jacobian(lambda u: predict(state, u), command)
            model.zero_grad()
            output.square().sum().backward()
            gradients = torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()])
            imagined = state
            loss = torch.zeros((), dtype=dtype)
            for _ in range(4):
                imagined = predict(imagined, command)
                loss = loss+imagined.square().sum()/4
            model.zero_grad()
            loss.backward()
            rollout_gradients = torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()])
            records[mode] = (output.detach(), sj, uj, gradients.detach(), loss.detach(), rollout_gradients)
        tolerance = 1e-12 if dtype == torch.float64 else 2e-6
        for mode in ('direct', 'lagged'):
            for left, right in zip(records['low'], records[mode]):
                torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
        if update == 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False, fused=False)
            optimizer.step()


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_two_network_calls_noop_rows_and_exact_activation(actuator):
    model, state, command = fixture(actuator)
    events = []
    handle = model.network.register_forward_hook(lambda *args: events.append('network_call'))
    prediction = model.predict(state.repeat(2, 1), command.repeat(2, 1), torch.tensor([[0.], [.04]], dtype=torch.float64))
    handle.remove()
    torch.testing.assert_close(prediction.next_state[0], state[0], atol=0, rtol=0)
    assert events == ['network_call', 'network_call']
    assert prediction.network_calls == 2
    assert prediction.scalar_candidate_evaluations == 2
    assert prediction.history_tokens_processed == 0
    assert model.predict(state, command, 0).network_calls == 0
    if actuator == 'lagged':
        expected = state[:, 4:] + (command-state[:, 4:])*(-np.expm1(-.04/.08))
        torch.testing.assert_close(prediction.next_state[1:, 4:], expected, atol=1e-12, rtol=1e-12)
        derivative = torch.autograd.functional.jacobian(lambda u: model.predict(state, u, .005).next_state[:, 4:], command)
        torch.testing.assert_close(derivative[0, :, 0, :], torch.eye(2, dtype=torch.float64)*(-np.expm1(-.005/.08)), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('actuator,mode,orders', [('direct','direct',(2,1)), ('lagged','lagged',(3,2)),
                                               ('direct','low',(1,1)), ('lagged','direct',(2,1))])
def test_affine_fixture_exposes_the_imposed_command_order(actuator, mode, orders):
    model, state, command = fixture(actuator)
    norms = []
    for h in (.0025, .005, .01):
        derivative = torch.autograd.functional.jacobian(lambda u: model.predict(state, u, h, power_mode=mode).next_state, command)[0, :, 0, :]
        norms.append([torch.linalg.matrix_norm(derivative[:2]).item(), torch.linalg.matrix_norm(derivative[2:4]).item()])
    slopes = [np.polyfit(np.log([.0025, .005, .01]), np.log(np.asarray(norms)[:, i]), 1)[0] for i in (0, 1)]
    np.testing.assert_allclose(slopes, orders, atol=1e-12, rtol=1e-12)


def test_zero_wrong_sign_and_swapped_jacobians_fail_absolute_gain_checks():
    # Matrix errors matter even where Frobenius gain is exactly right.
    reference = action_jacobian(.005, 'direct')[:4]
    for mutated in (np.zeros_like(reference), -reference, reference[:, ::-1]):
        error = np.linalg.norm(mutated-reference)/np.linalg.norm(reference)
        assert error > .5
    assert np.linalg.norm(-reference)/np.linalg.norm(reference) == 1


def test_fixed_state_assertion_catches_initializing_activation_from_command():
    assert importlib.util.find_spec('poc.models'), 'Fixed-state audit is not implemented'
    from poc.models.common import require_same_state
    state = np.array([.5, -.7, .3, -.1, .8, -.5])
    mutation = state.copy()
    mutation[4:] = [.9, -.6]
    require_same_state(state, state.copy())
    with pytest.raises(ValueError, match='initial state'):
        require_same_state(state, mutation)


def test_invalid_queries_and_history_are_read_only():
    model, state, command = fixture('direct')
    for h in (-.01, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            model.predict(state, command, h)
    with pytest.raises(ValueError):
        model.predict(state, command, .04, power_mode='invented')
    history = object()
    first = model.predict(state, command, .01, history=history).next_state
    model.predict(state, -command, .02, history=history)
    torch.testing.assert_close(model.predict(state, command, .01, history=history).next_state, first, atol=0, rtol=0)


def test_physical_command_gain_includes_normalization_chain_rule():
    model, state, command = fixture('direct')
    with torch.no_grad():
        model.network.weight.zero_()
        model.network.bias.zero_()
        for joint in (0, 1):
            model.network.weight[joint, 6+joint] = .04**2/model.scale_q[joint]
            model.network.weight[2+joint, 6+joint] = 2*.04/model.scale_v[joint]
    for h in (.0025, .005, .01, .04):
        measured = torch.autograd.functional.jacobian(lambda u: model.predict(state, u, h).next_state, command)[0, :, 0, :]
        np.testing.assert_allclose(measured.numpy(), action_jacobian(h, 'direct'), atol=1e-14, rtol=1e-10)


def test_m10_gate_cannot_pass_from_a_green_test_flag_without_numeric_arrays(tmp_path):
    import poc.milestones.m10 as milestone
    assert hasattr(milestone, 'evaluate'), 'M10 evidence evaluator is not implemented'
    with pytest.raises(ValueError):
        milestone.evaluate(tmp_path, [], {'tests': {'status': 'pass'}})


def test_raw_power_audit_serializes_every_mutation_and_gradient(tmp_path):
    from poc.mechanics import _power_evidence
    from poc.io import read_json
    measurements = _power_evidence(tmp_path)
    assert measurements['identity_max_abs_error'] <= 1e-12
    assert all(read_json(tmp_path / 'mutations.json')['detected'].values())
    with np.load(tmp_path / 'power_identity.npz', allow_pickle=False) as arrays:
        assert arrays['direct_outputs'].shape[:2] == (2, 3)
