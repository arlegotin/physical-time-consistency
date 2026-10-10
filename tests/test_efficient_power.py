"""Catch changed temporal gains, actuator interventions, and hidden neural work."""
import importlib.util

import pytest
import torch

from poc.models.common import Scales
from poc.models.factory import build_model
from poc.models.power import PowerHead


def implementation():
    assert importlib.util.find_spec('poc.efficient_power'), 'Equivalent head optimization is missing'
    from poc.efficient_power import EfficientPowerHead, neural_work
    return EfficientPowerHead, neural_work


def selected(actuator, dtype=torch.float64):
    scales = Scales(torch.tensor([.03, .07]), torch.tensor([.4, .8]), torch.tensor([1., 2.]))
    head = build_model('power_correct', actuator, scales, 8301).to(dtype).eval()
    state = torch.tensor([[.3, -.7, .8, -.2] + ([.9, -.4] if actuator == 'lagged' else [])], dtype=dtype)
    command = torch.tensor([[.5, -1.2]], dtype=dtype)
    return head, state, command


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
@pytest.mark.parametrize('mode', ['low', 'direct', 'lagged'])
def test_optimized_head_preserves_predictions_and_all_derivatives(actuator, mode):
    cls, _ = implementation()
    original, state, command = selected(actuator)
    optimized = cls.from_head(original)
    for h in (0., .0025, .005, .01, .02, .04, .06):
        predictions = []
        for model in (original, optimized):
            fn = lambda x, u: model.predict(x, u, h, power_mode=mode).next_state
            output = fn(state, command)
            sj = torch.autograd.functional.jacobian(lambda x: fn(x, command), state)
            uj = torch.autograd.functional.jacobian(lambda u: fn(state, u), command)
            if h:
                gradients = torch.autograd.grad(output.square().sum(), tuple(model.parameters()))
            else:
                gradients = ()
            predictions.append((output, sj, uj, *gradients))
        for expected, actual in zip(*predictions):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    h = torch.tensor([[0.], [.005], [.04]], dtype=state.dtype)
    torch.testing.assert_close(
        optimized.predict(state.repeat(3, 1), command.repeat(3, 1), h, power_mode=mode).next_state,
        original.predict(state.repeat(3, 1), command.repeat(3, 1), h, power_mode=mode).next_state,
        atol=1e-12, rtol=1e-12)
    trajectories = []
    for model in (original, optimized):
        value = state
        for _ in range(4):
            value = model.predict(value, command, .04, power_mode=mode).next_state
        loss = value.square().sum()
        trajectories.append((value, *torch.autograd.grad(loss, tuple(model.parameters()))))
    for expected, actual in zip(*trajectories):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(original.parameters(), optimized.parameters()))


@pytest.mark.parametrize('actuator,mode,expected_batches,expected_rows', [
    ('direct', None, [3], 3), ('direct', 'low', [3, 3], 6),
    ('direct', 'lagged', [3, 3], 6), ('lagged', None, [6], 6),
    ('lagged', 'direct', [6], 6), ('lagged', 'low', [6], 6)])
def test_direct_correct_uses_one_pass_and_lagged_batches_two_rows(actuator, mode, expected_batches, expected_rows):
    cls, work = implementation()
    head, state, command = selected(actuator)
    model = cls.from_head(head)
    batches = []
    handle = model.network.register_forward_hook(lambda module, inputs, output: batches.append(len(inputs[0])))
    result = model.predict(state.repeat(3, 1), command.repeat(3, 1), .005, power_mode=mode)
    handle.remove()
    assert batches == expected_batches
    assert result.network_calls == len(expected_batches)
    assert result.scalar_candidate_evaluations == 3
    assert work(result, 3, actuator, mode)['neural_rows'] == expected_rows
    zero = model.predict(state, command, 0., power_mode=mode)
    assert zero.network_calls == 0
    assert work(zero, 1, actuator, mode)['neural_rows'] == 0
    if actuator == 'lagged':
        expected = state[:, 4:] + (command - state[:, 4:])*(-torch.expm1(torch.tensor(-.005/.08)))
        torch.testing.assert_close(result.next_state[:1, 4:], expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_float32_smoke_and_bad_interval(actuator):
    cls, _ = implementation()
    head, state, command = selected(actuator, torch.float32)
    model = cls.from_head(head)
    torch.testing.assert_close(model.predict(state, command, .01).next_state,
                               head.predict(state, command, .01).next_state, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match='Interval'):
        model.predict(state, command, -.01)
    with pytest.raises(ValueError, match='power'):
        model.predict(state, command, .01, power_mode='invented')


def test_stateful_or_training_network_is_rejected():
    cls, _ = implementation()
    head, _, _ = selected('direct')
    head.train()
    with pytest.raises(ValueError, match='evaluation'):
        cls.from_head(head)
    for layer in (torch.nn.Dropout(.5), torch.nn.BatchNorm1d(4), torch.nn.ReLU()):
        bad = PowerHead(torch.nn.Sequential(torch.nn.Linear(9, 4), layer), 'direct',
                        Scales(torch.ones(2), torch.ones(2), torch.ones(2))).eval()
        with pytest.raises(ValueError, match='Linear.*Tanh'):
            cls.from_head(bad)
