"""Independent exact-mechanics answers, including cancellation-sensitive horizons."""
from decimal import Decimal, localcontext
import importlib

import numpy as np
import pytest
import torch


def api(name):
    assert importlib.util.find_spec('poc.analytic'), 'Exact analytic mechanics are not implemented'
    return getattr(importlib.import_module('poc.analytic'), name)


def decimal_coefficients(h, tau=.08):
    with localcontext() as context:
        context.prec = 60
        h, tau = Decimal(str(h)), Decimal(str(tau))
        kw = 1 - (-h / tau).exp()
        return tuple(float(value) for value in (kw, h - tau * kw, h*h/2 - tau*h + tau*tau*kw))


@pytest.mark.parametrize('h', [0., 1e-8, 1e-6, .000625, .00125, .0025, .005, .01, .02, .04, .08])
def test_stable_coefficients_against_60_digit_decimal(h):
    np.testing.assert_allclose(api('coefficients')(h), decimal_coefficients(h), rtol=1e-12, atol=0)


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
@pytest.mark.parametrize('joints', [1, 2])
def test_thousand_independent_exact_states_and_composition(actuator, joints):
    transition = api('transition')
    rng = np.random.Generator(np.random.PCG64(8001))
    dimension = joints * (3 if actuator == 'lagged' else 2)
    states = rng.uniform(-2, 2, (1000, dimension))
    commands = rng.uniform(-2, 2, (1000, joints))
    states[0] = 0
    commands[:4] = np.array([-2., 2., 0., -1.])[:, None]
    for h in (0., 1e-8, 1e-6, .0025, .005, .01, .02, .04, .08):
        kw, kv, kq = decimal_coefficients(h)
        for state, command in zip(states, commands):
            q, v = state[:joints], state[joints:2*joints]
            if actuator == 'direct':
                expected = np.concatenate((q+h*v+.5*h*h*command, v+h*command))
            else:
                w = state[2*joints:]
                expected = np.concatenate((q+h*v+.5*w*h*h+(command-w)*kq,
                                           v+w*h+(command-w)*kv, w+(command-w)*kw))
            actual = transition(state, command, h, actuator)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(transition(transition(state, command, h*.25, actuator), command, h*.75, actuator),
                                       actual, atol=1e-12, rtol=1e-12)
            assert not np.shares_memory(actual, state)


@pytest.mark.parametrize('actuator', ['direct', 'lagged'])
def test_new_command_jacobian_fixed_initial_activation_and_signal_resolution(actuator):
    transition, jacobian = api('transition'), api('action_jacobian')
    state = np.array([1., -1., .3, -.4] + ([.7, -.2] if actuator == 'lagged' else []))
    command = np.array([.5, -.6])
    unresolved = []
    for h in (0., 1e-8, 1e-6, .0025, .005, .01, .02, .04, .08):
        kw, kv, kq = decimal_coefficients(h)
        coefficients = (.5*h*h, h) if actuator == 'direct' else (kq, kv, kw)
        expected = np.concatenate([coefficient*np.eye(2) for coefficient in coefficients])
        np.testing.assert_allclose(jacobian(h, actuator), expected, atol=0, rtol=1e-10)
        # Independent differentiable affine map: initial w is a constant, never u.
        tensor_command = torch.tensor(command, dtype=torch.float64, requires_grad=True)
        reference = lambda u: torch.cat([torch.tensor(state[2*i:2*i+2], dtype=torch.float64) + c*u
                                         for i, c in enumerate(coefficients)])
        automatic = torch.autograd.functional.jacobian(reference, tensor_command).numpy()
        np.testing.assert_allclose(jacobian(h, actuator), automatic, atol=0, rtol=1e-10)
        epsilon = .02
        finite = np.column_stack([(transition(state, command+epsilon*np.eye(2)[j], h, actuator)
                                   - transition(state, command-epsilon*np.eye(2)[j], h, actuator))/(2*epsilon)
                                  for j in range(2)])
        for i, coefficient in enumerate(coefficients):
            uncertainty = 4*np.finfo(float).eps*max(1., np.max(np.abs(state)))/epsilon
            if coefficient <= 100*uncertainty:
                unresolved.append((h, i))
                continue
            np.testing.assert_allclose(finite[2*i:2*i+2], expected[2*i:2*i+2], rtol=1e-6, atol=uncertainty)
    assert (1e-8, 0) in unresolved


def test_piecewise_order_matters_and_bad_inputs_are_errors():
    transition = api('transition')
    state, command = np.array([1., 2.]), np.array([3.])
    np.testing.assert_allclose(transition(state, command, .04, 'direct'), [1.0824, 2.12], atol=1e-12)
    first = transition(transition(state, np.array([2.]), .02, 'direct'), np.array([-2.]), .02, 'direct')
    reverse = transition(transition(state, np.array([-2.]), .02, 'direct'), np.array([2.]), .02, 'direct')
    assert first[0] > reverse[0]
    for h in (-.01, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            transition(state, command, h, 'direct')
    for s, u, mode in ((state, [1, 2], 'direct'), ([1, float('nan')], [1], 'direct'), (state, [1], 'hidden')):
        with pytest.raises(ValueError):
            transition(s, u, .04, mode)

