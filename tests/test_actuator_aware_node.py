"""Response-aware quadrature is exact for constant mechanics, not nonlinear fields."""
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

from poc.actuator_aware_node import ActuatorAwareNODE
from poc.models.common import Scales
from poc.models.ode import NeuralODE
from poc.quadrature_node import QuadratureNODE


def scales():
    return Scales(torch.ones(2), torch.full((2,), .04), torch.ones(2))


def make(actor='lagged'):
    return ActuatorAwareNODE(actor, scales()).double()


class ConstantMechanics:
    def __init__(self, drift, matrix):
        self.drift, self.matrix = drift, matrix

    def __call__(self, qv, command, w=None):
        effort = command if w is None else w
        return torch.cat((qv[:, 2:4], self.drift+effort@self.matrix.T), 1)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_random_dense_constant_system_with_drift_and_initial_activation(actor):
    generator = torch.Generator().manual_seed(119)
    model = make(actor)
    drift = torch.randn(2, generator=generator, dtype=torch.float64)
    matrix = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    model.vector_field = ConstantMechanics(drift, matrix)
    state = torch.randn(7, 6 if actor == 'lagged' else 4, generator=generator, dtype=torch.float64)
    command = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    for h in (1e-8, 1e-6, .005, .008, .04, .08, .32):
        with localcontext() as context:
            context.prec = 80
            t, tau = Decimal(str(h)), Decimal('.08')
            decay = (-t/tau).exp()
            old_v, old_q = float(tau*(1-decay)), float(tau*t-tau*tau*(1-decay))
        s, u = state.numpy(), command.numpy()
        w = s[:, 4:6] if actor == 'lagged' else u
        q = s[:, :2]+h*s[:, 2:4]+h*h*drift.numpy()/2+(h*h*u/2+old_q*(w-u))@matrix.numpy().T
        v = s[:, 2:4]+h*drift.numpy()+(h*u+old_v*(w-u))@matrix.numpy().T
        values = [q, v]
        if actor == 'lagged':
            values.append(u+(w-u)*float(decay))
        output = model.predict(state, command, h)
        np.testing.assert_allclose(output.next_state, np.concatenate(values, 1), atol=2e-10, rtol=2e-7)
        assert output.network_calls == output.vector_field_calls == 2
        assert output.status == ('ok',)*7


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_leading_command_response_orders(actor):
    model = make(actor)
    matrix = torch.tensor([[1., .2], [-.3, 2.]], dtype=torch.float64)
    model.vector_field = ConstantMechanics(torch.zeros(2, dtype=torch.float64), matrix)
    state = torch.tensor([[.2, -.3, 1., -1., .9, -.7]], dtype=torch.float64)
    state = state if actor == 'lagged' else state[:, :4]
    command = torch.tensor([.4, -.6], dtype=torch.float64, requires_grad=True)
    h = 1e-7
    jac = torch.autograd.functional.jacobian(lambda u: model.predict(state, u[None], h).next_state[0], command)
    qgain = h**3/(6*.08) if actor == 'lagged' else h*h/2
    vgain = h*h/(2*.08) if actor == 'lagged' else h
    np.testing.assert_allclose(jac[:2], matrix*qgain, rtol=1e-6, atol=0)
    np.testing.assert_allclose(jac[2:4], matrix*vgain, rtol=1e-6, atol=0)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_all_three_nodes_share_initialization_and_mixed_zeros_are_immutable(actor):
    torch.manual_seed(911); model = make(actor)
    for cls in (NeuralODE, QuadratureNODE):
        torch.manual_seed(911)
        other = cls(actor, scales(), structured=True).double() if cls is NeuralODE else cls(actor, scales()).double()
        assert all(torch.equal(value, other.state_dict()[key]) for key, value in model.state_dict().items())
    state = torch.randn(3, 6 if actor == 'lagged' else 4, dtype=torch.float64)
    command = torch.randn(3, 2, dtype=torch.float64)
    old_state, old_command = state.clone(), command.clone()
    h = torch.tensor([[0.], [.04], [0.]], dtype=torch.float64, requires_grad=True)
    output = model.predict(state, command, h)
    assert torch.equal(output.next_state[[0, 2]], state[[0, 2]])
    assert torch.equal(state, old_state) and torch.equal(command, old_command)
    output.next_state.sum().backward()
    assert torch.isfinite(h.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    zero = model.predict(state, command, 0.)
    assert torch.equal(zero.next_state, state)
    assert zero.network_calls == zero.vector_field_calls == 0


def test_direct_torque_matches_historical_quadrature_on_nonlinear_network():
    model = make('direct')
    other = QuadratureNODE('direct', scales()).double()
    other.load_state_dict(model.state_dict())
    state, command = torch.randn(3, 4, dtype=torch.float64), torch.randn(3, 2, dtype=torch.float64)
    assert torch.equal(model.predict(state, command, .04).next_state, other.predict(state, command, .04).next_state)


def test_nonlinear_mechanics_are_not_claimed_exact():
    model = make()
    # q''=q² has exact solution q=6/(1-t)² for q0=6, v0=12.
    model.vector_field = lambda qv, command, w=None: torch.cat((qv[:, 2:4], qv[:, :2]**2), 1)
    state = torch.tensor([[6., 6., 12., 12., .8, -.6]], dtype=torch.float64)
    output = model.predict(state, torch.zeros(1, 2, dtype=torch.float64), .1)
    exact = torch.tensor([6/.9**2, 6/.9**2, 12/.9**3, 12/.9**3], dtype=torch.float64)
    assert (output.next_state[0, :4]-exact).abs().max() > .01
    assert torch.isfinite(output.next_state).all()


@pytest.mark.parametrize('tau', [.09, 0., -.08, float('nan'), float('inf'), [.08, .09], [.08, .08]])
def test_only_historical_shared_tau_supported(tau):
    with pytest.raises(ValueError):
        ActuatorAwareNODE('lagged', scales(), tau_s=tau)


@pytest.mark.parametrize('h', [-.1, float('nan'), float('inf')])
def test_invalid_interval_rejected(h):
    with pytest.raises(ValueError):
        make().predict(torch.zeros(1, 6, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), h)


def test_power_mode_rejected():
    with pytest.raises(ValueError):
        make().predict(torch.zeros(1, 6, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), .01, power_mode='lagged')
