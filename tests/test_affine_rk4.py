"""Torque-affine RK4 stages, units, initialization and constant-system audit."""
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

from poc.affine_flow import AffineFlow
from poc.affine_rk4 import AffineRK4
from poc.models.common import Scales


def make(actor='lagged'):
    return AffineRK4(actor, Scales(torch.ones(2), torch.full((2,), .04), torch.ones(2))).double()


def constant_system(model, drift, matrix):
    # Preserve the real coefficient decoder and its normalization.
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
        model.network[-1].bias[:2].copy_(drift/(model.scale_v/.04))
        model.network[-1].bias[2:].copy_((matrix/(model.scale_v/.08)[:, None]).flatten())


def exact_endpoint(state, command, h, actor, drift, matrix):
    # Integrate w(t)=u+(w0-u)e^-t/tau independently in Decimal arithmetic.
    with localcontext() as context:
        context.prec = 80
        t, tau = Decimal(str(h)), Decimal('.08')
        decay = (-t/tau).exp()
        old_v = float(tau*(1-decay))
        old_q = float(tau*t-tau*tau*(1-decay))
    s, u, f, b = [x.numpy() for x in (state, command, drift, matrix)]
    w = s[:, 4:6] if actor == 'lagged' else u
    iv = h*u + old_v*(w-u)
    iq = h*h*u/2 + old_q*(w-u)
    values = [s[:, :2]+h*s[:, 2:4]+h*h*f/2+iq@b.T, s[:, 2:4]+h*f+iv@b.T]
    if actor == 'lagged':
        values.append(u+(w-u)*float(decay))
    return np.concatenate(values, axis=1)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_dense_constant_drift_and_nonzero_activation_endpoints(actor):
    generator = torch.Generator().manual_seed(881)
    model = make(actor)
    drift = torch.randn(2, generator=generator, dtype=torch.float64)
    matrix = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    constant_system(model, drift, matrix)
    state = torch.randn(5, 6 if actor == 'lagged' else 4, generator=generator, dtype=torch.float64)
    command = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    for h in (1e-8, .005, .08, .32):
        output = model.predict(state, command, h, solver_substeps=64)
        np.testing.assert_allclose(output.next_state.detach(), exact_endpoint(state, command, h, actor, drift, matrix), atol=2e-10, rtol=2e-7)
        assert output.network_calls == output.vector_field_calls == 256
        assert output.status == ('ok',)*5


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_command_response_has_correct_leading_orders(actor):
    model = make(actor)
    matrix = torch.tensor([[1., .2], [-.3, 2.]], dtype=torch.float64)
    constant_system(model, torch.zeros(2, dtype=torch.float64), matrix)
    state = torch.tensor([[.2, -.3, 1., -1., .9, -.7]], dtype=torch.float64)
    state = state if actor == 'lagged' else state[:, :4]
    command = torch.tensor([.4, -.6], dtype=torch.float64, requires_grad=True)
    h = 1e-7
    jacobian = torch.autograd.functional.jacobian(lambda u: model.predict(state, u[None], h).next_state[0], command)
    qgain = h**3/(6*.08) if actor == 'lagged' else h*h/2
    vgain = h*h/(2*.08) if actor == 'lagged' else h
    np.testing.assert_allclose(jacobian[:2], matrix*qgain, rtol=1e-6, atol=0)
    np.testing.assert_allclose(jacobian[2:4], matrix*vgain, rtol=1e-6, atol=0)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_initialization_immutable_mixed_zeros_and_differentiability(actor):
    torch.manual_seed(199); model = make(actor)
    torch.manual_seed(199); other = AffineFlow(actor, Scales(torch.ones(2), torch.full((2,), .04), torch.ones(2))).double()
    assert all(torch.equal(value, other.state_dict()[key]) for key, value in model.state_dict().items())
    state = torch.randn(3, 6 if actor == 'lagged' else 4, dtype=torch.float64)
    command = torch.randn(3, 2, dtype=torch.float64)
    old_state, old_command = state.clone(), command.clone()
    h = torch.tensor([[0.], [.04], [0.]], dtype=torch.float64, requires_grad=True)
    result = model.predict(state, command, h, solver_substeps=3)
    assert torch.equal(result.next_state[[0, 2]], state[[0, 2]])
    assert torch.equal(state, old_state) and torch.equal(command, old_command)
    assert result.network_calls == result.vector_field_calls == 12
    result.next_state.sum().backward()
    assert torch.isfinite(h.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    zero = model.predict(state, command, 0.)
    assert torch.equal(zero.next_state, state)
    assert zero.network_calls == zero.vector_field_calls == 0


@pytest.mark.parametrize('bad', [0, -1, 1.5, True])
def test_invalid_substeps_rejected_even_at_zero(bad):
    with pytest.raises(ValueError):
        make().predict(torch.zeros(1, 6, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), 0., solver_substeps=bad)


@pytest.mark.parametrize('h', [-.1, float('nan'), float('inf')])
def test_invalid_interval_rejected(h):
    with pytest.raises(ValueError):
        make().predict(torch.zeros(1, 6, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), h)


def test_power_mode_rejected():
    with pytest.raises(ValueError):
        make().predict(torch.zeros(1, 6, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), .01, power_mode='lagged')
