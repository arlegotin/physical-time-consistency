"""Independent constant mechanics and interventions catch incorrect decoder timing."""
from decimal import Decimal, localcontext

import numpy as np
import pytest
import torch

from poc.analytic import transition
from poc.models.common import Scales


def model_for(actor):
    from poc.affine_flow import AffineFlow
    return AffineFlow(actor, Scales(torch.tensor([.04, .04]),
                                   torch.tensor([.04, .04]), torch.ones(2))).double()


def set_constant_field(model, drift=(0., 0.), matrix=((1., 0.), (0., 1.))):
    # f scale is one; B scale is 1/2. The independent fixture is unit mechanics.
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
        values = [*drift, *np.asarray(matrix).ravel()*2]
        model.network[-1].bias.copy_(torch.tensor(values, dtype=torch.float64))


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_constant_unit_field_matches_exact_held_command_at_all_horizons(actor):
    model = model_for(actor)
    set_constant_field(model)
    state = torch.tensor([[.2, -.3, 1.1, -.9, .8, -.6]], dtype=torch.float64)
    state = state[:, :4] if actor == 'direct' else state
    command = torch.tensor([[.7, -1.3]], dtype=torch.float64)
    for h in (0., 1e-8, 1e-6, .0025, .005, .01, .04, .08, .24):
        output = model.predict(state, command, h)
        expected = transition(state[0].numpy(), command[0].numpy(), h, actor)
        np.testing.assert_allclose(output.next_state[0].detach(), expected, atol=2e-10, rtol=2e-7)
        assert output.network_calls == (0 if h == 0 else 2)
        assert output.vector_field_calls == (0 if h == 0 else 2)
        assert output.status == ('ok',)


def test_lag_coefficients_resolve_cubic_response_without_cancellation():
    from poc.affine_flow import lag_coefficients
    with localcontext() as context:
        context.prec = 60
        for h in ('0.00000001', '0.000001', '.005', '.04', '.24'):
            t, tau = Decimal(h), Decimal('.08')
            kw = 1 - (-t/tau).exp()
            want_v = t - tau*kw
            want_q = t*t/2 - tau*t + tau*tau*kw
            v, q = lag_coefficients(torch.tensor([[float(h)]], dtype=torch.float64))
            assert float(v) == pytest.approx(float(want_v), rel=1e-12)
            assert float(q) == pytest.approx(float(want_q), rel=1e-12)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_new_command_jacobian_uses_physical_units_and_fixed_activation(actor):
    model = model_for(actor)
    set_constant_field(model, matrix=((1., .2), (-.3, 2.)))
    state = torch.tensor([[.3, -.2, 1., -1., .9, -.7]], dtype=torch.float64)
    state = state[:, :4] if actor == 'direct' else state
    original = state.clone()
    u = torch.tensor([.4, -.6], dtype=torch.float64, requires_grad=True)
    h = .005
    jac = torch.autograd.functional.jacobian(lambda x:model.predict(state, x[None], h).next_state[0], u)
    matrix = np.array([[1., .2], [-.3, 2.]])
    if actor == 'direct':
        qgain, vgain = .5*h*h, h
    else:
        with localcontext() as c:
            c.prec = 60
            t, tau = Decimal('.005'), Decimal('.08')
            kw = 1-(-t/tau).exp()
            vgain = float(t-tau*kw)
            qgain = float(t*t/2-tau*t+tau*tau*kw)
    np.testing.assert_allclose(jac[:2], qgain*matrix, rtol=2e-7, atol=1e-14)
    np.testing.assert_allclose(jac[2:4], vgain*matrix, rtol=2e-7, atol=1e-14)
    assert torch.equal(state, original)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_zero_rows_noop_and_query_does_not_mutate_input(actor):
    model = model_for(actor)
    state = torch.randn(3, 4 if actor == 'direct' else 6, dtype=torch.float64)
    command = torch.randn(3, 2, dtype=torch.float64)
    before = state.clone()
    output = model.predict(state, command, torch.tensor([[0.], [.005], [0.]], dtype=torch.float64))
    assert torch.equal(output.next_state[[0, 2]], state[[0, 2]])
    assert torch.equal(state, before)
    assert output.network_calls == 2
    assert torch.equal(model.predict(state, command, 0.).next_state, state)


@pytest.mark.parametrize('bad', [-.1, float('nan'), float('inf')])
def test_invalid_interval_rejected(bad):
    with pytest.raises(ValueError):
        model_for('direct').predict(torch.zeros(1, 4, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), bad)


def test_every_field_output_learns_and_parameter_budget_is_matched():
    model = model_for('lagged')
    assert sum(p.numel() for p in model.parameters()) == 18182
    state = torch.tensor([[.2, -.3, .4, -.5, .8, -.9]], dtype=torch.float64)
    command = torch.tensor([[.6, -.7]], dtype=torch.float64)
    model.predict(state, command, .04).next_state[:, :4].sum().backward()
    assert torch.isfinite(model.network[-1].bias.grad).all()
    assert (model.network[-1].bias.grad.abs() > 1e-10).all()
    with pytest.raises(ValueError):
        model.predict(state, command, .04, power_mode='lagged')
