"""Independent high-precision checks of held-command response coefficients."""
from decimal import Decimal, localcontext

import pytest
import torch

from poc.actuator_integrals import lag_integrals, response_ratios


def decimal_coefficients(r, tau='.08'):
    with localcontext() as context:
        context.prec = 80
        t = Decimal(tau)
        h = Decimal(str(r))*t
        decay = (-h/t).exp()
        beta = 1-(-h/(2*t)).exp()
        kv = h-t*(1-decay)
        kq = h*h/2-t*h+t*t*(1-decay)
        return tuple(float(x) for x in (kv, kq, kv/beta, kq/beta))


@pytest.mark.parametrize('r', [1e-8, 1e-6, .0625, .1, .5, 1, 2, 4])
@pytest.mark.parametrize('tau', [.08, .23])
def test_coefficients_match_decimal_reference(r, tau):
    h = torch.tensor(r*tau, dtype=torch.float64)
    got = (*lag_integrals(h, tau), *response_ratios(h, tau))
    for actual, expected in zip(got, decimal_coefficients(r, str(tau))):
        assert actual.item() == pytest.approx(expected, rel=1e-11, abs=0)


def test_float32_mixed_zero_outputs_and_interval_gradients_are_finite():
    h = torch.tensor([0., 8e-10, 8e-8, .005, .008, .32], requires_grad=True)
    values = (*lag_integrals(h, .08), *response_ratios(h, .08))
    assert all(torch.isfinite(value).all() for value in values)
    assert all(value[0].item() == 0 for value in values)
    sum(value.sum() for value in values).backward()
    assert torch.isfinite(h.grad).all()
    for function in (lag_integrals, response_ratios):
        x = torch.tensor([0., .005, .008], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda z: function(z[1:], .08), (x,))


def test_series_branch_has_continuous_values_and_derivatives():
    x = torch.tensor([.008*(1-1e-8), .008*(1+1e-8)], dtype=torch.float64, requires_grad=True)
    for values in (*lag_integrals(x, .08), *response_ratios(x, .08)):
        assert values[0].item() == pytest.approx(values[1].item(), rel=7e-8)
        gradient = torch.autograd.grad(values.sum(), x, retain_graph=True)[0]
        assert gradient[0].item() == pytest.approx(gradient[1].item(), rel=7e-8)


@pytest.mark.parametrize('tau', [0., -.08, float('nan'), float('inf'), [.08, .09]])
@pytest.mark.parametrize('function', [lag_integrals, response_ratios])
def test_invalid_tau_rejected(function, tau):
    with pytest.raises(ValueError):
        function(torch.tensor(.01), tau)


@pytest.mark.parametrize('h', [-.01, float('nan'), float('inf')])
@pytest.mark.parametrize('function', [lag_integrals, response_ratios])
def test_invalid_interval_rejected(function, h):
    with pytest.raises(ValueError):
        function(torch.tensor(h), .08)
