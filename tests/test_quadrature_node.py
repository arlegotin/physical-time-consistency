"""A cheap comparator must retain physical command order, not just low cost."""
import numpy as np
import pytest
import torch
from poc.models.common import Scales


def make(actor='lagged'):
    from poc.quadrature_node import QuadratureNODE
    return QuadratureNODE(actor, Scales(torch.ones(2), torch.full((2,), .04), torch.ones(2))).double()


class EffortField:
    def __call__(self, qv, command, w=None):
        effort = command if w is None else w
        matrix = torch.tensor([[1., .2], [-.3, 2.]], dtype=qv.dtype)
        return torch.cat((qv[:, 2:4], effort @ matrix.T), 1)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_correct_finite_quadrature_and_new_command_coefficients(actor):
    model = make(actor)
    model.vector_field = EffortField()
    s = torch.tensor([[.3, -.2, 1., -1., .9, -.7]], dtype=torch.float64)
    if actor == 'direct': s = s[:, :4]
    original = s.clone()
    u = torch.tensor([.4, -.6], dtype=torch.float64, requires_grad=True)
    matrix = np.array([[1., .2], [-.3, 2.]])
    for h in (1e-7, .005, .04):
        gain = 1 if actor == 'direct' else -np.expm1(-h/.16)
        jac = torch.autograd.functional.jacobian(lambda x:model.predict(s, x[None], h).next_state[0], u)
        qgain = .5*h*h if actor == 'direct' else h*h*gain/3
        vgain = h if actor == 'direct' else h*gain
        np.testing.assert_allclose(jac[:2], qgain*matrix, rtol=1e-12, atol=1e-24)
        np.testing.assert_allclose(jac[2:4], vgain*matrix, rtol=1e-12, atol=1e-24)
        if actor == 'lagged' and h == 1e-7:
            np.testing.assert_allclose(jac[:2]/h**3, matrix/(6*.08), rtol=1e-6)
            np.testing.assert_allclose(jac[2:4]/h**2, matrix/(2*.08), rtol=1e-6)
        out = model.predict(s, u[None], h)
        a0 = matrix@(u.detach().numpy() if actor=='direct' else s[0,4:].numpy())
        am = matrix@(u.detach().numpy() if actor=='direct' else s[0,4:].numpy()+gain*(u.detach().numpy()-s[0,4:].numpy()))
        np.testing.assert_allclose(out.next_state[0,:2].detach().numpy(), s[0,:2].numpy()+h*s[0,2:4].numpy()+h*h*(a0/6+am/3), rtol=1e-14)
        assert out.network_calls == out.vector_field_calls == 2
    assert torch.equal(s, original)


@pytest.mark.parametrize('actor', ['direct', 'lagged'])
def test_zero_mixed_rows_gradients_and_same_network_initialization(actor):
    from poc.models.ode import NeuralODE
    torch.manual_seed(123); model = make(actor)
    torch.manual_seed(123); other = NeuralODE(actor, Scales(torch.ones(2),torch.full((2,),.04),torch.ones(2)),True).double()
    assert all(torch.equal(v, other.state_dict()[k]) for k,v in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) == 17922
    s = torch.randn(3,4 if actor=='direct' else 6,dtype=torch.float64)
    u = torch.randn(3,2,dtype=torch.float64)
    old = s.clone(); out = model.predict(s,u,torch.tensor([[0.],[.04],[0.]],dtype=torch.float64))
    assert torch.equal(out.next_state[[0,2]],s[[0,2]])
    assert torch.equal(s,old)
    assert model.predict(s,u,0).network_calls == 0
    out.next_state[:,:4].sum().backward()
    assert torch.isfinite(model.network[-1].bias.grad).all()
    assert (model.network[-1].bias.grad.abs()>0).all()


@pytest.mark.parametrize('h', [-.1,float('nan'),float('inf')])
def test_invalid_intervals_rejected(h):
    with pytest.raises(ValueError): make().predict(torch.zeros(1,6,dtype=torch.float64),torch.zeros(1,2,dtype=torch.float64),h)
