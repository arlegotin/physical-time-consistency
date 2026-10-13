"""Nonempirical hybrid fixtures check shared fields and actual solver behavior."""
import importlib
import math

import numpy as np
import pytest
import torch
from torch import nn

from poc.calibration_models import build_model
from poc.models.common import Scales
from poc.physics_fit import PhysicsFit


PRIMARY = ('hybrid_affine_flow', 'hybrid_affine_rk4', 'hybrid_affine_aware')
REFERENCES = ('hybrid_accel_rk4', 'hybrid_accel_aware')
IDS = PRIMARY + REFERENCES


def scales(dtype=torch.float64):
    return Scales(torch.tensor([.03, .05], dtype=dtype),
                  torch.tensor([.2, .3], dtype=dtype),
                  torch.tensor([.7, .9], dtype=dtype))


def query(dtype=torch.float64):
    return (torch.tensor([[.2, -.7, .4, -.3, .6, -.5],
                          [-.4, .1, -.2, .5, -.3, .7]], dtype=dtype),
            torch.tensor([[-.4, .8], [.9, -.6]], dtype=dtype))


def legacy_acceleration(model, qv, effort):
    # Independent literal pre-refactor arithmetic: changing operation order is visible.
    p = model.physical_parameters()
    coupling = p[2] * torch.cos(qv[:, 1])
    m11 = p[0] + p[1] + 2 * coupling + .02
    m12 = p[1] + coupling
    m22 = torch.ones_like(coupling) * (p[1] + .02)
    mass = torch.stack((torch.stack((m11, m12), 1), torch.stack((m12, m22), 1)), 1)
    v1, v2 = qv[:, 2], qv[:, 3]
    cross = p[2] * torch.sin(qv[:, 1])
    coriolis = torch.stack((-cross * (2 * v1 * v2 + v2**2), cross * v1**2), 1)
    rhs = effort - coriolis - p[3:] * qv[:, 2:4]
    return torch.linalg.solve(mass, rhs.unsqueeze(2)).squeeze(2)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nominal_coefficients_reconstruct_legacy_dense_effort_and_preserve_arithmetic(dtype, monkeypatch):
    model = PhysicsFit(torch.tensor([.1, .02, .02, .05, .05], dtype=dtype), 'lagged', scales(dtype))
    generator = torch.Generator().manual_seed(781)
    qv = torch.randn(31, 4, generator=generator, dtype=dtype)
    effort = torch.randn(31, 2, generator=generator, dtype=dtype)
    expected = legacy_acceleration(model, qv, effort)
    assert torch.equal(model.acceleration(qv, effort), expected)
    assert hasattr(model, 'acceleration_coefficients'), 'nominal combined coefficient solve is missing'
    solve = torch.linalg.solve
    solves = []

    def observed_solve(matrix, rhs):
        solves.append(rhs.shape)
        return solve(matrix, rhs)

    monkeypatch.setattr(torch.linalg, 'solve', observed_solve)
    drift, gain = model.acceleration_coefficients(qv)
    assert solves == [torch.Size([31, 2, 3])]
    assert drift.shape == (31, 2) and gain.shape == (31, 2, 2)
    assert torch.count_nonzero(gain[:, 0, 1]) == 31
    torch.testing.assert_close(drift + torch.bmm(gain, effort.unsqueeze(-1)).squeeze(-1), expected)


@pytest.mark.parametrize('model_id', PRIMARY)
def test_zero_coefficient_residual_is_nominal_and_scaled_dense_rows(model_id):
    model = build_model(model_id, scales(), 781)
    state, _ = query()
    qv = state[:, :4]
    f, b = model.coefficients(qv)
    expected_f, expected_b = model.nominal.acceleration_coefficients(qv)
    assert torch.equal(f, expected_f) and torch.equal(b, expected_b)
    with torch.no_grad():
        model.network[-1].bias.copy_(torch.tensor([.1, -.2, .3, -.4, .5, -.6], dtype=torch.float64))
    f, b = model.coefficients(qv)
    torch.testing.assert_close(f - expected_f, torch.tensor([[.5, -1.5]]).double().expand(2, -1))
    torch.testing.assert_close(b - expected_b,
                               torch.tensor([[[.75, -1.], [1.875, -2.25]]]).double().expand(2, -1, -1))


@pytest.mark.parametrize('model_id', REFERENCES)
def test_zero_acceleration_residual_uses_existing_nominal_field(model_id):
    model = build_model(model_id, scales(), 781)
    state, effort = query()
    actual = model.acceleration(state[:, :4], effort)
    assert torch.equal(actual, legacy_acceleration(model.nominal, state[:, :4], effort))


class ConstantPrior(nn.Module):
    """Unit fixture with independently specified dense cross-joint coupling."""
    def acceleration_coefficients(self, qv):
        return (qv.new_tensor([.7, -.4]).expand(len(qv), -1),
                qv.new_tensor([[1.2, -.6], [.4, 1.7]]).expand(len(qv), -1, -1))

    def acceleration(self, qv, effort):
        f, b = self.acceleration_coefficients(qv)
        return f + effort @ b[0].T


def analytic_constant(state, command, h):
    # Independent exponential integrals, evaluated in float64 away from cancellation.
    decay = math.exp(-h / .08)
    initial = state.numpy()
    u = command.numpy()
    w = initial[:, 4:]
    b = np.array([[1.2, -.6], [.4, 1.7]])
    f = np.array([.7, -.4])
    iv = h * u + .08 * (1 - decay) * (w - u)
    iq = h*h/2 * u + (.08*h - .08**2*(1-decay)) * (w-u)
    return torch.from_numpy(np.c_[initial[:, :2] + h*initial[:, 2:4] + h*h/2*f + iq @ b.T,
                                  initial[:, 2:4] + h*f + iv @ b.T, u + (w-u)*decay])


@pytest.mark.parametrize('model_id', ['hybrid_affine_flow', 'hybrid_affine_aware', 'hybrid_accel_aware'])
@pytest.mark.parametrize('h', [.005, .04, .08])
def test_two_call_maps_match_independent_dense_constant_actuator_response(model_id, h):
    model = build_model(model_id, scales(), 781)
    model.nominal = ConstantPrior()
    state, command = query()
    actual = model.predict(state, command, h)
    torch.testing.assert_close(actual.next_state, analytic_constant(state, command, h), rtol=2e-14, atol=2e-14)
    assert actual.network_calls == actual.vector_field_calls == 2


@pytest.mark.parametrize('model_id', ['hybrid_affine_rk4', 'hybrid_accel_rk4'])
def test_rk4_matches_independent_dense_integration(model_id):
    model = build_model(model_id, scales(), 781)
    model.nominal = ConstantPrior()
    state, command = query()
    # Eight successive steps resolve the independent analytic response accurately.
    actual = state
    for _ in range(8):
        actual = model.predict(actual, command, .005).next_state
    torch.testing.assert_close(actual, analytic_constant(state, command, .04), rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize('model_id,calls', list(zip(IDS, (2, 4, 2, 4, 2))))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_real_call_counts_zero_mixed_tiny_queries_and_gradients(model_id, calls, dtype):
    model = build_model(model_id, scales(dtype), 781)
    state, command = query(dtype)
    original = state.clone()
    observations = []
    handle = model.network.register_forward_hook(lambda *args: observations.append(1))
    try:
        zero = model.predict(state, command, 0.)
        assert not observations and zero.network_calls == zero.vector_field_calls == 0
        assert torch.equal(zero.next_state, state) and zero.next_state.data_ptr() != state.data_ptr()
        mixed = model.predict(state, command, state.new_tensor([[0.], [.04]]))
        assert len(observations) == mixed.network_calls == mixed.vector_field_calls == calls
        assert torch.equal(mixed.next_state[0], state[0])
        observations.clear()
        tiny = model.predict(state, command, 1e-12)
        assert len(observations) == tiny.network_calls == calls
        assert torch.isfinite(tiny.next_state).all()
    finally:
        handle.remove()
    state.requires_grad_()
    command.requires_grad_()
    h = state.new_tensor(.04, requires_grad=True)
    model.predict(state, command, h).next_state[:, :4].square().sum().backward()
    assert torch.equal(state.detach(), original)
    for value in (state, command, h):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    assert (model.network[-1].weight.grad.abs().sum(1) > 0).all()
    assert (model.network[-1].bias.grad.abs() > 0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.network.parameters())
    assert model.nominal.raw.grad is None


@pytest.mark.parametrize('model_id', IDS)
def test_invalid_hybrid_queries(model_id):
    model = build_model(model_id, scales(), 781)
    state, command = query()
    for h in (-.04, float('nan'), float('inf'), torch.ones(2)):
        with pytest.raises(ValueError):
            model.predict(state, command, h)
    for bad_state, bad_command in ((state[:, :4], command), (state, command[:, :1]),
                                    (state * float('nan'), command), (state, command * float('inf'))):
        with pytest.raises(ValueError):
            model.predict(bad_state, bad_command, .04)
    with pytest.raises(ValueError):
        model.predict(state, command, .04, power_mode='fast')


def test_public_hybrid_classes_take_scales_and_rk4_substeps_validate():
    module = importlib.import_module('poc.hybrid_residual')
    for name in ('HybridAffineFlow', 'HybridAffineRK4', 'HybridAffineAware', 'HybridAccelRK4', 'HybridAccelAware'):
        before = torch.random.get_rng_state().clone()
        assert getattr(module, name)(scales()).actuator == 'lagged'
        assert torch.equal(torch.random.get_rng_state(), before)
    model = module.HybridAffineRK4(scales())
    state, command = query()
    for substeps in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            model.predict(state, command, .04, solver_substeps=substeps)
    packet = model.predict(state, command, .04, solver_substeps=2)
    assert packet.network_calls == packet.vector_field_calls == 8
