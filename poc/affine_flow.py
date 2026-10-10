"""Two-stage learned affine acceleration with exact frozen-coefficient lag integrals."""
from __future__ import annotations

import math

import torch
from torch import nn

from .models.common import Scales, activation, interval_tensor, prediction


def lag_coefficients(h):
    """New-command velocity/position coefficients; avoid small-r subtraction."""
    r = h/.08
    v = h+.08*torch.expm1(-r)
    q = .5*h*h-.08*h-.08**2*torch.expm1(-r)
    if bool((r >= .1).all()):
        return v, q
    # Kv/h = r/2-r²/6+...; Kq/h² = r/6-r²/24+...
    # At r<.1, terms through order 12 resolve float64 relative precision.
    pv, pq = torch.zeros_like(h), torch.zeros_like(h)
    for k in range(12, 0, -1):
        pv = pv*r+((-1)**(k+1)/math.factorial(k+1))
        pq = pq*r+((-1)**(k+1)/math.factorial(k+2))
    small_v, small_q = h*r*pv, h*h*r*pq
    return torch.where(r < .1, small_v, v), torch.where(r < .1, small_q, q)


class AffineFlow(nn.Module):
    def __init__(self, actuator: str, scales: Scales):
        super().__init__()
        if actuator not in {'direct', 'lagged'}:
            raise ValueError('Unknown actuator chain')
        self.actuator = actuator
        self.register_buffer('scale_q', scales.q.detach().clone())
        self.register_buffer('scale_v', scales.v.detach().clone())
        self.register_buffer('v_input', scales.v_input.detach().clone())
        self.network = nn.Sequential(nn.Linear(6, 128), nn.Tanh(), nn.Linear(128, 128),
                                     nn.Tanh(), nn.Linear(128, 6))

    def coefficients(self, qv):
        q, v = qv[:, :2], qv[:, 2:4]
        inputs = torch.cat((torch.stack((q[:, 0].sin(), q[:, 0].cos(),
                                        q[:, 1].sin(), q[:, 1].cos()), dim=1), v/self.v_input), dim=1)
        raw = self.network(inputs)
        f = raw[:, :2]*(self.scale_v/.04)
        matrix = raw[:, 2:].reshape(-1, 2, 2)*(self.scale_v/.08)[None, :, None]
        return f, matrix

    def _step(self, state, command, h, f, matrix):
        if self.actuator == 'direct':
            v_effort, q_effort = h*command, .5*h*h*command
        else:
            new_v, new_q = lag_coefficients(h)
            w = state[:, 4:6]
            v_effort = (h-new_v)*w+new_v*command
            q_effort = (.5*h*h-new_q)*w+new_q*command
        v_torque = torch.bmm(matrix, v_effort.unsqueeze(-1)).squeeze(-1)
        q_torque = torch.bmm(matrix, q_effort.unsqueeze(-1)).squeeze(-1)
        q = state[:, :2]+h*state[:, 2:4]+.5*h*h*f+q_torque
        v = state[:, 2:4]+h*f+v_torque
        values = [q, v]
        if self.actuator == 'lagged':
            values.append(activation(state, command, h))
        return torch.cat(values, dim=1)

    def predict(self, state, command, h_s, history=None, power_mode=None):
        if power_mode is not None:
            raise ValueError('Affine flow has no selectable power mode')
        h = interval_tensor(state, command, h_s, self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)
        f0, b0 = self.coefficients(state[:, :4])
        midpoint = self._step(state, command, h/2, f0, b0)
        fm, bm = self.coefficients(midpoint[:, :4])
        values = self._step(state, command, h, fm, bm)
        return prediction(torch.where(h == 0, state, values), 2, vector_fields=2)
