"""The protocol's two-pass discrete power head, with shared nominal weights."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import Prediction, Scales, activation, features, interval_tensor, prediction

POWERS = {'low': (1, 1), 'direct': (2, 1), 'lagged': (3, 2)}


class PowerHead(nn.Module):
    def __init__(self, network: nn.Module, actuator: str, scales: Scales):
        super().__init__()
        if actuator not in {'direct', 'lagged'}:
            raise ValueError('Unknown actuator chain')
        self.network, self.actuator = network, actuator
        self.register_buffer('scale_q', scales.q.detach().clone())
        self.register_buffer('scale_v', scales.v.detach().clone())
        self.register_buffer('v_input', scales.v_input.detach().clone())

    def predict(self, state: Tensor, command: Tensor, h_s: Tensor,
                history=None, power_mode: str | None = None) -> Prediction:
        mode = power_mode or self.actuator
        if mode not in POWERS:
            raise ValueError('Unknown power mode')
        h = interval_tensor(state, command, h_s, self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)
        zero = self.network(features(state, torch.zeros_like(command), h, self.v_input))
        selected = self.network(features(state, command, h, self.v_input))
        if zero.shape != (len(state), 4) or selected.shape != zero.shape:
            raise ValueError('Power-head network must return four outputs')
        delta = selected-zero
        pq, pv = POWERS[mode]
        # Nq=(scale_q/h0^2)*zq and Nv=(scale_v/h0)*zv, exactly S01.4.
        nq, nv = zero[:, :2]*(self.scale_q/.04**2), zero[:, 2:]*(self.scale_v/.04)
        q = state[:, :2]+h*state[:, 2:4]+h*h*nq + (h/.04)**pq*self.scale_q*delta[:, :2]
        v = state[:, 2:4]+h*nv + (h/.04)**pv*self.scale_v*delta[:, 2:]
        values = torch.cat((q, v, activation(state, command, h)), dim=1) if self.actuator == 'lagged' else torch.cat((q, v), dim=1)
        values = torch.where(h == 0, state, values)
        return prediction(values, 2)

