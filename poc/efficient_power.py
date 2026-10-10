"""Equivalent evaluation of the frozen head; historical two-pass code is intact."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .models.common import Scales, activation, features, interval_tensor, prediction
from .models.power import POWERS, PowerHead


class EfficientPowerHead(PowerHead):
    @classmethod
    def from_head(cls, head: PowerHead) -> 'EfficientPowerHead':
        if head.training or any(module.training for module in head.network.modules()):
            raise ValueError('Equivalent batching requires evaluation mode')
        if any(type(module) not in (nn.Sequential, nn.Linear, nn.Tanh)
               for module in head.network.modules()):
            raise ValueError('Only pointwise Linear and Tanh networks are qualified')
        return cls(deepcopy(head.network), head.actuator,
                   Scales(head.scale_q, head.scale_v, head.v_input)).eval()

    def predict(self, state, command, h_s, history=None, power_mode=None):
        mode = power_mode or self.actuator
        if mode not in POWERS:
            raise ValueError('Unknown power mode')
        h = interval_tensor(state, command, h_s, self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)
        if self.actuator == 'direct' and mode == 'direct':
            selected = self.network(features(state, command, h, self.v_input))
            if selected.shape != (len(state), 4):
                raise ValueError('Power-head network must return four outputs')
            q = state[:, :2]+h*state[:, 2:4]+(h/.04)**2*self.scale_q*selected[:, :2]
            v = state[:, 2:4]+(h/.04)*self.scale_v*selected[:, 2:]
            return prediction(torch.where(h == 0, state, torch.cat((q, v), 1)), 1)
        if self.actuator == 'direct':
            return super().predict(state, command, h, history, mode)
        # Both feature sets contain exactly the same initial w. Concatenation
        # reduces dispatches, while still evaluating two rows per candidate.
        inputs = torch.cat((features(state, torch.zeros_like(command), h, self.v_input),
                            features(state, command, h, self.v_input)), 0)
        raw = self.network(inputs)
        if raw.shape != (2*len(state), 4):
            raise ValueError('Power-head network must return four outputs')
        zero, selected = raw.split(len(state))
        delta = selected-zero
        pq, pv = POWERS[mode]
        q = (state[:, :2]+h*state[:, 2:4]+h*h*zero[:, :2]*(self.scale_q/.04**2)
             +(h/.04)**pq*self.scale_q*delta[:, :2])
        v = (state[:, 2:4]+h*zero[:, 2:]*(self.scale_v/.04)
             +(h/.04)**pv*self.scale_v*delta[:, 2:])
        values = torch.cat((q, v, activation(state, command, h)), 1)
        return prediction(torch.where(h == 0, state, values), 1)


def neural_work(result, batch_size: int, actuator: str, power_mode=None) -> dict:
    """Neural work for the optimized head, separate from transition counts."""
    if batch_size != len(result.status) or actuator not in {'direct', 'lagged'}:
        raise ValueError('Work accounting requires the actual candidate batch')
    mode = power_mode or actuator
    if mode not in POWERS:
        raise ValueError('Unknown power mode')
    multiplier = 2 if actuator == 'lagged' else 1
    return dict(network_dispatches=result.network_calls,
                vector_field_calls=result.vector_field_calls,
                neural_rows=batch_size*result.network_calls*multiplier)
