"""Two-evaluation exponential-response quadrature for shared actuator lag.

Exact for constant drift and dense constant effort gain, not general nonlinear
mechanics. Stage A fixes tau to the historical activation helper's .08 seconds.
"""
from __future__ import annotations

from numbers import Real

import torch

from .actuator_integrals import response_ratios
from .models.common import Prediction, Scales, activation, interval_tensor, prediction
from .models.ode import NeuralODE


class ActuatorAwareNODE(NeuralODE):
    stateless_query = True

    def __init__(self, actuator: str, scales: Scales, tau_s: float = .08):
        if isinstance(tau_s, bool) or not isinstance(tau_s, Real) or tau_s != .08:
            raise ValueError('Stage A requires the historical shared actuator tau of .08 seconds')
        super().__init__(actuator, scales, structured=True)
        self.tau_s = float(tau_s)

    def predict(self, state, command, h_s, history=None, power_mode=None) -> Prediction:
        if power_mode is not None:
            raise ValueError('Actuator-aware NODE has no selectable power mode')
        h = interval_tensor(state, command, h_s, self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)
        q, v = state[:, :2], state[:, 2:4]
        w = state[:, 4:6] if self.actuator == 'lagged' else None
        a0 = self.vector_field(state[:, :4], command, w)[:, 2:4]
        midpoint = torch.cat((q+h*v/2+h*h*a0/8, v+h*a0/2), 1)
        wm = activation(state, command, h/2) if self.actuator == 'lagged' else None
        am = self.vector_field(midpoint, command, wm)[:, 2:4]
        if self.actuator == 'lagged':
            rv, rq = response_ratios(h, self.tau_s)
            value = torch.cat((q+h*v+h*h*a0/2+rq*(am-a0),
                               v+h*a0+rv*(am-a0),
                               activation(state, command, h)), 1)
        else:
            value = torch.cat((q+h*v+h*h*(a0/6+am/3), v+h*am), 1)
        return prediction(torch.where(h == 0, state, value), 2, vector_fields=2)
