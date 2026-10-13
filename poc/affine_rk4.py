"""RK4 control sharing AffineFlow's torque-affine coefficient network."""
from __future__ import annotations

import torch

from .affine_flow import AffineFlow
from .models.common import Prediction, Scales, activation, interval_tensor, prediction


class AffineRK4(AffineFlow):
    stateless_query = True

    def __init__(self, actuator: str, scales: Scales):
        super().__init__(actuator, scales)

    def predict(self, state, command, h_s, history=None, power_mode=None,
                solver_substeps: int = 1) -> Prediction:
        if power_mode is not None:
            raise ValueError('Affine RK4 has no selectable power mode')
        h = interval_tensor(state, command, h_s, self.actuator)
        if type(solver_substeps) is not int or solver_substeps <= 0:
            raise ValueError('RK4 requires a positive integer substep count')
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)
        x, step = state[:, :4], h/solver_substeps

        def field(qv, t):
            f, matrix = self.coefficients(qv)
            effort = activation(state, command, t) if self.actuator == 'lagged' else command
            acceleration = f+torch.bmm(matrix, effort.unsqueeze(-1)).squeeze(-1)
            return torch.cat((qv[:, 2:4], acceleration), 1)

        for index in range(solver_substeps):
            t = index*step
            k1 = field(x, t)
            k2 = field(x+step*k1/2, t+step/2)
            k3 = field(x+step*k2/2, t+step/2)
            k4 = field(x+step*k3, t+step)
            x = x+step*(k1+2*k2+2*k3+k4)/6
        values = torch.cat((x, activation(state, command, h)), 1) if self.actuator == 'lagged' else x
        calls = 4*solver_substeps
        return prediction(torch.where(h == 0, state, values), calls, vector_fields=calls)
