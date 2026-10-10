"""Two acceleration evaluations, preserving mechanical new-command order."""
import torch
from .models.common import activation, interval_tensor, prediction
from .models.ode import NeuralODE


class QuadratureNODE(NeuralODE):
    stateless_query = True

    def __init__(self, actuator, scales):
        super().__init__(actuator, scales, structured=True)

    def predict(self, state, command, h_s, history=None, power_mode=None):
        if power_mode is not None:
            raise ValueError('Quadrature NODE has no selectable power mode')
        h = interval_tensor(state, command, h_s, self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(), 0)
        q, v = state[:, :2], state[:, 2:4]
        w = state[:, 4:6] if self.actuator == 'lagged' else None
        a0 = self.vector_field(state[:, :4], command, w)[:, 2:4]
        midpoint = torch.cat((q+h*v/2+h*h*a0/8, v+h*a0/2), 1)
        wm = activation(state, command, h/2) if self.actuator == 'lagged' else None
        am = self.vector_field(midpoint, command, wm)[:, 2:4]
        value = torch.cat((q+h*v+h*h*(a0/6+am/3), v+h*am), 1)
        if self.actuator == 'lagged':
            value = torch.cat((value, activation(state, command, h)), 1)
        return prediction(torch.where(h == 0, state, value), 2, vector_fields=2)
