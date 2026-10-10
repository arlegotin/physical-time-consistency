"""Small interval-conditioned residual predictors and shared tanh MLPs."""
from torch import nn
import torch

from .common import activation,features,interval_tensor,prediction


def tanh_mlp(input_size: int, output_size: int = 4):
    return nn.Sequential(nn.Linear(input_size,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,output_size))


class TransitionPredictor(nn.Module):
    def __init__(self,actuator,scales):
        super().__init__()
        if actuator not in {'direct','lagged'}:
            raise ValueError('Unknown actuator chain')
        self.actuator = actuator
        self.register_buffer('scale_q',scales.q.detach().clone())
        self.register_buffer('scale_v',scales.v.detach().clone())
        self.register_buffer('v_input',scales.v_input.detach().clone())

    def decode(self,state,command,h,z,scaled=False):
        factor = h/.04 if scaled else 1.
        q = state[:,:2]+h*state[:,2:4]+factor*self.scale_q*z[:,:2]
        v = state[:,2:4]+factor*self.scale_v*z[:,2:]
        values = torch.cat((q,v,activation(state,command,h)),dim=1) if self.actuator == 'lagged' else torch.cat((q,v),dim=1)
        return torch.where(h == 0,state,values)


class MLP(TransitionPredictor):
    def __init__(self,actuator,scales,time_scaled=False,time_blind=False):
        super().__init__(actuator,scales)
        self.network = tanh_mlp(9 if actuator == 'direct' else 11)
        self.time_scaled,self.time_blind = time_scaled,time_blind

    def predict(self,state,command,h_s,history=None,power_mode=None):
        h = interval_tensor(state,command,h_s,self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(),0)
        inputs = features(state,command,h,self.v_input)
        if self.time_blind:
            inputs = torch.cat((inputs[:,:-1],torch.ones_like(h)),dim=1)
        z = self.network(inputs)
        return prediction(self.decode(state,command,h,z,self.time_scaled),1)
