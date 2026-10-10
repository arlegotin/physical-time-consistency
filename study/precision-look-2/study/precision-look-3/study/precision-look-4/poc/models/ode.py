"""Explicit RK4 neural fields with independent command/activation semantics."""
import torch

from .common import activation,features,interval_tensor,prediction
from .mlp import TransitionPredictor,tanh_mlp


class NeuralODE(TransitionPredictor):
    def __init__(self,actuator,scales,structured=False):
        super().__init__(actuator,scales)
        self.structured = structured
        input_size = 8 if structured or actuator == 'direct' else 10
        self.network = tanh_mlp(input_size,2 if structured else 4)

    def vector_field(self,qv,command,w=None):
        if self.structured and self.actuator == 'lagged':
            # The newly selected command is excluded; stage activation supplies effort.
            inputs = features(qv,w,torch.zeros_like(command[:,:1]),self.v_input)[:,:-1]
        else:
            state = torch.cat((qv,w),dim=1) if self.actuator == 'lagged' else qv
            inputs = features(state,command,torch.zeros_like(command[:,:1]),self.v_input)[:,:-1]
        z = self.network(inputs)
        if self.structured:
            return torch.cat((qv[:,2:4],self.scale_v/.04*z),dim=1)
        return torch.cat((qv[:,2:4]+self.scale_q/.04*z[:,:2],self.scale_v/.04*z[:,2:]),dim=1)

    def predict(self,state,command,h_s,history=None,power_mode=None,solver_substeps=1):
        h = interval_tensor(state,command,h_s,self.actuator)
        if type(solver_substeps) is not int or solver_substeps <= 0:
            raise ValueError('RK4 requires a positive integer substep count')
        if bool((h == 0).all()):
            return prediction(state.clone(),0)
        x,step = state[:,:4],h/solver_substeps
        def field(qv,t):
            w = activation(state,command,t) if self.actuator == 'lagged' else None
            return self.vector_field(qv,command,w)
        for index in range(solver_substeps):
            t = index*step
            k1 = field(x,t)
            k2 = field(x+step*k1/2,t+step/2)
            k3 = field(x+step*k2/2,t+step/2)
            k4 = field(x+step*k3,t+step)
            x = x+step*(k1+2*k2+2*k3+k4)/6
        values = torch.cat((x,activation(state,command,h)),dim=1) if self.actuator == 'lagged' else x
        return prediction(torch.where(h == 0,state,values),4*solver_substeps,vector_fields=4*solver_substeps)
