"""Declared model recipes and seeded default PyTorch initialization."""
import torch

from .mlp import MLP,tanh_mlp
from .ode import NeuralODE
from .power import PowerHead
from .sequence import Recurrent

CORE_MODELS = ('mlp_dt','mlp_h','gru_dt','lstm_dt','power_correct','node_generic','node_structured')


def build_model(model_id,actuator,scales,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if model_id in {'mlp_dt','mlp_h','mlp_blind'}:
            return MLP(actuator,scales,time_scaled=model_id == 'mlp_h',time_blind=model_id == 'mlp_blind')
        if model_id in {'gru_dt','lstm_dt'}:
            return Recurrent(actuator,scales,'gru' if model_id == 'gru_dt' else 'lstm')
        if model_id == 'power_correct':
            return PowerHead(tanh_mlp(9 if actuator == 'direct' else 11),actuator,scales)
        if model_id in {'node_generic','node_structured'}:
            return NeuralODE(actuator,scales,structured=model_id == 'node_structured')
        if model_id == 'attention_dt':
            from .attention import Attention
            return Attention(actuator,scales)
        if model_id in {'ssm_dt','ssm_fixed'}:
            from .ssm import StateSpace
            return StateSpace(actuator,scales,fixed=model_id == 'ssm_fixed')
        raise ValueError('Unregistered model: '+model_id)
