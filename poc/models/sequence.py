"""Recomputed finite physical histories for ordinary recurrent predictors."""
import torch
from torch import nn

from ..history import HistoryBatch
from .common import features,interval_tensor,prediction
from .mlp import TransitionPredictor


def history_features(state,command,h,history: HistoryBatch | None,v_input,cache=None):
    """Padding follows each query; selecting valid lengths excludes its effect."""
    query = features(state,command,h,v_input)
    if history is None:
        return query[:,None,:],torch.ones(len(state),dtype=torch.int64,device=state.device),len(state)
    if len(history.tokens) != len(state):
        raise ValueError('History batch and current state candidates disagree')
    sequences,lengths = [],[]
    normalization = (str(state.dtype),str(state.device),v_input.detach().cpu().numpy().tobytes()) if cache is not None else None
    for index,(tokens,query_time) in enumerate(zip(history.tokens,history.query_time_ns)):
        inputs = []
        for token in tokens:
            if not (int(query_time)-320_000_000 <= token.start_ns < int(query_time)
                    and token.start_ns+token.duration_ns <= int(query_time)):
                continue
            key = None
            if cache is not None and not isinstance(token.state,torch.Tensor) and not v_input.requires_grad:
                key = (token.packet_id,token.start_ns,token.duration_ns,token.state.tobytes(),
                       token.mean_command.tobytes(),normalization)
            value = cache.get(key) if key is not None else None
            if value is None:
                token_state = token.state.to(state) if isinstance(token.state,torch.Tensor) else torch.tensor(token.state.copy(),dtype=state.dtype,device=state.device)
                effort = torch.tensor(token.mean_command.copy(),dtype=state.dtype,device=state.device)
                duration = torch.tensor([[token.duration_ns/1e9]],dtype=state.dtype,device=state.device)
                value = features(token_state[None,:],effort[None,:],duration,v_input)[0]
                if key is not None:
                    cache[key] = value
            inputs.append(value)
        sequences.append(torch.stack([*inputs,query[index]]))
        lengths.append(len(inputs)+1)
    values = nn.utils.rnn.pad_sequence(sequences,batch_first=True)
    return values,torch.tensor(lengths,dtype=torch.int64,device=state.device),sum(lengths)


class Recurrent(TransitionPredictor):
    def __init__(self,actuator,scales,kind='gru'):
        super().__init__(actuator,scales)
        size = 9 if actuator == 'direct' else 11
        width = 64 if kind == 'gru' else 56
        self.encoder = (nn.GRU if kind == 'gru' else nn.LSTM)(size,width,batch_first=True,num_layers=1,dropout=0)
        self.head = nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,4))
        self._history_feature_cache = {}

    def predict(self,state,command,h_s,history=None,power_mode=None):
        h = interval_tensor(state,command,h_s,self.actuator)
        if bool((h == 0).all()):
            return prediction(state.clone(),0)
        values,lengths,work = history_features(state,command,h,history,self.v_input,cache=self._history_feature_cache)
        encoded,_ = self.encoder(values)
        last = encoded[torch.arange(len(state),device=state.device),lengths-1]
        z = self.head(last)
        return prediction(self.decode(state,command,h,z),1,history_tokens=work)
