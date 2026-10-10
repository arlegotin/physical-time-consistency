"""Exact architecture, units, time stages and physical history query tests."""
import importlib
import numpy as np
import pytest
import torch
from torch import nn

from poc.history import HistoryBatch,HistoryToken
from poc.models.common import Scales,features

MODELS = ('mlp_dt','mlp_h','gru_dt','lstm_dt','power_correct','node_generic','node_structured')
COUNTS = {'mlp_dt':18308,'mlp_h':18308,'gru_dt':18820,'lstm_dt':18428,
          'power_correct':18308,'node_generic':18180,'node_structured':17922}


def scales():
    return Scales(torch.tensor([.1,.2]),torch.tensor([.3,.4]),torch.tensor([1.,2.]))


def build(name,actuator='direct',seed=800):
    assert importlib.util.find_spec('poc.models.factory'), 'Core model factory is not implemented'
    return importlib.import_module('poc.models.factory').build_model(name,actuator,scales(),seed)


@pytest.mark.parametrize('name',MODELS)
@pytest.mark.parametrize('actuator',['direct','lagged'])
def test_parameter_counts_mixed_zero_intervals_and_independent_batch_parity(name,actuator):
    model = build(name,actuator).double()
    count = sum(x.numel() for x in model.parameters())
    expected = COUNTS[name]
    if actuator == 'lagged':
        expected += {'mlp_dt':256,'mlp_h':256,'gru_dt':384,'lstm_dt':448,
                     'power_correct':256,'node_generic':256,'node_structured':0}[name]
    assert count == expected
    state = torch.tensor([[.2,-.4,.5,-.6]+([.3,-.2] if actuator == 'lagged' else [])]*3,dtype=torch.float64)
    command = torch.tensor([[.7,-.8],[-.3,.4],[.2,.1]],dtype=torch.float64,requires_grad=True)
    h = torch.tensor([[0.],[.01],[.04]],dtype=torch.float64)
    result = model.predict(state,command,h)
    torch.testing.assert_close(result.next_state[0],state[0],rtol=0,atol=0)
    assert result.status == ('ok',)*3
    for i in range(3):
        scalar = model.predict(state[i:i+1],command[i:i+1],h[i:i+1])
        torch.testing.assert_close(result.next_state[i],scalar.next_state[0],atol=1e-12,rtol=1e-12)
    result.next_state[:,0:4].sum().backward()
    assert torch.isfinite(command.grad).all() and torch.linalg.norm(command.grad[1:]) > 0
    if actuator == 'lagged':
        expected_w = state[:,4:]+(command-state[:,4:])*(-torch.expm1(-h/.08))
        torch.testing.assert_close(result.next_state[:,4:],expected_w,rtol=0,atol=0)
    if name.startswith('node_'):
        assert result.vector_field_calls == 4 and result.network_calls == 4
        assert model.predict(state,command,h,solver_substeps=2).vector_field_calls == 8
    if name == 'power_correct':
        assert result.network_calls == 2


def test_feature_order_and_physical_command_conversion_are_explicit():
    state = torch.tensor([[.2,.3,4.,6.,.5,-.5]],dtype=torch.float64)
    command = torch.tensor([[.6,-.8]],dtype=torch.float64,requires_grad=True)
    f = features(state,command,torch.tensor([[.02]]),torch.tensor([2.,3.]))
    torch.testing.assert_close(f,torch.tensor([[np.sin(.2),np.cos(.2),np.sin(.3),np.cos(.3),2.,2.,.25,-.25,.3,-.4,.5]],dtype=torch.float64))
    assert torch.autograd.grad(f[0,8],command)[0][0,0] == .5


@pytest.mark.parametrize('name', ['gru_dt','lstm_dt'])
def test_sequence_history_counts_eligibility_read_only_and_candidate_permutation(name):
    model = build(name).double()
    state = torch.tensor([[.2,.3,.1,-.2]]*3,dtype=torch.float64)
    command = torch.tensor([[.1,.2],[.3,.4],[-.5,.6]],dtype=torch.float64)
    sequence = tuple(HistoryToken(i*20_000_000,np.array([.1,.2,.3,.4]),np.array([.2,-.1]),20_000_000,'observed',f'p{i}') for i in range(17))
    histories = HistoryBatch((sequence,(),sequence[:8]),np.array([340_000_000]*3,dtype=np.int64))
    original = model.predict(state,command,.005,history=histories)
    assert original.history_tokens_processed == 16+7+3  # Start 0 is outside the .34-.32 window in both branches.
    permutation = [2,0,1]
    shuffled = model.predict(state[permutation],command[permutation],.005,
                     history=HistoryBatch(tuple(histories.tokens[i] for i in permutation),histories.query_time_ns[permutation]))
    torch.testing.assert_close(original.next_state,shuffled.next_state[np.argsort(permutation)],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(original.next_state,model.predict(state,command,.005,history=histories).next_state,rtol=0,atol=0)
    torch.testing.assert_close(original.next_state[1],model.predict(state[1:2],command[1:2],.005).next_state[0],atol=1e-12,rtol=1e-12)
    altered = HistoryToken(0,np.ones(4)*100,np.array([2.,-2.]),20_000_000,'observed','p0')
    masked = HistoryBatch(((altered,*sequence[1:]),(),(altered,*sequence[1:8])),histories.query_time_ns)
    torch.testing.assert_close(original.next_state,model.predict(state,command,.005,history=masked).next_state,rtol=0,atol=0)
    assert len(histories.tokens[0]) == 17


def test_lagged_structured_field_has_no_new_command_shortcut_at_fixed_activation():
    model = build('node_structured','lagged').double()
    qv = torch.tensor([[.2,.3,.4,.5]],dtype=torch.float64)
    w = torch.tensor([[.6,.7]],dtype=torch.float64)
    command = torch.tensor([[.8,.9]],dtype=torch.float64,requires_grad=True)
    field = model.vector_field(qv,command,w)
    derivative = torch.autograd.grad(field.sum()+command.sum()*0,command)[0]
    torch.testing.assert_close(derivative,torch.zeros_like(command),rtol=0,atol=0)
    torch.testing.assert_close(field[:,:2],qv[:,2:],rtol=0,atol=0)


def test_structured_rk4_stage_activation_matches_exact_lagged_constant_acceleration_field():
    model = build('node_structured','lagged').double()
    network = nn.Linear(8,2,bias=False).double()
    with torch.no_grad():
        network.weight.zero_()
        network.weight[0,6] = 2*.04/model.scale_v[0]
        network.weight[1,7] = 2*.04/model.scale_v[1]
    model.network = network
    state = torch.tensor([[.2,-.3,.4,-.5,.6,-.7]],dtype=torch.float64)
    command = torch.tensor([[.8,-.9]],dtype=torch.float64)
    h = .01
    result = model.predict(state,command,h)
    qv = state[:,:4]
    def exact_field(x,t):
        w = state[:,4:]+(command-state[:,4:])*(-torch.expm1(torch.tensor(-t/.08,dtype=torch.float64)))
        return torch.cat((x[:,2:],w),dim=1)
    k1 = exact_field(qv,0)
    k2 = exact_field(qv+h*k1/2,h/2)
    k3 = exact_field(qv+h*k2/2,h/2)
    k4 = exact_field(qv+h*k3,h)
    expected = qv+h*(k1+2*k2+2*k3+k4)/6
    torch.testing.assert_close(result.next_state[:,:4],expected,atol=1e-15,rtol=1e-15)


def test_initialization_is_reproducible_without_repurposing_global_seed():
    first,second = build('mlp_dt'),build('mlp_dt')
    for a,b in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match='model'):
        build('unregistered_architecture')


def test_history_feature_cache_reuses_only_identical_completed_physical_inputs(monkeypatch):
    import poc.models.sequence as sequence
    state = torch.tensor([[.2,.3,.4,.5]],dtype=torch.float64)
    command = torch.tensor([[.7,-.8]],dtype=torch.float64,requires_grad=True)
    h = torch.tensor([[.005]],dtype=torch.float64)
    token = HistoryToken(0,np.array([.1,.2,.3,.4]),np.array([.5,-.6]),40_000_000,'observed','p0')
    history = HistoryBatch(((token,),),np.array([40_000_000],dtype=np.int64))
    original = sequence.history_features(state,command,h,history,scales().v_input)
    calls = []
    old_features = sequence.features
    def counted(*args):
        calls.append(len(args[0]))
        return old_features(*args)
    monkeypatch.setattr(sequence,'features',counted)
    cache = {}
    first = sequence.history_features(state,command,h,history,scales().v_input,cache=cache)
    assert len(calls) == 2
    second = sequence.history_features(state,command,h,history,scales().v_input,cache=cache)
    assert len(calls) == 3  # The new query is always recomputed; the past token is unchanged.
    torch.testing.assert_close(original[0],first[0],rtol=0,atol=0)
    torch.testing.assert_close(original[0],second[0],rtol=0,atol=0)
    derivative = torch.autograd.grad(second[0].sum(),command)[0]
    torch.testing.assert_close(derivative,torch.ones_like(command)/2,rtol=0,atol=0)
    altered = HistoryToken(0,np.ones(4),token.mean_command,40_000_000,'observed','p0')
    changed = HistoryBatch(((altered,),),history.query_time_ns)
    recomputed = sequence.history_features(state,command,h,changed,scales().v_input,cache=cache)
    torch.testing.assert_close(recomputed[0],sequence.history_features(state,command,h,changed,scales().v_input)[0],rtol=0,atol=0)
    assert not torch.equal(recomputed[0],original[0])
    normalized = sequence.history_features(state,command,h,history,scales().v_input*2,cache=cache)
    torch.testing.assert_close(normalized[0],sequence.history_features(state,command,h,history,scales().v_input*2)[0],rtol=0,atol=0)
    pending = history.at_time(np.array([20_000_000],dtype=np.int64))
    result = sequence.history_features(state,command,h,pending,scales().v_input,cache=cache)
    assert result[1].tolist() == [1] and result[2] == 1


def test_history_feature_cache_keeps_imagined_state_gradients_and_fresh_backward_graphs():
    from poc.models.sequence import history_features
    imagined = torch.tensor([.1,.2,.3,.4],dtype=torch.float64,requires_grad=True)
    history = HistoryBatch(((HistoryToken(0,imagined,np.array([.5,-.6]),40_000_000,'imagined','p0'),),),
                           np.array([40_000_000],dtype=np.int64))
    cache = {}
    for _ in range(2):
        values,_,_ = history_features(torch.zeros((1,4),dtype=torch.float64),torch.zeros((1,2),dtype=torch.float64),
                                     torch.tensor([[.005]],dtype=torch.float64),history,torch.ones(2),cache=cache)
        gradient = torch.autograd.grad(values.sum(),imagined,retain_graph=True)[0]
        torch.testing.assert_close(gradient,torch.tensor([np.cos(.1)-np.sin(.1),np.cos(.2)-np.sin(.2),1.,1.],dtype=torch.float64))
    assert not cache
