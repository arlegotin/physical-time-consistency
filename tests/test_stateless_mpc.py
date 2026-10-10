"""Shared fast evaluation must preserve physical scoring and failed candidates."""
import numpy as np
import pytest
import torch
from poc import mpc
from poc.models.common import Scales, prediction
from poc.models.ode import NeuralODE
from poc.affine_flow import AffineFlow
from poc.history import HistoryBatch


def cfg():
    return dict(candidate_count=16, iterations=2, elite_count=2, initial_std=1., std_floor=.1)


@pytest.mark.parametrize('name',['affine','rk4','quadrature'])
@pytest.mark.parametrize('step',[5_000_000,40_000_000])
def test_fast_and_original_candidate_paths_costs_work_and_selection_agree(name,step):
    from poc.quadrature_node import QuadratureNODE
    from poc.stateless_mpc import rollout_candidates, search
    torch.manual_seed(2)
    scales=Scales(torch.full((2,),.04),torch.full((2,),.04),torch.ones(2))
    model={'affine':lambda:AffineFlow('lagged',scales),'rk4':lambda:NeuralODE('lagged',scales,True),
           'quadrature':lambda:QuadratureNODE('lagged',scales)}[name]().double()
    state=torch.tensor([.2,-.3,.1,-.2,.8,-.4],dtype=torch.float64)
    target=np.array([.5,-.2]); context=HistoryBatch(((),),np.array([320_000_000],dtype=np.int64))
    samples=np.random.default_rng(4).normal(size=(2,16,6,2)); bank=np.clip(samples[0],-2,2)
    args=(model,state,target,context,bank,(step,)*(40_000_000//step),40_000_000)
    expected=mpc.rollout_candidates(*args); actual=rollout_candidates(*args)
    np.testing.assert_array_equal(actual['anchors'],expected['anchors'])
    np.testing.assert_allclose(actual['costs'],expected['costs'],rtol=1e-14,atol=1e-14)
    for key in ('statuses','reasons','path_checksums','work'): assert actual[key]==expected[key]
    args=(model,state,torch.tensor(target),context,None,samples,cfg(),(step,)*(40_000_000//step),40_000_000)
    expected=mpc.search(*args); actual=search(*args)
    assert actual['selected_global_id']==expected['selected_global_id']
    np.testing.assert_array_equal(actual['knots'],expected['knots'])
    np.testing.assert_array_equal(actual['candidates']['knots'],expected['candidates']['knots'])


class Failure:
    stateless_query=True
    def predict(self,state,command,h_s,history=None):
        values=state.clone(); values[command[:,0]>0,0]=float('nan')
        return prediction(values,1)


def test_failures_and_all_failure_fallback_remain_explicit():
    from poc.stateless_mpc import rollout_candidates, search
    bank=np.zeros((2,6,2)); bank[0,:,0]=1
    out=rollout_candidates(Failure(),torch.zeros(4),np.zeros(2),None,bank,(40_000_000,),40_000_000)
    assert out['costs'][0]==np.inf and out['statuses'][0]=='model_failure'
    assert out['reasons'][0] and out['costs'][1]==0
    class All(Failure):
        def predict(self,state,command,h_s,history=None):return prediction(torch.full_like(state,float('nan')),1)
    out=search(All(),torch.zeros(4),torch.zeros(2),None,None,np.ones((2,16,6,2)),cfg(),(40_000_000,),40_000_000)
    assert out['status']=='all_candidates_failed' and out['fallback']=='zero_command'
    assert not out['knots'].any() and out['selected_global_id'] is None


def test_invalid_path_or_history_dependent_model_rejected():
    from poc.stateless_mpc import rollout_candidates
    class Sequence(Failure): stateless_query=False
    bank=np.zeros((2,6,2)); args=(torch.zeros(4),np.zeros(2),None,bank,(40_000_000,),40_000_000)
    with pytest.raises(ValueError):rollout_candidates(Sequence(),*args)
    bank[0,0,0]=3
    with pytest.raises(ValueError):rollout_candidates(Failure(),*args)
    with pytest.raises(ValueError):rollout_candidates(Failure(),torch.zeros(4),np.zeros(2),None,np.zeros((2,6,2)),(40_000_000,),20_000_000)
