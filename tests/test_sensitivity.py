"""New-command derivatives, complete matrix errors and numerical slope intervals."""
import importlib
import numpy as np
import pytest
import torch
from torch import nn

from poc.analytic import action_jacobian,coefficients
from poc.envs import Plant
from poc.models.common import prediction


def modules():
    assert importlib.util.find_spec('poc.metrics'), 'Jacobian metrics are not implemented'
    assert importlib.util.find_spec('poc.model_sensitivity'), 'Learned derivative audit is not implemented'
    return importlib.import_module('poc.metrics'),importlib.import_module('poc.model_sensitivity')


class ExactPredictor(nn.Module):
    def __init__(self,actuator):
        super().__init__()
        self.actuator = actuator
        self.anchor = nn.Parameter(torch.tensor(0.))

    def predict(self,state,command,h_s,history=None):
        h = float(h_s)
        q,v = state[:,:2],state[:,2:4]
        if self.actuator == 'direct':
            output = torch.cat((q+h*v+.5*h*h*command,v+h*command),dim=1)
        else:
            kw,kv,kq = coefficients(h)
            w = state[:,4:]
            output = torch.cat((q+h*v+.5*h*h*w+kq*(command-w),
                               v+h*w+kv*(command-w),w+kw*(command-w)),dim=1)
        return prediction(output+0*self.anchor,1)


@pytest.mark.parametrize('actuator',['direct','lagged'])
def test_autograd_exact_response_is_in_physical_command_units_at_fixed_state(actuator):
    _,audit = modules()
    model = ExactPredictor(actuator)
    state = np.array([.2,-.3,.4,-.5]+([.7,-.8] if actuator == 'lagged' else []))
    original = state.copy()
    result = audit.action_jacobians(model,state,np.array([.8,-.9]),.005)
    expected = action_jacobian(.005,actuator)
    np.testing.assert_allclose(result['q_jac'],expected[:2],rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(result['v_jac'],expected[2:4],rtol=1e-12,atol=1e-15)
    if actuator == 'lagged':
        np.testing.assert_allclose(result['w_jac'],expected[4:],rtol=1e-12,atol=1e-15)
    np.testing.assert_array_equal(state,original)
    assert next(model.parameters()).dtype == torch.float32


def test_full_matrix_error_detects_zero_gain_wrong_sign_and_command_column_swap():
    metrics,_ = modules()
    reference = np.diag([1.,2.])
    exact = metrics.jacobian_metrics(reference,reference,{'reference':0.,'model':0.})
    assert exact['error'] == 0 and exact['gain'] == 1 and exact['gain_certain_inside']
    zero = metrics.jacobian_metrics(np.zeros((2,2)),reference,{'reference':0.,'model':0.})
    assert zero['error'] == 1 and zero['gain'] == 0 and not zero['gain_possible_inside']
    reversed_response = metrics.jacobian_metrics(-reference,reference,{'reference':0.,'model':0.})
    assert reversed_response['gain'] == 1 and reversed_response['error'] == 2
    assert reversed_response['alignment'] == pytest.approx(-1)
    swapped = metrics.jacobian_metrics(reference[:,::-1],reference,{'reference':0.,'model':0.})
    assert swapped['gain'] == 1 and swapped['error'] > 1


def test_absolute_gain_and_error_bounds_expand_across_frozen_threshold():
    metrics,_ = modules()
    reference = np.eye(2)
    result = metrics.jacobian_metrics(.5*reference,reference,{'reference':.01,'model':.002})
    assert result['error_bounds'][0] < .5 < result['error_bounds'][1]
    assert result['gain_bounds'][0] < .5 < result['gain_bounds'][1]
    assert not result['gain_certain_inside'] and result['gain_possible_inside']
    n,a,d = np.linalg.norm(reference),np.linalg.norm(.5*reference),np.linalg.norm(-.5*reference)
    np.testing.assert_allclose(result['error_bounds'],[(d-.012)/(n+.01),(d+.012)/(n-.01)])
    np.testing.assert_allclose(result['gain_bounds'],[(a-.002)/(n+.01),(a+.002)/(n-.01)])


def test_unresolved_reference_and_failed_model_remain_explicit_rows():
    metrics,_ = modules()
    unresolved = metrics.jacobian_metrics(np.eye(2),np.eye(2)*1e-8,{'reference':1e-9,'model':0.})
    assert unresolved['status'] == 'unresolved_reference' and unresolved['error'] is None
    failure = metrics.jacobian_metrics(np.full((2,2),np.nan),np.eye(2),{'reference':0.,'model':0.})
    assert failure['status'] == 'model_failure' and failure['error'] is None


@pytest.mark.parametrize('power',[1,2,3])
def test_slope_and_signed_log_interval_propagation_match_independent_endpoint_calculation(power):
    metrics,_ = modules()
    h = np.array([.0025,.005,.01])
    norms = 3*h**power
    bounds = np.c_[norms*.99,norms*1.01]
    result = metrics.fit_slope(h,norms,bounds)
    assert result['slope'] == pytest.approx(power,abs=1e-12)
    lower = (np.log(bounds[2,0])-np.log(bounds[0,1]))/np.log(4)
    upper = (np.log(bounds[2,1])-np.log(bounds[0,0]))/np.log(4)
    np.testing.assert_allclose(result['slope_bounds'],[lower,upper],rtol=1e-12,atol=1e-12)
    assert result['residual_sum_squares'] < 1e-24
    zero = metrics.fit_slope(h,np.zeros(3),np.zeros((3,2)))
    assert zero['status'] == 'zero_model_gain' and zero['slope'] is None


def test_reference_branches_replay_same_activation_state_and_leave_real_plant_untouched():
    _,audit = modules()
    plant = Plant('lagged',250_000)
    state = np.array([.3,-.4,.5,-.6,.7,-.8])
    plant.reset(state,time_s=.32)
    before = plant.snapshot()
    result = audit.reference_jacobians(plant,state,np.array([.8,-.9]),5_000_000,.02)
    assert result['endpoints'].shape == (2,2,6)
    assert result['status'] == 'ok'
    assert plant.snapshot() == before
    assert result['q_jac'].shape == (2,2) and result['v_jac'].shape == (2,2)
    for axis in range(2):
        difference = result['endpoints'][axis,1]-result['endpoints'][axis,0]
        np.testing.assert_allclose(difference/(2*.02),result['state_jac'][:,axis],rtol=0,atol=0)


def test_metric_rows_reject_unclassified_missing_values_and_unknown_fields():
    metrics,_ = modules()
    assert hasattr(metrics,'validate_row'), 'Metric schema validation is not implemented'
    row = {'run_id':'synthetic','experiment_id':'E01','phase':'unit','model_id':'exact',
           'actuator':'direct','condition':'F','loss_mode':'one_step','seed':0,'workload_id':'fixture',
           'metric':'normalized_jacobian_error','statistic':'median','value':0.,'unit':'dimensionless',
           'sample_count':16,'status':'ok','evidence':'fixture/raw.npz'}
    metrics.validate_row(row)
    with pytest.raises(ValueError,match='missing|value'):
        metrics.validate_row({**row,'value':None})
    metrics.validate_row({**row,'value':None,'status':'unresolved'})
    with pytest.raises(ValueError):
        metrics.validate_row({**row,'value':float('nan')})
    with pytest.raises(ValueError,match='unknown'):
        metrics.validate_row({**row,'unregistered':True})


def test_in_place_activation_initialization_from_command_is_an_invalid_intervention():
    _,audit = modules()
    class WrongInitialization(ExactPredictor):
        def predict(self,state,command,h_s,history=None):
            state[:,4:] = command
            return super().predict(state,command,h_s,history)
    with pytest.raises(ValueError,match='initial state'):
        audit.action_jacobians(WrongInitialization('lagged'),np.zeros(6),np.array([.5,.6]),.005)


def test_power_head_physical_input_chain_agrees_with_saved_float64_central_differences():
    from poc.models.common import Scales
    from poc.models.power import PowerHead
    _,audit = modules()
    network = nn.Linear(9,4,bias=False)
    with torch.no_grad():
        network.weight.zero_()
        network.weight[0,6],network.weight[1,7] = 2.,2.
        network.weight[2,6],network.weight[3,7] = 4.,4.
    model = PowerHead(network,'direct',Scales(torch.ones(2)*.001,torch.ones(2)*.01,torch.ones(2)))
    result = audit.action_jacobians(model,np.array([.2,-.3,.4,-.5]),np.array([.8,-.9]),.005)
    assert 'central_jacobians' in result and 'model_numerical_envelope' in result
    np.testing.assert_allclose(result['q_jac'],np.eye(2)*(.005/.04)**2*float(model.scale_q[0]),atol=1e-15,rtol=1e-12)
    np.testing.assert_allclose(result['v_jac'],np.eye(2)*(.005/.04)*2*float(model.scale_v[0]),atol=1e-15,rtol=1e-12)
    assert result['central_endpoints'].shape == (2,2,2,4)
    np.testing.assert_allclose(result['central_jacobians'],np.broadcast_to(result['state_jac'],(2,4,2)),atol=1e-10,rtol=1e-8)
    assert min(result['model_numerical_envelope']) >= 0
def test_two_half_step_node_audit_retains_the_same_field_and_counts_all_calls():
    from poc.models.factory import build_model
    from poc.models.common import Scales
    from poc.model_sensitivity import action_jacobians
    import torch
    import numpy as np
    scales = Scales(torch.ones(2),torch.ones(2),torch.ones(2))
    model = build_model('node_structured','direct',scales,1).double()
    field = torch.nn.Linear(8,2,bias=False).double()
    with torch.no_grad():
        field.weight.zero_(); field.weight[0,6] = .08; field.weight[1,7] = .08
    model.network = field
    state = np.array([.1,-.2,.3,-.4]); command = np.array([.6,-.7])
    coarse = action_jacobians(model,state,command,.01)
    refined = action_jacobians(model,state,command,.01,solver_substeps=2)
    np.testing.assert_allclose(refined['q_jac'],.5*.01**2*np.eye(2),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(refined['v_jac'],.01*np.eye(2),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(refined['q_jac'],coarse['q_jac'],rtol=1e-12,atol=1e-12)
    assert coarse['vector_field_calls'] == 36 and refined['vector_field_calls'] == 72
