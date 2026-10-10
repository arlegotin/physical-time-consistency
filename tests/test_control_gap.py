"""Selected-plan error uses its actual next anchor, rather than later replanned motion."""
import copy
import math
import numpy as np
import pytest


def workload():
    rows=[];anchors=np.zeros((40,4));segments=[]
    predicted=[.1,-.2,.3,.4]
    for i in range(40):
        rows.append(dict(time_ns=i*40_000_000,status='ok',
            predicted_anchors=[predicted]*6,actual_first_interval_state=[0.,0.,0.,0.],
            predicted_first_interval_cost=.00258,actual_first_interval_cost=.00008))
        segments.append(dict(start_ns=i*40_000_000,duration_ns=40_000_000,command=[1.,1.]))
    return rows,anchors,segments,np.zeros(2)


def test_gap_reconstructs_one_next_anchor_and_no_overlapping_horizon_or_terminal_costs():
    from poc.control_gap import first_interval_metrics
    rows,truth,segments,target=workload()
    result=first_interval_metrics(rows,truth,segments,target)
    assert result['n_decisions']==40 and result['n_predictions']==40
    assert result['q_prediction_rmse_rad']==pytest.approx(math.sqrt(.025))
    assert result['v_prediction_rmse_rad_s']==pytest.approx(math.sqrt(.125))
    assert result['mean_real_minus_predicted_cost']==pytest.approx(-.0025)
    assert result['sum_actual_interval_costs']==pytest.approx(.0032)
    assert result['terminal_cost_included'] is False
    assert len(result['decisions'])==40


@pytest.mark.parametrize('mutation',['extra','clock','state','cost','prediction'])
def test_gap_rejects_inconsistent_real_anchor_or_logged_cost(mutation):
    from poc.control_gap import first_interval_metrics
    rows,truth,segments,target=workload()
    if mutation=='extra':rows.append(copy.deepcopy(rows[0]))
    elif mutation=='clock':rows[1]['time_ns']=5_000_000
    elif mutation=='state':rows[0]['actual_first_interval_state'][0]=.01
    elif mutation=='cost':rows[0]['actual_first_interval_cost']+=.001
    else:rows[0]['predicted_anchors'][0][0]=float('nan')
    with pytest.raises(ValueError):first_interval_metrics(rows,truth,segments,target)


def test_fallback_retains_real_cost_and_explicit_missing_prediction():
    from poc.control_gap import first_interval_metrics
    rows,truth,segments,target=workload()
    rows[0].update(status='all_candidates_failed',predicted_anchors=None,predicted_first_interval_cost=None,
                   actual_first_interval_cost=0.)
    segments[0]['command']=[0.,0.]
    result=first_interval_metrics(rows,truth,segments,target)
    assert result['n_predictions']==39 and result['fallback_count']==1
    assert result['sum_actual_interval_costs']==pytest.approx(.00312)
    assert result['decisions'][0]['q_error_rad'] is None
    assert result['decisions'][0]['prediction_status']=='missing_due_to_fallback'


def test_all_failed_predictions_cannot_become_zero_error():
    from poc.control_gap import first_interval_metrics
    rows,truth,segments,target=workload()
    for row,segment in zip(rows,segments):
        row.update(status='all_candidates_failed',predicted_anchors=None,predicted_first_interval_cost=None,
                   actual_first_interval_cost=0.)
        segment['command']=[0.,0.]
    result=first_interval_metrics(rows,truth,segments,target)
    assert result['n_predictions']==0 and result['fallback_count']==40
    assert result['q_prediction_rmse_rad'] is None
    assert result['mean_real_minus_predicted_cost'] is None
