"""Honest results require complete paired cells and uncensored compute/tradeoffs."""
import numpy as np
import pytest


def rows(seeds=(1,2,3,4,5),episodes=2):
    from poc.compute_study import CONFIGS
    return [dict(model_id=name,step_ns=step,seed=s,episode_id=e,complete=True,realized_cost=1.+s/10,
        planning_median_seconds=.1,planning_p95_seconds=.12,q_rmse_rad=.2,terminal_q_error_rad=.01,
        velocity_rms_rad_s=.3,effort_integral=.4,settings={'candidate_count':16},
        work={'network_calls':40,'vector_field_calls':40,'scalar_candidate_evaluations':640,'history_tokens_processed':0},
        success=True,fallback_count=0,target=[.1,.2],initial_state=[0.]*6)
        for s in seeds for e in range(episodes) for name,step in CONFIGS]


def test_coverage_rejects_missing_duplicate_or_unpaired_tasks():
    from poc.compute_study_report import control_table
    data=rows();result=control_table(data,[1,2,3,4,5],2)
    assert len(result['configurations'])==6 and result['episode_count']==60
    for bad in (data[:-1],data+[data[0]], [{**r,'target':[.4,.2]} if i==0 else r for i,r in enumerate(data)]):
        with pytest.raises(ValueError):control_table(bad,[1,2,3,4,5],2)


def test_control_table_reports_measured_work_and_rejects_invalid_counters():
    from poc.compute_study_report import control_table
    data=rows();result=control_table(data,[1,2,3,4,5],2)
    row=result['configurations'][0]
    assert row['network_calls_per_decision']==1
    assert row['neural_rows_per_decision']==16
    assert row['candidate_transitions_per_decision']==16
    assert row['velocity_rms_rad_s']==pytest.approx(.3)
    assert row['effort_integral']==pytest.approx(.4)
    data[0]['work']['network_calls']=-1
    with pytest.raises(ValueError):control_table(data,[1,2,3,4,5],2)


def test_eight_contrast_holm_supports_positive_and_preserves_losses():
    from poc.compute_study_report import holm_eight
    from poc.affine_confirmation import paired_comparison
    baseline=np.arange(1.,12.)
    positive=[paired_comparison(baseline,.8*baseline,1,0.) for _ in range(8)]
    assert all(r['holm']['supported'] for r in holm_eight(positive,1,0.))
    negative=[paired_comparison(baseline,1.2*baseline,1,0.) for _ in range(8)]
    assert not any(r['holm']['supported'] for r in holm_eight(negative,1,0.))
    assert all(r['point_improvement']<0 for r in holm_eight(negative,1,0.))
    with pytest.raises(ValueError):holm_eight(positive[:-1],1,0.)


def test_independent_cost_has_one_terminal_and_every_physical_anchor():
    from poc.compute_study_report import independent_cost
    raw=dict(time_ns=np.arange(0,1_600_000_001,250000),state=np.zeros((6401,6)),
             anchors=np.zeros((40,6)),commands=np.zeros((40,2)),target=np.ones(2))
    assert independent_cost(raw)==pytest.approx(13.2,rel=1e-14)
    raw['time_ns']=raw['time_ns'][:-1]
    with pytest.raises(ValueError):independent_cost(raw)


def test_nonfinite_truth_target_cannot_evade_independent_arithmetic():
    from poc.compute_study_report import independent_cost
    raw=dict(time_ns=np.arange(0,1_600_000_001,250000),state=np.zeros((6401,6)),
             anchors=np.zeros((40,6)),commands=np.zeros((40,2)),target=np.array([float('nan'),1.]))
    with pytest.raises(ValueError):independent_cost(raw)


def test_failed_candidates_cannot_hide_behind_complete_episode_summary():
    from poc.compute_study_report import audit_candidates
    raw=dict(candidate_knots=np.zeros((40,8,6,2)),candidate_costs=np.ones((40,8)),
        candidate_statuses=np.full((40,8),'ok',dtype='U20'),candidate_reasons=np.full((40,8),'',dtype='U30'),
        selected_ids=np.zeros(40,dtype=int),selected_knots=np.zeros((40,6,2)),commands=np.zeros((40,2)))
    assert audit_candidates(raw,0)==0
    raw['candidate_costs'][0,0]=float('inf')
    with pytest.raises(ValueError):audit_candidates(raw,0)
    raw['candidate_statuses'][0,0]='model_failure';raw['candidate_reasons'][0,0]='known failed'
    with pytest.raises(ValueError):audit_candidates(raw,0) # Executed candidate is failed.
    raw['candidate_costs'][0,:]=float('inf');raw['candidate_statuses'][0,:]='model_failure';raw['candidate_reasons'][0,:]='known failed'
    raw['selected_ids'][0]=-1
    with pytest.raises(ValueError):audit_candidates(raw,0) # Undisclosed fallback.
    assert audit_candidates(raw,1)==1


def test_moving_state_losses_and_compute_mismatch_remain_visible():
    from poc.compute_study_report import reduction, compute_match
    assert reduction(1.,1.12)==pytest.approx(-.12)
    assert reduction(0.,0.) is None
    assert compute_match([.1,.15,.11,.09,.10,.10])['matched'] is False
    with pytest.raises(ValueError):compute_match([.1,float('nan')])


def test_demo_clips_validate_measured_motion_and_paired_targets():
    from poc.compute_study_report import validate_clips
    from poc.compute_study import CONFIGS
    data=[dict(model_id=n,step_ns=h,seed=1,episode_id=0,time_s=(np.arange(161)/100).tolist(),
               q=np.zeros((161,2)).tolist(),target=[.1,.2],cost=1.,planner_seconds=.1,candidates=16) for n,h in CONFIGS]
    assert len(validate_clips(data,[1],1))==6
    data[0]['q']=data[0]['q'][:-1]
    with pytest.raises(ValueError):validate_clips(data,[1],1)
