import copy

import numpy as np
import pytest

from poc.affine_confirmation import (fresh_probes, prediction_rows, paired_comparison, control_summary,
                                    reserve_analysis_look, validate_profiles, retrain_metrics)


def corpus():
    states=np.arange(2*33*6,dtype=float).reshape(2,33,6)
    return dict(state=states,time_ns=np.tile(np.arange(33,dtype=np.int64)*20000000,(2,1)),
                trajectory_id=np.arange(2,dtype=np.int64))


def test_fresh_probes_use_unchanged_cold_and_completed_warm_states():
    values=fresh_probes(corpus())
    assert np.array_equal(values['states'][0],corpus()['state'][0,0])
    assert np.array_equal(values['states'][1],corpus()['state'][0,16])
    assert values['strata'].tolist()==['cold','warm','cold','warm']
    assert values['probe_ids'].tolist()==[0,1,2,3]
    assert np.array_equal(values['commands'],fresh_probes(corpus())['commands'])
    assert np.max(np.abs(values['commands'])) < 1.5


def raw_predictions():
    shape=(4,2,6)
    return dict(probe_ids=np.arange(4),strata=np.array(['cold','warm']*2),
                horizons_ns=np.array([5000000,10000000]),truth_zero=np.zeros(shape),
                truth_command=np.zeros(shape),model_zero=np.zeros(shape),
                model_command=np.ones(shape)*.01,model_composed=np.ones(shape)*.02)


def test_prediction_reaggregation_keeps_all_coordinates_and_components():
    raw=raw_predictions(); expected={key:raw[key] for key in ('probe_ids','strata','horizons_ns')}
    rows=prediction_rows(raw,expected)
    assert len(rows)==8
    assert rows[0]['total_error_mean']==pytest.approx(np.sqrt(2)*.01)
    damaged=copy.deepcopy(raw); damaged['strata'][0]='warm'
    with pytest.raises(ValueError,match='coordinate'):
        prediction_rows(damaged,expected)
    damaged=copy.deepcopy(raw); damaged['model_command'][0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        prediction_rows(damaged,expected)


def test_cluster_comparison_does_not_invent_certainty_or_zero_denominator_gain():
    b=np.array([1.,1.1,1.05,.95,1.2]); m=.5*b
    result=paired_comparison(b,m,1,0.)
    assert result['contrast']['lower'] > 0
    assert result['ratio']['estimate']==pytest.approx(.5)
    constant=paired_comparison(np.ones(5),np.ones(5)*.5,1,0.)
    assert constant['contrast']['status']=='unresolved_variance'
    zero=paired_comparison(np.zeros(5),np.zeros(5),1,0.)
    assert zero['ratio']['status']=='unresolved_denominator'
    with pytest.raises(ValueError):
        paired_comparison(b,m[:-1],1,0.)


def episode(model,step,seed,task,cost,fallback=0):
    return dict(model_id=model,model_partition_ns=[step]*(40000000//step),seed=seed,episode_id=task,
                actuator='lagged',actuator_interval_ns=40000000,phase='confirmatory',complete=True,
                initial_state=[0.]*6,target=[.1,.2],realized_cost=cost,fallback_count=fallback,
                planner_latency_median_seconds=.2,q_rmse_rad=.1,success=False)


def test_control_reaggregation_retains_failed_policies_and_matches_every_paired_task():
    configs=[('affine_flow',5000000),('node_structured',5000000),('node_structured',40000000)]
    seeds=[2026100800,2026100801]
    rows=[episode(name,step,seed,task,1. if name=='affine_flow' else 2.,int(task==0 and name=='affine_flow'))
          for seed in seeds for task in range(2) for name,step in configs]
    result=control_summary(rows,seeds,2)
    assert len(result['seed_metrics'])==6
    assert sum(r['fallback_count'] for r in result['seed_metrics'])==2
    with pytest.raises(ValueError,match='coverage'):
        control_summary(rows[:-1],seeds,2)
    damaged=copy.deepcopy(rows); damaged[1]['target']=[.3,.2]
    with pytest.raises(ValueError,match='paired'):
        control_summary(damaged,seeds,2)


def test_statistical_looks_are_immutable_and_reaggregation_does_not_reset_alpha(tmp_path):
    first=reserve_analysis_look(tmp_path,1,{'measurement':'a','control':None})
    assert first['alpha']==.025
    assert reserve_analysis_look(tmp_path,1,{'measurement':'a','control':None})==first
    with pytest.raises(ValueError,match='look'):
        reserve_analysis_look(tmp_path,1,{'measurement':'a','control':'b'})
    second=reserve_analysis_look(tmp_path,2,{'measurement':'a','control':'b'})
    assert second['alpha']==.0125
    with pytest.raises(ValueError,match='look'):
        reserve_analysis_look(tmp_path,4,{'measurement':'c','control':'b'})


def test_timing_reaggregation_rejects_invented_speed_and_missing_queries():
    rows=[dict(model_id=name,seed=2026100800,batch_size=batch,h_s=h,
               samples_ns=[100]* (500 if batch==1 else 100),median_ns=100.,p95_ns=100.)
          for name in ('affine_flow','node_structured') for batch in (1,128) for h in (.005,.01,.04)]
    validate_profiles(rows,[2026100800])
    with pytest.raises(ValueError,match='coverage'):
        validate_profiles(rows[:-1],[2026100800])
    damaged=copy.deepcopy(rows);damaged[0]['median_ns']=50.
    with pytest.raises(ValueError,match='raw'):
        validate_profiles(damaged,[2026100800])


def test_training_reproduction_compares_canonical_tensors_and_nominal_selection():
    first={'weight':np.array([1.,-2.]),'bias':np.array([.1])}
    assert retrain_metrics(first,first,.2,.2)['pass']
    changed={**first,'weight':np.array([1.,-2.1])}
    assert not retrain_metrics(first,changed,.2,.2)['pass']
    assert not retrain_metrics(first,first,.2,.21)['pass']
    with pytest.raises(ValueError,match='tensor'):
        retrain_metrics(first,{'weight':first['weight']},.2,.2)


def test_control_claim_requires_holm_and_retains_numerical_uncertainty():
    from poc.affine_confirmation import holm_control_comparisons
    b=np.array([1.,1.1,1.2,1.3,1.4])
    good=paired_comparison(b,b-np.array([.10,.12,.11,.13,.12]),1,0.)
    unresolved=paired_comparison(b,b-np.array([.1,-.1,.05,-.05,.01]),1,0.)
    rows=holm_control_comparisons([good,unresolved],1,0.)
    assert rows[0]['holm']['supported']
    assert not rows[1]['holm']['supported']
    assert rows[0]['holm']['alpha']==.0125
    rows=holm_control_comparisons([good,good],1,.1)
    assert not any(r['holm']['supported'] for r in rows)


def test_control_refinement_envelope_uses_absolute_costs_and_full_subset():
    from poc.affine_confirmation import refinement_envelope
    configs=[('affine_flow',5000000),('node_structured',5000000),('node_structured',40000000)]
    seeds=[2026100800,2026100801]
    old=[episode(name,step,seed,0,1.) for seed in seeds for name,step in configs]
    new=copy.deepcopy(old)
    for r in new:
        r['realized_cost']=1.01;r['phase']='replication'
    result=refinement_envelope(old,new,seeds)
    assert result['absolute_cost_envelope']==pytest.approx(.01)
    assert result['converged']
    new[0]['realized_cost']=1.03
    assert not refinement_envelope(old,new,seeds)['converged']
    with pytest.raises(ValueError,match='coverage'):
        refinement_envelope(old,new[:-1],seeds)


def test_coarse_affine_case_gets_paired_coverage_and_three_contrast_holm():
    from poc.affine_confirmation import holm_control_comparisons
    configs=[('affine_flow',5000000),('node_structured',5000000),
             ('node_structured',40000000),('affine_flow',40000000)]
    seeds=[2026100800,2026100801]
    rows=[episode(name,step,seed,task,1.) for seed in seeds for task in range(2) for name,step in configs]
    summary=control_summary(rows,seeds,2,configurations=configs)
    assert summary['episode_count']==16
    with pytest.raises(ValueError,match='coverage'):
        control_summary(rows[:-1],seeds,2,configurations=configs)
    b=np.array([1.,1.1,1.2,1.3,1.4]);m=b-np.array([.1,.12,.11,.13,.12])
    pair=paired_comparison(b,m,1,0.)
    result=holm_control_comparisons([pair,pair,pair],1,0.)
    assert result[0]['holm']['alpha']==pytest.approx(.025/3)
    assert all(r['holm']['supported'] for r in result)
