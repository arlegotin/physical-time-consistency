"""Raw evidence and provisional simulator screens, never empirical fixtures."""
import importlib.util
import numpy as np
import pytest
from poc.calibration_study import prospective_recipe


def module():
    assert importlib.util.find_spec('poc.calibration_report'), 'Reporting stage is missing'
    from poc import calibration_report
    return calibration_report


@pytest.mark.parametrize('matrix,neural,physical', [('pilot',18,3),('development',180,15)])
def test_prospective_coverage_does_not_depend_on_outcomes(matrix,neural,physical):
    rows = module().expected_coordinates(matrix)
    assert len(rows) == neural+physical
    assert sum(c['model_id']=='physics_fit' for c in rows)==physical
    assert len({str(c) for c in rows})==len(rows)
    assert not any(c['model_id']=='physics_nominal' for c in rows)


def metrics(q=.01,v=.1,pq=.03,pv=.3,failures=0,envelope=0.):
    return [{'component':c,'rmse':r,'p95':p,'conservative_rmse_upper':r+envelope,
             'conservative_p95_upper':p+envelope,'failure_count':failures}
            for c,r,p in [('q',q,pq),('v',v,pv)]]


def outcome(**changes):
    return {'decision_failure':False,'no_true_feasible_bank':False,'regret':.01,
            'regret_upper':.01,'zero_command_truth_cost':1.,**changes}


def test_exact_physical_thresholds_failures_and_conservative_bounds():
    screen=module().configuration_screen
    assert screen(metrics(),[outcome()])['status']=='pass'
    for changes in [dict(q=.0100001),dict(v=.100001),dict(pq=.0300001),dict(pv=.300001),dict(failures=1),dict(envelope=1e-9)]:
        assert screen(metrics(**changes),[outcome()])['status']=='fail'
    assert screen(metrics(),[outcome(regret=.010001,regret_upper=.010001)])['status']=='fail'
    decisions=[outcome() for _ in range(100)]
    decisions[0]=outcome(decision_failure=True,regret=None,regret_upper=None)
    assert screen(metrics(),decisions)['status']=='pass'
    decisions[1]=decisions[0]
    assert screen(metrics(),decisions)['status']=='fail'
    for changes in [dict(zero_command_truth_cost=1e-8),dict(no_true_feasible_bank=True)]:
        assert screen(metrics(),[outcome(**changes)])['status']=='unresolved'


def test_budget_thresholds_keep_never_passing_and_nonmonotone_curves():
    curve=module().threshold_curve
    result=curve([{'budget_s':b,'status':s} for b,s in [(30,'fail'),(120,'pass'),(600,'fail')]])
    assert result['bracket_s']==[30,120] and result['nonmonotone']
    assert len(result['rows'])==3
    assert curve([{'budget_s':b,'status':'fail'} for b in [30,120,600]])['bracket_s']==[600,None]
    assert curve([{'budget_s':30,'status':'pass'}])['bracket_s']==[0,30]
    assert curve([{'budget_s':30,'status':'unresolved'},{'budget_s':120,'status':'pass'}])['bracket_s']==[0,120]


def test_paired_bootstrap_averages_replicas_without_inventing_sessions():
    m=module()
    rows=[{'session_id':s,'model_id':model,'budget_s':30,'query_step_ns':5_000_000,
           'initialization_id':i,'value':value+offset} for s,value in enumerate([1.,2.,4.,8.,16.])
           for model,offset in [('a',0.),('b',10.)] for i in range(4)]
    result=m.cluster_bootstrap(rows,['model_id','budget_s','query_step_ns'],['value'],prospective_recipe()['aggregation'])
    assert result['independent_session_count']==5 and result['draws']==10000
    assert result==m.cluster_bootstrap(rows*2,['model_id','budget_s','query_step_ns'],['value'],prospective_recipe()['aggregation'])
    a,b=sorted(result['groups'],key=lambda x:x['coordinate']['model_id'])
    assert a['session_values']['value']==[1.,2.,4.,8.,16.]
    assert np.asarray(b['intervals']['value'])-np.asarray(a['intervals']['value'])==pytest.approx([10.,10.])
    assert result['powered_claim'] is False


def config(model,budget=30,time=.005,status='pass',regret=.001):
    return {'model_id':model,'budget_s':budget,'query_step_ns':5_000_000,'screen':{'status':status},
            'decision_failure_fraction':0.,'mean_valid_regret':regret,'mean_regret_upper':regret,
            'scorer_wall_median_s':time,'optimization_limited':False}


@pytest.mark.parametrize('rows,want', [
    ([config('physics_fit'),config('affine_flow',120)],'no_practical_headroom'),
    ([config('affine_flow'),config('affine_rk4'),config('node_structured',120)],'representation_only'),
    ([config('affine_flow'),config('node_actuator_aware')],'cheap_solver_explains_gain'),
    ([config('affine_flow'),config('node_structured',120)],'proceed_to_deadline_spec'),
    ([config('affine_flow',time=.1),config('node_structured',time=.01)],'no_advantage'),
])
def test_gate_scopes_physics_representation_cheap_solver_and_slow_scoring(rows,want):
    result=module().gate_decision({'coverage':{'complete':True},'empirical_claims_allowed':True,
                                  'configurations':rows})
    assert result['decision']==want and result['reasons']
    assert module().gate_decision({'coverage':{'complete':False},'configurations':rows})['decision']=='unresolved'
    assert module().gate_decision({'coverage':{'complete':True},'empirical_claims_allowed':False,
                                  'configurations':rows})['decision']=='unresolved'


def test_matrix_timing_selection_ignores_expansion_and_filtered_overlap_without_fastest_selection():
    m=module()
    pilot=m.expected_coordinates('pilot')
    development=m.expected_coordinates('development')
    def product(name,coords):
        return ({'job_key':name},None,{'stage':'timing','request':{'coordinates':coords}})
    p=product('pilot',pilot);d=product('development',development);subset=product('filtered',pilot[:1])
    selected,ignored=m.select_timing_products([subset,d,p],pilot)
    assert [x[0]['job_key'] for x in selected]==['pilot']
    assert len(ignored)==2
    selected,ignored=m.select_timing_products([subset,p,d],development)
    assert [x[0]['job_key'] for x in selected]==['development']
    with pytest.raises(ValueError,match='Ambiguous'):
        m.select_timing_products([p,product('different',pilot)],pilot)
    with pytest.raises(ValueError,match='Overlapping'):
        m.select_timing_products([product('a',pilot[:3]),product('b',pilot[2:5])],pilot)


@pytest.mark.parametrize('matrix', ['pilot','development'])
def test_legacy_disjoint_timing_subsets_still_close_selected_membership(matrix):
    m=module()
    wanted=m.expected_coordinates(matrix)
    subsets=[({'job_key':name},None,{'stage':'timing','request':{'coordinates':coords}})
             for name,coords in [('first',wanted[:3]),('remaining',wanted[3:])]]
    selected,ignored=m.select_timing_products(subsets,wanted)
    assert [e[0]['job_key'] for e in selected]==['first','remaining']
    assert ignored==[]
    assert sum(len(e[2]['request']['coordinates']) for e in selected)==len(wanted)


def test_fitted_physics_can_remove_headroom_even_when_flow_never_passes():
    m=module()
    assert m.gate_decision({'coverage':{'complete':True},'empirical_claims_allowed':True,
        'configurations':[config('physics_fit'),config('affine_flow',status='fail',time=.02)]})['decision']=='no_practical_headroom'


def test_threshold_ratios_remain_interval_censored():
    m=module()
    flow=m.threshold_curve([{'budget_s':30,'status':'fail'},{'budget_s':120,'status':'pass'},{'budget_s':600,'status':'pass'}])
    alternative=m.threshold_curve([{'budget_s':30,'status':'fail'},{'budget_s':120,'status':'fail'},{'budget_s':600,'status':'pass'}])
    ratio=m.threshold_ratio(flow,alternative)
    assert ratio['observed_budget_ratio']==5 and ratio['ratio_bracket']==[1.,20.]
    never=m.threshold_curve([{'budget_s':b,'status':'fail'} for b in [30,120,600]])
    assert m.threshold_ratio(flow,never)['observed_budget_ratio'] is None
    assert m.threshold_ratio(flow,never)['ratio_bracket']==[5.,None]


def test_sub_envelope_regret_difference_cannot_support_continuation():
    m=module()
    f=config('affine_flow');other=config('node_structured',120)
    other['mean_regret_lower']=0.
    assert m.gate_decision({'coverage':{'complete':True},'empirical_claims_allowed':True,
                          'configurations':[f,other]})['decision']!='proceed_to_deadline_spec'


def test_development_screen_uses_replica_average_and_preserves_any_prediction_failure():
    m=module()
    rows=[]
    for initialization,q in enumerate([.009,.011]):
        screen=m.configuration_screen(metrics(q=q),[outcome()])
        rows.append({'session_id':0,'model_id':'affine_flow','budget_s':30,'query_step_ns':5_000_000,
            'initialization_id':initialization,'q_rmse':q,'v_rmse':.1,'q_p95':.03,'v_p95':.3,
            'q_conservative_rmse_upper':q,'v_conservative_rmse_upper':.1,
            'q_conservative_p95_upper':.03,'v_conservative_p95_upper':.3,
            'scorer_wall_median_s':.005,'scorer_wall_p95_s':.006,'screen':screen,
            'optimization_limited':False,**{k:screen[k] for k in ['decision_failure_fraction','mean_valid_regret','mean_regret_upper','mean_regret_lower','mean_zero_command_cost']}})
    configs,uncertainty=m.summarize_configurations(rows,prospective_recipe()['aggregation'])
    assert configs[0]['q_rmse']==pytest.approx(.01) and configs[0]['screen']['status']=='pass'
    assert configs[0]['independent_session_count']==1 and configs[0]['initializations_per_session']==2
    rows[0]['screen']['truth_feasibility_unresolved']=True
    assert m.summarize_configurations(rows,prospective_recipe()['aggregation'])[0][0]['screen']['status']=='unresolved'
    rows[0]['screen']['truth_feasibility_unresolved']=False
    rows[0]['screen']['prediction_failure_count']=1
    assert m.summarize_configurations(rows,prospective_recipe()['aggregation'])[0][0]['screen']['status']=='fail'


def test_earlier_screen_can_beat_a_never_passing_learned_alternative():
    m=module()
    rows=[config('affine_flow',120),*[config('node_structured',budget,status='fail') for budget in [30,120,600]]]
    decision=m.gate_decision({'coverage':{'complete':True},'empirical_claims_allowed':True,'configurations':rows})
    assert decision['decision']=='proceed_to_deadline_spec'
    assert 'earlier_screen_budget' in decision['reasons']


@pytest.mark.parametrize('alternative_budget,alternative_time', [(600,.005),(120,.01)])
@pytest.mark.parametrize('missing_decision_metrics', [False,True])
def test_continuation_cannot_ignore_unresolved_learned_comparator(
        alternative_budget,alternative_time,missing_decision_metrics):
    m=module()
    unresolved=config('node_actuator_aware',30,time=None,status='unresolved')
    if missing_decision_metrics:
        unresolved.update(decision_failure_fraction=None,mean_valid_regret=None,mean_regret_upper=None)
    rows=[config('affine_flow',120),config('node_structured',alternative_budget,time=alternative_time),unresolved]
    result=m.gate_decision({'coverage':{'complete':True},'empirical_claims_allowed':True,'configurations':rows})
    assert result['decision']=='unresolved'
    assert result['reasons']==['alternative_screen_or_matching_timing_unresolved']
    assert result['stage_c_launched'] is False


@pytest.mark.parametrize('matching_model,want', [
    ('physics_fit','no_practical_headroom'),
    ('affine_rk4','unresolved'),
    ('node_actuator_aware','cheap_solver_explains_gain'),
])
def test_resolved_scoped_negative_outcomes_remain_visible_with_unresolved_comparator(matching_model,want):
    m=module()
    unresolved=config('node_quadrature',30,time=None,status='unresolved')
    unresolved.update(decision_failure_fraction=None,mean_valid_regret=None,mean_regret_upper=None)
    rows=[config('affine_flow',120),config(matching_model,120),unresolved]
    result=m.gate_decision({'coverage':{'complete':True},'empirical_claims_allowed':True,'configurations':rows})
    assert result['decision']==want and result['stage_c_launched'] is False


def full_table(*passing):
    from poc.calibration_models import MODEL_IDS
    rows = [{**config(m,b,status='fail'), 'query_step_ns':h}
            for m in (*MODEL_IDS,'physics_nominal') for b in (30,120,600)
            for h in (5_000_000,10_000_000,20_000_000,40_000_000)]
    for replacement in passing:
        for i,row in enumerate(rows):
            if all(row[k]==replacement[k] for k in ('model_id','budget_s','query_step_ns')):
                rows[i]=replacement
    return {'coverage':{'complete':True},'empirical_claims_allowed':True,'configurations':rows}


@pytest.mark.parametrize('physical', ['physics_fit','physics_nominal'])
@pytest.mark.parametrize('status', ['pass','unresolved'])
def test_full_gate_requires_resolved_physical_reference(physical,status):
    result=full_table(config('affine_flow',120),config('node_structured',600),
                      config(physical,30,time=None,status=status))
    assert module().gate_decision(result)['decision']=='unresolved'


@pytest.mark.parametrize('alternative', ['node_structured','node_quadrature','physics_residual'])
def test_full_gate_keeps_lower_budget_and_other_partition_frontier(alternative):
    cheaper={**config(alternative,30,time=.002),'query_step_ns':40_000_000}
    result=full_table(config('affine_flow',120),config(alternative,120,time=.01),cheaper)
    assert module().gate_decision(result)['decision']=='no_advantage'


def test_full_gate_representation_parity_requires_advantage_over_unrestricted():
    result=full_table(config('affine_flow'),config('affine_rk4'),config('node_structured',time=.002))
    assert module().gate_decision(result)['decision']=='no_advantage'


@pytest.mark.parametrize('physical', ['physics_fit','physics_nominal'])
def test_later_flow_configuration_cannot_bypass_physical_runtime_frontier(physical):
    result=full_table(config('affine_flow',30,time=.1,regret=.005),
        config('affine_flow',120,time=.005),config(physical,120,time=.002),
        config('node_structured',120,time=.01))
    assert module().gate_decision(result)['decision']=='no_advantage'
