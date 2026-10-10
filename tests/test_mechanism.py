import copy
import numpy as np
import pytest

from poc.mechanism import development_cells, confirmation_cells, development_screen, sensitivity_contrasts
from poc.milestones.m30 import transition as development_transition
from poc.milestones.mr import transition as rejection_transition


def test_explicit_development_and_eventual_confirmation_cells():
    cells = development_cells()
    assert len(cells) == 30 and len({c['cell_id'] for c in cells}) == 30
    assert {c['seed'] for c in cells} == {9000}
    assert sum(c['condition'] == 'A' and c['loss_mode'] == 'shared' for c in cells) == 10
    full = confirmation_cells()
    assert len(full) == 180 and len({c['cell_id'] for c in full}) == 180
    assert {c['seed'] for c in full} == {0,1,2,3,4}
    assert all(c['model_id'] != 'power_low' for c in full)


def metrics(primary,gain=1.):
    return {str(h):{'primary_error':primary,'gain_q_median':gain,'gain_v_median':gain}
            for h in (5_000_000,10_000_000)}


def development_observations():
    result = {}
    for actuator in ('direct','lagged'):
        for model in ('mlp_dt','gru_dt','node_generic','power_correct'):
            result[(actuator,'A','shared',model,None)] = {s:metrics(.3 if model == 'power_correct' else 1.) for s in ('cold','warm')}
        for mode,error in (('low',2.),('direct',.4 if actuator == 'direct' else .7),('lagged',.4)):
            result[(actuator,'F','one_step','power_correct',mode)] = {s:metrics(error) for s in ('cold','warm')}
    return result


def test_single_development_seed_can_only_screen_or_route_to_confirmation():
    observations = development_observations()
    assert development_screen(observations)['outcome'] == 'GO'
    assert development_transition('GO',[])['permitted_next_actions'][0]['milestone_id'] == 'M40'
    observations[('lagged','A','shared','power_correct',None)]['warm']['5000000']['primary_error'] = .9
    result = development_screen(observations)
    assert result['outcome'] == 'INCONCLUSIVE'
    assert development_transition(result['outcome'],[])['permitted_next_actions'][0]['milestone_id'] == 'MR'
    assert result.get('terminal_verdict') is None


def test_missing_cell_or_bad_gain_cannot_screen_positive():
    observations = development_observations()
    del observations[('direct','A','shared','gru_dt',None)]
    with pytest.raises(ValueError,match='missing'):
        development_screen(observations)
    observations = development_observations()
    observations[('direct','A','shared','power_correct',None)]['cold']['5000000']['gain_q_median'] = .1
    assert development_screen(observations)['outcome'] == 'INCONCLUSIVE'


def test_rejection_subset_shares_full_family_and_never_claims_pass():
    next_step = rejection_transition('GO',[])
    assert next_step['terminal_verdict'] is None
    assert next_step['permitted_next_actions'][0]['milestone_id'] == 'M40'
    failed = rejection_transition('NO-GO',[])
    assert failed['terminal_verdict'] == 'POC FAIL'
    assert failed['permitted_next_actions'][0]['milestone_id'] == 'MF'
    assert rejection_transition('INCONCLUSIVE',[])['permitted_next_actions'][0]['branch'] == 'precision'


def test_registered_history_cache_repair_cannot_expand_its_change_scope():
    from poc.milestones.mx import validate_request
    request = {'contract':{str(i):'specified' for i in range(1,13)},'branch':'repair',
       'repair_kind':'history_cache','changed_files':['poc/history.py','poc/models/sequence.py',
             'tests/test_history.py','tests/test_models.py','tests/test_training.py'],
       'request_id':'MX-history-0003','unresolved_question':'Does immutable-input caching preserve exact training?',
       'counterexample':'Measured repeated physical-history work','invalidated_run_ids':['interrupted'],
       'regression_tests':['tests/test_history.py','tests/test_models.py','tests/test_training.py'],
       'resource_projection':{'wall_seconds':15},'return_milestone':'M30','protocol_version':'PTCO-1.0'}
    assert validate_request(request) == request
    request['changed_files'].append('poc/metrics.py')
    with pytest.raises(ValueError,match='files|scope'):
        validate_request(request)


def test_paired_sensitivity_contrast_and_numerical_envelope_preserve_seed_order():
    from poc.criteria import expand_criteria
    criterion = next(c for c in expand_criteria() if c['criterion_id'].startswith('K02.relative/')
                     and c['dimensions']['contrast'] == 'relative')
    d = criterion['dimensions']
    method = np.array([.1,.2,.3,.4,.5])
    baseline = np.array([1.,1.1,1.2,1.3,1.4])
    records = {}
    for seed in range(5):
        for model,x in (('power_correct',method[seed]),(d['competitor'],baseline[seed])):
            records[(d['actuator'],'A','shared',model,None,seed)] = {
                'metrics':{d['stratum']:{str(d['h_ns']):{
                  'primary_error':x,'primary_error_lower':x-.01,'primary_error_upper':x+.02}}}}
    summary = sensitivity_contrasts(criterion,records,[0,1,2,3,4])
    np.testing.assert_allclose(summary['values'],.5*baseline-method)
    assert summary['numerical_envelope']['lower'] == pytest.approx(.025)
    assert summary['numerical_envelope']['upper'] == pytest.approx(.02)
    rho = method.mean()/baseline.mean()
    assert summary['ratio_numerical_envelope']['lower'] == pytest.approx((method.mean()+.02)/(baseline.mean()-.01)-rho)
    assert summary['ratio_numerical_envelope']['upper'] == pytest.approx(rho-(method.mean()-.01)/(baseline.mean()+.02))
    assert summary['validity'] == 'valid'
    with pytest.raises(ValueError,match='missing'):
        sensitivity_contrasts(criterion,records,[0,1,2,3,4,5])


def test_unresolved_numerical_baseline_error_cannot_supply_ratio_precision():
    from poc.criteria import expand_criteria
    criterion = next(c for c in expand_criteria() if c['criterion_id'].startswith('K02.relative/')
                     and c['dimensions']['contrast'] == 'relative')
    d = criterion['dimensions']; records = {}
    for seed in range(5):
        for model,value in (('power_correct',1.+seed*.01),(d['competitor'],.01)):
            records[(d['actuator'],'A','shared',model,None,seed)] = {'metrics':{d['stratum']:{str(d['h_ns']):{
                'primary_error':value,'primary_error_lower':max(0,value-.02),'primary_error_upper':value+.02}}}}
    assert sensitivity_contrasts(criterion,records,range(5))['ratio_numerical_envelope'] is None


def test_underqualified_nominal_fit_and_solver_sensitive_comparator_are_unresolved():
    from poc.criteria import expand_criteria
    from poc.criterion_gates import evaluate_criterion
    criterion = next(c for c in expand_criteria() if c['criterion_id'].startswith('K02.relative/')
                     and c['dimensions']['contrast']=='relative' and c['dimensions']['competitor']=='node_generic')
    d=criterion['dimensions']; records={}
    for seed in range(5):
        for model,error in (('power_correct',.4+seed*.001),('node_generic',1.+seed*.001)):
            records[(d['actuator'],'A','shared',model,None,seed)]={'training_qualified':True,
                'metrics':{d['stratum']:{str(d['h_ns']):{'primary_error':error,
                    'primary_error_lower':error-.001,'primary_error_upper':error+.001}}}}
    key=(d['actuator'],'A','shared','node_generic',None,2)
    records[key]['training_qualified']=False
    summary=sensitivity_contrasts(criterion,records,range(5))
    assert summary['qualification_status']=='unresolved'
    assert evaluate_criterion(criterion,summary,1)['status']=='unresolved'
    records[key]['training_qualified']=True
    for seed in range(5):
        records[(d['actuator'],'A','shared','node_generic',None,seed)]['solver_metrics']={
            d['stratum']:{str(d['h_ns']):{'primary_error':.2}}}
    summary=sensitivity_contrasts(criterion,records,range(5))
    assert summary['qualification_status']=='unresolved' and summary['needs_branch']=='solver'
    assert evaluate_criterion(criterion,summary,1)['status']=='unresolved'
    for seed in range(5):
        records[(d['actuator'],'A','shared','node_generic',None,seed)]['solver_metrics'][d['stratum']][str(d['h_ns'])]['primary_error']=1.+seed*.001
    assert sensitivity_contrasts(criterion,records,range(5))['validity']=='valid'


def test_actual_gate_records_mr_transition_and_cache_restores_pointer(tmp_path,monkeypatch):
    from pathlib import Path
    import shutil
    import poc.gates
    import poc.contracts
    import poc.milestones.m30
    from poc.io import atomic_json,read_json
    project = Path(__file__).resolve().parents[1]
    (tmp_path/'schemas').mkdir()
    shutil.copyfile(project/'schemas/decision.schema.json',tmp_path/'schemas/decision.schema.json')
    folder = tmp_path/'evidence/M30/fixture'
    folder.mkdir(parents=True)
    record = {'milestone_id':'M30','run_id':'fixture','phase':'exploratory',
              'code_tree_sha256':'a'*64,'configuration_sha256':'b'*64}
    atomic_json(folder/'manifest.json',record)
    atomic_json(folder/'validity.json',{'validity_status':'valid','checks':[]})
    # This tests the decision-schema adapter, not any experiment validity.
    monkeypatch.setattr(poc.gates,'verify_run',lambda *_:None)
    monkeypatch.setattr(poc.contracts,'verify_run',lambda *_:None)
    monkeypatch.setattr(poc.milestones.m30,'evaluate',lambda *_:('INCONCLUSIVE','valid',[
        {'criterion_id':'fixture','status':'unresolved','estimate':None,'bounds':None,'threshold':.25,
         'operator':'>=','reason':'Synthetic weak screen','pass_bounds':{'unused_debug_detail':1}}]))
    decision = poc.gates.decide('G30',[record],tmp_path)
    assert decision['permitted_next_actions'][0]['milestone_id'] == 'MR'
    assert 'pass_bounds' not in decision['criterion_evaluations'][0]
    pointer = read_json(tmp_path/'decisions/next_action.json')
    assert pointer['permitted_next_actions'] == decision['permitted_next_actions']
    atomic_json(tmp_path/'decisions/next_action.json',{'synthetic_interrupted_pointer':True})
    assert poc.gates.decide('G30',[record],tmp_path) == decision
    assert read_json(tmp_path/'decisions/next_action.json')['permitted_next_actions'] == decision['permitted_next_actions']
