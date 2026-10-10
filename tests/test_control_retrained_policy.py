"""Repeated training preserves tasks and checks costs without adding statistical looks."""
import copy

import pytest
from poc.io import atomic_json,read_json,sha256_file
from test_control_refinement import private_freeze
from test_control_replay import known_episode


def fit_receipt():
    return dict(state='complete',expected_fits=12,completed_fits=12,diagnosis_required=False,fits=[
        dict(cell_id=f'{a}__A__shared__{m}__seed{s}',manifest=f'fresh/{a}/{m}/{s}/manifest.json',
            manifest_sha256='c'*64,checkpoint_sha256='d'*64,
            comparison=dict(status='pass',diagnosis_required=False))
        for a in ('direct','lagged') for m in ('mlp_dt','node_generic','power_correct') for s in (0,4)])


def test_retrained_control_uses_all_twenty_tasks_both_fixed_seeds_and_clocks(tmp_path):
    from poc.control_retrained_policy import retrained_requests
    primary=private_freeze(tmp_path);fits=fit_receipt();rows=retrained_requests(tmp_path,primary,fits)
    assert len(rows)==480 and len({r['cell_id'] for r in rows})==480
    assert {r['request']['seed'] for r in rows}=={0,4}
    assert {r['request']['episode_id'] for r in rows}==set(range(20))
    parents={j['cell_id']:j for j in primary['jobs']};fresh={j['cell_id']:j for j in fits['fits']}
    for row in rows:
        before=read_json(tmp_path/parents[row['parent_cell_id']]['path']);request=row['request']
        fit=fresh[f"{before['actuator']}__A__shared__{before['model_id']}__seed{before['seed']}"]
        assert request==dict(before,phase='replication',model_manifest=fit['manifest'],
            manifest_sha256=fit['manifest_sha256'],checkpoint_sha256=fit['checkpoint_sha256'])


@pytest.mark.parametrize('mutation',['missing_primary','duplicate_primary','missing_fit','duplicate_fit','failed_fit','partial_fit'])
def test_retrained_policy_cannot_select_or_omit_a_primary_task_or_fresh_fit(tmp_path,mutation):
    from poc.control_retrained_policy import retrained_requests
    primary=private_freeze(tmp_path);fits=fit_receipt()
    if mutation=='missing_primary':primary['jobs'].pop()
    elif mutation=='duplicate_primary':primary['jobs'][-1]=primary['jobs'][0]
    elif mutation=='missing_fit':fits['fits'].pop()
    elif mutation=='duplicate_fit':fits['fits'][-1]=fits['fits'][0]
    elif mutation=='failed_fit':fits['fits'][0]['comparison']['status']='outside_tolerance'
    else:fits['state']='running'
    with pytest.raises(ValueError):retrained_requests(tmp_path,primary,fits)


def summaries():
    first=known_episode()['summary'];fresh=copy.deepcopy(first);fresh['phase']='replication'
    fresh['checkpoint_hash']='b'*64
    return first,fresh


def test_retrained_policy_cost_is_compared_with_original_tolerance_and_failure_accounting():
    from poc.control_retrained_policy import compare_control
    first,fresh=summaries();result=compare_control(first,fresh)
    assert result['status']=='pass' and result['relative_cost_difference']==0.
    assert result['new_independent_training_clusters']==0
    fresh['realized_cost']*=1.002
    assert compare_control(first,fresh)['status']=='outside_tolerance'
    fresh['realized_cost']=first['realized_cost'];fresh['success']=not first['success']
    assert compare_control(first,fresh)['status']=='outside_tolerance'


@pytest.mark.parametrize('mutation',['target','seed','phase','clock','nan','missing','negative'])
def test_invalid_retrained_control_is_not_an_acceptable_or_numerical_match(mutation):
    from poc.control_retrained_policy import compare_control
    first,fresh=summaries()
    if mutation=='target':fresh['target'][0]+=.1
    elif mutation=='seed':fresh['seed']=1
    elif mutation=='phase':fresh['phase']='confirmatory'
    elif mutation=='clock':fresh['physics_step_ns']=125_000
    elif mutation=='nan':fresh['realized_cost']=float('nan')
    elif mutation=='missing':fresh['complete']=False
    else:fresh['realized_cost']=-1
    with pytest.raises(ValueError):compare_control(first,fresh)


def stability_fixture():
    from test_control_utility import private_episodes,registered_design,envelopes
    from poc.control_utility import seed_costs,utility_bounds
    primary=private_episodes();design=registered_design();numerics=envelopes(design)
    repeated=[dict(copy.deepcopy(r),phase='replication') for r in primary if r['seed'] in (0,4)]
    result=utility_bounds(seed_costs(primary,design),design,1,numerics)
    return primary,repeated,design,result,numerics


def test_reproduced_seed_substitution_preserves_original_cluster_count_and_statistical_look():
    from poc.control_retrained_policy import criterion_stability
    primary,repeated,design,result,numerics=stability_fixture();before=copy.deepcopy(primary)
    audit=criterion_stability(primary,repeated,design,result,numerics)
    assert audit['status']=='pass' and audit['criteria_unchanged'] is True
    assert audit['new_independent_training_clusters']==0 and audit['alpha_spent']==0
    assert audit['look_index']==1 and audit['recomputed']['criteria']==result['criteria']
    assert primary==before


def test_retraining_that_changes_criterion_decisions_requires_diagnosis():
    from poc.control_retrained_policy import criterion_stability
    primary,repeated,design,result,numerics=stability_fixture()
    for row in repeated:
        if row['model_id']=='power_correct':row['realized_cost']+=4.
    audit=criterion_stability(primary,repeated,design,result,numerics)
    assert audit['status']=='outside_tolerance' and audit['criteria_unchanged'] is False
    assert audit['diagnosis_required'] is True


def test_incomplete_reproduced_task_matrix_cannot_certify_criterion_stability():
    from poc.control_retrained_policy import criterion_stability
    primary,repeated,design,result,numerics=stability_fixture();repeated.pop()
    with pytest.raises(ValueError):criterion_stability(primary,repeated,design,result,numerics)


def test_retrained_control_prepare_rejects_pending_weights_before_writing_requests(tmp_path):
    from poc.control_retrained_policy import prepare
    atomic_json(tmp_path/'reports/control_retraining.json',dict(state='running'))
    with pytest.raises(ValueError,match='Complete.*training'):prepare(tmp_path)
    assert not (tmp_path/'config/control_retrained_policy_jobs.json').exists()


def test_retrained_control_runner_reuses_raw_receipts_and_preserves_cost_comparison(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import poc.control_retrained_policy as module
    parent=tmp_path/'parent/manifest.json';parent.parent.mkdir();parent.write_text('{}')
    fresh=tmp_path/'fresh/manifest.json';fresh.parent.mkdir();fresh.write_text('{}')
    request=tmp_path/'request.json';atomic_json(request,{'private':'validated separately'})
    job=dict(cell_id='fixed-policy-cell',path='request.json',sha256=sha256_file(request),
        parent_manifest='parent/manifest.json',parent_manifest_sha256=sha256_file(parent))
    registry=dict(jobs=[job],input_hashes={})
    monkeypatch.setattr(module,'verify_registry',lambda root,value:value)
    monkeypatch.setattr(module,'_observe_policy',lambda root,job,path:dict(cell_id=job['cell_id'],
        manifest=str(path.relative_to(root)),manifest_sha256=sha256_file(path),
        comparison={'status':'pass','diagnosis_required':False},summary={'actual':'repeated physical episode'}))
    monkeypatch.setattr('poc.io._process_start',lambda pid:'private-metadata')
    calls=[]
    def child(*args,**kwargs):
        calls.append(args);return SimpleNamespace(returncode=0,stdout=str(fresh)+'\n',stderr='')
    monkeypatch.setattr(module.subprocess,'run',child)
    result=module.run_policies(tmp_path,registry,'a'*64)
    assert result['state']=='complete' and result['criterion_stability_checked'] is False
    digest=sha256_file(tmp_path/'reports/control_retrained_policy_batch.json')
    assert module.run_policies(tmp_path,registry,'a'*64,resume=True)==result
    assert len(calls)==1 and sha256_file(tmp_path/'reports/control_retrained_policy_batch.json')==digest
    fresh.write_text('changed')
    with pytest.raises(ValueError):module.run_policies(tmp_path,registry,'a'*64,resume=True)


def test_retrained_policy_cli_consumes_actual_primary_audit_contract_without_reserving_a_look(tmp_path,monkeypatch):
    import poc.control_retrained_policy as module
    import poc.control_audit_closure as closure
    primary,repeated,design,result,numerics=stability_fixture()
    atomic_json(tmp_path/'config/authorization.json',dict(local_compute_authorized=True,paid_services_authorized=False))
    atomic_json(tmp_path/'locks/environment.json',dict(versions={}))
    atomic_json(tmp_path/'config/control_retrained_policy_jobs.json',dict(input_hashes={}))
    atomic_json(tmp_path/'config/control_confirmation_design.json',design)
    atomic_json(tmp_path/'reports/control_confirmation_result.json',result)
    atomic_json(tmp_path/'reports/control_confirmation_numerical.json',dict(envelopes=numerics))
    atomic_json(tmp_path/'reports/control_retrained_policy_batch.json',dict(private_fixture=True))
    monkeypatch.setattr(module,'verify_registry',lambda root,value:value)
    monkeypatch.setattr(module,'run_policies',lambda *a,**k:dict(diagnosis_required=False,
        episodes=[dict(summary=r) for r in repeated]))
    monkeypatch.setattr(closure,'verify_primary_audit',lambda root:dict(design=design,
        entries=[dict(summary=r) for r in primary],audit={'private':'actual envelope shape'}))
    assert module.main(['--root',str(tmp_path),'--resume'])==0
    observed=read_json(tmp_path/'reports/control_retrained_policy_result.json')
    assert observed['criteria_unchanged'] is True and observed['alpha_spent']==0
    assert observed['original_ptco_terminal_verdict'] is None
    assert not (tmp_path/'decisions/control_utility_looks').exists()
