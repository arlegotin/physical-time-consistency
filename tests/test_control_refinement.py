"""Finer-plant jobs must preserve the exact frozen controller and paired task."""
import pytest
from poc.io import atomic_json,sha256_file,read_json


def private_freeze(root):
    jobs=[]
    for a in ('direct','lagged'):
        for m in ('mlp_dt','node_generic','power_correct'):
            for s in range(5):
                for h in (40_000_000,5_000_000):
                    for e in range(20):
                        cell=f'{a}/{m}/{s}/{h}/{e}'
                        path=root/'requests'/f'{a}_{m}_{s}_{h}_{e}.json'
                        request=dict(schema_version=1,phase='confirmatory',model_id=m,actuator=a,seed=s,
                            episode_id=e,condition='clean',model_partition_ns=[h]*(40_000_000//h),
                            actuator_interval_ns=40_000_000,physics_step_ns=250_000,power_mode=None,
                            model_manifest=f'fits/{a}_{m}_{s}/manifest.json',manifest_sha256='a'*64,
                            checkpoint_sha256='b'*64,planner_settings=dict(candidate_count=128,iterations=4,
                            elite_count=16,initial_std=1.,std_floor=.1),resource_projection={'fixture':True})
                        atomic_json(path,request)
                        jobs.append(dict(cell_id=cell,path=str(path.relative_to(root)),sha256=sha256_file(path)))
    return dict(state='complete',expected_episodes=1200,jobs=jobs)


def test_refinement_requests_use_first_five_tasks_and_only_change_physics_tick(tmp_path):
    from poc.control_refinement import refinement_requests
    freeze=private_freeze(tmp_path)
    rows=refinement_requests(tmp_path,freeze)
    assert len(rows)==300
    jobs={j['cell_id']:j for j in freeze['jobs']}
    assert {r['request']['episode_id'] for r in rows}==set(range(5))
    for row in rows:
        before=read_json(tmp_path/jobs[row['parent_cell_id']]['path'])
        assert row['request']=={**before,'physics_step_ns':125_000}
        assert row['cell_id']!=row['parent_cell_id']
        assert row['parent_request_sha256']==jobs[row['parent_cell_id']]['sha256']


def test_full_refinement_requests_do_not_replace_the_registered_subset(tmp_path):
    from poc.control_refinement import refinement_requests
    freeze=private_freeze(tmp_path)
    rows=refinement_requests(tmp_path,freeze,scope='full')
    assert len(rows)==1200
    assert len({r['cell_id'] for r in rows})==1200


@pytest.mark.parametrize('mutation',['missing','duplicate','hash','phase','step'])
def test_refinement_cannot_be_created_from_incomplete_or_changed_primary_requests(tmp_path,mutation):
    from poc.control_refinement import refinement_requests
    freeze=private_freeze(tmp_path)
    if mutation=='missing':freeze['jobs'].pop()
    elif mutation=='duplicate':freeze['jobs'][-1]=freeze['jobs'][0]
    else:
        job=freeze['jobs'][0];path=tmp_path/job['path'];request=read_json(path)
        if mutation=='hash':request['seed']=1
        elif mutation=='phase':request['phase']='exploratory'
        else:request['physics_step_ns']=125_000
        atomic_json(path,request)
        if mutation!='hash':job['sha256']=sha256_file(path)
    with pytest.raises(ValueError):refinement_requests(tmp_path,freeze)


def test_refinement_prepare_refuses_a_partial_primary_batch_before_writing_jobs(tmp_path):
    from poc.control_refinement import prepare
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='running',expected_episodes=1200,completed_episodes=1,episodes=[]))
    with pytest.raises(ValueError,match='complete primary'):
        prepare(tmp_path)
    assert not (tmp_path/'config/control_refinement_subset.json').exists()


def test_refinement_runner_reuses_verified_receipts_and_keeps_refined_raw_cost(tmp_path,monkeypatch):
    from test_control_confirmation_batch import completed_episode
    from poc.control_confirmation_batch import episode_receipt
    import poc.control_refinement as module
    first,manifest=completed_episode(tmp_path,0)
    prior=episode_receipt(tmp_path,first,manifest)
    path=tmp_path/'reports/control_refinement_subset_batch.json'
    atomic_json(path,dict(state='interrupted',registry_sha256='f'*64,episodes=[prior]))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('A verified complete job must not relaunch'))
    result=module.run_refinement_jobs(tmp_path,[first],'f'*64,scope='subset',resume=True)
    assert result['state']=='complete' and result['completed_episodes']==1
    assert result['episodes'][0]['summary']['realized_cost']==1.
    assert result['scientific_decision']=='not_evaluated'


def test_partial_refinement_receipt_never_becomes_a_reused_cost(tmp_path,monkeypatch):
    from test_control_confirmation_batch import completed_episode
    from poc.control_confirmation_batch import episode_receipt
    import poc.control_refinement as module
    first,manifest=completed_episode(tmp_path,0)
    prior=episode_receipt(tmp_path,first,manifest)
    (manifest.parent/'COMPLETED').unlink()
    atomic_json(tmp_path/'reports/control_refinement_subset_batch.json',
                dict(state='interrupted',registry_sha256='f'*64,episodes=[prior]))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Invalid prior evidence fails before launch'))
    with pytest.raises(ValueError,match='completion marker'):
        module.run_refinement_jobs(tmp_path,[first],'f'*64,scope='subset',resume=True)


def test_interrupted_refinement_keeps_completed_jobs_and_child_diagnostics(tmp_path,monkeypatch):
    from test_control_confirmation_batch import completed_episode
    from types import SimpleNamespace
    import poc.control_refinement as module
    first,manifest=completed_episode(tmp_path,0)
    second,_=completed_episode(tmp_path,1)
    monkeypatch.setattr(module.subprocess,'run',lambda argv,**k:SimpleNamespace(
        returncode=0 if argv[-1]==first['path'] else 1,
        stdout=str(manifest)+'\n' if argv[-1]==first['path'] else '',stderr='interrupted test'))
    with pytest.raises(RuntimeError,match='interrupted test'):
        module.run_refinement_jobs(tmp_path,[first,second],'f'*64,scope='subset',resume=False)
    receipt=read_json(tmp_path/'reports/control_refinement_subset_batch.json')
    assert receipt['state']=='interrupted' and receipt['completed_episodes']==1
    assert 'interrupted test' in (tmp_path/'reports/control_refinement_attempts.jsonl').read_text()


def test_reverification_of_complete_refinement_keeps_the_measured_receipt_hash(tmp_path,monkeypatch):
    from test_control_confirmation_batch import completed_episode
    from types import SimpleNamespace
    import poc.control_refinement as module
    job,manifest=completed_episode(tmp_path,0)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:
                        SimpleNamespace(returncode=0,stdout=str(manifest)+'\n',stderr=''))
    first=module.run_refinement_jobs(tmp_path,[job],'f'*64,scope='subset',resume=False)
    path=tmp_path/'reports/control_refinement_subset_batch.json';original=sha256_file(path)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:pytest.fail('Completed refinement cannot relaunch'))
    assert module.run_refinement_jobs(tmp_path,[job],'f'*64,scope='subset',resume=True)==first
    assert sha256_file(path)==original


def test_full_prepare_revalidates_existing_raw_audit_without_rewriting_subset_dependency(tmp_path,monkeypatch):
    import poc.control_refinement as refinement
    import poc.control_utility as utility
    import poc.control_audit_closure as closure
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='complete',expected_episodes=1200,completed_episodes=1200,episodes=[{}]*1200))
    raw_path=tmp_path/'reports/control_confirmation_raw_audit.json'
    atomic_json(raw_path,dict(state='complete',fixture='Already pinned by the subset freeze'))
    before=sha256_file(raw_path);calls=[]
    monkeypatch.setattr(utility,'audit_batch',lambda root:pytest.fail('A second audit write changes the subset dependency'))
    monkeypatch.setattr(closure,'verify_primary_audit',lambda root:calls.append(root),raising=False)
    atomic_json(tmp_path/'config/control_confirmation_cells.json',{})
    def after_audit(*args,**kwargs):
        raise RuntimeError('Reached request derivation after read-only audit')
    monkeypatch.setattr(refinement,'refinement_requests',after_audit)
    with pytest.raises(RuntimeError,match='Reached request derivation'):
        refinement.prepare(tmp_path,scope='full')
    assert calls==[tmp_path.resolve()] and sha256_file(raw_path)==before
